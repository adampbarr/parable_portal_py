from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from db import Base, engine

app = FastAPI()

# Create DB tables on startup
Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("parable")


# -------------------------
# Helpers / config
# -------------------------
def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


COOKIE_SECURE = env_bool("COOKIE_SECURE", True)
MAX_HISTORY = 8
MAX_MESSAGE_LENGTH = 1500
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

STATE_CLEANUP_INTERVAL_SECONDS = 15 * 60
_last_cleanup_at = 0.0


def now_ts() -> float:
    return time.time()


def maybe_cleanup_state() -> None:
    global _last_cleanup_at

    now = now_ts()
    if now - _last_cleanup_at < STATE_CLEANUP_INTERVAL_SECONDS:
        return

    _last_cleanup_at = now

    # Clean expired auth sessions
    expired_auth = [sid for sid, exp in _authed_sids.items() if exp <= now]
    for sid in expired_auth:
        _authed_sids.pop(sid, None)

    # Clean expired login lockouts
    expired_lockouts = [key for key, unlock_at in _login_lockouts.items() if unlock_at <= now]
    for key in expired_lockouts:
        _login_lockouts.pop(key, None)
        _login_attempts.pop(key, None)

    # Clean expired rate limit windows
    for bucket in (_chat_rate_windows, _upload_rate_windows):
        stale_keys = []
        for key, timestamps in bucket.items():
            fresh = [ts for ts in timestamps if now - ts <= 3600]
            if fresh:
                bucket[key] = fresh
            else:
                stale_keys.append(key)
        for key in stale_keys:
            bucket.pop(key, None)

    # Clean very old sessions
    stale_sessions = [sid for sid, sess in sessions.items() if now - sess.get("last_seen", now) > 7 * 24 * 3600]
    for sid in stale_sessions:
        sessions.pop(sid, None)

    logger.info(
        "State cleanup complete | expired_auth=%s expired_lockouts=%s stale_sessions=%s",
        len(expired_auth),
        len(expired_lockouts),
        len(stale_sessions),
    )


def is_api_path(path: str) -> bool:
    return path.startswith("/api/")


def api_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def is_reasonable_sid(sid: str) -> bool:
    if not sid:
        return False
    if len(sid) < 10 or len(sid) > 128:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(ch in allowed for ch in sid)


# -------------------------
# OpenAI client
# -------------------------
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


# -------------------------
# Session memory
# -------------------------
class Turn(TypedDict):
    role: str
    content: str


class Session(TypedDict):
    platform: Optional[str]
    history: List[Turn]
    last_seen: float


sessions: Dict[str, Session] = {}


def get_session(sid: str) -> Session:
    sess = sessions.get(sid)
    if not sess:
        sess = {"platform": None, "history": [], "last_seen": now_ts()}
        sessions[sid] = sess

    if "platform" not in sess:
        sess["platform"] = None
    if "history" not in sess:
        sess["history"] = []
    sess["last_seen"] = now_ts()

    return sess


# -------------------------
# Auth config
# -------------------------
APP_USERNAME = os.getenv("PARABLE_USERNAME")
APP_PASSWORD = os.getenv("PARABLE_PASSWORD")

if not APP_USERNAME or not APP_PASSWORD:
    raise RuntimeError("PARABLE_USERNAME and PARABLE_PASSWORD must be set")

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_SECONDS = 30 * 60  # 30 minutes
AUTH_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

_login_attempts: Dict[str, int] = {}
_login_lockouts: Dict[str, float] = {}
_authed_sids: Dict[str, float] = {}


# -------------------------
# Simple in-memory rate limiting
# -------------------------
CHAT_RATE_LIMIT_COUNT = 20
CHAT_RATE_LIMIT_WINDOW_SECONDS = 5 * 60

UPLOAD_RATE_LIMIT_COUNT = 10
UPLOAD_RATE_LIMIT_WINDOW_SECONDS = 10 * 60

_chat_rate_windows: Dict[str, List[float]] = {}
_upload_rate_windows: Dict[str, List[float]] = {}


def rate_limit_check(
    bucket: Dict[str, List[float]],
    key: str,
    max_count: int,
    window_seconds: int,
) -> Tuple[bool, int]:
    now = now_ts()
    history = bucket.get(key, [])
    history = [ts for ts in history if now - ts <= window_seconds]

    if len(history) >= max_count:
        retry_after = max(1, int(window_seconds - (now - history[0])))
        bucket[key] = history
        return False, retry_after

    history.append(now)
    bucket[key] = history
    return True, 0


def get_login_key(request: Request, sid: str) -> str:
    return f"{get_client_ip(request)}:{sid}"


def get_rate_key(request: Request, sid: str) -> str:
    return f"{get_client_ip(request)}:{sid}"


# -------------------------
# Cookie helpers
# -------------------------
def set_sid_cookie(resp: JSONResponse | HTMLResponse | RedirectResponse, sid: str) -> None:
    resp.set_cookie(
        key="sid",
        value=sid,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


def set_auth_cookie(resp: JSONResponse | HTMLResponse | RedirectResponse) -> None:
    resp.set_cookie(
        key="parable_auth",
        value="1",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
        max_age=AUTH_TTL_SECONDS,
    )


def clear_auth_cookie(resp: JSONResponse | HTMLResponse | RedirectResponse) -> None:
    resp.delete_cookie(
        key="parable_auth",
        path="/",
        samesite="lax",
        secure=COOKIE_SECURE,
    )


# -------------------------
# SID resolver
# -------------------------
def get_sid(request: Request) -> str:
    header_sid = request.headers.get("x-parable-sid", "").strip()
    if is_reasonable_sid(header_sid):
        return header_sid

    cookie_sid = (request.cookies.get("sid") or "").strip()
    if is_reasonable_sid(cookie_sid):
        return cookie_sid

    return uuid.uuid4().hex


# -------------------------
# Logged-in helpers
# -------------------------
def _sid_is_authed(sid: str) -> bool:
    exp = _authed_sids.get(sid)
    if not exp:
        return False

    if exp <= now_ts():
        _authed_sids.pop(sid, None)
        return False

    return True


def is_logged_in(request: Request, sid: str) -> bool:
    cookie_ok = request.cookies.get("parable_auth") == "1"
    return cookie_ok and _sid_is_authed(sid)


def mark_sid_authed(sid: str) -> None:
    _authed_sids[sid] = now_ts() + AUTH_TTL_SECONDS


def unmark_sid_authed(sid: str) -> None:
    _authed_sids.pop(sid, None)


# -------------------------
# Lockout helpers
# -------------------------
def is_locked_out(login_key: str) -> Tuple[bool, int]:
    unlock_at = _login_lockouts.get(login_key)
    if not unlock_at:
        return False, 0

    remaining = int(unlock_at - now_ts())
    if remaining <= 0:
        _login_lockouts.pop(login_key, None)
        _login_attempts.pop(login_key, None)
        return False, 0

    return True, remaining


def register_failed_login(login_key: str) -> Tuple[int, bool, int]:
    attempts = _login_attempts.get(login_key, 0) + 1
    _login_attempts[login_key] = attempts

    if attempts >= MAX_LOGIN_ATTEMPTS:
        unlock_at = now_ts() + LOCKOUT_SECONDS
        _login_lockouts[login_key] = unlock_at
        remaining = int(unlock_at - now_ts())
        return attempts, True, remaining

    return attempts, False, 0


def reset_login_attempts(login_key: str) -> None:
    _login_attempts.pop(login_key, None)
    _login_lockouts.pop(login_key, None)


# -------------------------
# API models
# -------------------------
class ChatIn(BaseModel):
    message: str = Field(default="", max_length=MAX_MESSAGE_LENGTH)
    image_url: Optional[str] = None


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


# -------------------------
# Middleware
# -------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    maybe_cleanup_state()
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store" if is_api_path(request.url.path) else "public, max-age=300"

    return response


# -------------------------
# Exception handlers
# -------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error | path=%s | detail=%s", request.url.path, exc.errors())
    if is_api_path(request.url.path):
        return api_error("Invalid request.", 422)
    return HTMLResponse("<h1>Invalid request</h1>", status_code=422)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    if exc.status_code >= 500:
        logger.error("HTTPException | path=%s | status=%s | detail=%s", request.url.path, exc.status_code, detail)
    else:
        logger.info("HTTPException | path=%s | status=%s | detail=%s", request.url.path, exc.status_code, detail)

    if is_api_path(request.url.path):
        return api_error(detail, exc.status_code)
    return HTMLResponse(f"<h1>{detail}</h1>", status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error | path=%s", request.url.path)
    if is_api_path(request.url.path):
        return api_error("Something went wrong. Please try again.", 500)
    return HTMLResponse("<h1>Something went wrong. Please try again.</h1>", status_code=500)


# -------------------------
# Manifest
# -------------------------
@app.get("/manifest.webmanifest")
def manifest():
    manifest_path = BASE_DIR / "manifest.webmanifest"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    return FileResponse(
        str(manifest_path),
        media_type="application/manifest+json",
    )


# -------------------------
# Auth endpoints
# -------------------------
@app.post("/api/login")
def login_api(payload: LoginIn, request: Request):
    username = (payload.username or "").strip()
    password = payload.password or ""
    sid = get_sid(request)
    login_key = get_login_key(request, sid)

    locked, remaining = is_locked_out(login_key)
    if locked:
        minutes = max(1, remaining // 60)
        logger.warning("Login blocked | sid=%s ip=%s remaining=%ss", sid, get_client_ip(request), remaining)
        resp = JSONResponse(
            {
                "ok": False,
                "error": f"Too many wrong tries. Try again in about {minutes} minutes.",
            },
            status_code=429,
        )
        set_sid_cookie(resp, sid)
        return resp

    if username == APP_USERNAME and password == APP_PASSWORD:
        reset_login_attempts(login_key)
        mark_sid_authed(sid)

        logger.info("Login success | sid=%s ip=%s", sid, get_client_ip(request))

        resp = JSONResponse(
            {
                "ok": True,
                "message": "Logged in",
                "redirect": "/chat",
            }
        )
        set_sid_cookie(resp, sid)
        set_auth_cookie(resp)
        return resp

    attempts, locked_now, remaining = register_failed_login(login_key)
    logger.warning(
        "Login failed | sid=%s ip=%s attempts=%s locked_now=%s",
        sid,
        get_client_ip(request),
        attempts,
        locked_now,
    )

    if locked_now:
        minutes = max(1, remaining // 60)
        resp = JSONResponse(
            {
                "ok": False,
                "error": f"Too many wrong tries. Login locked for about {minutes} minutes.",
            },
            status_code=429,
        )
        set_sid_cookie(resp, sid)
        return resp

    tries_left = MAX_LOGIN_ATTEMPTS - attempts
    resp = JSONResponse(
        {
            "ok": False,
            "error": f"Wrong username or password. {tries_left} attempt(s) left.",
        },
        status_code=401,
    )
    set_sid_cookie(resp, sid)
    return resp


@app.post("/api/logout")
def logout_api(request: Request):
    sid = get_sid(request)
    unmark_sid_authed(sid)

    logger.info("Logout | sid=%s ip=%s", sid, get_client_ip(request))

    resp = JSONResponse({"ok": True})
    set_sid_cookie(resp, sid)
    clear_auth_cookie(resp)
    return resp


@app.get("/api/me")
def me_api(request: Request):
    sid = get_sid(request)
    logged_in = is_logged_in(request, sid)

    resp = JSONResponse(
        {
            "ok": True,
            "logged_in": logged_in,
        }
    )
    set_sid_cookie(resp, sid)
    return resp


# -------------------------
# Upload endpoint
# -------------------------
@app.post("/api/upload-image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    sid = get_sid(request)

    if not is_logged_in(request, sid):
        raise HTTPException(status_code=401, detail="Login required")

    allowed, retry_after = rate_limit_check(
        _upload_rate_windows,
        get_rate_key(request, sid),
        UPLOAD_RATE_LIMIT_COUNT,
        UPLOAD_RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many uploads. Please wait about {retry_after} seconds and try again.",
        )

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="File required")

    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File was empty")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 8MB)")

    filename = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / filename
    out_path.write_bytes(data)

    full_url = str(request.base_url).rstrip("/") + f"/static/uploads/{filename}"
    logger.info(
        "Image uploaded | sid=%s ip=%s file=%s bytes=%s",
        sid,
        get_client_ip(request),
        filename,
        len(data),
    )
    return {"ok": True, "url": full_url}


# -------------------------
# Chat endpoint
# -------------------------
@app.post("/api/chat")
def chat_api(payload: ChatIn, request: Request):
    sid = get_sid(request)
    sess = get_session(sid)

    message = (payload.message or "").strip()
    image_url = (payload.image_url or "").strip() or None

    if not message and not image_url:
        return JSONResponse({"ok": False, "error": "Message or image required."}, status_code=400)

    if len(message) > MAX_MESSAGE_LENGTH:
        return JSONResponse(
            {"ok": False, "error": f"Message is too long. Keep it under {MAX_MESSAGE_LENGTH} characters."},
            status_code=400,
        )

    allowed, retry_after = rate_limit_check(
        _chat_rate_windows,
        get_rate_key(request, sid),
        CHAT_RATE_LIMIT_COUNT,
        CHAT_RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Too many messages too quickly. Please wait about {retry_after} seconds and try again.",
            },
            status_code=429,
        )

    history = sess["history"]
    platform = sess["platform"]

    lower_message = message.lower() if message else ""

    if "iphone" in lower_message or "ios" in lower_message:
        sess["platform"] = "iphone"
        platform = "iphone"

    if "android" in lower_message:
        sess["platform"] = "android"
        platform = "android"

    try:
        system_text = (
            "You are Parable Smartphone Support. Talk like a calm, friendly helper.\n"
            "Rules:\n"
            "- Use simple words. No tech jargon.\n"
            "- Keep it short: 3 to 6 steps max.\n"
            "- One step per line. Start each line with a verb: Tap, Open, Turn on, Turn off, Go to, Try, Restart.\n"
            "- Ask at most one question, only if needed.\n"
            "- If you mention a setting, include the exact path like: Settings > Accessibility > Zoom.\n"
            "- Avoid acronyms. If you must use one, explain it in 3 words.\n"
            "- If scams or pop-ups: start with 'Don't click anything.'\n"
            "\n"
            "If a photo is provided, do this structure:\n"
            "1) Start with: 'What I see in your screenshot:' and list 3 to 6 bullets of exact visible details.\n"
            "2) Then: 'Is it suspicious?' and answer Yes or No with one sentence why.\n"
            "3) Then give 3 to 6 safe steps the user should do next.\n"
            "- If the screenshot is unreadable, say so and ask them to upload a clearer one.\n"
            "- End with: 'Did that work?' when appropriate.\n"
        )

        if image_url and image_url.startswith("/"):
            image_url = str(request.base_url).rstrip("/") + image_url

        user_text = message
        if not user_text and image_url:
            user_text = "Please check this screenshot. Is it suspicious? What should I do?"

        input_messages: List[dict] = [{"role": "system", "content": system_text}]

        if platform:
            input_messages.append({"role": "system", "content": f"User is on {platform}."})

        for turn in history[-MAX_HISTORY:]:
            input_messages.append(turn)

        if image_url:
            input_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            )
        else:
            input_messages.append({"role": "user", "content": user_text})

        logger.info(
            "Chat request | sid=%s ip=%s logged_in=%s has_image=%s platform=%s",
            sid,
            get_client_ip(request),
            is_logged_in(request, sid),
            bool(image_url),
            platform,
        )

        ai_response = get_client().responses.create(
            model="gpt-4.1-mini",
            input=input_messages,
        )

        answer = (ai_response.output_text or "").strip()
        if not answer:
            answer = "I had trouble answering that. Please try again."

        if message:
            history.append({"role": "user", "content": message})
        elif image_url:
            history.append({"role": "user", "content": "[Uploaded a photo]"})

        history.append({"role": "assistant", "content": answer})
        sess["history"] = history[-MAX_HISTORY:]

        resp = JSONResponse(
            {
                "ok": True,
                "answer": answer,
                "logged_in": is_logged_in(request, sid),
            }
        )
        set_sid_cookie(resp, sid)
        return resp

    except HTTPException:
        raise
    except Exception:
        logger.exception("AI service error | sid=%s ip=%s", sid, get_client_ip(request))
        resp = JSONResponse(
            {"ok": False, "error": "AI service error. Please try again."},
            status_code=502,
        )
        set_sid_cookie(resp, sid)
        return resp


# -------------------------
# Health endpoints
# -------------------------
@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "parable-portal",
        "time": int(now_ts()),
        "sessions": len(sessions),
        "authed_sessions": len(_authed_sids),
    }


# -------------------------
# Simple pages
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Parable Portal ✅</h1>
    <p><a href="/dashboard">Go to Dashboard</a></p>
    <p><a href="/chat">Go to Chat</a></p>
    <p><a href="/ping">Ping Test</a></p>
    <p><a href="/health">Health Check</a></p>
    """


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    sid = get_sid(request)
    if not is_logged_in(request, sid):
        resp = RedirectResponse(url="/chat", status_code=302)
        set_sid_cookie(resp, sid)
        return resp

    html = """
    <h1>Customer Dashboard</h1>
    <ul>
      <li>📱 Smartphone Insurance: Active</li>
      <li>🛡️ Antivirus: Active</li>
      <li>🔒 VPN: Active</li>
      <li>🧾 Identity Guard: Active</li>
    </ul>
    <p><a href="/chat">Open Chatbot</a></p>
    <p><a href="/">Back Home</a></p>
    """
    resp = HTMLResponse(html)
    set_sid_cookie(resp, sid)
    return resp


# -------------------------
# Chat UI
# -------------------------
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    sid = get_sid(request)
    logged_in = is_logged_in(request, sid)

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover"
  />
  <title>Parable Chat</title>
  <link rel="manifest" href="/manifest.webmanifest">
  <meta name="theme-color" content="#ea580c">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <link rel="apple-touch-icon" href="/static/icon-192.png">

  <style>
    :root {{
      --navy: #020617;
      --orange: #ea580c;
      --bg: #f4f6fb;
      --card: #ffffff;
      --soft: #f8fafc;
      --line: #e5e7eb;
      --danger: #b91c1c;
      --text: #111827;
      --muted: #475569;
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      height: 100%;
      overflow-x: hidden;
    }}

    body {{
      font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--bg) 55%);
      color: var(--text);
      -webkit-text-size-adjust: 100%;
      overscroll-behavior-x: none;
    }}

    img {{
      max-width: 100%;
      height: auto;
    }}

    .page {{
      width: 100%;
      max-width: 100%;
      min-height: 100dvh;
      overflow-x: hidden;
      padding: 12px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}

    .wrap {{
      width: 100%;
      max-width: 920px;
      margin: 0 auto;
    }}

    .card {{
      width: 100%;
      max-width: 100%;
      background: var(--card);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 14px 40px rgba(2, 6, 23, 0.10);
      border: 2px solid rgba(234, 88, 12, 0.55);
      position: relative;
      overflow: hidden;
    }}

    .card::before {{
      content: "";
      position: absolute;
      inset: 10px;
      border-radius: 14px;
      border: 1px solid rgba(234, 88, 12, 0.22);
      pointer-events: none;
    }}

    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      color: var(--navy);
      letter-spacing: 0.2px;
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    .sub {{
      margin: 0;
      color: var(--muted);
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    .welcome {{
      margin: 0 0 14px;
      padding: 14px 16px;
      background: #fff7ed;
      border: 1px solid rgba(234, 88, 12, 0.25);
      border-radius: 14px;
      color: var(--navy);
      font-weight: 700;
      text-align: center;
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}

    .topbar > div {{
      min-width: 0;
    }}

    .status {{
      font-size: 14px;
      color: var(--muted);
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    .smallbtn {{
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #ffffff;
      cursor: pointer;
      font-weight: 600;
      color: var(--text);
    }}

    .chatbox {{
      width: 100%;
      max-width: 100%;
      height: min(58dvh, 520px);
      min-height: 320px;
      overflow-y: auto;
      overflow-x: hidden;
      -webkit-overflow-scrolling: touch;
      background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      word-break: break-word;
      overflow-wrap: break-word;
    }}

    .row {{
      display: flex;
      margin: 10px 0;
      width: 100%;
      max-width: 100%;
    }}

    .bubble {{
      padding: 10px 12px;
      border-radius: 14px;
      max-width: 85%;
      white-space: pre-wrap;
      line-height: 1.35;
      overflow-wrap: break-word;
      word-wrap: break-word;
      word-break: break-word;
    }}

    .you {{
      justify-content: flex-end;
    }}

    .you .bubble {{
      background: var(--navy);
      color: #ffffff;
    }}

    .bot {{
      justify-content: flex-start;
    }}

    .bot .bubble {{
      background: var(--soft);
      color: var(--text);
      border: 1px solid rgba(234, 88, 12, 0.22);
    }}

    .quick {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 10px 0 0;
      width: 100%;
      max-width: 100%;
    }}

    .q {{
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--navy);
      cursor: pointer;
      font-weight: 600;
      max-width: 100%;
    }}

    .q:hover {{
      border-color: var(--orange);
    }}

    .controls {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      align-items: stretch;
      flex-wrap: wrap;
      width: 100%;
      max-width: 100%;
    }}

    .controls > * {{
      min-width: 0;
    }}

    input#msg,
    input.login-input {{
      flex: 1 1 220px;
      min-width: 0;
      width: 100%;
      max-width: 100%;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid #d1d5db;
      outline: none;
      font-size: 16px;
    }}

    input#msg:focus,
    input.login-input:focus {{
      border-color: rgba(234, 88, 12, 0.8);
      box-shadow: 0 0 0 4px rgba(234, 88, 12, 0.15);
    }}

    button.action {{
      min-width: 110px;
      padding: 11px 16px;
      border-radius: 12px;
      border: 1px solid rgba(234, 88, 12, 0.35);
      background: var(--orange);
      color: #ffffff;
      cursor: pointer;
      font-weight: 700;
      flex-shrink: 0;
    }}

    button.action:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}

    .iconbtn {{
      min-width: 56px;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid rgba(234, 88, 12, 0.35);
      background: #ffffff;
      cursor: pointer;
      font-weight: 800;
      color: var(--navy);
      flex-shrink: 0;
    }}

    .iconbtn:hover {{
      border-color: var(--orange);
    }}

    .overlay {{
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, 0.55);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 16px;
      z-index: 9999;
    }}

    .overlay.show {{
      display: flex;
    }}

    .login-card {{
      width: min(420px, 100%);
      max-width: 100%;
      background: #ffffff;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(2, 6, 23, 0.25);
      border: 2px solid rgba(234, 88, 12, 0.35);
    }}

    .login-card h2 {{
      margin: 0 0 8px;
      color: var(--navy);
      font-size: 20px;
    }}

    .login-card p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}

    .login-grid {{
      display: grid;
      gap: 10px;
    }}

    .login-actions {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }}

    .error {{
      color: var(--danger);
      font-size: 14px;
      min-height: 20px;
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    .preview {{
      margin-top: 10px;
      display: none;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      padding: 10px;
      border: 1px dashed rgba(234, 88, 12, 0.35);
      border-radius: 12px;
      background: #ffffff;
      width: 100%;
      max-width: 100%;
      overflow: hidden;
    }}

    .preview img {{
      max-height: 72px;
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      flex-shrink: 0;
    }}

    .muted {{
      color: #64748b;
      font-size: 13px;
      overflow-wrap: break-word;
      word-break: break-word;
    }}

    @media (max-width: 640px) {{
      .page {{
        padding: 8px;
      }}

      .card {{
        padding: 12px;
        border-radius: 14px;
      }}

      h1 {{
        font-size: 20px;
      }}

      .chatbox {{
        height: min(56dvh, 460px);
        min-height: 280px;
        padding: 10px;
      }}

      .bubble {{
        max-width: 92%;
      }}

      .controls {{
        gap: 8px;
      }}

      input#msg {{
        flex: 1 1 100%;
      }}

      button.action {{
        min-width: 96px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="wrap">
      <div class="card" id="card">
        <div class="topbar">
          <div>
            <h1>Parable Chatbot</h1>
            <p class="sub">Ask a question about your phone.</p>
          </div>
          <div>
            <span id="loginStatus" class="status">{'Logged in' if logged_in else 'Not logged in'}</span>
            <button id="openLoginBtn" class="smallbtn" type="button">{'Account' if logged_in else 'Log in'}</button>
          </div>
        </div>

        <div class="welcome">Thank you, How can we help you today?</div>

        <div id="usageStatus" class="status" style="margin-bottom:10px;"></div>

        <div id="chatbox" class="chatbox"></div>

        <div class="quick">
          <button class="q" type="button" onclick="quick('iPhone')">I'm on iPhone</button>
          <button class="q" type="button" onclick="quick('Android')">I'm on Android</button>
        </div>

        <div id="preview" class="preview">
          <img id="previewImg" alt="preview" />
          <div style="min-width:0;">
            <div><strong>Photo ready to send</strong></div>
            <div class="muted" id="previewNote">It will upload when you press Send.</div>
          </div>
          <button id="clearPhoto" class="smallbtn" type="button">Remove</button>
        </div>

        <div class="controls">
          <input
            id="msg"
            placeholder="Type here (or choose Voice first)..."
            autocomplete="off"
            autocapitalize="sentences"
            autocorrect="on"
            spellcheck="true"
            maxlength="1500"
          />
          <input id="photoInput" type="file" accept="image/*" style="display:none" />
          <button id="attach" class="iconbtn" type="button" title="Upload photo">📎</button>
          <button id="mic" class="action" type="button">🎤 Speak</button>
          <button id="btn" class="action" type="button">Send</button>
        </div>
      </div>
    </div>
  </div>

  <div id="loginOverlay" class="overlay">
    <div class="login-card">
      <h2>Log in</h2>
      <p>Enter your username and password.</p>

      <div class="login-grid">
        <input
          id="loginUser"
          name="username"
          class="login-input"
          type="text"
          placeholder="Username"
          autocomplete="username"
          autocapitalize="none"
          autocorrect="off"
          spellcheck="false"
          inputmode="text"
        />
        <input
          id="loginPass"
          name="password"
          class="login-input"
          type="password"
          placeholder="Password"
          autocomplete="current-password"
          autocapitalize="none"
          autocorrect="off"
          spellcheck="false"
        />
      </div>

      <div id="loginError" class="error"></div>

      <div class="login-actions">
        <button id="loginSubmit" class="action" type="button">Log in</button>
        <button id="loginClose" class="smallbtn" type="button">Close</button>
      </div>
    </div>
  </div>

<script>
  const chatbox = document.getElementById("chatbox");
  const input = document.getElementById("msg");
  const btn = document.getElementById("btn");
  const micBtn = document.getElementById("mic");
  const card = document.getElementById("card");
  const usageStatus = document.getElementById("usageStatus");

  const attachBtn = document.getElementById("attach");
  const photoInput = document.getElementById("photoInput");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("previewImg");
  const clearPhoto = document.getElementById("clearPhoto");

  const loginOverlay = document.getElementById("loginOverlay");
  const openLoginBtn = document.getElementById("openLoginBtn");
  const loginSubmit = document.getElementById("loginSubmit");
  const loginClose = document.getElementById("loginClose");
  const loginUser = document.getElementById("loginUser");
  const loginPass = document.getElementById("loginPass");
  const loginError = document.getElementById("loginError");
  const loginStatus = document.getElementById("loginStatus");

  let greeted = false;
  let recognition = null;
  let preferVoice = null;
  let selectedVoice = null;
  let micReady = false;
  let loggedIn = {str(logged_in).lower()};

  let selectedFile = null;
  let uploadedUrl = null;

  btn.disabled = true;
  micBtn.disabled = true;

  function getOrCreateSid() {{
    const key = "parable_sid";
    let sid = localStorage.getItem(key);
    if (!sid || sid.length < 10) {{
      sid = (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "") : (Date.now() + "-" + Math.random()).replace(/[^a-zA-Z0-9-_]/g, ""));
      localStorage.setItem(key, sid);
    }}
    return sid;
  }}

  function updateUsageUi(data) {{
    if (!data) return;
    usageStatus.textContent = data.logged_in ? "Logged in account active." : "Chat is ready.";
  }}

  async function loadUsage() {{
    try {{
      const res = await fetch("/api/me", {{
        headers: {{
          "X-Parable-SID": getOrCreateSid()
        }},
        credentials: "same-origin"
      }});
      const data = await res.json();
      if (res.ok) {{
        loggedIn = data.logged_in;
        updateLoginUi();
        updateUsageUi(data);
      }}
    }} catch (e) {{
      console.error("Status load failed", e);
    }}
  }}

  function scrollChatToBottom() {{
    chatbox.scrollTop = chatbox.scrollHeight;
  }}

  function addBubble(text, who) {{
    const row = document.createElement("div");
    row.className = "row " + who;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    chatbox.appendChild(row);
    scrollChatToBottom();

    if (who === "bot" && preferVoice === true) {{
      speak(text);
    }}
  }}

  function greetOnce() {{
    if (greeted) return;
    greeted = true;
    addBubble("Thank you, How can we help you today?", "bot");
    addBubble("You can type, speak, or upload a photo.", "bot");
    btn.disabled = false;
    micBtn.disabled = false;
    input.focus();
  }}

  function pickVoice() {{
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;

    const preferred = [
      "Microsoft Aria Online",
      "Microsoft Jenny Online",
      "Microsoft Guy Online",
      "Microsoft Zira",
      "Microsoft David",
      "Google US English",
      "Samantha"
    ];

    for (const name of preferred) {{
      const v = voices.find(x => x.name && x.name.includes(name));
      if (v) return v;
    }}

    return voices.find(v => (v.lang || "").startsWith("en")) || voices[0];
  }}

  function speak(text) {{
    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();

    if (!selectedVoice) {{
      selectedVoice = pickVoice();
    }}

    let t = (text || "").replace(/\\n+/g, ". ").replace(/\\s+/g, " ").trim();
    if (t.length > 260) {{
      t = t.slice(0, 260) + "...";
    }}

    const u = new SpeechSynthesisUtterance(t);
    if (selectedVoice) {{
      u.voice = selectedVoice;
    }}
    u.rate = 0.98;
    u.pitch = 1.02;
    u.volume = 1.0;

    window.speechSynthesis.speak(u);
  }}

  if ("speechSynthesis" in window) {{
    window.speechSynthesis.onvoiceschanged = () => {{
      selectedVoice = pickVoice();
    }};
  }}

  async function askMicPermission() {{
    try {{
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
        micReady = false;
        return false;
      }}

      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      stream.getTracks().forEach(track => track.stop());

      micReady = true;
      return true;
    }} catch (err) {{
      console.error("Microphone permission error:", err);
      micReady = false;
      return false;
    }}
  }}

  async function startMic() {{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {{
      addBubble("Voice input is not supported in this browser.", "bot");
      return;
    }}

    if (!micReady) {{
      const ok = await askMicPermission();
      if (!ok) {{
        addBubble("Microphone permission was blocked. Please allow microphone access and try again.", "bot");
        return;
      }}
    }}

    if (!recognition) {{
      recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = function () {{
        micBtn.disabled = true;
        micBtn.textContent = "🎤 Listening...";
      }};

      recognition.onend = function () {{
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Speak";
      }};

      recognition.onerror = function (event) {{
        console.error("Speech recognition error:", event.error);

        let message = "Microphone error. Please try again.";

        if (event.error === "not-allowed") {{
          message = "Microphone permission was denied. Please allow mic access and try again.";
          micReady = false;
        }} else if (event.error === "no-speech") {{
          message = "I did not hear anything. Try speaking again.";
        }} else if (event.error === "audio-capture") {{
          message = "No microphone was found or it is unavailable.";
        }}

        addBubble(message, "bot");
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Speak";
      }};

      recognition.onresult = function (event) {{
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
        send();
      }};
    }}

    try {{
      recognition.start();
    }} catch (err) {{
      console.error("Recognition start error:", err);
      addBubble("Could not start microphone. Please try again.", "bot");
    }}
  }}

  function showLogin() {{
    loginError.textContent = "";
    loginOverlay.classList.add("show");
    loginUser.focus();
  }}

  function hideLogin() {{
    loginOverlay.classList.remove("show");
  }}

  function updateLoginUi() {{
    loginStatus.textContent = loggedIn ? "Logged in" : "Not logged in";
    openLoginBtn.textContent = loggedIn ? "Account" : "Log in";
  }}

  async function submitLogin() {{
    const username = loginUser.value.trim();
    const password = loginPass.value;

    loginError.textContent = "";

    if (!username || !password) {{
      loginError.textContent = "Enter username and password.";
      return;
    }}

    try {{
      const res = await fetch("/api/login", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "X-Parable-SID": getOrCreateSid()
        }},
        credentials: "same-origin",
        body: JSON.stringify({{ username, password }})
      }});

      const data = await res.json();

      if (!res.ok) {{
        loginError.textContent = data.error || "Login failed.";
        return;
      }}

      loggedIn = true;
      updateLoginUi();
      hideLogin();
      loginPass.value = "";
      loadUsage();

      if (data.redirect) {{
        window.location.href = data.redirect;
        return;
      }}

      addBubble("You are logged in.", "bot");
    }} catch (e) {{
      loginError.textContent = "Login error. Try again.";
    }}
  }}

  function showPreview(file) {{
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    preview.style.display = "flex";
  }}

  function clearSelectedPhoto() {{
    selectedFile = null;
    uploadedUrl = null;
    photoInput.value = "";
    previewImg.src = "";
    preview.style.display = "none";
  }}

  async function uploadSelectedPhotoIfNeeded() {{
    if (!selectedFile) return null;
    if (uploadedUrl) return uploadedUrl;

    const fd = new FormData();
    fd.append("file", selectedFile);

    const res = await fetch("/api/upload-image", {{
      method: "POST",
      headers: {{
        "X-Parable-SID": getOrCreateSid()
      }},
      credentials: "same-origin",
      body: fd
    }});

    if (res.status === 401) {{
      addBubble("Please log in to upload photos.", "bot");
      showLogin();
      return null;
    }}

    const data = await res.json();
    if (!res.ok || !data.ok) {{
      addBubble(data.error || data.detail || "Upload failed.", "bot");
      return null;
    }}

    uploadedUrl = data.url;
    return uploadedUrl;
  }}

  async function send() {{
    const message = input.value.trim();

    greetOnce();

    if (!message && !selectedFile) return;

    const label = message || "[Photo sent]";
    addBubble(label, "you");

    if (selectedFile) {{
      const imgRow = document.createElement("div");
      imgRow.className = "row you";

      const img = document.createElement("img");
      img.src = URL.createObjectURL(selectedFile);
      img.style.maxWidth = "140px";
      img.style.maxHeight = "160px";
      img.style.objectFit = "cover";
      img.style.borderRadius = "12px";
      img.style.border = "1px solid #e5e7eb";
      img.style.cursor = "pointer";
      img.onclick = () => window.open(img.src, "_blank");

      imgRow.appendChild(img);
      chatbox.appendChild(imgRow);
      scrollChatToBottom();
    }}

    input.value = "";
    input.focus();
    btn.disabled = true;

    try {{
      const photoUrl = await uploadSelectedPhotoIfNeeded();

      if (selectedFile && !photoUrl) {{
        btn.disabled = false;
        return;
      }}

      const res = await fetch("/api/chat", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "X-Parable-SID": getOrCreateSid()
        }},
        credentials: "same-origin",
        body: JSON.stringify({{
          message: message || "",
          image_url: photoUrl
        }})
      }});

      const data = await res.json();
      addBubble(data.answer || data.error || ("HTTP " + res.status), "bot");
      updateUsageUi(data);

      if (photoUrl) {{
        clearSelectedPhoto();
      }}
    }} catch (e) {{
      addBubble("Network or server error.", "bot");
    }} finally {{
      btn.disabled = false;
    }}
  }}

  function quick(text) {{
    input.value = text;
    send();
  }}

  btn.addEventListener("click", send);

  input.addEventListener("keydown", (e) => {{
    if (e.key === "Enter") {{
      send();
    }}
  }});

  micBtn.addEventListener("click", async () => {{
    greetOnce();
    preferVoice = true;
    await startMic();
  }});

  openLoginBtn.addEventListener("click", () => {{
    if (loggedIn) {{
      addBubble("You are already logged in.", "bot");
      return;
    }}
    showLogin();
  }});

  loginSubmit.addEventListener("click", submitLogin);
  loginClose.addEventListener("click", hideLogin);

  loginPass.addEventListener("keydown", (e) => {{
    if (e.key === "Enter") {{
      submitLogin();
    }}
  }});

  attachBtn.addEventListener("click", () => {{
    greetOnce();
    photoInput.click();
  }});

  photoInput.addEventListener("change", () => {{
    const f = photoInput.files && photoInput.files[0];
    if (!f) return;
    selectedFile = f;
    uploadedUrl = null;
    showPreview(f);
  }});

  clearPhoto.addEventListener("click", clearSelectedPhoto);

  card.addEventListener("click", () => {{
    greetOnce();
    input.focus();
  }});

  window.addEventListener("load", () => {{
    updateLoginUi();
    greetOnce();
    loadUsage();
  }});

  if ("serviceWorker" in navigator) {{
    navigator.serviceWorker.register("/static/sw.js");
  }}
</script>
</body>
</html>
"""

    resp = HTMLResponse(html)
    set_sid_cookie(resp, sid)
    return resp
  
