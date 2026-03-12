from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from db import SessionLocal, engine
from models import Base, UsageCount

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


sessions: Dict[str, Session] = {}
MAX_HISTORY = 8


def get_session(sid: str) -> Session:
    sess = sessions.get(sid)
    if not sess:
        sess = {"platform": None, "history": []}
        sessions[sid] = sess

    if "platform" not in sess:
        sess["platform"] = None
    if "history" not in sess:
        sess["history"] = []

    return sess


# -------------------------
# Free usage limit
# -------------------------
MAX_FREE_QUERIES = 2


# -------------------------
# Auth config
# -------------------------
APP_USERNAME = os.getenv("PARABLE_USERNAME")
APP_PASSWORD = os.getenv("PARABLE_PASSWORD")

if not APP_USERNAME or not APP_PASSWORD:
    raise RuntimeError("PARABLE_USERNAME and PARABLE_PASSWORD must be set")

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_SECONDS = 30 * 60  # 30 minutes

_login_attempts: Dict[str, int] = {}
_login_lockouts: Dict[str, float] = {}

AUTH_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
_authed_sids: Dict[str, float] = {}


# -------------------------
# Cookie helpers
# -------------------------
def set_sid_cookie(resp: JSONResponse | HTMLResponse, sid: str) -> None:
    resp.set_cookie(
        key="sid",
        value=sid,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


def set_auth_cookie(resp: JSONResponse | HTMLResponse) -> None:
    resp.set_cookie(
        key="parable_auth",
        value="1",
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=AUTH_TTL_SECONDS,
    )


def clear_auth_cookie(resp: JSONResponse | HTMLResponse) -> None:
    resp.delete_cookie(
        key="parable_auth",
        path="/",
        samesite="lax",
        secure=True,
    )


# -------------------------
# SID resolver
# -------------------------
def get_sid(request: Request) -> str:
    sid = request.headers.get("x-parable-sid")
    if sid and len(sid) >= 10:
        return sid

    sid = request.cookies.get("sid")
    if sid and len(sid) >= 10:
        return sid

    return str(uuid.uuid4())


# -------------------------
# Logged-in helpers
# -------------------------
def _sid_is_authed(sid: str) -> bool:
    exp = _authed_sids.get(sid)
    if not exp:
        return False

    if exp <= time.time():
        _authed_sids.pop(sid, None)
        return False

    return True


def is_logged_in(request: Request, sid: str) -> bool:
    cookie_ok = request.cookies.get("parable_auth") == "1"
    return cookie_ok and _sid_is_authed(sid)


def mark_sid_authed(sid: str) -> None:
    _authed_sids[sid] = time.time() + AUTH_TTL_SECONDS


def unmark_sid_authed(sid: str) -> None:
    _authed_sids.pop(sid, None)


# -------------------------
# Usage count helpers (DB-backed)
# -------------------------
def get_usage_key(request: Request, sid: str) -> str:
    if is_logged_in(request, sid):
        return f"user:{APP_USERNAME}"
    return f"sid:{sid}"


def get_query_count(key: str) -> int:
    db = SessionLocal()
    try:
        row = db.query(UsageCount).filter(UsageCount.key == key).first()
        return row.count if row else 0
    finally:
        db.close()


def increment_query_count(key: str) -> int:
    db = SessionLocal()
    try:
        row = db.query(UsageCount).filter(UsageCount.key == key).first()

        if not row:
            row = UsageCount(key=key, count=1)
            db.add(row)
            db.commit()
            db.refresh(row)
            return row.count

        row.count += 1
        db.commit()
        db.refresh(row)
        return row.count
    finally:
        db.close()


def enforce_free_limit(sid: str, request: Request) -> None:
    if is_logged_in(request, sid):
        return

    key = get_usage_key(request, sid)
    count = increment_query_count(key)

    if count > MAX_FREE_QUERIES:
        raise HTTPException(status_code=402, detail="Login required")


# -------------------------
# Lockout helpers
# -------------------------
def is_locked_out(sid: str) -> Tuple[bool, int]:
    unlock_at = _login_lockouts.get(sid)
    if not unlock_at:
        return False, 0

    remaining = int(unlock_at - time.time())
    if remaining <= 0:
        _login_lockouts.pop(sid, None)
        _login_attempts.pop(sid, None)
        return False, 0

    return True, remaining


def register_failed_login(sid: str) -> Tuple[int, bool, int]:
    attempts = _login_attempts.get(sid, 0) + 1
    _login_attempts[sid] = attempts

    if attempts >= MAX_LOGIN_ATTEMPTS:
        unlock_at = time.time() + LOCKOUT_SECONDS
        _login_lockouts[sid] = unlock_at
        remaining = int(unlock_at - time.time())
        return attempts, True, remaining

    return attempts, False, 0


def reset_login_attempts(sid: str) -> None:
    _login_attempts.pop(sid, None)
    _login_lockouts.pop(sid, None)


# -------------------------
# API models
# -------------------------
class ChatIn(BaseModel):
    message: str
    image_url: Optional[str] = None


class LoginIn(BaseModel):
    username: str
    password: str


# -------------------------
# Manifest
# -------------------------
@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(
        str(BASE_DIR / "manifest.webmanifest"),
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

    locked, remaining = is_locked_out(sid)
    if locked:
        minutes = max(1, remaining // 60)
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
        reset_login_attempts(sid)
        mark_sid_authed(sid)

        resp = JSONResponse({"ok": True, "message": "Logged in"})
        set_sid_cookie(resp, sid)
        set_auth_cookie(resp)
        return resp

    attempts, locked_now, remaining = register_failed_login(sid)

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

    resp = JSONResponse({"ok": True})
    set_sid_cookie(resp, sid)
    clear_auth_cookie(resp)
    return resp


@app.get("/api/me")
def me_api(request: Request):
    sid = get_sid(request)
    logged_in = is_logged_in(request, sid)
    usage_key = get_usage_key(request, sid)
    used = get_query_count(usage_key)

    resp = JSONResponse(
        {
            "ok": True,
            "logged_in": logged_in,
            "used_count": used,
            "free_limit": MAX_FREE_QUERIES,
            "remaining_free": max(0, MAX_FREE_QUERIES - used),
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

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="file required")

    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"

    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 8MB)")

    filename = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / filename
    out_path.write_bytes(data)

    full_url = str(request.base_url).rstrip("/") + f"/static/uploads/{filename}"
    return {"ok": True, "url": full_url}


# -------------------------
# Chat endpoint
# -------------------------
@app.post("/api/chat")
def chat_api(payload: ChatIn, request: Request):
    message = (payload.message or "").strip()
    image_url = (payload.image_url or "").strip() or None

    if not message and not image_url:
        return JSONResponse({"error": "message or image required"}, status_code=400)

    sid = get_sid(request)
    enforce_free_limit(sid, request)

    usage_key = get_usage_key(request, sid)
    used_count = get_query_count(usage_key)

    sess = get_session(sid)
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
            input_messages.append(
                {"role": "system", "content": f"User is on {platform}."}
            )

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

        ai_response = get_client().responses.create(
            model="gpt-4.1-mini",
            input=input_messages,
        )

        answer = ai_response.output_text or ""

        if message:
            history.append({"role": "user", "content": message})
        elif image_url:
            history.append({"role": "user", "content": "[Uploaded a photo]"})

        history.append({"role": "assistant", "content": answer})
        sess["history"] = history[-MAX_HISTORY:]

        resp = JSONResponse(
            {
                "answer": answer,
                "logged_in": is_logged_in(request, sid),
                "used_count": used_count,
                "free_limit": MAX_FREE_QUERIES,
                "remaining_free": max(0, MAX_FREE_QUERIES - used_count),
            }
        )
        set_sid_cookie(resp, sid)
        return resp

    except HTTPException:
        raise
    except Exception:
        resp = JSONResponse(
            {"error": "AI service error. Please try again."},
            status_code=502,
        )
        set_sid_cookie(resp, sid)
        return resp


@app.get("/ping")
def ping():
    return {"ok": True}


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
    """


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
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


# -------------------------
# Chat UI
# -------------------------
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    sid = get_sid(request)
    logged_in = is_logged_in(request, sid)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Parable Chat</title>
  <link rel="manifest" href="/manifest.webmanifest">
  <meta name="theme-color" content="#ea580c">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <link rel="apple-touch-icon" href="/static/icon-192.png">
  <style>
    :root {{
      --navy:#020617;
      --orange:#ea580c;
      --bg:#f4f6fb;
      --card:#fff;
      --soft:#f8fafc;
      --line:#e5e7eb;
      --danger:#b91c1c;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      font-family:system-ui,-apple-system,"Segoe UI",Arial,sans-serif;
      background:radial-gradient(1200px 600px at 50% 0%, #fff 0%, var(--bg) 55%);
      margin:0;
    }}
    .wrap {{
      max-width:920px;
      margin:0 auto;
      padding:28px 16px 40px;
    }}
    .card {{
      background:var(--card);
      border-radius:18px;
      padding:18px;
      box-shadow:0 14px 40px rgba(2,6,23,.10);
      border:2px solid rgba(234,88,12,.55);
      position:relative;
    }}
    .card:before {{
      content:"";
      position:absolute;
      inset:10px;
      border-radius:14px;
      border:1px solid rgba(234,88,12,.22);
      pointer-events:none;
    }}
    h1 {{
      margin:0 0 6px;
      font-size:22px;
      color:var(--navy);
      letter-spacing:.2px;
    }}
    .sub {{
      margin:0 0 14px;
      color:#475569;
    }}
    .topbar {{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      margin-bottom:12px;
      flex-wrap:wrap;
    }}
    .status {{
      font-size:14px;
      color:#475569;
    }}
    .smallbtn {{
      padding:8px 12px;
      border-radius:10px;
      border:1px solid var(--line);
      background:#fff;
      cursor:pointer;
      font-weight:600;
    }}
    .chatbox {{
      height:520px;
      overflow:auto;
      background:linear-gradient(180deg,#fff 0%,#fbfbfd 100%);
      border:1px solid var(--line);
      border-radius:14px;
      padding:12px;
    }}
    .row {{
      display:flex;
      margin:10px 0;
    }}
    .bubble {{
      padding:10px 12px;
      border-radius:14px;
      max-width:85%;
      white-space:pre-wrap;
      line-height:1.35;
    }}
    .you {{
      justify-content:flex-end;
    }}
    .you .bubble {{
      background:var(--navy);
      color:#fff;
    }}
    .bot {{
      justify-content:flex-start;
    }}
    .bot .bubble {{
      background:var(--soft);
      color:#111827;
      border:1px solid rgba(234,88,12,.22);
    }}
    .quick {{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      margin:10px 0 0;
    }}
    .q {{
      padding:8px 10px;
      border-radius:999px;
      border:1px solid var(--line);
      background:#fff;
      color:var(--navy);
      cursor:pointer;
      font-weight:600;
    }}
    .q:hover {{
      border-color:var(--orange);
    }}
    .controls {{
      display:flex;
      gap:10px;
      margin-top:12px;
      align-items:center;
      flex-wrap:wrap;
    }}
    input#msg,
    input.login-input {{
      flex:1;
      padding:12px;
      border-radius:14px;
      border:1px solid #d1d5db;
      outline:none;
      width:100%;
      min-width:220px;
    }}
    input#msg:focus,
    input.login-input:focus {{
      border-color:rgba(234,88,12,.8);
      box-shadow:0 0 0 4px rgba(234,88,12,.15);
    }}
    button.action {{
      min-width:110px;
      padding:11px 16px;
      border-radius:12px;
      border:1px solid rgba(234,88,12,.35);
      background:var(--orange);
      color:#fff;
      cursor:pointer;
      font-weight:700;
    }}
    button.action:disabled {{
      opacity:.45;
      cursor:not-allowed;
    }}
    .iconbtn {{
      min-width:56px;
      padding:11px 12px;
      border-radius:12px;
      border:1px solid rgba(234,88,12,.35);
      background:#fff;
      cursor:pointer;
      font-weight:800;
      color:var(--navy);
    }}
    .iconbtn:hover {{
      border-color:var(--orange);
    }}
    .overlay {{
      position:fixed;
      inset:0;
      background:rgba(2,6,23,.55);
      display:none;
      align-items:center;
      justify-content:center;
      padding:16px;
      z-index:9999;
    }}
    .overlay.show {{
      display:flex;
    }}
    .login-card {{
      width:min(420px, 100%);
      background:#fff;
      border-radius:18px;
      padding:18px;
      box-shadow:0 20px 50px rgba(2,6,23,.25);
      border:2px solid rgba(234,88,12,.35);
    }}
    .login-card h2 {{
      margin:0 0 8px;
      color:var(--navy);
      font-size:20px;
    }}
    .login-card p {{
      margin:0 0 12px;
      color:#475569;
      font-size:14px;
    }}
    .login-grid {{
      display:grid;
      gap:10px;
    }}
    .login-actions {{
      display:flex;
      gap:10px;
      margin-top:12px;
      flex-wrap:wrap;
    }}
    .error {{
      color:var(--danger);
      font-size:14px;
      min-height:20px;
    }}
    .preview {{
      margin-top:10px;
      display:none;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
      padding:10px;
      border:1px dashed rgba(234,88,12,.35);
      border-radius:12px;
      background:#fff;
    }}
    .preview img {{
      max-height:72px;
      border-radius:10px;
      border:1px solid var(--line);
    }}
    .muted {{
      color:#64748b;
      font-size:13px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card" id="card">
      <div class="topbar">
        <div>
          <h1>Parable Chatbot</h1>
          <p class="sub">Ask a question about your phone.</p>
        </div>
        <div>
          <span id="loginStatus" class="status">{'Logged in' if logged_in else 'Free mode'}</span>
          <button id="openLoginBtn" class="smallbtn" type="button">{'Account' if logged_in else 'Log in'}</button>
        </div>
      </div>

      <div id="usageStatus" class="status" style="margin-bottom:10px;"></div>

      <div id="chatbox" class="chatbox"></div>

      <div class="quick">
        <button class="q" type="button" onclick="quick('iPhone')">I'm on iPhone</button>
        <button class="q" type="button" onclick="quick('Android')">I'm on Android</button>
      </div>

      <div id="preview" class="preview">
        <img id="previewImg" alt="preview" />
        <div>
          <div><strong>Photo ready to send</strong></div>
          <div class="muted" id="previewNote">It will upload when you press Send.</div>
        </div>
        <button id="clearPhoto" class="smallbtn" type="button">Remove</button>
      </div>

      <div class="controls">
        <input id="msg" placeholder="Type here (or choose Voice first)..." />
        <input id="photoInput" type="file" accept="image/*" style="display:none" />
        <button id="attach" class="iconbtn" type="button" title="Upload photo">📎</button>
        <button id="mic" class="action" type="button">🎤 Speak</button>
        <button id="btn" class="action" type="button">Send</button>
      </div>
    </div>
  </div>

  <div id="loginOverlay" class="overlay">
    <div class="login-card">
      <h2>Log in</h2>
      <p>Enter your username and password to keep chatting.</p>

      <div class="login-grid">
        <input id="loginUser" class="login-input" placeholder="Username" autocomplete="username" />
        <input id="loginPass" class="login-input" type="password" placeholder="Password" autocomplete="current-password" />
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
      sid = (crypto.randomUUID ? crypto.randomUUID() : (Date.now() + "-" + Math.random()));
      localStorage.setItem(key, sid);
    }}
    return sid;
  }}

  function updateUsageUi(data) {{
    if (!data) return;

    if (data.logged_in) {{
      usageStatus.textContent = "Logged in account active.";
      return;
    }}

    usageStatus.textContent = `Free questions left: ${{Math.max(0, data.remaining_free)}} of ${{data.free_limit}}`;
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
      console.error("Usage load failed", e);
    }}
  }}

  function addBubble(text, who) {{
    const row = document.createElement("div");
    row.className = "row " + who;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    chatbox.appendChild(row);
    chatbox.scrollTop = chatbox.scrollHeight;

    if (who === "bot" && preferVoice === true) {{
      speak(text);
    }}
  }}

  function greetOnce() {{
    if (greeted) return;
    greeted = true;
    addBubble("Hi, and welcome to Parable Chat.", "bot");
    addBubble("Would you like to use text or voice?", "bot");
    addBubble("Tip: Tap 📎 to upload a photo (like an error message or scam pop-up).", "bot");
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
    loginStatus.textContent = loggedIn ? "Logged in" : "Free mode";
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
      addBubble("You are logged in. You can keep chatting now.", "bot");
      loginPass.value = "";
      loadUsage();
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
      addBubble(data.error || "Upload failed.", "bot");
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
      chatbox.scrollTop = chatbox.scrollHeight;
    }}

    input.value = "";
    input.focus();
    btn.disabled = true;

    try {{
      const photoUrl = await uploadSelectedPhotoIfNeeded();

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

      if (res.status === 402) {{
        addBubble("Free limit reached — please log in to continue.", "bot");
        showLogin();
        loadUsage();
        return;
      }}

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
  
