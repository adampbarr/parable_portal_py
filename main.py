from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, Float, Integer, String, Text

from db import AuthSession, Base, ChatSession, LoginAttempt, SessionLocal, engine

app = FastAPI()

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

DEFAULT_SECURITY_PORTAL_URL = os.getenv("BITDEFENDER_PORTAL_URL", "")
DEFAULT_IDENTITY_PORTAL_URL = os.getenv("NORTON_PORTAL_URL", "")
DEFAULT_SUPPORT_URL = os.getenv("PARABLE_SUPPORT_URL", "https://calendar.app.google/3ySUu9E6ogv41mgEA")


def now_ts() -> float:
    return time.time()


def truthy_text(value: Optional[bool]) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def format_ts_value(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


# -------------------------
# Database context manager
# -------------------------
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# -------------------------
# Parable dashboard models
# -------------------------
class VendorAccount(Base):
    __tablename__ = "vendor_accounts"

    id = Column(Integer, primary_key=True)
    sid = Column(String(128), index=True, nullable=False)
    vendor = Column(String(50), index=True, nullable=False)
    external_account_id = Column(String(255), nullable=True)
    portal_url = Column(String(500), nullable=True)
    subscription_active = Column(Boolean, nullable=True)
    display_name = Column(String(255), nullable=True)
    created_at = Column(Float, nullable=False, default=now_ts)
    updated_at = Column(Float, nullable=False, default=now_ts, onupdate=now_ts)


class DeviceHealthSnapshot(Base):
    __tablename__ = "device_health_snapshots"

    id = Column(Integer, primary_key=True)
    sid = Column(String(128), index=True, nullable=False)
    vendor = Column(String(50), index=True, nullable=False)
    status = Column(String(50), nullable=False, default="unknown")
    covered_devices = Column(Integer, nullable=False, default=0)
    threats_found = Column(Integer, nullable=False, default=0)
    definitions_current = Column(Boolean, nullable=True)
    last_seen_at = Column(Float, nullable=True)
    created_at = Column(Float, nullable=False, default=now_ts)


class IdentityHealthSnapshot(Base):
    __tablename__ = "identity_health_snapshots"

    id = Column(Integer, primary_key=True)
    sid = Column(String(128), index=True, nullable=False)
    vendor = Column(String(50), index=True, nullable=False)
    monitoring_active = Column(Boolean, nullable=True)
    alerts_open = Column(Integer, nullable=False, default=0)
    risk_summary = Column(String(255), nullable=True)
    id_lock_status = Column(String(50), nullable=False, default="unknown")
    last_checked_at = Column(Float, nullable=True)
    created_at = Column(Float, nullable=False, default=now_ts)


class VendorAlert(Base):
    __tablename__ = "vendor_alerts"

    id = Column(Integer, primary_key=True)
    sid = Column(String(128), index=True, nullable=False)
    vendor = Column(String(50), index=True, nullable=False)
    severity = Column(String(50), nullable=False, default="info")
    title = Column(String(255), nullable=False)
    detail = Column(Text, nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(Float, nullable=False, default=now_ts)


class VendorSyncLog(Base):
    __tablename__ = "vendor_sync_logs"

    id = Column(Integer, primary_key=True)
    sid = Column(String(128), index=True, nullable=True)
    vendor = Column(String(50), index=True, nullable=False)
    event_type = Column(String(100), nullable=False)
    success = Column(Boolean, nullable=False, default=True)
    detail = Column(Text, nullable=True)
    created_at = Column(Float, nullable=False, default=now_ts)


# Create DB tables after all imported and local models exist
Base.metadata.create_all(bind=engine)


# -------------------------
# Cleanup / misc helpers
# -------------------------
def maybe_cleanup_state() -> None:
    global _last_cleanup_at

    now = now_ts()
    if now - _last_cleanup_at < STATE_CLEANUP_INTERVAL_SECONDS:
        return

    _last_cleanup_at = now

    with get_db() as db:
        expired_auth = db.query(AuthSession).filter(AuthSession.expires_at <= now).delete()
        expired_lockouts = db.query(LoginAttempt).filter(
            LoginAttempt.locked_until != None,
            LoginAttempt.locked_until <= now,
        ).delete()
        stale_sessions = db.query(ChatSession).filter(
            ChatSession.last_seen <= now - 7 * 24 * 3600
        ).delete()

    for bucket in (_chat_rate_windows, _upload_rate_windows, _login_rate_windows):
        stale_keys: List[str] = []
        for key, timestamps in bucket.items():
            fresh = [ts for ts in timestamps if now - ts <= 3600]
            if fresh:
                bucket[key] = fresh
            else:
                stale_keys.append(key)
        for key in stale_keys:
            bucket.pop(key, None)

    logger.info(
        "State cleanup complete | expired_auth=%s expired_lockouts=%s stale_sessions=%s",
        expired_auth,
        expired_lockouts,
        stale_sessions,
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


_SID_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def is_reasonable_sid(sid: str) -> bool:
    if not sid:
        return False
    if len(sid) < 10 or len(sid) > 128:
        return False
    if not all(ch in _SID_ALLOWED_CHARS for ch in sid):
        return False
    if ".." in sid or "/" in sid or "\\" in sid:
        return False
    return True


def severity_rank(value: str) -> int:
    order = {
        "not_connected": 0,
        "unknown": 1,
        "protected": 2,
        "active": 2,
        "warning": 3,
        "critical": 4,
        "offline": 4,
        "expired": 4,
    }
    return order.get((value or "unknown").lower(), 1)


def normalize_status(value: Optional[str], default: str = "unknown") -> str:
    raw = (value or default).strip().lower()
    allowed = {
        "unknown",
        "protected",
        "warning",
        "critical",
        "offline",
        "expired",
        "active",
        "not_connected",
    }
    return raw if raw in allowed else default


def safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps({"error": "unserializable payload"})


# -------------------------
# OpenAI client
# -------------------------
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


# -------------------------
# Session memory (DB-backed)
# -------------------------
class Turn(TypedDict):
    role: str
    content: str


class Session(TypedDict):
    platform: Optional[str]
    history: List[Turn]
    last_seen: float


def get_session(sid: str) -> Session:
    with get_db() as db:
        record = db.query(ChatSession).filter(ChatSession.sid == sid).first()
        if record:
            history = json.loads(record.history_json or "[]")
            return {"platform": record.platform, "history": history, "last_seen": record.last_seen}

        db.add(ChatSession(sid=sid, platform=None, history_json="[]", last_seen=now_ts()))
        return {"platform": None, "history": [], "last_seen": now_ts()}


def save_session(sid: str, sess: Session) -> None:
    with get_db() as db:
        record = db.query(ChatSession).filter(ChatSession.sid == sid).first()
        history_json = json.dumps(sess.get("history", []))
        if record:
            record.platform = sess.get("platform")
            record.history_json = history_json
            record.last_seen = now_ts()
        else:
            db.add(
                ChatSession(
                    sid=sid,
                    platform=sess.get("platform"),
                    history_json=history_json,
                    last_seen=now_ts(),
                )
            )


# -------------------------
# Auth config
# -------------------------
APP_USERNAME = os.getenv("PARABLE_USERNAME")
APP_PASSWORD = os.getenv("PARABLE_PASSWORD")

if not APP_USERNAME or not APP_PASSWORD:
    raise RuntimeError("PARABLE_USERNAME and PARABLE_PASSWORD must be set")

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_SECONDS = 30 * 60
AUTH_TTL_SECONDS = 60 * 60 * 24 * 7

# Pre-hash credentials for constant-time comparison
_APP_USERNAME_HASH = hashlib.sha256(APP_USERNAME.encode()).digest()
_APP_PASSWORD_HASH = hashlib.sha256(APP_PASSWORD.encode()).digest()


def check_credentials(username: str, password: str) -> bool:
    """Timing-safe credential check to prevent side-channel leaks."""
    u_hash = hashlib.sha256(username.encode()).digest()
    p_hash = hashlib.sha256(password.encode()).digest()
    u_ok = hmac.compare_digest(u_hash, _APP_USERNAME_HASH)
    p_ok = hmac.compare_digest(p_hash, _APP_PASSWORD_HASH)
    return u_ok and p_ok


# -------------------------
# Simple in-memory rate limiting
# -------------------------
CHAT_RATE_LIMIT_COUNT = 20
CHAT_RATE_LIMIT_WINDOW_SECONDS = 5 * 60
UPLOAD_RATE_LIMIT_COUNT = 10
UPLOAD_RATE_LIMIT_WINDOW_SECONDS = 10 * 60
LOGIN_RATE_LIMIT_COUNT = 10
LOGIN_RATE_LIMIT_WINDOW_SECONDS = 15 * 60
MAX_RATE_KEYS = 10000

_chat_rate_windows: Dict[str, List[float]] = {}
_upload_rate_windows: Dict[str, List[float]] = {}
_login_rate_windows: Dict[str, List[float]] = {}


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

    # Prevent unbounded growth from many unique keys
    if key not in bucket and len(bucket) >= MAX_RATE_KEYS:
        return False, 60

    history.append(now)
    bucket[key] = history
    return True, 0


def get_login_key(request: Request, sid: str) -> str:
    return get_client_ip(request)


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
    cookie_sid = (request.cookies.get("sid") or "").strip()
    if is_reasonable_sid(cookie_sid):
        return cookie_sid

    header_sid = request.headers.get("x-parable-sid", "").strip()
    if is_reasonable_sid(header_sid):
        return header_sid

    return uuid.uuid4().hex


# -------------------------
# Logged-in helpers
# -------------------------
def _sid_is_authed(sid: str) -> bool:
    with get_db() as db:
        record = db.query(AuthSession).filter(
            AuthSession.sid == sid,
            AuthSession.expires_at > now_ts(),
        ).first()
        return record is not None


def is_logged_in(request: Request, sid: str) -> bool:
    cookie_ok = request.cookies.get("parable_auth") == "1"
    return cookie_ok and _sid_is_authed(sid)


def mark_sid_authed(sid: str) -> None:
    expires = now_ts() + AUTH_TTL_SECONDS
    with get_db() as db:
        record = db.query(AuthSession).filter(AuthSession.sid == sid).first()
        if record:
            record.expires_at = expires
        else:
            db.add(AuthSession(sid=sid, expires_at=expires))


def unmark_sid_authed(sid: str) -> None:
    with get_db() as db:
        db.query(AuthSession).filter(AuthSession.sid == sid).delete()


# -------------------------
# Lockout helpers
# -------------------------
def is_locked_out(login_key: str) -> Tuple[bool, int]:
    with get_db() as db:
        record = db.query(LoginAttempt).filter(LoginAttempt.login_key == login_key).first()
        if not record or not record.locked_until:
            return False, 0
        remaining = int(record.locked_until - now_ts())
        if remaining <= 0:
            record.locked_until = None
            record.attempts = 0
            return False, 0
        return True, remaining


def register_failed_login(login_key: str) -> Tuple[int, bool, int]:
    with get_db() as db:
        record = db.query(LoginAttempt).filter(LoginAttempt.login_key == login_key).first()
        if not record:
            record = LoginAttempt(login_key=login_key, attempts=0, locked_until=None)
            db.add(record)
        record.attempts = (record.attempts or 0) + 1
        attempts = record.attempts
        if attempts >= MAX_LOGIN_ATTEMPTS:
            unlock_at = now_ts() + LOCKOUT_SECONDS
            record.locked_until = unlock_at
            return attempts, True, int(unlock_at - now_ts())
        return attempts, False, 0


def reset_login_attempts(login_key: str) -> None:
    with get_db() as db:
        db.query(LoginAttempt).filter(LoginAttempt.login_key == login_key).delete()


# -------------------------
# Dashboard helpers
# -------------------------
def require_integration_access(request: Request, sid: str) -> None:
    integration_token = os.getenv("INTEGRATION_ADMIN_TOKEN", "").strip()
    provided = request.headers.get("x-integration-token", "").strip()

    if integration_token:
        if not provided or not hmac.compare_digest(provided, integration_token):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return

    if not is_logged_in(request, sid):
        raise HTTPException(status_code=401, detail="Login required")


def get_vendor_account(db, sid: str, vendor: str) -> Optional[VendorAccount]:
    return (
        db.query(VendorAccount)
        .filter(VendorAccount.sid == sid, VendorAccount.vendor == vendor)
        .order_by(VendorAccount.id.desc())
        .first()
    )


def upsert_vendor_account(
    db,
    sid: str,
    vendor: str,
    external_account_id: Optional[str] = None,
    portal_url: Optional[str] = None,
    subscription_active: Optional[bool] = None,
    display_name: Optional[str] = None,
) -> VendorAccount:
    record = get_vendor_account(db, sid, vendor)
    if not record:
        record = VendorAccount(sid=sid, vendor=vendor)
        db.add(record)

    if external_account_id is not None:
        record.external_account_id = external_account_id
    if portal_url is not None:
        record.portal_url = portal_url
    if subscription_active is not None:
        record.subscription_active = subscription_active
    if display_name is not None:
        record.display_name = display_name
    record.updated_at = now_ts()
    return record


def create_sync_log(db, sid: Optional[str], vendor: str, event_type: str, success: bool, detail: str) -> None:
    db.add(
        VendorSyncLog(
            sid=sid,
            vendor=vendor,
            event_type=event_type,
            success=success,
            detail=detail,
            created_at=now_ts(),
        )
    )


def find_sid_for_vendor_reference(db, vendor: str, external_account_id: Optional[str]) -> Optional[str]:
    if not external_account_id:
        return None
    account = (
        db.query(VendorAccount)
        .filter(
            VendorAccount.vendor == vendor,
            VendorAccount.external_account_id == external_account_id,
        )
        .first()
    )
    return account.sid if account else None


def build_security_summary_for_sid(sid: str) -> Dict[str, Any]:
    with get_db() as db:
        vendor_account = get_vendor_account(db, sid, "bitdefender") or get_vendor_account(db, sid, "norton")
        latest = (
            db.query(DeviceHealthSnapshot)
            .filter(DeviceHealthSnapshot.sid == sid)
            .order_by(DeviceHealthSnapshot.created_at.desc())
            .first()
        )
        open_critical = (
            db.query(VendorAlert)
            .filter(
                VendorAlert.sid == sid,
                VendorAlert.resolved == False,
                VendorAlert.severity == "critical",
            )
            .count()
        )

        if not latest:
            return {
                "ok": True,
                "vendor": "Bitdefender",
                "status": "not_connected",
                "last_seen": None,
                "threats_found": 0,
                "definitions_current": None,
                "subscription_active": vendor_account.subscription_active if vendor_account else None,
                "covered_devices": 0,
                "critical_alerts": open_critical,
                "portal_url": (
                    vendor_account.portal_url if vendor_account and vendor_account.portal_url else DEFAULT_SECURITY_PORTAL_URL
                ),
            }

        vendor_name = (latest.vendor or "Bitdefender").title()
        status = normalize_status(latest.status)
        if open_critical > 0 and status in {"protected", "unknown", "not_connected"}:
            status = "critical"

        return {
            "ok": True,
            "vendor": vendor_name,
            "status": status,
            "last_seen": format_ts_value(latest.last_seen_at),
            "threats_found": int(latest.threats_found or 0),
            "definitions_current": latest.definitions_current,
            "subscription_active": vendor_account.subscription_active if vendor_account else None,
            "covered_devices": int(latest.covered_devices or 0),
            "critical_alerts": open_critical,
            "portal_url": (
                vendor_account.portal_url if vendor_account and vendor_account.portal_url else DEFAULT_SECURITY_PORTAL_URL
            ),
        }


def build_identity_summary_for_sid(sid: str) -> Dict[str, Any]:
    with get_db() as db:
        vendor_account = get_vendor_account(db, sid, "norton") or get_vendor_account(db, sid, "bitdefender")
        latest = (
            db.query(IdentityHealthSnapshot)
            .filter(IdentityHealthSnapshot.sid == sid)
            .order_by(IdentityHealthSnapshot.created_at.desc())
            .first()
        )
        open_alerts = (
            db.query(VendorAlert)
            .filter(
                VendorAlert.sid == sid,
                VendorAlert.resolved == False,
                VendorAlert.vendor == "norton",
            )
            .count()
        )

        if not latest:
            return {
                "ok": True,
                "vendor": "Norton",
                "monitoring_active": None,
                "alerts_open": open_alerts,
                "risk_summary": "Not connected yet",
                "id_lock_status": "unknown",
                "last_checked": None,
                "subscription_active": vendor_account.subscription_active if vendor_account else None,
                "portal_url": (
                    vendor_account.portal_url if vendor_account and vendor_account.portal_url else DEFAULT_IDENTITY_PORTAL_URL
                ),
            }

        vendor_name = (latest.vendor or "Norton").title()
        return {
            "ok": True,
            "vendor": vendor_name,
            "monitoring_active": latest.monitoring_active,
            "alerts_open": max(int(latest.alerts_open or 0), open_alerts),
            "risk_summary": latest.risk_summary or "No summary available",
            "id_lock_status": latest.id_lock_status or "unknown",
            "last_checked": format_ts_value(latest.last_checked_at),
            "subscription_active": vendor_account.subscription_active if vendor_account else None,
            "portal_url": (
                vendor_account.portal_url if vendor_account and vendor_account.portal_url else DEFAULT_IDENTITY_PORTAL_URL
            ),
        }


def build_dashboard_summary_for_sid(sid: str) -> Dict[str, Any]:
    security = build_security_summary_for_sid(sid)
    identity = build_identity_summary_for_sid(sid)
    return {
        "ok": True,
        "security": security,
        "identity": identity,
        "support_url": DEFAULT_SUPPORT_URL,
    }


# -------------------------
# API models
# -------------------------
class ChatIn(BaseModel):
    message: str = Field(default="", max_length=MAX_MESSAGE_LENGTH)
    image_url: Optional[str] = None


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class IntegrationConnectIn(BaseModel):
    target_sid: str = Field(..., min_length=10, max_length=128)
    external_account_id: Optional[str] = Field(default=None, max_length=255)
    portal_url: Optional[str] = Field(default=None, max_length=500)
    display_name: Optional[str] = Field(default=None, max_length=255)
    subscription_active: Optional[bool] = None
    status: Optional[str] = Field(default=None, max_length=50)
    covered_devices: Optional[int] = Field(default=None, ge=0, le=100000)
    threats_found: Optional[int] = Field(default=None, ge=0, le=100000)
    definitions_current: Optional[bool] = None
    last_seen_at: Optional[float] = None
    monitoring_active: Optional[bool] = None
    alerts_open: Optional[int] = Field(default=None, ge=0, le=100000)
    risk_summary: Optional[str] = Field(default=None, max_length=255)
    id_lock_status: Optional[str] = Field(default=None, max_length=50)
    last_checked_at: Optional[float] = None


# -------------------------
# Middleware
# -------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    maybe_cleanup_state()
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "ALLOW-FROM https://www.parablesmartphone.com"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'self' https://www.parablesmartphone.com https://parablesmartphone.com;"
    )
    response.headers["Cache-Control"] = "no-store"

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
    return FileResponse(str(manifest_path), media_type="application/manifest+json")


# -------------------------
# Auth endpoints
# -------------------------
@app.post("/api/login")
def login_api(payload: LoginIn, request: Request):
    username = (payload.username or "").strip()
    password = payload.password or ""
    sid = get_sid(request)
    login_key = get_login_key(request, sid)
    client_ip = get_client_ip(request)

    # Global per-IP login rate limit (before lockout check)
    allowed, retry_after = rate_limit_check(
        _login_rate_windows,
        client_ip,
        LOGIN_RATE_LIMIT_COUNT,
        LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    )
    if not allowed:
        logger.warning("Login rate limited | sid=%s ip=%s", sid, client_ip)
        resp = JSONResponse(
            {
                "ok": False,
                "error": f"Too many login attempts. Please wait about {retry_after} seconds.",
            },
            status_code=429,
        )
        set_sid_cookie(resp, sid)
        return resp

    locked, remaining = is_locked_out(login_key)
    if locked:
        minutes = max(1, remaining // 60)
        logger.warning("Login blocked | sid=%s ip=%s remaining=%ss", sid, client_ip, remaining)
        resp = JSONResponse(
            {
                "ok": False,
                "error": f"Too many wrong tries. Try again in about {minutes} minutes.",
            },
            status_code=429,
        )
        set_sid_cookie(resp, sid)
        return resp

    if check_credentials(username, password):
        reset_login_attempts(login_key)
        mark_sid_authed(sid)

        logger.info("Login success | sid=%s ip=%s", sid, client_ip)

        resp = JSONResponse({"ok": True, "message": "Logged in", "redirect": "/dashboard"})
        set_sid_cookie(resp, sid)
        set_auth_cookie(resp)
        return resp

    attempts, locked_now, remaining = register_failed_login(login_key)
    logger.warning(
        "Login failed | sid=%s ip=%s attempts=%s locked_now=%s",
        sid,
        client_ip,
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

    resp = JSONResponse({"ok": True, "logged_in": logged_in})
    set_sid_cookie(resp, sid)
    return resp


# -------------------------
# Dashboard APIs
# -------------------------
@app.get("/api/dashboard/security")
def dashboard_security_api(request: Request):
    sid = get_sid(request)
    if not is_logged_in(request, sid):
        raise HTTPException(status_code=401, detail="Login required")

    resp = JSONResponse(build_security_summary_for_sid(sid))
    set_sid_cookie(resp, sid)
    return resp


@app.get("/api/dashboard/identity")
def dashboard_identity_api(request: Request):
    sid = get_sid(request)
    if not is_logged_in(request, sid):
        raise HTTPException(status_code=401, detail="Login required")

    resp = JSONResponse(build_identity_summary_for_sid(sid))
    set_sid_cookie(resp, sid)
    return resp


@app.get("/api/dashboard/summary")
def dashboard_summary_api(request: Request):
    sid = get_sid(request)
    if not is_logged_in(request, sid):
        raise HTTPException(status_code=401, detail="Login required")

    resp = JSONResponse(build_dashboard_summary_for_sid(sid))
    set_sid_cookie(resp, sid)
    return resp


@app.post("/api/integrations/bitdefender/connect")
def bitdefender_connect_api(payload: IntegrationConnectIn, request: Request):
    caller_sid = get_sid(request)
    require_integration_access(request, caller_sid)

    with get_db() as db:
        account = upsert_vendor_account(
            db,
            sid=payload.target_sid,
            vendor="bitdefender",
            external_account_id=payload.external_account_id,
            portal_url=payload.portal_url,
            subscription_active=payload.subscription_active,
            display_name=payload.display_name,
        )

        if payload.status is not None or payload.covered_devices is not None or payload.threats_found is not None:
            db.add(
                DeviceHealthSnapshot(
                    sid=payload.target_sid,
                    vendor="bitdefender",
                    status=normalize_status(payload.status, "unknown"),
                    covered_devices=int(payload.covered_devices or 0),
                    threats_found=int(payload.threats_found or 0),
                    definitions_current=payload.definitions_current,
                    last_seen_at=payload.last_seen_at or now_ts(),
                    created_at=now_ts(),
                )
            )

        create_sync_log(
            db,
            sid=payload.target_sid,
            vendor="bitdefender",
            event_type="connect",
            success=True,
            detail=safe_json_dumps(
                {
                    "external_account_id": account.external_account_id,
                    "portal_url": account.portal_url,
                    "subscription_active": account.subscription_active,
                }
            ),
        )

    return {"ok": True, "message": "Bitdefender connection saved"}


@app.post("/api/integrations/norton/connect")
def norton_connect_api(payload: IntegrationConnectIn, request: Request):
    caller_sid = get_sid(request)
    require_integration_access(request, caller_sid)

    with get_db() as db:
        account = upsert_vendor_account(
            db,
            sid=payload.target_sid,
            vendor="norton",
            external_account_id=payload.external_account_id,
            portal_url=payload.portal_url,
            subscription_active=payload.subscription_active,
            display_name=payload.display_name,
        )

        if (
            payload.monitoring_active is not None
            or payload.alerts_open is not None
            or payload.risk_summary is not None
            or payload.id_lock_status is not None
        ):
            db.add(
                IdentityHealthSnapshot(
                    sid=payload.target_sid,
                    vendor="norton",
                    monitoring_active=payload.monitoring_active,
                    alerts_open=int(payload.alerts_open or 0),
                    risk_summary=payload.risk_summary,
                    id_lock_status=(payload.id_lock_status or "unknown").strip().lower(),
                    last_checked_at=payload.last_checked_at or now_ts(),
                    created_at=now_ts(),
                )
            )

        create_sync_log(
            db,
            sid=payload.target_sid,
            vendor="norton",
            event_type="connect",
            success=True,
            detail=safe_json_dumps(
                {
                    "external_account_id": account.external_account_id,
                    "portal_url": account.portal_url,
                    "subscription_active": account.subscription_active,
                }
            ),
        )

    return {"ok": True, "message": "Norton connection saved"}


@app.post("/api/integrations/bitdefender/webhook")
async def bitdefender_webhook_api(request: Request):
    caller_sid = get_sid(request)
    require_integration_access(request, caller_sid)
    payload = await request.json()

    target_sid = (payload.get("target_sid") or payload.get("sid") or "").strip()
    if not target_sid:
        with get_db() as db:
            target_sid = (
                find_sid_for_vendor_reference(db, "bitdefender", payload.get("external_account_id")) or ""
            )
    if not target_sid:
        raise HTTPException(status_code=400, detail="target_sid or known external_account_id required")

    status = normalize_status(payload.get("status"), "unknown")
    threats_found = int(payload.get("threats_found") or 0)
    covered_devices = int(payload.get("covered_devices") or 0)
    definitions_current = payload.get("definitions_current")
    last_seen_at = payload.get("last_seen_at") or now_ts()
    severity = (payload.get("severity") or "info").strip().lower()
    title = (payload.get("title") or "Bitdefender update").strip()
    detail = payload.get("detail")

    with get_db() as db:
        db.add(
            DeviceHealthSnapshot(
                sid=target_sid,
                vendor="bitdefender",
                status=status,
                covered_devices=covered_devices,
                threats_found=threats_found,
                definitions_current=definitions_current,
                last_seen_at=last_seen_at,
                created_at=now_ts(),
            )
        )

        if detail or title:
            db.add(
                VendorAlert(
                    sid=target_sid,
                    vendor="bitdefender",
                    severity=severity,
                    title=title,
                    detail=detail,
                    resolved=bool(payload.get("resolved", False)),
                    created_at=now_ts(),
                )
            )

        create_sync_log(
            db,
            sid=target_sid,
            vendor="bitdefender",
            event_type="webhook",
            success=True,
            detail=safe_json_dumps(payload),
        )

    return {"ok": True}


@app.post("/api/integrations/norton/webhook")
async def norton_webhook_api(request: Request):
    caller_sid = get_sid(request)
    require_integration_access(request, caller_sid)
    payload = await request.json()

    target_sid = (payload.get("target_sid") or payload.get("sid") or "").strip()
    if not target_sid:
        with get_db() as db:
            target_sid = find_sid_for_vendor_reference(db, "norton", payload.get("external_account_id")) or ""
    if not target_sid:
        raise HTTPException(status_code=400, detail="target_sid or known external_account_id required")

    monitoring_active = payload.get("monitoring_active")
    alerts_open = int(payload.get("alerts_open") or 0)
    risk_summary = payload.get("risk_summary") or "No summary available"
    id_lock_status = (payload.get("id_lock_status") or "unknown").strip().lower()
    last_checked_at = payload.get("last_checked_at") or now_ts()
    severity = (payload.get("severity") or "info").strip().lower()
    title = (payload.get("title") or "Norton identity update").strip()
    detail = payload.get("detail")

    with get_db() as db:
        db.add(
            IdentityHealthSnapshot(
                sid=target_sid,
                vendor="norton",
                monitoring_active=monitoring_active,
                alerts_open=alerts_open,
                risk_summary=risk_summary,
                id_lock_status=id_lock_status,
                last_checked_at=last_checked_at,
                created_at=now_ts(),
            )
        )

        if detail or title:
            db.add(
                VendorAlert(
                    sid=target_sid,
                    vendor="norton",
                    severity=severity,
                    title=title,
                    detail=detail,
                    resolved=bool(payload.get("resolved", False)),
                    created_at=now_ts(),
                )
            )

        create_sync_log(
            db,
            sid=target_sid,
            vendor="norton",
            event_type="webhook",
            success=True,
            detail=safe_json_dumps(payload),
        )

    return {"ok": True}


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
            "You are a friendly support helper at Parable, a company that helps people "
            "stay safe and get the most out of their smartphones. You genuinely care about "
            "the person you're talking to.\n"
            "\n"
            "How to sound:\n"
            "- Write like a patient friend who's good with phones, not a manual.\n"
            "- Use everyday language. Skip jargon and acronyms.\n"
            "- Vary your phrasing naturally. Don't start every sentence the same way.\n"
            "- A little warmth goes a long way — brief acknowledgments like "
            "'Good question' or 'That's a common one' feel human. But don't overdo it.\n"
            "- Keep answers concise. A few clear steps beat a wall of text.\n"
            "- When you mention a setting, include the path (e.g. Settings > Accessibility > Zoom) "
            "so they can find it easily.\n"
            "- If you need more info, ask one simple question.\n"
            "- If something sounds like a scam or suspicious pop-up, lead with safety: "
            "'Don't tap anything on that screen yet.' Then explain why and what to do.\n"
            "\n"
            "If someone shares a screenshot:\n"
            "- Describe what you notice in the image so they know you're looking at the right thing.\n"
            "- If it looks suspicious, say so plainly and explain what tipped you off.\n"
            "- Give them clear next steps to stay safe.\n"
            "- If the image is too blurry or cut off, just ask for a clearer one.\n"
            "\n"
            "After giving steps, check in naturally — something like 'Let me know if that helps' "
            "or 'Does that make sense?' Vary it so it doesn't feel scripted.\n"
        )

        if image_url:
            base = str(request.base_url).rstrip("/")
            if image_url.startswith("/"):
                image_url = base + image_url
            elif not image_url.startswith(base):
                image_url = None

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

        ai_response = get_client().responses.create(model="gpt-4.1-mini", input=input_messages)
        answer = (ai_response.output_text or "").strip() or "I had trouble answering that. Please try again."

        if message:
            history.append({"role": "user", "content": message})
        elif image_url:
            history.append({"role": "user", "content": "[Uploaded a photo]"})

        history.append({"role": "assistant", "content": answer})
        sess["history"] = history[-MAX_HISTORY:]
        save_session(sid, sess)

        resp = JSONResponse({"ok": True, "answer": answer, "logged_in": is_logged_in(request, sid)})
        set_sid_cookie(resp, sid)
        return resp

    except HTTPException:
        raise
    except Exception:
        logger.exception("AI service error | sid=%s ip=%s", sid, get_client_ip(request))
        resp = JSONResponse({"ok": False, "error": "AI service error. Please try again."}, status_code=502)
        set_sid_cookie(resp, sid)
        return resp


# -------------------------
# Health endpoints
# -------------------------
@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/health")
def health(request: Request):
    health_token = os.getenv("HEALTH_TOKEN")
    if health_token:
        provided = request.headers.get("x-health-token", "")
        if provided != health_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

    with get_db() as db:
        session_count = db.query(ChatSession).count()
        authed_count = db.query(AuthSession).filter(AuthSession.expires_at > now_ts()).count()
        vendor_accounts = db.query(VendorAccount).count()

    return {
        "ok": True,
        "service": "parable-portal",
        "time": int(now_ts()),
        "sessions": session_count,
        "authed_sessions": authed_count,
        "vendor_accounts": vendor_accounts,
    }


# -------------------------
# Simple pages
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return chat_page(request)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    sid = get_sid(request)
    if not is_logged_in(request, sid):
        resp = RedirectResponse(url="/", status_code=302)
        set_sid_cookie(resp, sid)
        return resp

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover" />
  <title>Parable Dashboard</title>
  <meta name="theme-color" content="#ea580c">
  <style>
    :root {
      --navy: #020617;
      --orange: #ea580c;
      --orange-dark: #c2410c;
      --bg: #f4f6fb;
      --card: #ffffff;
      --soft: #fff7ed;
      --line: #e5e7eb;
      --text: #111827;
      --muted: #475569;
      --success: #166534;
      --warning: #b45309;
      --danger: #b91c1c;
      --neutral: #64748b;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      min-height: 100%;
      overflow-x: hidden;
    }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--bg) 55%);
      color: var(--text);
      -webkit-text-size-adjust: 100%;
    }
    .page {
      width: 100%;
      min-height: 100dvh;
      padding: 18px;
      display: flex;
      justify-content: center;
    }
    .wrap { width: 100%; max-width: 1120px; }
    .card {
      background: var(--card);
      border-radius: 24px;
      padding: 24px;
      border: 2px solid rgba(234, 88, 12, 0.45);
      box-shadow: 0 14px 40px rgba(2, 6, 23, 0.10);
    }
    .topbar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .brand {
      font-size: 20px;
      font-weight: 800;
      color: var(--navy);
    }
    .top-actions {
      margin-left: auto;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn, .btn-outline, .mini-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 50px;
      padding: 12px 16px;
      border-radius: 14px;
      font-size: 18px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
      border: 1px solid rgba(234, 88, 12, 0.35);
    }
    .btn { background: var(--orange); color: #ffffff; }
    .btn-outline, .mini-btn { background: #ffffff; color: var(--navy); }
    h1 {
      margin: 0 0 10px;
      font-size: 40px;
      line-height: 1.15;
      color: var(--navy);
    }
    .lead {
      margin: 0 0 18px;
      font-size: 21px;
      line-height: 1.6;
      color: var(--muted);
    }
    .welcome {
      margin-bottom: 22px;
      padding: 18px 20px;
      border-radius: 18px;
      background: var(--soft);
      border: 1px solid rgba(234, 88, 12, 0.25);
      font-size: 22px;
      font-weight: 700;
      color: var(--navy);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin: 22px 0;
    }
    .tile {
      display: block;
      text-decoration: none;
      color: inherit;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 26px 22px;
      box-shadow: 0 8px 22px rgba(2, 6, 23, 0.05);
      min-height: 250px;
      transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .tile:hover {
      transform: translateY(-3px);
      border-color: rgba(234, 88, 12, 0.45);
      box-shadow: 0 16px 34px rgba(2, 6, 23, 0.10);
    }
    .tile-icon {
      width: 64px;
      height: 64px;
      border-radius: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 30px;
      margin-bottom: 18px;
      background: linear-gradient(135deg, rgba(234, 88, 12, 0.12), rgba(234, 88, 12, 0.22));
      color: var(--orange-dark);
    }
    .tile-title {
      margin: 0 0 10px;
      font-size: 30px;
      font-weight: 800;
      color: var(--navy);
      line-height: 1.2;
    }
    .tile-text {
      margin: 0;
      font-size: 19px;
      line-height: 1.55;
      color: var(--muted);
      white-space: pre-line;
    }
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 15px;
      font-weight: 800;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--neutral);
    }
    .tile[data-state="protected"] .status-chip,
    .tile[data-state="active"] .status-chip {
      color: var(--success);
      border-color: rgba(22, 101, 52, 0.25);
      background: rgba(22, 101, 52, 0.07);
    }
    .tile[data-state="warning"] .status-chip {
      color: var(--warning);
      border-color: rgba(180, 83, 9, 0.25);
      background: rgba(180, 83, 9, 0.07);
    }
    .tile[data-state="critical"],
    .tile[data-state="expired"],
    .tile[data-state="offline"] {
      border-color: rgba(185, 28, 28, 0.25);
    }
    .tile[data-state="critical"] .status-chip,
    .tile[data-state="expired"] .status-chip,
    .tile[data-state="offline"] .status-chip {
      color: var(--danger);
      border-color: rgba(185, 28, 28, 0.25);
      background: rgba(185, 28, 28, 0.07);
    }
    .tile-actions {
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    .footer-note {
      margin-top: 24px;
      font-size: 18px;
      color: var(--muted);
      line-height: 1.6;
      text-align: center;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .tile { min-height: auto; }
    }
    @media (max-width: 760px) {
      .page { padding: 12px; }
      .card { padding: 18px; border-radius: 18px; }
      h1 { font-size: 32px; }
      .lead, .welcome, .tile-text, .btn, .btn-outline, .mini-btn { font-size: 19px; }
      .tile-title { font-size: 26px; }
      .top-actions { width: 100%; margin-left: 0; }
      .btn, .btn-outline { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="wrap">
      <div class="card">
        <div class="topbar">
          <div class="brand">Parable Smartphone</div>
          <div class="top-actions">
            <a class="btn" href="/appchat">Open Chat</a>
            <button class="btn-outline" id="logoutBtn" type="button">Log out</button>
          </div>
        </div>

        <h1>Welcome to your Parable Dashboard</h1>
        <p class="lead">Choose one of the big options below to get help fast.</p>

        <div class="welcome">Big, simple help for your phone and identity.</div>

        <div class="grid">
          <div class="tile" id="securityTile" data-state="unknown">
            <div class="tile-icon">\U0001f6e1\ufe0f</div>
            <div class="status-chip" id="securityChip">Loading security...</div>
            <div class="tile-title">Security Health</div>
            <p class="tile-text" id="securityText">Checking vendor, threats, definitions, and subscription status.</p>
            <div class="tile-actions">
              <a class="mini-btn" id="securityPortalBtn" href="#" target="_blank" rel="noopener noreferrer" style="display:none;">Open vendor portal</a>
            </div>
          </div>

          <div class="tile" id="identityTile" data-state="unknown">
            <div class="tile-icon">\U0001faaa</div>
            <div class="status-chip" id="identityChip">Loading identity...</div>
            <div class="tile-title">Identity Health</div>
            <p class="tile-text" id="identityText">Checking monitoring, alerts, risk summary, and ID lock status.</p>
            <div class="tile-actions">
              <a class="mini-btn" id="identityPortalBtn" href="#" target="_blank" rel="noopener noreferrer" style="display:none;">Review alert</a>
            </div>
          </div>

          <a class="tile" href="/appchat">
            <div class="tile-icon">\U0001f4ac</div>
            <div class="tile-title">Parable Chat</div>
            <p class="tile-text">Ask a question, speak into the microphone, or get step-by-step help for your phone.</p>
          </a>

          <a class="tile" href="/appchat?topic=I got a scam pop-up or suspicious message on my phone">
            <div class="tile-icon">\u26a0\ufe0f</div>
            <div class="tile-title">ScamShield</div>
            <p class="tile-text">Get help with scam pop-ups, fake messages, suspicious links, and phone safety concerns.</p>
          </a>

          <a class="tile" href="https://calendar.app.google/3ySUu9E6ogv41mgEA" target="_blank" rel="noopener noreferrer">
            <div class="tile-icon">\U0001f4c5</div>
            <div class="tile-title">Contact Parable Support</div>
            <p class="tile-text">Book one-on-one help with Parable when you want direct support from a real person.</p>
          </a>

          <a class="tile" href="https://www.parablesmartphone.com" target="_blank" rel="noopener noreferrer">
            <div class="tile-icon">\u2b50</div>
            <div class="tile-title">Parable Subscriptions</div>
            <p class="tile-text">View plan information and account options on the Parable website.</p>
          </a>
        </div>

        <p class="footer-note">Need help right now? Tap <strong>Open Chat</strong>.</p>
      </div>
    </div>
  </div>

  <script>
    function formatTime(ts) {
      if (!ts) return "Unknown";
      try {
        return new Date(ts * 1000).toLocaleString();
      } catch (e) {
        return "Unknown";
      }
    }

    function yesNoUnknown(value) {
      if (value === true) return "Yes";
      if (value === false) return "No";
      return "Unknown";
    }

    function titleCase(text) {
      const raw = (text || "unknown").replace(/_/g, " ");
      return raw.charAt(0).toUpperCase() + raw.slice(1);
    }

    function setState(el, state) {
      el.setAttribute("data-state", state || "unknown");
    }

    async function loadDashboardCards() {
      try {
        const [securityRes, identityRes] = await Promise.all([
          fetch("/api/dashboard/security", { credentials: "same-origin" }),
          fetch("/api/dashboard/identity", { credentials: "same-origin" })
        ]);

        const security = await securityRes.json();
        const identity = await identityRes.json();

        const securityTile = document.getElementById("securityTile");
        const securityChip = document.getElementById("securityChip");
        const securityText = document.getElementById("securityText");
        const securityPortalBtn = document.getElementById("securityPortalBtn");

        const securityState = security.status || "unknown";
        setState(securityTile, securityState);
        securityChip.textContent = titleCase(security.vendor || "Vendor") + " \u2022 " + titleCase(securityState);
        securityText.textContent = [
          "Vendor: " + (security.vendor || "Unknown"),
          "Threats found: " + (security.threats_found ?? 0),
          "Definitions current: " + yesNoUnknown(security.definitions_current),
          "Subscription active: " + yesNoUnknown(security.subscription_active),
          "Covered devices: " + (security.covered_devices ?? 0),
          "Last seen: " + formatTime(security.last_seen)
        ].join("\\n");
        if (security.portal_url) {
          securityPortalBtn.href = security.portal_url;
          securityPortalBtn.style.display = "inline-flex";
        }

        const identityTile = document.getElementById("identityTile");
        const identityChip = document.getElementById("identityChip");
        const identityText = document.getElementById("identityText");
        const identityPortalBtn = document.getElementById("identityPortalBtn");

        let identityState = "unknown";
        if (identity.alerts_open > 0) {
          identityState = "warning";
        } else if (identity.monitoring_active === true) {
          identityState = "active";
        }
        setState(identityTile, identityState);
        identityChip.textContent = titleCase(identity.vendor || "Vendor") + " \u2022 " + titleCase(identityState);
        identityText.textContent = [
          "Monitoring active: " + yesNoUnknown(identity.monitoring_active),
          "Alerts open: " + (identity.alerts_open ?? 0),
          "Risk summary: " + (identity.risk_summary || "No summary available"),
          "ID Lock / Freeze: " + titleCase(identity.id_lock_status || "unknown"),
          "Subscription active: " + yesNoUnknown(identity.subscription_active),
          "Last checked: " + formatTime(identity.last_checked)
        ].join("\\n");
        if (identity.portal_url) {
          identityPortalBtn.href = identity.portal_url;
          identityPortalBtn.style.display = "inline-flex";
        }
      } catch (e) {
        console.error("Dashboard load failed", e);
      }
    }

    async function logout() {
      try {
        await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
      } catch (e) {
        console.error("Logout failed", e);
      }
      window.location.href = "/appchat";
    }

    document.getElementById("logoutBtn").addEventListener("click", logout);
    window.addEventListener("load", loadDashboardCards);
  </script>
</body>
</html>
"""
    resp = HTMLResponse(html)
    set_sid_cookie(resp, sid)
    return resp


# -------------------------
# Chat UI
# -------------------------
@app.get("/appchat", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    sid = get_sid(request)
    logged_in = is_logged_in(request, sid)
    topic = (request.query_params.get("topic") or "").strip()

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover" />
  <title>Parable Chat</title>
  <link rel="manifest" href="/manifest.webmanifest">
  <meta name="theme-color" content="#ea580c">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <link rel="apple-touch-icon" href="/static/icon-192.png">
  <style>
    :root {
      --navy: #020617;
      --orange: #ea580c;
      --bg: #f4f6fb;
      --card: #ffffff;
      --soft: #f8fafc;
      --line: #e5e7eb;
      --danger: #b91c1c;
      --text: #111827;
      --muted: #475569;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      height: 100%;
      overflow-x: hidden;
    }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--bg) 55%);
      color: var(--text);
      -webkit-text-size-adjust: 100%;
      overscroll-behavior-x: none;
      font-size: 20px;
      line-height: 1.6;
    }
    img { max-width: 100%; height: auto; }
    .page {
      width: 100%;
      max-width: 100%;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      padding: 8px;
      padding-top: max(8px, env(safe-area-inset-top));
      padding-bottom: max(8px, env(safe-area-inset-bottom));
      padding-left: max(8px, env(safe-area-inset-left));
      padding-right: max(8px, env(safe-area-inset-right));
      display: flex;
      justify-content: center;
      align-items: stretch;
    }
    .wrap {
      width: 100%;
      max-width: 620px;
      margin: 0 auto;
      display: flex;
      flex: 1 1 0;
      min-height: 0;
    }
    .card {
      width: 100%;
      background: var(--card);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 14px 40px rgba(2, 6, 23, 0.10);
      border: 2px solid rgba(234, 88, 12, 0.55);
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      flex: 1 1 0;
      min-height: 0;
    }
    .card::before {
      content: "";
      position: absolute;
      inset: 10px;
      border-radius: 16px;
      border: 1px solid rgba(234, 88, 12, 0.22);
      pointer-events: none;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 38px;
      line-height: 1.15;
      color: var(--navy);
      overflow-wrap: break-word;
      word-break: break-word;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 21px;
      line-height: 1.6;
      overflow-wrap: break-word;
      word-break: break-word;
    }
    .welcome {
      margin: 0 0 12px;
      padding: 14px 16px;
      background: #fff7ed;
      border: 1px solid rgba(234, 88, 12, 0.25);
      border-radius: 16px;
      color: var(--navy);
      font-weight: 800;
      font-size: 22px;
      text-align: center;
    }
    .topbar {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 10px;
      flex-wrap: wrap;
      flex-shrink: 0;
    }
    .topbar-main {
      min-width: 0;
      flex: 1 1 auto;
    }
    .topbar-actions {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .status {
      font-size: 18px;
      color: var(--muted);
      overflow-wrap: break-word;
      word-break: break-word;
    }
    .smallbtn {
      min-height: 50px;
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #ffffff;
      cursor: pointer;
      font-weight: 700;
      font-size: 18px;
      color: var(--text);
    }
    .chatbox {
      width: 100%;
      min-height: 120px;
      flex: 1 1 auto;
      overflow-y: auto;
      overflow-x: hidden;
      -webkit-overflow-scrolling: touch;
      background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      word-break: break-word;
      overflow-wrap: break-word;
      font-size: 20px;
      line-height: 1.65;
    }
    .row {
      display: flex;
      margin: 12px 0;
      width: 100%;
      max-width: 100%;
      flex-shrink: 0;
    }
    .bubble {
      padding: 14px 16px;
      border-radius: 16px;
      max-width: 88%;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 20px;
      overflow-wrap: break-word;
      word-wrap: break-word;
      word-break: break-word;
    }
    .you { justify-content: flex-end; }
    .you .bubble { background: var(--navy); color: #ffffff; }
    .bot { justify-content: flex-start; }
    .bot .bubble {
      background: var(--soft);
      color: var(--text);
      border: 1px solid rgba(234, 88, 12, 0.22);
    }
    .quick {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 10px 0 0;
      width: 100%;
      max-width: 100%;
      flex-shrink: 0;
    }
    .q {
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--navy);
      cursor: pointer;
      font-weight: 700;
      font-size: 18px;
      max-width: 100%;
    }
    .q:hover { border-color: var(--orange); }
    .controls {
      display: flex;
      gap: 10px;
      margin-top: 10px;
      align-items: stretch;
      flex-wrap: wrap;
      width: 100%;
      max-width: 100%;
      flex-shrink: 0;
    }
    .controls > * { min-width: 0; }
    input#msg, input.login-input {
      flex: 1 1 240px;
      min-width: 0;
      width: 100%;
      max-width: 100%;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid #d1d5db;
      outline: none;
      font-size: 20px;
      line-height: 1.4;
    }
    input#msg:focus, input.login-input:focus {
      border-color: rgba(234, 88, 12, 0.8);
      box-shadow: 0 0 0 4px rgba(234, 88, 12, 0.15);
    }
    button.action {
      min-width: 118px;
      min-height: 56px;
      padding: 14px 18px;
      border-radius: 14px;
      border: 1px solid rgba(234, 88, 12, 0.35);
      background: var(--orange);
      color: #ffffff;
      cursor: pointer;
      font-weight: 800;
      font-size: 19px;
      flex-shrink: 0;
    }
    button.action:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .iconbtn {
      min-width: 58px;
      min-height: 56px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(234, 88, 12, 0.35);
      background: #ffffff;
      cursor: pointer;
      font-weight: 800;
      font-size: 22px;
      color: var(--navy);
      flex-shrink: 0;
    }
    .overlay {
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, 0.55);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 16px;
      z-index: 9999;
    }
    .overlay.show { display: flex; }
    .login-card {
      width: min(460px, 100%);
      max-width: 100%;
      background: #ffffff;
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 20px 50px rgba(2, 6, 23, 0.25);
      border: 2px solid rgba(234, 88, 12, 0.35);
    }
    .login-card h2 {
      margin: 0 0 8px;
      color: var(--navy);
      font-size: 30px;
      line-height: 1.2;
    }
    .login-card p {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 19px;
      line-height: 1.6;
    }
    .login-grid { display: grid; gap: 12px; }
    .login-actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .error {
      color: var(--danger);
      font-size: 17px;
      min-height: 24px;
      margin-top: 8px;
      overflow-wrap: break-word;
      word-break: break-word;
    }
    .preview {
      margin-top: 12px;
      display: none;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px;
      border: 1px dashed rgba(234, 88, 12, 0.35);
      border-radius: 14px;
      background: #ffffff;
      width: 100%;
      max-width: 100%;
      overflow: hidden;
      flex-shrink: 0;
    }
    .preview img {
      max-height: 80px;
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      flex-shrink: 0;
    }
    .muted {
      color: #64748b;
      font-size: 16px;
      line-height: 1.5;
      overflow-wrap: break-word;
      word-break: break-word;
    }
    @media (max-width: 640px) {
      .page {
        padding: 6px;
        padding-top: max(6px, env(safe-area-inset-top));
        padding-bottom: max(6px, env(safe-area-inset-bottom));
        padding-left: max(6px, env(safe-area-inset-left));
        padding-right: max(6px, env(safe-area-inset-right));
      }
      .card { padding: 12px; border-radius: 16px; }
      h1 { font-size: 30px; }
      .sub, .welcome, .status, .bubble, .q, .smallbtn, .login-card p, .muted { font-size: 18px; }
      .chatbox { min-height: 100px; padding: 12px; }
      .bubble { max-width: 94%; }
      .controls { gap: 8px; }
      input#msg { flex: 1 1 100%; }
      button.action { min-width: 100px; }
      .topbar-actions { width: 100%; margin-left: 0; justify-content: flex-end; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="wrap">
      <div class="card" id="card">
        <div class="topbar">
          <div class="topbar-main">
            <h1>Parable Chat</h1>
            <p class="sub">Ask a question about your phone.</p>
          </div>
          <div class="topbar-actions">
            <span id="loginStatus" class="status">__LOGIN_STATUS__</span>
            <button id="openLoginBtn" class="smallbtn" type="button">__LOGIN_BUTTON__</button>
          </div>
        </div>

        <div class="welcome">Thank you. How can we help you today?</div>
        <div id="usageStatus" class="status" style="margin-bottom:10px;"></div>
        <div id="chatbox" class="chatbox"></div>

        <div class="quick">
          <button class="q" type="button" onclick="quick('I am on iPhone')">I'm on iPhone</button>
          <button class="q" type="button" onclick="quick('I am on Android')">I'm on Android</button>
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
            placeholder="Type here or choose Voice..."
            autocomplete="off"
            autocapitalize="sentences"
            autocorrect="on"
            spellcheck="true"
            maxlength="1500"
          />
          <input id="photoInput" type="file" accept="image/*" style="display:none" />
          <button id="attach" class="iconbtn" type="button" title="Upload photo">\U0001f4ce</button>
          <button id="mic" class="action" type="button">\U0001f3a4 Speak</button>
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
  let loggedIn = __LOGGED_IN__;
  let initialTopic = __SAFE_TOPIC__;
  let selectedFile = null;
  let uploadedUrl = null;

  btn.disabled = true;
  micBtn.disabled = true;

  function getOrCreateSid() {
    const key = "parable_sid";
    let sid = localStorage.getItem(key);
    if (!sid || sid.length < 10) {
      sid = (crypto.randomUUID
        ? crypto.randomUUID().replace(/-/g, "")
        : (Date.now() + "-" + Math.random()).replace(/[^a-zA-Z0-9-_]/g, ""));
      localStorage.setItem(key, sid);
    }
    return sid;
  }

  function updateUsageUi(data) {
    if (!data) return;
    usageStatus.textContent = data.logged_in ? "Logged in account active." : "Chat is ready.";
  }

  async function loadUsage() {
    try {
      const res = await fetch("/api/me", {
        headers: { "X-Parable-SID": getOrCreateSid() },
        credentials: "same-origin"
      });
      const data = await res.json();
      if (res.ok) {
        loggedIn = data.logged_in;
        updateLoginUi();
        updateUsageUi(data);
      }
    } catch (e) {
      console.error("Status load failed", e);
    }
  }

  function scrollChatToBottom() {
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  function addBubble(text, who) {
    const row = document.createElement("div");
    row.className = "row " + who;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatbox.appendChild(row);
    scrollChatToBottom();
    if (who === "bot" && preferVoice === true) {
      speak(text);
    }
  }

  function greetOnce() {
    if (greeted) return;
    greeted = true;
    addBubble("You can type, speak, or upload a photo.", "bot");
    btn.disabled = false;
    micBtn.disabled = false;
    input.focus();
  }

  function pickVoice() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;
    // Prefer neural / natural voices — these sound human, not robotic.
    // Avoid legacy voices like Microsoft Zira, David, etc.
    const preferred = [
      "Microsoft Aria Online",
      "Microsoft Jenny Online",
      "Microsoft Guy Online",
      "Microsoft Ana Online",
      "Microsoft Andrew Online",
      "Microsoft Emma Online",
      "Google US English",
      "Samantha",
      "Karen",
      "Daniel"
    ];
    for (const name of preferred) {
      const v = voices.find(x => x.name && x.name.includes(name));
      if (v) return v;
    }
    // Pick any "Online" or "Natural" voice before falling back to legacy
    const neural = voices.find(v =>
      (v.lang || "").startsWith("en") &&
      (/online|natural|neural/i.test(v.name))
    );
    if (neural) return neural;
    return voices.find(v => (v.lang || "").startsWith("en")) || voices[0];
  }

  function speakCleanText(raw) {
    // Turn step-format text into something that sounds natural when read aloud.
    let t = (raw || "").trim();
    // Strip markdown-style bold/italic markers
    t = t.replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1");
    // Turn numbered list items ("1. Do this" / "1) Do this") into plain sentences
    t = t.replace(/^\d+[.)]\s*/gm, "");
    // Turn bullet dashes / asterisks into natural pauses
    t = t.replace(/^[\-\u2022\*]\s*/gm, "");
    // Turn setting paths like "Settings > Accessibility > Zoom" into spoken form
    t = t.replace(/\s*>\s*/g, ", then ");
    // Collapse multiple newlines into a period + space (sentence break)
    t = t.replace(/\\n{2,}/g, ". ");
    // Single newlines become a short pause (comma)
    t = t.replace(/\\n/g, ", ");
    // Clean up doubled punctuation from the replacements above
    t = t.replace(/[.,]{2,}/g, ".").replace(/,\s*\./g, ".").replace(/\.\s*,/g, ".");
    // Collapse whitespace
    t = t.replace(/\s+/g, " ").trim();
    return t;
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    if (!selectedVoice) {
      selectedVoice = pickVoice();
    }
    const cleaned = speakCleanText(text);
    // Split into sentences so the synthesizer can breathe between them.
    // This avoids the monotone run-on that makes TTS sound robotic.
    const sentences = cleaned.match(/[^.!?]+[.!?]+/g) || [cleaned];
    // Cap total spoken length — read at most ~500 chars so it stays useful
    let charBudget = 500;
    for (const sentence of sentences) {
      const s = sentence.trim();
      if (!s) continue;
      if (charBudget <= 0) break;
      charBudget -= s.length;
      const u = new SpeechSynthesisUtterance(s);
      if (selectedVoice) {
        u.voice = selectedVoice;
      }
      u.rate = 0.92;
      u.pitch = 1.0;
      u.volume = 1.0;
      window.speechSynthesis.speak(u);
    }
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      selectedVoice = pickVoice();
    };
  }

  async function askMicPermission() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        micReady = false;
        return false;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      micReady = true;
      return true;
    } catch (err) {
      console.error("Microphone permission error:", err);
      micReady = false;
      return false;
    }
  }

  async function startMic() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      addBubble("Voice input is not supported in this browser.", "bot");
      return;
    }
    if (!micReady) {
      const ok = await askMicPermission();
      if (!ok) {
        addBubble("Microphone permission was blocked. Please allow microphone access and try again.", "bot");
        return;
      }
    }
    if (!recognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      recognition.onstart = function () {
        micBtn.disabled = true;
        micBtn.textContent = "\U0001f3a4 Listening...";
      };
      recognition.onend = function () {
        micBtn.disabled = false;
        micBtn.textContent = "\U0001f3a4 Speak";
      };
      recognition.onerror = function (event) {
        console.error("Speech recognition error:", event.error);
        let message = "Microphone error. Please try again.";
        if (event.error === "not-allowed") {
          message = "Microphone permission was denied. Please allow mic access and try again.";
          micReady = false;
        } else if (event.error === "no-speech") {
          message = "I did not hear anything. Try speaking again.";
        } else if (event.error === "audio-capture") {
          message = "No microphone was found or it is unavailable.";
        }
        addBubble(message, "bot");
        micBtn.disabled = false;
        micBtn.textContent = "\U0001f3a4 Speak";
      };
      recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
        send();
      };
    }
    try {
      recognition.start();
    } catch (err) {
      console.error("Recognition start error:", err);
      addBubble("Could not start microphone. Please try again.", "bot");
    }
  }

  function showLogin() {
    loginError.textContent = "";
    loginOverlay.classList.add("show");
    loginUser.focus();
  }

  function hideLogin() {
    loginOverlay.classList.remove("show");
  }

  function updateLoginUi() {
    loginStatus.textContent = loggedIn ? "Logged in" : "Not logged in";
    openLoginBtn.textContent = loggedIn ? "Dashboard" : "Log in";
  }

  async function submitLogin() {
    const username = loginUser.value.trim();
    const password = loginPass.value;
    loginError.textContent = "";
    if (!username || !password) {
      loginError.textContent = "Enter username and password.";
      return;
    }
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Parable-SID": getOrCreateSid()
        },
        credentials: "same-origin",
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (!res.ok) {
        loginError.textContent = data.error || "Login failed.";
        return;
      }
      loggedIn = true;
      updateLoginUi();
      hideLogin();
      loginPass.value = "";
      loadUsage();
      if (data.redirect) {
        window.location.href = data.redirect;
        return;
      }
      addBubble("You are logged in.", "bot");
    } catch (e) {
      loginError.textContent = "Login error. Try again.";
    }
  }

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    preview.style.display = "flex";
  }

  function clearSelectedPhoto() {
    selectedFile = null;
    uploadedUrl = null;
    photoInput.value = "";
    previewImg.src = "";
    preview.style.display = "none";
  }

  async function uploadSelectedPhotoIfNeeded() {
    if (!selectedFile) return null;
    if (uploadedUrl) return uploadedUrl;

    const fd = new FormData();
    fd.append("file", selectedFile);

    const res = await fetch("/api/upload-image", {
      method: "POST",
      headers: { "X-Parable-SID": getOrCreateSid() },
      credentials: "same-origin",
      body: fd
    });

    if (res.status === 401) {
      addBubble("Please log in to upload photos.", "bot");
      showLogin();
      return null;
    }

    const data = await res.json();
    if (!res.ok || !data.ok) {
      addBubble(data.error || data.detail || "Upload failed.", "bot");
      return null;
    }

    uploadedUrl = data.url;
    return uploadedUrl;
  }

  async function send() {
    const message = input.value.trim();
    greetOnce();
    if (!message && !selectedFile) return;

    const label = message || "[Photo sent]";
    addBubble(label, "you");

    if (selectedFile) {
      const imgRow = document.createElement("div");
      imgRow.className = "row you";
      const img = document.createElement("img");
      img.src = URL.createObjectURL(selectedFile);
      img.style.maxWidth = "160px";
      img.style.maxHeight = "180px";
      img.style.objectFit = "cover";
      img.style.borderRadius = "12px";
      img.style.border = "1px solid #e5e7eb";
      imgRow.appendChild(img);
      chatbox.appendChild(imgRow);
      scrollChatToBottom();
    }

    input.value = "";
    input.focus();
    btn.disabled = true;

    try {
      const photoUrl = await uploadSelectedPhotoIfNeeded();
      if (selectedFile && !photoUrl) {
        btn.disabled = false;
        return;
      }
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Parable-SID": getOrCreateSid()
        },
        credentials: "same-origin",
        body: JSON.stringify({ message: message || "", image_url: photoUrl })
      });
      const data = await res.json();
      addBubble(data.answer || data.error || ("HTTP " + res.status), "bot");
      updateUsageUi(data);
      if (photoUrl) {
        clearSelectedPhoto();
      }
    } catch (e) {
      addBubble("Network or server error.", "bot");
    } finally {
      btn.disabled = false;
    }
  }

  function quick(text) {
    input.value = text;
    send();
  }

  btn.addEventListener("click", send);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      send();
    }
  });

  micBtn.addEventListener("click", async () => {
    greetOnce();
    preferVoice = true;
    await startMic();
  });

  openLoginBtn.addEventListener("click", () => {
    if (loggedIn) {
      window.location.href = "/dashboard";
      return;
    }
    showLogin();
  });

  loginSubmit.addEventListener("click", submitLogin);
  loginClose.addEventListener("click", hideLogin);

  loginPass.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      submitLogin();
    }
  });

  attachBtn.addEventListener("click", () => {
    greetOnce();
    photoInput.click();
  });

  photoInput.addEventListener("change", () => {
    const f = photoInput.files && photoInput.files[0];
    if (!f) return;
    selectedFile = f;
    uploadedUrl = null;
    showPreview(f);
  });

  clearPhoto.addEventListener("click", clearSelectedPhoto);

  card.addEventListener("click", () => {
    greetOnce();
    input.focus();
  });

  window.addEventListener("load", () => {
    updateLoginUi();
    greetOnce();
    loadUsage();
    if (initialTopic) {
      setTimeout(() => {
        input.value = initialTopic;
        send();
      }, 450);
    }
  });
</script>
</body>
</html>
"""

    html = (
        html.replace("__LOGGED_IN__", "true" if logged_in else "false")
        .replace("__LOGIN_STATUS__", "Logged in" if logged_in else "Not logged in")
        .replace("__LOGIN_BUTTON__", "Dashboard" if logged_in else "Log in")
        .replace("__SAFE_TOPIC__", json.dumps(topic))
    )
    resp = HTMLResponse(html)
    set_sid_cookie(resp, sid)
    return resp
