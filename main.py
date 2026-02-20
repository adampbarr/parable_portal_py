from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional, TypedDict

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Only mount if the folder exists (prevents Render crash)
if STATIC_DIR.exists():
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
# Session memory (simple)
# -------------------------
class Turn(TypedDict):
    role: str
    content: str


class Session(TypedDict):
    platform: Optional[str]  # "iphone" | "android" | None
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
# API models
# -------------------------
class ChatIn(BaseModel):
    message: str


# -------------------------
# PWA manifest
# -------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(
        str(BASE_DIR / "manifest.webmanifest"),
        media_type="application/manifest+json",
    )


# -------------------------
# API endpoints
# -------------------------
@app.post("/api/chat")
def chat_api(payload: ChatIn, request: Request):
    message = (payload.message or "").strip()
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    m = message.lower()
    sid = request.cookies.get("sid") or str(uuid.uuid4())

    sess = get_session(sid)
    history = sess["history"]

    # Platform capture (lightweight)
    if "iphone" in m or "ios" in m:
        sess["platform"] = "iphone"
        if len(m) <= 12:
            return {"answer": "Got it — iPhone. What issue are you having?"}

    if "android" in m:
        sess["platform"] = "android"
        if len(m) <= 12:
            return {"answer": "Got it — Android. What issue are you having?"}

    platform = sess["platform"]
    ask_platform = "Quick question — are you on iPhone or Android?" if not platform else ""

    # Common fast answers (cheap + snappy)
    if "zoom" in m or "zoomed" in m or "magnif" in m:
        if platform == "iphone":
            return {
                "answer": (
                    "Try this (iPhone):\n"
                    "1) Double-tap with 3 fingers (often turns Zoom off).\n"
                    "2) Settings > Accessibility > Zoom > Off.\n"
                    "Did that work?"
                )
            }
        if platform == "android":
            return {
                "answer": (
                    "Try this (Android):\n"
                    "1) Triple-tap to turn Magnification off.\n"
                    "2) Settings > Accessibility > Magnification > Off.\n"
                    "Did that work?"
                )
            }
        return {
            "answer": (
                "Try this:\n"
                "1) iPhone: Double-tap with 3 fingers.\n"
                "2) iPhone: Settings > Accessibility > Zoom > Off.\n"
                "3) Android: Settings > Accessibility > Magnification > Off.\n"
                + ask_platform
            )
        }

    if (
        "no sound" in m
        or "can't hear" in m
        or "cannot hear" in m
        or "volume" in m
        or "speaker" in m
    ):
        if platform == "iphone":
            return {
                "answer": (
                    "Let’s get your iPhone sound back.\n"
                    "1) Press Volume Up a few times.\n"
                    "2) Flip the Silent switch (orange showing = silent).\n"
                    "3) Control Center: turn Bluetooth off.\n"
                    "4) Settings > Sounds & Haptics: raise Ringer/Alerts.\n"
                    "5) Restart the iPhone.\n"
                    "Is it no sound at all, or just phone calls?"
                )
            }
        if platform == "android":
            return {
                "answer": (
                    "Let’s get your Android sound back.\n"
                    "1) Press Volume Up, then raise Media and Ring.\n"
                    "2) Turn Do Not Disturb off.\n"
                    "3) Turn Bluetooth off.\n"
                    "4) Restart the phone.\n"
                    "Is it no sound at all, or just phone calls?"
                )
            }
        return {
            "answer": (
                "Let’s fix the sound.\n"
                "1) Press Volume Up a few times.\n"
                "2) Turn Bluetooth off.\n"
                "3) Turn Do Not Disturb off.\n"
                "4) Restart the phone.\n"
                "Are you on iPhone or Android? And is it all sound, or just calls?"
            )
        }

    if "scam" in m or "pop-up" in m or "popup" in m or "virus" in m:
        return {
            "answer": (
                "Don’t click anything.\n"
                "1) Close the tab or app.\n"
                "2) iPhone: Settings > Safari > Clear History and Website Data.\n"
                "3) Android: Chrome > Settings > Privacy > Clear browsing data.\n"
                + ask_platform
            )
        }

    if "storage" in m or "not enough space" in m or "low space" in m or ("full" in m and "storage" in m):
        return {
            "answer": (
                "Storage full? Try this:\n"
                "1) Delete big videos you don’t need.\n"
                "2) Remove unused apps.\n"
                "3) Delete message attachments you don’t need.\n"
                "4) Empty Recently Deleted photos.\n"
                "5) Turn on iCloud Photos or Google Photos.\n"
                "Did that help?"
            )
        }

    # Fallback (OpenAI, short memory)
    try:
        convo: List[dict] = [
            {
                "role": "system",
                "content": (
                    "You are Parable Smartphone Support. Talk like a calm, friendly helper.\n"
                    "Rules:\n"
                    "- Use simple words. No tech jargon.\n"
                    "- Keep it short: 3–6 steps max.\n"
                    "- One step per line. Start each line with a verb: Tap, Open, Turn on, Turn off, Go to, Try, Restart.\n"
                    "- Ask at most ONE question, only if needed.\n"
                    "- If you mention a setting, include the exact path like: Settings > Accessibility > Zoom.\n"
                    "- Avoid acronyms. If you must use one, explain it in 3 words.\n"
                    "- If scams/pop-ups: start with 'Don’t click anything.'\n"
                    "- End with: 'Did that work?' when appropriate.\n"
                ),
            }
        ]

        if platform:
            convo.append({"role": "system", "content": f"User is on {platform}."})

        for turn in history[-MAX_HISTORY:]:
            convo.append(turn)

        convo.append({"role": "user", "content": message})

        resp = get_client().responses.create(
            model="gpt-4.1-mini",
            input=convo,
        )

        answer = resp.output_text or ""

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        sess["history"] = history[-MAX_HISTORY:]

        return {"answer": answer}

    except Exception as e:
        return JSONResponse({"error": f"AI service error: {str(e)}"}, status_code=502)


@app.get("/api/hello")
def api_hello():
    return {"answer": "Hi! I’m Parable. Are you on iPhone or Android?"}


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
# Chat UI (single clean HTML)
# -------------------------
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    sid = request.cookies.get("sid") or str(uuid.uuid4())

    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Parable Chat</title>

  <!-- PWA -->
  <link rel="manifest" href="/manifest.webmanifest">
  <meta name="theme-color" content="#ea580c">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <link rel="apple-touch-icon" href="/static/icon-192.png">

  <style>
    :root{
      --navy: #020617;
      --orange: #ea580c;
      --bg: #f4f6fb;
      --card: #ffffff;
      --soft: #f8fafc;
      --line: #e5e7eb;
    }

    body {
      font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1200px 600px at 50% 0%, #ffffff 0%, var(--bg) 55%);
      margin: 0;
    }

    .wrap {
      max-width: 920px;
      margin: 0 auto;
      padding: 28px 16px 40px;
    }

    .card {
      background: var(--card);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 40px rgba(2, 6, 23, .10);
      border: 2px solid rgba(234, 88, 12, .55);
      position: relative;
    }

    .card:before {
      content: "";
      position: absolute;
      inset: 10px;
      border-radius: 14px;
      border: 1px solid rgba(234, 88, 12, .22);
      pointer-events: none;
    }

    h1 { margin: 0 0 6px; font-size: 22px; color: var(--navy); letter-spacing: .2px; }
    .sub { margin: 0 0 14px; color:#475569; }

    .chatbox {
      height: 420px;
      overflow: auto;
      background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }

    .row { display:flex; margin: 10px 0; }
    .bubble { padding:10px 12px; border-radius:14px; max-width: 85%; white-space: pre-wrap; line-height:1.35; }

    .you { justify-content:flex-end; }
    .you .bubble { background: var(--navy); color: white; }

    .bot { justify-content:flex-start; }
    .bot .bubble {
      background: var(--soft);
      color: #111827;
      border: 1px solid rgba(234, 88, 12, .22);
    }

    .quick { display:flex; gap:8px; flex-wrap:wrap; margin: 10px 0 0; }

    .q {
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--navy);
      cursor: pointer;
      font-weight: 600;
    }
    .q:hover { border-color: var(--orange); }

    .controls { display:flex; gap:10px; margin-top:12px; align-items: center; }

    input#msg{
      flex: 1;
      padding: 12px 12px;
      border-radius: 14px;
      border: 1px solid #d1d5db;
      outline: none;
    }

    input#msg:focus {
      border-color: rgba(234, 88, 12, .8);
      box-shadow: 0 0 0 4px rgba(234, 88, 12, .15);
    }

    button.action{
      min-width: 110px;
      padding: 11px 16px;
      border-radius: 12px;
      border: 1px solid rgba(234, 88, 12, .35);
      background: var(--orange);
      color: white;
      cursor: pointer;
      font-weight: 700;
    }
    button.action:disabled{ opacity: .45; cursor: not-allowed; }
    /* Mobile: stack Speak + Send nicely */
@media (max-width: 520px) {
  .controls {
    flex-wrap: wrap;
  }

  input#msg {
    width: 100%;
    flex: 1 1 100%;
  }

  button.action {
    width: 100%;
    min-width: 0;
    flex: 1 1 100%;
  }
}

  </style>
</head>

<body>
  <div class="wrap">
    <div class="card" id="card">
      <h1>Parable Chatbot</h1>
      <p class="sub">Ask a question about your phone.</p>

      <div id="chatbox" class="chatbox"></div>

      <div class="quick">
        <button class="q" type="button" onclick="quick('iPhone')">I’m on iPhone</button>
        <button class="q" type="button" onclick="quick('Android')">I’m on Android</button>
      </div>

      <div class="controls">
        <input id="msg" placeholder="Type here (or choose Voice first)..." />
        <button id="mic" class="action" type="button">🎤 Speak</button>
        <button id="btn" class="action" type="button">Send</button>
      </div>
    </div>
  </div>

<script>
  const chatbox = document.getElementById("chatbox");
  const input   = document.getElementById("msg");
  const btn     = document.getElementById("btn");
  const micBtn  = document.getElementById("mic");
  const card    = document.getElementById("card");

  let greeted = false;
  let recognition = null;
  let preferVoice = null; // null = not chosen, true = voice, false = text
  let selectedVoice = null;

  // Disable send/mic until user chooses Text or Voice
  btn.disabled = true;
  micBtn.disabled = true;

  function addBubble(text, who) {
    const row = document.createElement("div");
    row.className = "row " + who;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    chatbox.appendChild(row);
    chatbox.scrollTop = chatbox.scrollHeight;

    if (who === "bot" && preferVoice === true) speak(text);
  }

  function addChoiceButtons() {
    const row = document.createElement("div");
    row.className = "row bot";

    const wrap = document.createElement("div");
    wrap.className = "bubble";

    const label = document.createElement("div");
    label.textContent = "Would you like to use text or voice?";
    label.style.marginBottom = "8px";

    const btnRow = document.createElement("div");
    btnRow.style.display = "flex";
    btnRow.style.gap = "8px";
    btnRow.style.flexWrap = "wrap";

    const bText = document.createElement("button");
    bText.type = "button";
    bText.textContent = "Text";
    bText.className = "q";
    bText.onclick = () => {
      preferVoice = false;
      btn.disabled = false;
      micBtn.disabled = true;
      addBubble("Perfect — type your question below.", "bot");
      input.focus();
    };

    const bVoice = document.createElement("button");
    bVoice.type = "button";
    bVoice.textContent = "Voice";
    bVoice.className = "q";
    bVoice.onclick = () => {
      preferVoice = true;
      btn.disabled = false;
      micBtn.disabled = false;
      input.focus();

      // This is a user click -> speech is allowed
      speak("Great — tap the microphone and tell me what’s going on.");
    };

    btnRow.appendChild(bText);
    btnRow.appendChild(bVoice);

    wrap.appendChild(label);
    wrap.appendChild(btnRow);

    row.appendChild(wrap);
    chatbox.appendChild(row);
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  function greetOnce() {
    if (greeted) return;
    greeted = true;
    addBubble("Hi, and welcome to Parable Chat.", "bot");
    addChoiceButtons();
  }

  function pickVoice() {
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

    for (const name of preferred) {
      const v = voices.find(x => x.name && x.name.includes(name));
      if (v) return v;
    }

    return voices.find(v => (v.lang || "").startsWith("en")) || voices[0];
  }

  // More human-sounding voice
  function speak(text) {
    if (!("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();
    if (!selectedVoice) selectedVoice = pickVoice();

    let t = (text || "")
      .replace(/\n+/g, ". ")
      .replace(/\s+/g, " ")
      .trim();

    if (t.length > 260) t = t.slice(0, 260) + "...";

    const u = new SpeechSynthesisUtterance(t);
    if (selectedVoice) u.voice = selectedVoice;

    // Slightly slower + small variation
    u.rate = 0.97 + (Math.random() * 0.04 - 0.02);
    u.pitch = 1.03 + (Math.random() * 0.06 - 0.03);
    u.volume = 1.0;

    window.speechSynthesis.speak(u);
  }

  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      selectedVoice = pickVoice();
    };
  }

  function startMic() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      addBubble("Voice input not supported in this browser. Try Chrome or Edge.", "bot");
      return;
    }

    if (!recognition) {
      recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        micBtn.disabled = true;
        micBtn.textContent = "🎤 Listening...";
      };

      recognition.onend = () => {
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Speak";
      };

      recognition.onerror = () => {
        addBubble("Mic error. Check permission and try again.", "bot");
        micBtn.disabled = false;
        micBtn.textContent = "🎤 Speak";
      };

      recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        input.value = text;
        send();
      };
    }

    recognition.start();
  }

  async function send() {
    const message = input.value.trim();
    if (!message) return;

    greetOnce();

    addBubble(message, "you");
    input.value = "";
    input.focus();

    btn.disabled = true;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ message })
      });

      const raw = await res.text();
      let data;
      try { data = JSON.parse(raw); } catch { data = { raw }; }

      addBubble(
        data.answer || data.error || data.message || data.raw || ("HTTP " + res.status),
        "bot"
      );
    } catch (e) {
      addBubble("Network/server error — check the uvicorn terminal.", "bot");
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
    if (e.key === "Enter") send();
  });

  micBtn.addEventListener("click", () => {
    greetOnce();
    preferVoice = true;
    btn.disabled = false;
    micBtn.disabled = false;
    startMic();
  });

  card.addEventListener("click", () => {
    greetOnce();
    input.focus();
  });

  window.addEventListener("load", greetOnce);

  // Register service worker (PWA)
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/static/sw.js");
  }
</script>

</body>
</html>
"""

    resp = HTMLResponse(html)
    resp.set_cookie(
        key="sid",
        value=sid,
        httponly=True,
        samesite="lax",
    )
    return resp
