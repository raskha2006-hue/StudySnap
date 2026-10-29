"""
StudySnap - backend

Flask server that does two jobs:
  1. serves index.html
  2. talks to Gemma 4 through the Gemini API (google-genai SDK)

Endpoints
  GET  /            -> index.html
  GET  /health      -> {"ok": true}
  POST /api/analyze -> image in, structured study material out
  POST /api/chat    -> follow-up doubts about the analysed material

Env vars
  GEMINI_API_KEY   required
  GEMMA_MODEL      optional, default gemma-4-26b-a4b-it  (or gemma-4-31b-it)
"""
import json
import logging
import os
import re
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

try:
    from google.genai import errors as genai_errors
except Exception:  # pragma: no cover
    genai_errors = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
LEVELS = {"Class 6-8", "Class 9-10", "Class 11-12", "College"}

RATE_LIMIT = 12        # requests
RATE_WINDOW = 60       # per this many seconds, per IP

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 512 * 1024
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("studysnap")

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
# This is the system instruction sent to Gemma 4 on every /api/analyze call.
# The JSON shape here is what the frontend renders, so keep them in sync.
ANALYZE_SYSTEM = """You are StudySnap, a patient tutor who helps students understand
what is written in a photo of their study material (handwritten notes, textbook page,
diagram, or a problem to solve).

Look at the image carefully. If some part is unreadable, say so inside the explanation
instead of guessing. If the image is not study material at all, set "topic" to
"Not study material" and use the explanation to tell the student what you can see.

Reply with ONE JSON object and nothing else - no markdown fences, no commentary.
Exact shape:
{
  "topic": "short title, max 8 words",
  "explanation": "clear explanation using simple words and one everyday analogy. Use **bold** for important terms. Paragraphs separated by \\n\\n",
  "key_points": ["4 to 6 short points a student should remember"],
  "solution_steps": ["numbered-style steps, ONLY if the image contains a problem to solve, otherwise []"],
  "quiz": [
    {"q": "question text", "options": ["option A", "option B", "option C", "option D"], "answer": 0, "why": "one-line reason"}
  ],
  "flashcards": [{"front": "term or question", "back": "short answer"}]
}

Rules:
- Exactly 5 quiz questions, 4 options each. "answer" is the 0-based index of the right option.
- Exactly 6 flashcards.
- Quiz questions must be answerable from the material, not trivia.
- Write in clear, friendly English suited to the student level you are given.
"""

CHAT_SYSTEM = """You are StudySnap, a patient tutor. The student is asking a follow-up
doubt about study material they uploaded earlier. A summary of that material is below.
Answer briefly and clearly (under 150 words unless a derivation needs more), in the same
language and style the student writes in. Use **bold** for key terms. If the question has
nothing to do with the material or studying, gently steer back.

Material summary:
"""

# --------------------------------------------------------------------------
# Gemini / Gemma client
# --------------------------------------------------------------------------
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set on the server.")
        _client = genai.Client(api_key=key)
    return _client


def call_gemma_vision(image_bytes, mime, level, deep_think):
    """Send the photo + instructions to Gemma 4 and return raw text."""
    cfg = {"system_instruction": ANALYZE_SYSTEM, "temperature": 0.4}
    if deep_think:
        # Gemma 4 only supports thinking on/off; "high" turns it on.
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level="high")

    prompt = (
        f"Student level: {level}\n"
        "Analyse the attached study material and return the JSON object."
    )
    resp = get_client().models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(**cfg),
    )
    return resp.text or ""


def call_gemma_chat(material, history, question):
    """Multi-turn doubt solving. History is replayed into a chat session."""
    past = []
    for item in history[-10:]:
        role = "user" if item.get("role") == "user" else "model"
        text = str(item.get("text", ""))[:2000]
        if text:
            past.append(types.Content(role=role, parts=[types.Part(text=text)]))

    chat = get_client().chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=CHAT_SYSTEM + material[:6000], temperature=0.5
        ),
        history=past,
    )
    return (chat.send_message(question).text or "").strip()


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def extract_json(text):
    """Models sometimes wrap JSON in fences or add a sentence around it."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start : end + 1])


def clean_list(value, limit):
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def normalise(data):
    """Make sure the frontend always gets the shape it expects."""
    quiz = []
    for q in data.get("quiz", []) if isinstance(data.get("quiz"), list) else []:
        try:
            options = clean_list(q.get("options"), 4)
            answer = int(q.get("answer"))
            if len(options) >= 2 and 0 <= answer < len(options) and q.get("q"):
                quiz.append(
                    {"q": str(q["q"]).strip(), "options": options,
                     "answer": answer, "why": str(q.get("why", "")).strip()}
                )
        except (TypeError, ValueError, AttributeError):
            continue

    cards = []
    for c in data.get("flashcards", []) if isinstance(data.get("flashcards"), list) else []:
        if isinstance(c, dict) and c.get("front") and c.get("back"):
            cards.append({"front": str(c["front"]).strip(), "back": str(c["back"]).strip()})

    return {
        "topic": str(data.get("topic") or "Untitled topic").strip()[:120],
        "explanation": str(data.get("explanation") or "").strip(),
        "key_points": clean_list(data.get("key_points"), 8),
        "solution_steps": clean_list(data.get("solution_steps"), 15),
        "quiz": quiz[:8],
        "flashcards": cards[:10],
    }


# --------------------------------------------------------------------------
# Tiny per-IP rate limiter (in memory - fine for a single small instance)
# --------------------------------------------------------------------------
_hits = defaultdict(deque)


def too_many_requests():
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "?").split(",")[0].strip()
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def friendly_error(exc):
    """Turn SDK / network exceptions into something a student can read."""
    code = getattr(exc, "code", None)
    msg = str(exc).lower()
    if isinstance(exc, RuntimeError) and "api_key" in msg:
        return 500, "The server has no API key configured. Please tell the site admin."
    if code == 429 or "quota" in msg or "rate" in msg:
        return 429, "Gemma's free usage limit is full right now. Try again in a minute."
    if code in (400, 403) and ("api key" in msg or "permission" in msg):
        return 500, "The API key is invalid or expired."
    if code == 400:
        return 400, "Gemma couldn't process this image. Try a different photo."
    return 502, "Couldn't get a response from Gemma. Please try again shortly."


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, model=MODEL)


@app.post("/api/analyze")
def analyze():
    if too_many_requests():
        return jsonify(error="Too many requests. Please slow down a little."), 429

    file = request.files.get("image")
    if not file:
        return jsonify(error="Please attach an image."), 400
    mime = (file.mimetype or "").lower()
    if mime not in ALLOWED_MIME:
        return jsonify(error="Only JPG, PNG or WebP images are supported."), 400
    image_bytes = file.read()
    if not image_bytes:
        return jsonify(error="The image file is empty."), 400
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return jsonify(error="The image is larger than 8 MB."), 413

    level = request.form.get("level", "Class 11-12")
    if level not in LEVELS:
        level = "Class 11-12"
    deep_think = request.form.get("deep_think") == "true"

    last_err = None
    for attempt in (1, 2):  # one retry if the model returns broken JSON
        try:
            raw = call_gemma_vision(image_bytes, mime, level, deep_think)
            result = normalise(extract_json(raw))
            if not result["explanation"]:
                raise ValueError("empty explanation")
            return jsonify(result)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            log.warning("attempt %s: bad model output (%s)", attempt, exc)
        except Exception as exc:
            log.exception("gemma call failed")
            status, message = friendly_error(exc)
            return jsonify(error=message), status

    log.error("giving up after bad JSON: %s", last_err)
    return jsonify(error="Gemma's reply came back incomplete. Please try again."), 502


@app.post("/api/chat")
def chat():
    if too_many_requests():
        return jsonify(error="Too many requests. Please slow down a little."), 429

    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    material = str(body.get("material", "")).strip()
    history = body.get("history") if isinstance(body.get("history"), list) else []
    if not question:
        return jsonify(error="Please type a question first."), 400
    if len(question) > 1000:
        return jsonify(error="Keep the question under 1000 characters."), 400

    try:
        return jsonify(answer=call_gemma_chat(material, history, question))
    except Exception as exc:
        log.exception("chat failed")
        status, message = friendly_error(exc)
        return jsonify(error=message), status


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="The image is larger than 8 MB."), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
