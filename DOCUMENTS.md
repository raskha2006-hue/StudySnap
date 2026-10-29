# StudySnap: design notes

These notes explain why the project is built the way it is, how the pieces talk to each other, and where Gemma 4 sits in all of it.

## 1. The problem

A student has a page of notes or a textbook problem they don't understand. The usual options are:

1. Retype it into a chatbot. Slow, and diagrams, formulas and handwriting get lost.
2. Search for it. Results are generic and rarely at the right level.
3. Ask a friend or teacher. Not always available.

StudySnap takes the photo as-is and returns material to study from, not just an answer.

## 2. Who it's for

School and early college students. The level selector (Class 6-8 up to College) changes how deep and how simple the explanation is.

## 3. Features

| Feature | Where it happens |
|---|---|
| Photo upload, drag and drop, camera capture | `index.html` |
| Client-side image shrinking (max 1600 px, JPEG 0.88) | `shrink()` in `index.html` |
| Explanation, key points, step-by-step solution | Gemma 4, rendered by `render()` |
| Quiz with instant feedback and score | Gemma 4 output, `renderQuiz()` |
| Flashcards that flip | Gemma 4 output, `renderCards()` |
| Doubt chat with memory of earlier turns | `/api/chat`, `call_gemma_chat()` |
| Notes export (.md) | Built in the browser, no server call |

## 4. Architecture

```
 Browser (index.html)                    Flask (app.py)                      Gemini API
 ────────────────────                    ──────────────                      ──────────
 pick photo
 shrink to <=1600px  ── multipart ───▶   /api/analyze
                                         validate type, size, options
                                         call_gemma_vision() ── image + prompt ──▶  Gemma 4
                                                              ◀── JSON text ──────
                                         extract_json() + normalise()
 render tabs         ◀── JSON ────────   (retry once on bad JSON)

 ask a doubt         ── JSON ────────▶   /api/chat
                                         call_gemma_chat() ── history + question ─▶ Gemma 4
 show answer         ◀── {answer} ────
```

One Flask process serves the page and the API, so there is no CORS setup and nothing to configure between frontend and backend. The browser never sees the API key.

## 5. How Gemma 4 is used

Both Gemma calls are in `app.py`, using the `google-genai` SDK.

### 5.1 Vision call: `call_gemma_vision()`

```python
client.models.generate_content(
    model=MODEL,                               # gemma-4-26b-a4b-it
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type=mime),
        "Student level: Class 11-12\n..."
    ],
    config=types.GenerateContentConfig(
        system_instruction=ANALYZE_SYSTEM,
        temperature=0.4,
        thinking_config=types.ThinkingConfig(thinking_level="high"),  # only if toggle is on
    ),
)
```

What each part does:

- **Image as inline bytes.** The browser sends a JPEG, the server forwards the raw bytes. No file upload API, no storage, no cleanup.
- **System instruction.** Defines the tutor role, the exact JSON schema, and rules such as "exactly 5 quiz questions, 4 options each, `answer` is a zero-based index".
- **Temperature 0.4.** Low enough to keep the JSON shape stable, high enough that explanations don't read like a template.
- **Thinking mode.** Gemma 4 supports thinking on or off, and `"high"` switches it on. It is off by default because it adds latency, and a plain explanation request doesn't need it. It is exposed as a toggle for physics and maths problems.

### 5.2 Chat call: `call_gemma_chat()`

```python
chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(system_instruction=CHAT_SYSTEM + material),
    history=past_turns,
)
chat.send_message(question)
```

The server is stateless. The browser keeps the conversation and sends the last 10 turns with each question. The topic, explanation and key points from the original analysis go into the system instruction, so the model knows what "this" refers to when a student says "explain this part again".

### 5.3 Prompt design decisions

- **Ask for JSON in the system instruction, not the user turn.** The schema stays fixed while the user turn only carries the student level.
- **Tell the model what to do when the image is bad.** If text is unreadable, say so in the explanation. If it's not study material, set the topic to "Not study material". Without this, models tend to invent content.
- **Quiz questions must be answerable from the material.** Stops the model from asking general trivia.

### 5.4 Handling imperfect output

The model returns text, and text can be wrong in shape even when the content is fine. Three layers handle that:

1. `extract_json()` removes code fences and takes everything between the first `{` and last `}`.
2. `normalise()` rebuilds the object with the fields the frontend expects. A quiz question with a missing option list or an out-of-range answer index is dropped rather than displayed broken.
3. `/api/analyze` retries once if parsing fails or the explanation is empty.

## 6. API reference

### `POST /api/analyze`

`multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPEG, PNG or WebP, up to 8 MB |
| `level` | string | `Class 6-8`, `Class 9-10`, `Class 11-12` (default), `College` |
| `deep_think` | string | `"true"` to enable thinking mode |

Response `200`:

```json
{
  "topic": "Newton's Second Law",
  "explanation": "...",
  "key_points": ["...", "..."],
  "solution_steps": [],
  "quiz": [{ "q": "...", "options": ["...", "...", "...", "..."], "answer": 2, "why": "..." }],
  "flashcards": [{ "front": "...", "back": "..." }]
}
```

Errors return `{ "error": "message" }` with status 400 (bad input), 413 (too large), 429 (rate limit), 500 (server config) or 502 (Gemma failed).

### `POST /api/chat`

```json
{
  "question": "explain this more simply",
  "material": "topic + explanation + key points as text",
  "history": [{ "role": "user", "text": "..." }, { "role": "model", "text": "..." }]
}
```

Response: `{ "answer": "..." }`

### `GET /health`

Returns `{ "ok": true, "model": "gemma-4-26b-a4b-it" }`. Useful for hosting health checks.

## 7. Frontend notes

- Single file, no build step, no framework. It loads two Google Fonts and nothing else from outside.
- Everything the model returns is inserted with `textContent` and DOM nodes, never `innerHTML`. The only formatting supported is `**bold**`, handled by `rich()`. That keeps model output from injecting markup.
- Design is neubrutalist: thick black borders, hard offset shadows, flat bright colours, slightly rotated blocks. Motion is limited to button presses, a marquee and the loader, and is turned off for users who prefer reduced motion.
- Option pickers are real buttons with `aria-pressed`, tabs use `role="tab"` and `aria-selected`, and the loader and error boxes are live regions.

## 8. Safety and limits

| Concern | What the project does |
|---|---|
| API key exposure | Key lives in a server environment variable only. |
| Large or wrong uploads | MIME type whitelist, 8 MB cap (Flask `MAX_CONTENT_LENGTH` plus an explicit check). |
| Abuse of a free API key | In-memory limit of 12 requests per minute per IP. Fine for one instance. With several instances, move it to Redis or a platform-level limiter. |
| Prompt injection from text inside the photo | Output is treated as data only. It is parsed as JSON, validated and rendered as text. It cannot trigger actions. |
| Data retention | Nothing is saved by the app. |

## 9. Trade-offs I made

- **One model for everything.** Simpler than adding a separate OCR step, and Gemma 4 reads handwriting and diagrams well enough that it wasn't needed. The cost is that bad photos give bad results with no fallback.
- **JSON from a prompt instead of a schema-enforced mode.** Keeps the code identical across both Gemma 4 sizes. The cost is the cleanup code in section 5.4.
- **Stateless server.** The browser owns the chat history. Nothing to clean up server side, but history is lost on refresh.
- **No accounts, no database.** Fast to ship and nothing sensitive to protect. The cost is no saved notes or progress tracking.

## 10. Known weaknesses

- Generated quiz answers are sometimes debatable, especially in subjects with several valid framings.
- Dense full-page photos give shallower output than a cropped section.
- The same page can be phrased differently from one run to the next.

## 11. Ideas for later

- Save sessions and track which quiz questions a student gets wrong
- Spaced-repetition scheduling for flashcards
- Crop tool before upload so students can select just one problem
- Read-aloud for the explanation
- Compare `gemma-4-26b-a4b-it` and `gemma-4-31b-it` on the same set of photos and publish the difference

## 12. Manual test checklist

1. Upload a clear typed textbook page. Expect a sensible topic, 4 to 6 key points, 5 questions, 6 cards.
2. Upload a handwritten page. Expect an explanation that mentions anything unreadable.
3. Upload a maths problem with Deep thinking on. Expect `solution_steps` to be filled.
4. Upload a photo of something unrelated, such as a pet. Expect topic "Not study material".
5. Try a PDF or a 20 MB image. Expect a clear error message, not a crash.
6. Ask three doubts in a row, the last one referring to the first ("what did you say earlier about it?"). Expect the history to carry through.
7. Send 13 requests in a minute. Expect the 429 message.
8. Unset `GEMINI_API_KEY` and restart. Expect a readable server-config error.
9. 
