# StudySnap

Photo of your notes or textbook goes in. A plain-language explanation, a quiz, flashcards and a doubt chat come out.

Built for the **Best Use of Gemma 4** track. The whole project runs on one model, Gemma 4, called through the Gemini API.

## Why I made it

Most students don't struggle because the material is missing. They struggle because they're staring at a page of notes at 11 pm and nothing makes sense. Typing that page into a chatbot is slow, and diagrams or handwriting don't survive the typing. So the input here is just a photo, and the output is the things you actually need to study: an explanation, practice questions and revision cards.

The level is selectable (Class 6-8 up to College) because the same page needs to be explained differently to a Class 8 kid and a first-year student.

## What it does

- Reads a photo of notes, a textbook page, a diagram or a problem
- Explains it in simple words with one everyday analogy
- Lists the key points to remember
- Solves it step by step if the photo contains a problem
- Generates a 5-question multiple choice quiz with instant feedback and a score
- Generates 6 flip-style flashcards
- Answers follow-up doubts in a chat, aware of what was in the photo
- Exports the notes as a Markdown file

## How Gemma 4 is used

Model: `gemma-4-26b-a4b-it` (default). Set `GEMMA_MODEL=gemma-4-31b-it` to switch to the bigger one. Both are served by the Gemini API, so there is no model to host.

All Gemma calls live in `app.py`. There are two of them.

| Where | Gemma 4 feature | What it's for |
|---|---|---|
| `call_gemma_vision()` | Image input (`types.Part.from_bytes`) | The photo and the text prompt go in the same request. Gemma reads handwriting, printed text, formulas and diagrams directly. There is no separate OCR step. |
| `call_gemma_vision()` | System instruction (`ANALYZE_SYSTEM`) | Tells the model it is a tutor and fixes the JSON shape of the reply (topic, explanation, key points, steps, quiz, flashcards). The frontend renders straight from that JSON. |
| `call_gemma_vision()` | Thinking mode (`thinking_level="high"`) | Optional. The "Deep thinking mode" switch in the UI turns this on. Helps on maths and physics problems where the model needs to work through steps before answering. |
| `call_gemma_chat()` | Chat session (`client.chats.create`) | Powers the Doubt tab. Earlier turns are replayed as history, and a summary of the uploaded material is placed in the system instruction so answers stay on topic. |

Things around the model that make it dependable:

- `extract_json()` strips code fences or stray text if the model wraps its JSON.
- `normalise()` checks every quiz question (valid options, answer index in range) and every flashcard, and drops broken ones instead of crashing the page.
- If the reply can't be parsed, `/api/analyze` retries once before giving up.

More detail, including the exact prompt design, is in `documents.md`.

## Project structure

```
studysnap/
├── app.py            Flask backend + all Gemma 4 calls
├── index.html        Entire frontend (HTML, CSS, JS in one file)
├── documents.md      Design notes, architecture, API reference
├── readme.md         This file
├── requirements.txt  Python dependencies
├── Procfile          Start command for Render / Railway / Heroku-style hosts
└── .gitignore
```

## Run it locally

You need Python 3.10+ and a Gemini API key (free from https://aistudio.google.com/apikey).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your_key"  # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
python app.py
```

Open http://localhost:5000.

## Environment variables

| Name | Required | Default | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | none | Server side only. Never sent to the browser. |
| `GEMMA_MODEL` | no | `gemma-4-26b-a4b-it` | Or `gemma-4-31b-it`. |
| `PORT` | no | `5000` | Hosting platforms set this themselves. |

## Deploying

Any host that can run a Python web process works. For Render: push this folder to GitHub, create a Web Service, build command `pip install -r requirements.txt`, start command `gunicorn app:app --workers 2 --threads 4 --timeout 120`, and add `GEMINI_API_KEY` as an environment variable.

## Privacy

Uploaded images are held in memory for the length of one request and sent to the Gemini API. They are not written to disk and not stored by this app. Check Google's terms for how the Gemini API itself handles data.

## Known limits

- The free Gemini API tier has rate limits. The app has its own small per-IP limit (12 requests a minute) and shows a readable message when Google's limit is hit.
- Very blurry or badly lit photos give weaker results. The explanation will say when part of the page was unreadable.
- Quiz and flashcards are generated, so they can occasionally contain mistakes. It is a study aid, not an answer key.
- HEIC photos won't load in most desktop browsers. JPG, PNG and WebP are fine.
- 
