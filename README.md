# auto-annotate

POC: can the Qwen VLM auto-annotate medicine images with region-level boxes?

Two classes: `name` (product name) and `data` (batch no, MRP, mfg/expiry).

## Run

```
uv venv .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python app.py          # http://127.0.0.1:7860
.venv/bin/python test_app.py     # parser self-check
```

Model and endpoint are configured in `.env` (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`).

## Tuning notes (measured, not guessed)

- A control test proved the coordinate math is correct: on large objects
  (bottle / cap / label) the boxes are tight. Early bad results were the model
  mis-picking regions, not a scaling bug.
- Small photos are scaled **up** to a 1280px long edge before sending. A
  471x768 phone crop gave loose, wrong boxes at native size and correct ones
  once enlarged.
- The prompt names the exact traps seen in real photos: it must not pick
  warning text ("DO NOT MIX WITH WATER") as the name, and the `data` box must
  include the field labels, not just the values column.
- `name` is English-only; it is omitted when a label carries no English
  product name (e.g. Devanagari-only branding).
- The model occasionally replies with no JSON at all. That surfaces as an error
  with the raw response, never as invented boxes.

## Notes

- The model returns coords **normalized 0-1000**, not pixels. `app.py` rescales;
  the UI radio overrides the guess if a render looks wrong.
- `LLM_REASONING_EFFORT` is ignored — this endpoint 400s on `reasoning_effort`.
- Invalid JSON shows the raw response and an error. No boxes are ever invented.
