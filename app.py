"""Medicine image auto-annotation POC.

Tests whether the configured Qwen model can produce region-level bounding boxes
for two classes: `name` (product name) and `data` (batch/MRP/mfg/expiry).
"""

import base64
import io
import json
import os
import time

import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "").strip('"')
MODEL = os.environ.get("LLM_MODEL", "openai/qwen")
# LLM_REASONING_EFFORT is deliberately NOT sent: this endpoint returns
# HTTP 400 litellm.UnsupportedParamsError for reasoning_effort on this model.

TARGET_EDGE = 1280  # long edge we send; small photos are scaled UP (see call_model)
COLORS = {"name": "#e53935", "data": "#1e88e5"}

PROMPT = """This is a photo of a medicine package or bottle.

Find TWO regions on the printed label:

- "name": the BRAND / PRODUCT NAME of the medicine, in ENGLISH only. It is the
  largest, most prominent English text on the label, usually at the top.
  Consider ONLY English / Latin-script text. Ignore any text printed in
  Devanagari/Hindi or other non-English scripts -- never box it. Do NOT choose
  warning or instruction text such as "DO NOT MIX WITH WATER" or "Keep all
  medicines out of reach of children", and do NOT choose the manufacturer name.
  If no English product name is visible, omit "name" entirely.

- "data": the block of batch and price information. Include BOTH the field
  labels ("Batch No.", "B.No.", "Mfg. Date", "Expiry Date", "EXP", "Maximum
  Retail Price", "MRP") AND their printed values. Return ONE box covering the
  ENTIRE block, from the leftmost field label to the rightmost value, top row to
  bottom row.

Return exactly one box per label, covering the whole region. Do not return one
box per text line or per word. If a label is genuinely absent, omit it; never guess.

Reply with ONLY a JSON array, no prose:
[{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "name"},
 {"bbox_2d": [x_min, y_min, x_max, y_max], "label": "data"}]

Coordinates must be integers normalized to 0-1000 on both axes."""


def extract_json(text):
    """Pull the first balanced {...} or [...] block out of a model reply."""
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError("no JSON object or array found in response")
    opener = text[start]
    closer = {"{": "}", "[": "]"}[opener]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON in response")


def parse_annotations(raw, width, height):
    """Model reply -> [{"class": str, "bbox": [x0,y0,x1,y1]}] in image pixels.

    Accepts the model's native `bbox_2d`/`label` shape and the `bbox`/`class`
    shape, bare list or wrapped in {"annotations": [...]}. Raises on anything
    it cannot understand -- it never invents boxes.
    """
    data = extract_json(raw)
    if isinstance(data, dict):
        items = data.get("annotations", data.get("objects"))
        if items is None:
            raise ValueError("JSON object has no 'annotations' list")
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("expected a list of annotations")

    boxes = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError(f"annotation is not an object: {it!r}")
        bbox = it.get("bbox_2d", it.get("bbox"))
        label = it.get("label", it.get("class"))
        if bbox is None or label is None:
            raise ValueError(f"annotation missing bbox/label: {it!r}")
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 values: {bbox!r}")
        boxes.append((str(label).strip().lower(), [float(v) for v in bbox]))

    # This endpoint returns 0-1000 normalized coords (verified against a control
    # test on large objects); a value >1000 means it switched to raw pixels.
    normalized = all(v <= 1000 for _, b in boxes for v in b)

    out = []
    for label, (x0, y0, x1, y1) in boxes:
        if normalized:
            x0, x1 = x0 / 1000 * width, x1 / 1000 * width
            y0, y1 = y0 / 1000 * height, y1 / 1000 * height
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        box = [
            max(0, round(x0)),
            max(0, round(y0)),
            min(width, round(x1)),
            min(height, round(y1)),
        ]
        if box[2] > box[0] and box[3] > box[1]:  # drop empty/off-image boxes
            out.append({"class": label, "bbox": box})
    return out


def call_model(image):
    """POST the image to the endpoint. Returns (raw_text, seconds)."""
    # Scale the long edge to TARGET_EDGE -- UP for small photos as well as down.
    # Measured: a 471x768 phone crop produced loose, mis-picked boxes at native
    # size and correct ones once enlarged; the encoder gets more image tokens.
    scale = TARGET_EDGE / max(image.size)
    small = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    buf = io.BytesIO()
    small.convert("RGB").save(buf, "JPEG", quality=95)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
    return r.json()["choices"][0]["message"]["content"], elapsed


def draw(image, annotations):
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w = max(2, round(min(out.size) / 250))
    for a in annotations:
        x0, y0, x1, y1 = a["bbox"]
        color = COLORS.get(a["class"], "#43a047")
        d.rectangle([x0, y0, x1, y1], outline=color, width=w)
        d.text((x0 + w + 2, max(0, y0 - 14 * w)), a["class"], fill=color)
    return out


def annotate(image):
    if image is None:
        return None, "", "", "", "Upload an image first."
    try:
        raw, elapsed = call_model(image)
    except Exception as e:
        return None, "", "", "", f"Request failed: {e}"

    timing = f"{elapsed:.2f} s"
    try:
        annotations = parse_annotations(raw, image.width, image.height)
    except Exception as e:
        # Explicit requirement: show the raw response, draw nothing, invent nothing.
        return None, raw, "", timing, f"Could not parse model response: {e}"

    parsed = {
        "image_width": image.width,
        "image_height": image.height,
        "annotations": annotations,
    }
    warn = "" if annotations else "Model returned no usable boxes."
    return draw(image, annotations), raw, json.dumps(parsed, indent=2), timing, warn


with gr.Blocks(title="Medicine auto-annotation POC") as demo:
    gr.Markdown(f"### Medicine auto-annotation POC\nModel: `{MODEL}` @ `{BASE_URL}`")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload medicine image")
            btn = gr.Button("Annotate", variant="primary")
            err = gr.Textbox(label="Error", interactive=False)
            ms = gr.Textbox(label="Inference time", interactive=False)
        with gr.Column():
            out_img = gr.Image(label="Annotated (name=red, data=blue)")
            out_parsed = gr.Code(label="Parsed annotations (pixels)", language="json")
            out_raw = gr.Textbox(label="Raw model response", lines=12, max_lines=30)

    btn.click(annotate, inp, [out_img, out_raw, out_parsed, ms, err])

if __name__ == "__main__":
    demo.launch()
