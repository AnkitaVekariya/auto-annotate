"""Parser self-check. Fixtures are real replies captured from the endpoint.

Run: .venv/bin/python test_app.py
"""

from app import parse_annotations

# 1. The model's native shape, in a ```json fence -- what it actually returned
#    for a 1600x400 image during probing.
NATIVE = """```json
[
\t{"bbox_2d": [575, 25, 875, 125], "label": "name"},
\t{"bbox_2d": [50, 575, 250, 875], "label": "data"}
]
```"""
got = parse_annotations(NATIVE, 1600, 400)
assert got == [
    {"class": "name", "bbox": [920, 10, 1400, 50]},
    {"class": "data", "bbox": [80, 230, 400, 350]},
], got

# 2. The spec shape, wrapped in an object -- also seen during probing (1000x600).
SPEC = """```json
{"image_width": 1000, "image_height": 600, "annotations": [
  {"class": "name", "bbox": [72, 161, 529, 320]},
  {"class": "data", "bbox": [72, 656, 529, 857]}]}
```"""
got = parse_annotations(SPEC, 1000, 600)
assert got == [
    {"class": "name", "bbox": [72, 97, 529, 192]},
    {"class": "data", "bbox": [72, 394, 529, 514]},
], got

# 3. A value >1000 means the reply is already in pixels; auto must not rescale.
PIXELS = '[{"bbox": [100, 200, 1500, 900], "class": "name"}]'
assert parse_annotations(PIXELS, 1600, 1000) == [
    {"class": "name", "bbox": [100, 200, 1500, 900]}
]

# 4. Pixel coords running past the image edge are clipped to it.
assert parse_annotations('[{"bbox":[100,200,9999,9999],"class":"name"}]', 1600, 1000) == [
    {"class": "name", "bbox": [100, 200, 1600, 1000]}
]

# 5. Prose around the JSON is tolerated; a trailing brace must not break it.
assert parse_annotations(
    'Here are the regions: [{"bbox_2d":[0,0,500,500],"label":"NAME"}] (approx)',
    200, 200,
) == [{"class": "name", "bbox": [0, 0, 100, 100]}]

# 6. Inverted coords are normalized rather than silently dropped.
assert parse_annotations('[{"bbox_2d":[800,600,200,100],"label":"data"}]', 100, 100) == [
    {"class": "data", "bbox": [20, 10, 80, 60]}
]

# 7. Degenerate (zero-area) boxes are dropped, not drawn.
assert parse_annotations('[{"bbox_2d":[500,500,500,900],"label":"data"}]', 100, 100) == []

# 8. Anything unparseable must RAISE -- never return invented boxes.
for bad in (
    "I could not find any medicine name in this image.",
    '[{"label": "name"}]',                    # no bbox
    '[{"bbox_2d": [1, 2, 3], "label": "n"}]', # wrong arity
    '{"result": "none"}',                     # no annotations key
    '["name", "data"]',                       # not objects
):
    try:
        parse_annotations(bad, 100, 100)
    except (ValueError, KeyError, TypeError):
        pass
    else:
        raise AssertionError(f"should have raised: {bad!r}")

print("all parser checks passed")
