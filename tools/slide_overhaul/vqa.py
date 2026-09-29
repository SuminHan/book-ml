"""슬라이드 PDF 를 쪽마다 렌더링해 VL 모델로 시각 검수. usage: vqa.py slides.pdf out.json"""
import sys, json, base64, subprocess, tempfile, pathlib, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
pdf, out = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8000/v1/chat/completions"; MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
SCHEMA = {"type": "object", "additionalProperties": False,
  "required": ["text_cut_off", "overlap", "too_crowded", "blank_or_broken", "note"],
  "properties": {"text_cut_off": {"type": "boolean"}, "overlap": {"type": "boolean"},
                 "too_crowded": {"type": "boolean"}, "blank_or_broken": {"type": "boolean"},
                 "note": {"type": "string", "maxLength": 120}}}
PROMPT = """You are checking one lecture slide image (16:9) for LAYOUT DEFECTS only. Do not judge content.
- text_cut_off: some text, formula, table or figure is clipped by the slide edge, runs past the bottom, or runs into the bottom footer bar.
- overlap: two elements visibly overlap each other (text over a figure, text over text, content over the footer/page number).
- too_crowded: text is so dense or small that a student in a classroom could not read it.
- blank_or_broken: the slide body is empty/near-empty when it clearly should have content, shows a missing-image box, or has invisible/garbled text.
A slide that is only a title, a section divider, or a single photo with a caption is NORMAL.
Be conservative: answer true only if you clearly see the defect. note: one short phrase describing what you saw (Korean or English)."""
tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["pdftoppm", "-png", "-r", "110", pdf, str(tmp / "p")], check=True)
imgs = sorted(tmp.glob("p-*.png"), key=lambda p: int(re.search(r"-(\d+)\.png", p.name).group(1)))
def ask(p):
    b64 = base64.b64encode(p.read_bytes()).decode()
    body = {"model": MODEL, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": PROMPT}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "qa", "schema": SCHEMA, "strict": True}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    d = json.loads(r["choices"][0]["message"]["content"]); d["page"] = int(re.search(r"-(\d+)\.png", p.name).group(1)); return d
with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, imgs))
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
bad = [r for r in res if any(r[k] for k in ("text_cut_off", "overlap", "too_crowded", "blank_or_broken"))]
print(f"{len(res)} pages checked, {len(bad)} flagged")
for r in bad: print(f"  p{r['page']}: " + ",".join(k for k in ("text_cut_off","overlap","too_crowded","blank_or_broken") if r[k]) + f" | {r['note']}")
