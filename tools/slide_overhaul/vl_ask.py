"""로컬 VL 모델에게 이미지 질문. usage: vl_ask.py <image.png|jpg|pdf[:page]> "질문"  -> 답(텍스트)"""
import sys, base64, json, subprocess, tempfile, pathlib, urllib.request
src, q = sys.argv[1], sys.argv[2]
p = pathlib.Path(src.split(":")[0])
if p.suffix.lower() == ".pdf":
    page = src.split(":")[1] if ":" in src else "1"; t = pathlib.Path(tempfile.mkdtemp())
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", page, "-l", page, "-singlefile", str(p), str(t / "x")], check=True); p = t / "x.png"
else:
    t = pathlib.Path(tempfile.mkdtemp()); o = t / "x.png"   # 너무 큰 이미지는 축소
    subprocess.run(["convert", str(p) + "[0]", "-resize", "1280x1280>", "-background", "white", "-flatten", str(o)], check=True); p = o
b64 = base64.b64encode(p.read_bytes()).decode()
body = {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "temperature": 0, "max_tokens": 300,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": q}]}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
print(r["choices"][0]["message"]["content"].strip())
