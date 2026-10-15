"""docs/datashop/worldcup/index.html 의 후보 목록 · 사진 출처 · 카드 사진을 candidates.json · covers.json 에 맞춘다.
   ~/Documents/manim_shorts/_env/v/bin/python worldcup/update_page.py"""
import io, json, os, re, urllib.request
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
PAGE = os.path.join(ROOT, "docs", "datashop", "worldcup", "index.html"); IMG = os.path.join(ROOT, "docs", "datashop", "worldcup", "img")
cands = json.load(open(os.path.join(HERE, "candidates.json"))); covers = json.load(open(os.path.join(HERE, "..", "covers.json")))
credits = {}
for c in cands:
    i = c["id"]; path = os.path.join(IMG, i + ".jpg")
    if i in covers and covers[i].get("local"):
        cv = covers[i]
        credits[i] = dict(title=cv["title"], artist=cv["artist"], license=cv["license"], page=cv["page"]) if cv.get("page") else None
        im = Image.open(os.path.join(HERE, "..", cv["local"])).convert("RGB"); w, h = 480, 300
        r = min(w / im.width, h / im.height); im = im.resize((round(im.width * r), round(im.height * r)))
        canvas = Image.new("RGB", (w, h), "white"); canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); canvas.save(path, quality=80)
    elif i in covers:
        cv = covers[i]
        credits[i] = dict(title=cv["title"].replace("File:", ""), artist=cv["artist"] or "작자 미상", license=cv["license"], page=cv["page"])
        if not os.path.exists(path):
            raw = urllib.request.urlopen(urllib.request.Request(cv["thumb"], headers={"User-Agent": "ML course datashop"}), timeout=120).read()
            im = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = 480, 300
            r = max(w / im.width, h / im.height); im = im.resize((round(im.width * r), round(im.height * r)))
            l, t = (im.width - w) // 2, (im.height - h) // 2; im.crop((l, t, l + w, t + h)).save(path, quality=72)
            print("image", i)
    else:
        credits[i] = None          # drawn from the data (자금세탁 패턴 · 법인카드 지출 그래프)
s = open(PAGE, encoding="utf-8").read()
s = re.sub(r"const CANDS = \[.*?\];\n", lambda m: "const CANDS = " + json.dumps(cands, ensure_ascii=False) + ";\n", s, count=1, flags=re.S)
s = re.sub(r"const CREDITS = \{.*?\};\n", lambda m: "const CREDITS = " + json.dumps(credits, ensure_ascii=False) + ";\n", s, count=1, flags=re.S)
n = len(cands)
s = re.sub(r"공개 데이터 \d+개\.", f"공개 데이터 {n}개.", s)
s = re.sub(r'\[\[\d+, "\d+강", "[^"]*"\], \[\d+, "\d+강", "[^"]*"\]\]', f'[[32, "32강", "{n}개 중 무작위 32개 · 31경기"], [16, "16강", "{n}개 중 무작위 16개 · 15경기"]]', s)
s = re.sub(r"let mode = \d+;", "let mode = 32;", s)
s = s.replace('if (!c) { li.textContent = `${byId[id].title}: 직접 그린 패턴 그림 (원형 · 쪼개기와 모으기 · 한곳에 모으기)`; }',
              'if (!c) { li.textContent = `${byId[id].title}: 데이터로 직접 그린 그림`; }')
open(PAGE, "w", encoding="utf-8").write(s)
print("candidates", n)
