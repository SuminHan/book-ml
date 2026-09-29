"""xelatex 로그에서 '슬라이드 아래로 넘친' 쪽 번호 추출. usage: overfull.py slides-only.log"""
import re, sys
log = open(sys.argv[1], encoding='utf-8', errors='replace').read()
page = 0; hits = []
for m in re.finditer(r'\[(\d+)(?=[\s\]<{])|Overfull \\vbox \(([\d.]+)pt too high\)', log):
    if m.group(1): page = int(m.group(1))
    elif float(m.group(2)) > 2: hits.append((page + 1, float(m.group(2))))
print("overflow pages (>2pt):", ", ".join(f"p{p}(+{h:.0f}pt)" for p, h in hits) if hits else "none")
