"""넘친 쪽 -> 해당 page 파일의 frame 에 shrink 옵션. usage: fix_overflow.py MAIN.tex slides-only.log slides-only.nav"""
import re, sys, pathlib
main, log, nav = [open(f, encoding='utf-8', errors='replace').read() for f in sys.argv[1:4]]
page = 0; over = {}
for m in re.finditer(r'\[(\d+)(?=[\s\]<{])|Overfull \\vbox \(([\d.]+)pt too high\)', log):
    if m.group(1): page = int(m.group(1))
    elif float(m.group(2)) > 2: over[page + 1] = max(over.get(page + 1, 0), float(m.group(2)))
fp = [int(a) for a, b in re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', nav)]
divs = {int(x) for x in re.findall(r'\\sectionentry \{\d+\}\{.*?\}\{(\d+)\}', nav)}
files = []
for name in re.findall(r'\\input\{pages/([^}]+)\}', main):
    t = pathlib.Path(f'pages/{name}.tex').read_text(encoding='utf-8'); files += [name] * len(re.findall(r'\\begin\{frame\}', t))
start2file = {}; k = 0
for a in fp:
    if a in divs: continue
    if k < len(files): start2file[a] = files[k]
    k += 1
done = []
for pg, h in over.items():
    f = start2file.get(pg)
    if not f: continue
    p = pathlib.Path(f'pages/{f}.tex'); t = p.read_text(encoding='utf-8')
    if 'shrink' in t or 'allowframebreaks' in t: continue
    pct = min(40, int(h / (215 + h) * 100) + 4)
    t2 = re.sub(r'\\begin\{frame\}\[([^\]]*)\]', lambda m: '\\begin{frame}[' + m.group(1) + f',shrink={pct}]', t, count=1)
    if t2 == t: t2 = t.replace('\\begin{frame}', f'\\begin{{frame}}[shrink={pct}]', 1)
    p.write_text(t2, encoding='utf-8'); done.append(f'{f}(p{pg},+{h:.0f}pt->shrink{pct})')
print('shrunk:', ', '.join(done) if done else 'none')
