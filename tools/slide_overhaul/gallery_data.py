"""덱별 추가 이미지 수집 -> gallery/data.json (슬라이드 썸네일 포함)"""
import re, json, sys, base64, subprocess, pathlib
S = pathlib.Path(sys.argv[1]); decks = sys.argv[2:]
out = []
def plain(t): return re.sub(r'\s+', ' ', re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?|[{}$\\]', ' ', t)).strip()
for ID in decks:
    C, W = ID.split('-'); D = S / 'decks' / ID / 'slides/kor' / C / W
    src = {}
    sp = D / 'figs/SOURCES.md'
    if sp.exists():
        for line in sp.read_text(encoding='utf-8').splitlines():
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 2 and '.' in parts[0] and not parts[0].startswith('파일'): src[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else '')
    nav = (D / 'slides-only.nav').read_text(encoding='utf-8')
    fp = [(int(a), int(b)) for a, b in re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', nav)]
    divs = {int(x) for x in re.findall(r'\\sectionentry \{\d+\}\{.*?\}\{(\d+)\}', nav)}
    fids = (D / 'frame_ids.txt').read_text().split('\n'); page_of = {}; k = 0
    for a, b in fp:
        if a in divs: continue
        if k < len(fids): page_of[fids[k].rsplit('-', 1)[0]] = a
        k += 1
    for p in sorted((D / 'pages').glob('*.tex'), key=lambda x: x.name):
        t = p.read_text(encoding='utf-8'); stem = p.stem
        imgs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', t)
        new = 'g' in re.sub(r'^p\d+[a-z]?', '', stem) and re.search(r'g\d+$', stem)
        thin = (D / 'pages' / (stem + '.thin'))
        old = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', thin.read_text(encoding='utf-8'))) if thin.exists() else set()
        added = [i for i in imgs if new or i not in old]
        if not added: continue
        m = re.search(r'\\begin\{frame\}(?:\[[^\]]*\])?\{([^}]*)\}', t)
        page = page_of.get(stem)
        thumb = ''
        if page:
            subprocess.run(['pdftoppm', '-jpeg', '-jpegopt', 'quality=72', '-r', '72', '-f', str(page), '-l', str(page), '-singlefile', str(D / 'slides-only.pdf'), '/tmp/_g'], check=True)
            thumb = 'data:image/jpeg;base64,' + base64.b64encode(open('/tmp/_g.jpg', 'rb').read()).decode()
        for i in added:
            u, what = src.get(i, (None, ''))
            out.append({'deck': ID, 'page_file': stem, 'pdf_page': page, 'title': plain(m.group(1)) if m else '(사진 슬라이드)',
                        'image': i, 'kind': '사진' if u else '도식', 'source': u, 'what': what, 'new_frame': bool(new),
                        'caption': plain(re.sub(r'\\note\{.*', '', t, flags=re.S))[:160], 'thumb': thumb})
json.dump(out, open(S / 'gallery/data.json', 'w'), ensure_ascii=False)
print(len(out), 'images;', {d: sum(1 for o in out if o['deck'] == d) for d in decks}, '| photos', sum(o['kind'] == '사진' for o in out))
