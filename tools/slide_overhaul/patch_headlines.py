"""이미 이미지 작업이 끝난 덱: 페이지 안의 옛 headline 만 제자리 교체. usage: patch_headlines.py DECKROOT..."""
import sys, os, re, json, pathlib; sys.path.insert(0, os.path.dirname(__file__))
from reassemble import esc, headline_latex, restore
from split_frames import split_frame
for R in sys.argv[1:]:
    R = pathlib.Path(R); ID = R.name; C, W = ID.split('-'); P = R / 'slides/kor' / C / W / 'pages'
    L = json.load(open(R / 'labels.json', encoding='utf-8')); fixed = dropped = missing = 0
    for r in L:
        if r.get('skip'): continue
        orig = P / (r['page'] + '.orig')
        if not orig.exists(): continue
        sf = split_frame(orig.read_text(encoding='utf-8'))
        for u, l in zip(sf['units'], r['units']):
            if l['label'] != 'KEEP' or not l.get('headline'): continue
            old = r'\textbf{' + esc(l['headline']) + '}'
            lat = restore(u['latex'], sf['fixed']); hl = headline_latex(l['headline'], lat)
            for fn in (P / (r['page'] + '.tex'), P / (r['page'] + '.thin')):
                t = fn.read_text(encoding='utf-8')
                if old not in t:
                    if fn.suffix == '.tex': missing += 1
                    continue
                if hl:
                    t = t.replace(old, r'\textbf{' + hl + '}', 1); fixed += fn.suffix == '.tex'
                else:   # 원문을 화면으로 되돌리고 노트의 중복 항목 제거
                    t = t.replace(old, lat, 1)
                    k = t.find(r'\note{')
                    if k >= 0: t = t[:k] + t[k:].replace(r'\item ' + lat, '', 1)
                    dropped += fn.suffix == '.tex'
                t = re.sub(r'\\note\{\\begin\{itemize\}\s*\\end\{itemize\}\}', '', t)
                fn.write_text(t, encoding='utf-8')
    print(ID, 'restored-math', fixed, 'dropped->original', dropped, 'not-found', missing)
