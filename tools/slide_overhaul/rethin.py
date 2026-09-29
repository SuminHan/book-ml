"""labels.json 으로 .orig 에서 재조립 다시 (Qwen 재실행 없음). usage: rethin.py DECKROOT..."""
import sys, os, json, pathlib; sys.path.insert(0, os.path.dirname(__file__))
from reassemble import reassemble, center_frame
for R in sys.argv[1:]:
    R = pathlib.Path(R); ID = R.name; C, W = ID.split('-'); P = R / 'slides/kor' / C / W / 'pages'
    L = {r['page']: r for r in json.load(open(R / 'labels.json', encoding='utf-8'))}; n = f = 0
    for o in sorted(P.glob('*.orig')):
        src = o.read_text(encoding='utf-8'); r = L.get(o.stem)
        if r and not r.get('skip'):
            try: src, _ = reassemble(src, r); n += 1
            except Exception as e: f += 1
        (P / (o.stem + '.thin')).write_text(src, encoding='utf-8')
        (P / (o.stem + '.tex')).write_text(center_frame(src), encoding='utf-8')
    print(ID, 'rethinned', n, 'fail', f)
