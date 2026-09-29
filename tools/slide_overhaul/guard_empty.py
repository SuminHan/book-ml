"""글만 있는 프레임이 KEEP=0 이면 첫 항목을 KEEP 으로 (labels.json 보정) + 에이전트가 안 건드린 쪽만 재조립."""
import sys, os, json, glob, pathlib; sys.path.insert(0, os.path.dirname(__file__))
from reassemble import reassemble, center_frame
S = sys.argv[1]; busy = set(sys.argv[2:])
for lf in sorted(glob.glob(f"{S}/decks/*/labels.json")):
    R = pathlib.Path(lf).parent; ID = R.name
    if ID in busy: continue
    C, W = ID.split('-'); P = R / 'slides/kor' / C / W / 'pages'
    L = json.load(open(lf, encoding='utf-8')); changed = []
    for r in L:
        if r.get('skip') or r['has_visual'] or any(u['label'] == 'KEEP' for u in r['units']): continue
        old = dict(r, units=[dict(u) for u in r['units']])
        r['units'][0]['label'] = 'KEEP'; r['units'][0]['headline'] = None
        orig = P / (r['page'] + '.orig'); tex = P / (r['page'] + '.tex'); thin = P / (r['page'] + '.thin')
        src = orig.read_text(encoding='utf-8')
        try: old_thin, _ = reassemble(src, old)
        except Exception: old_thin = None
        untouched = tex.read_text(encoding='utf-8') in (thin.read_text(encoding='utf-8'), center_frame(thin.read_text(encoding='utf-8')))
        new_thin, _ = reassemble(src, r)
        thin.write_text(new_thin, encoding='utf-8')
        if untouched: tex.write_text(center_frame(new_thin), encoding='utf-8'); changed.append(r['page'])
        else: changed.append(r['page'] + '(에이전트 수정본 유지)')
    if changed:
        json.dump(L, open(lf, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(ID, changed)
