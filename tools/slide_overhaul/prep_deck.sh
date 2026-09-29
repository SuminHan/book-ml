#!/bin/bash
# usage: prep_deck.sh ml1 week02   -> scratch copy + Qwen v2 labels + thinned/centered pages
set -e
P=$(cd "$(dirname "$0")" && pwd); S=$(dirname "$P"); C=$1; W=$2; ID=$C-$W
REPO=/home/smhan/book-ml; R=$S/decks/$ID; D=$R/slides/kor/$C/$W
rm -rf "$R"; mkdir -p "$R/slides/kor/$C"
cp -r $REPO/slides/theme "$R/slides/theme"; cp -r $REPO/slides/figs "$R/slides/figs"
sed -i "s/^\\\\usepackage{listings}$/\\\\usepackage{listings}\\n\\\\usepackage{varwidth}/" "$R/slides/theme/ksa-theme.tex"
cp -r $REPO/slides/kor/$C/$W "$R/slides/kor/$C/"
rm -f "$D"/*.pdf "$D"/*.aux "$D"/*.nav "$D"/*.snm "$D"/*.toc "$D"/*.out "$D"/*.log "$D"/*.vrb
PY=~/qwen-setup/venv/bin/python
SYSTEM_FILE=$P/system_v2.txt EXEMPT='정리|확인 문제|개요|목차|자주 묻는|실습' S=$P \
  $PY $P/run_week.py "$D/pages" "$R/labels.json" > "$R/qwen.log" 2>&1
S=$P $PY - "$D" "$R/labels.json" <<'PYEOF'
import os, sys, json, pathlib; sys.path.insert(0, os.environ["S"])
from reassemble import reassemble, center_frame
D = pathlib.Path(sys.argv[1]); P = D / "pages"
R = {r["page"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}
for p in sorted(P.glob("*.tex")):
    src = p.read_text(encoding="utf-8"); (P / (p.stem + ".orig")).write_text(src, encoding="utf-8")
    r = R.get(p.stem)
    if r and not r.get("skip"):
        try: src, _ = reassemble(src, r)
        except Exception as e: print("reassemble-fail", p.stem, e)
    (P / (p.stem + ".thin")).write_text(src, encoding="utf-8")
    p.write_text(center_frame(src), encoding="utf-8")
PYEOF
echo "prepped $ID: $(tail -n +1 "$R/qwen.log" | head -1)"
