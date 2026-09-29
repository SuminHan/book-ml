#!/bin/bash
# usage: build_deck.sh ml1 week02  -> out/<id>-슬라이드.pdf, out/<id>-노트포함.pdf
P=$(cd "$(dirname "$0")" && pwd); S=$(dirname "$P"); C=$1; W=$2; ID=$C-$W
R=${DECKS:-$S/decks}/$ID; D=$R/slides/kor/$C/$W; M=$C-$W.tex
cd "$D" || exit 1
grep -q 'usepackage{varwidth}' $M || sed -i 's/\\begin{document}/\\usepackage{varwidth}\n\\begin{document}/' $M
: > fallback.txt
for round in $(seq 1 40); do
  xelatex -interaction=nonstopmode -halt-on-error -jobname=slides-only $M > b1.log 2>&1 && break
  err=$(grep -m1 -E '^! ' b1.log); pg=$(grep -oE '\(\./pages/p[0-9]+[a-z0-9_]*\.tex' b1.log | tail -1 | sed 's/.*\/\(p[0-9a-z_]*\)\.tex/\1/')
  [ -z "$pg" ] && { echo "UNLOCATED: $err" >> fallback.txt; break; }
  if [ -e pages/$pg.thin ] && ! cmp -s pages/$pg.tex pages/$pg.thin; then cp pages/$pg.tex pages/$pg.failed; cp pages/$pg.thin pages/$pg.tex; echo "$pg -> thin ($err)" >> fallback.txt
  elif [ -e pages/$pg.orig ] && ! cmp -s pages/$pg.tex pages/$pg.orig; then cp pages/$pg.orig pages/$pg.tex; echo "$pg -> orig ($err)" >> fallback.txt
  else echo "$pg -> STUCK ($err)" >> fallback.txt; break; fi
done
xelatex -interaction=nonstopmode -jobname=slides-only $M > b2.log 2>&1
python3 $P/fix_overflow.py $M b2.log slides-only.nav | tee -a fallback.txt
xelatex -interaction=nonstopmode -jobname=slides-only $M > b2.log 2>&1; xelatex -interaction=nonstopmode -jobname=slides-only $M > b2.log 2>&1
# notes-only with invisible frame IDs
rm -rf pagesN; mkdir pagesN
python3 - $M <<'PYEOF'
import sys, re, pathlib
main = open(sys.argv[1], encoding="utf-8").read(); frames = []
for name in re.findall(r'\\input\{pages/([^}]+)\}', main):
    t = pathlib.Path(f"pages/{name}.tex").read_text(encoding="utf-8")
    parts = re.split(r'(\\begin\{frame\})', t); out = parts[0]
    for i, j in enumerate(range(1, len(parts), 2)):
        seg = parts[j] + parts[j+1]
        if r'\note{' not in seg:
            k = seg.rfind(r'\end{frame}'); seg = seg[:k] + "\\note{}\n" + seg[k:]
        k = seg.find(r'\note{')
        if k >= 0: seg = seg[:k] + re.sub(r'\\verb\*?(.)(.*?)\1', lambda m: r'\texttt{' + m.group(2).replace('_', r'\_').replace('#', r'\#').replace('%', r'\%').replace('&', r'\&') + '}', seg[k:])
        seg = seg.replace(r'\note{', r'\note{{\color{white}\tiny NID:%s-%d:NID}' % (name, i), 1)
        frames.append(f"{name}-{i}"); out += seg
    pathlib.Path(f"pagesN/{name}.tex").write_text(out, encoding="utf-8")
open("frame_ids.txt", "w").write("\n".join(frames))
n = main.replace("\\input{pages/", "\\input{pagesN/").replace("\\begin{document}",
    "\\setbeameroption{show only notes}\n\\setbeamertemplate{note page}{\\small\\insertnote}\n\\begin{document}", 1)
open("notes-only.tex", "w", encoding="utf-8").write(n)
# 쪽->프레임 대응용: 슬라이드 쪽에도 보이지 않는 표식(높이 0)을 넣은 별도 빌드
import os
os.makedirs("pagesM", exist_ok=True)
for name in re.findall(r'\\input\{pages/([^}]+)\}', main):
    t = pathlib.Path(f"pages/{name}.tex").read_text(encoding="utf-8")
    parts = re.split(r'(\\end\{frame\})', t); out = ""
    for i, j in enumerate(range(0, len(parts) - 1, 2)):
        out += parts[j] + "\\gdef\\SID{%s-%d}\n" % (name, i) + parts[j+1]
    out += parts[-1]
    pathlib.Path(f"pagesM/{name}.tex").write_text(out, encoding="utf-8")
mk = main.replace("\\input{pages/", "\\input{pagesM/").replace("\\begin{document}", "\\gdef\\SID{none}\n\\setbeamertemplate{background}{\\tiny\\color{white}SID:\\SID:SID}\n\\begin{document}", 1)
open("slides-marked.tex", "w", encoding="utf-8").write(mk)
PYEOF
xelatex -interaction=nonstopmode slides-marked.tex > b5.log 2>&1
xelatex -interaction=nonstopmode notes-only.tex > b3.log 2>&1; xelatex -interaction=nonstopmode notes-only.tex > b3.log 2>&1
python3 - <<'PYEOF'
import re, subprocess
def pages(f): return int(re.search(r'Pages:\s+(\d+)', subprocess.run(["pdfinfo",f],capture_output=True,text=True).stdout).group(1))
def size(f):
    w,h = re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', subprocess.run(["pdfinfo",f],capture_output=True,text=True).stdout).groups(); return float(w), float(h)
nav = open("slides-only.nav", encoding="utf-8").read()
fp = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', nav)
divs = {int(x) for x in re.findall(r'\\sectionentry \{\d+\}\{.*?\}\{(\d+)\}', nav)}
note_of = {}
for q in range(1, pages("notes-only.pdf")+1):
    t = re.sub(r'[\u2010-\u2015\u2212]', '-', subprocess.run(["pdftotext","-f",str(q),"-l",str(q),"notes-only.pdf","-"],capture_output=True,text=True).stdout)
    m = re.search(r'NID:(\S+?):NID', t)
    if m: note_of[m.group(1)] = q
frames = open("frame_ids.txt").read().split("\n")
W, H = size("slides-only.pdf"); body = []; miss = 0
NP = pages("slides-only.pdf")
if pages("slides-marked.pdf") != NP: print("WARN marked/slides page mismatch", pages("slides-marked.pdf"), NP)
sid = []
for p in range(1, NP+1):
    t = re.sub(r'[\u2010-\u2015\u2212]', '-', subprocess.run(["pdftotext","-f",str(p),"-l",str(p),"slides-marked.pdf","-"],capture_output=True,text=True).stdout)
    m = re.search(r'SID:(\S+?):SID', t); sid.append(m.group(1) if m else None)
for p in range(NP-2, -1, -1):          # 표식 없는 쪽(allowframebreaks 앞쪽) = 다음 표식의 프레임
    if sid[p] is None and (p+1) not in divs and sid[p+1] is not None:
        nxt = sid[p+1]; sid[p] = nxt
for p in range(1, NP+1):
    fid = None if p in divs else sid[p-1]; q = note_of.get(fid) if fid else None
    if fid and q is None: miss += 1
    right = r"\includegraphics[page=%d]{notes-only.pdf}" % q if q else r"\rule{%.2fbp}{0pt}" % W
    body.append(r"\noindent\includegraphics[page=%d]{slides-only.pdf}%s" % (p, right))
open("merged.tex","w").write(r"""\documentclass{article}
\usepackage[paperwidth=%.2fbp,paperheight=%.2fbp,margin=0pt]{geometry}
\usepackage{graphicx}\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\topskip}{0pt}
\begin{document}
""" % (2*W, H) + "\n\\newpage\n".join(body) + "\n\\end{document}\n")
print(f"pages {NP} dividers {len(divs)} unmarked {sum(1 for s in sid if s is None)} note-miss {miss}")
PYEOF
pdflatex -interaction=nonstopmode merged.tex > b4.log 2>&1
OUTD=${OUT:-$S/out}; mkdir -p $OUTD
cp slides-only.pdf "$OUTD/$ID-슬라이드.pdf"
gs -q -o "$OUTD/$ID-노트포함.pdf" -sDEVICE=pdfwrite -dDetectDuplicateImages=true -dCompressFonts=true -dSubsetFonts=true -dPDFSETTINGS=/prepress merged.pdf
echo "built $ID: slides $(pdfinfo slides-only.pdf | awk '/^Pages/{print $2}')p, merged $(pdfinfo "$OUTD/$ID-노트포함.pdf" | awk '/^Pages/{print $2}')p, fallbacks $(wc -l < fallback.txt)"
~/qwen-setup/venv/bin/python $P/vqa.py "$D/slides-only.pdf" "$R/vqa.json" > "$R/vqa.txt" 2>&1 && head -1 "$R/vqa.txt"
python3 $P/overfull.py "$D/b2.log" | tee "$R/overflow.txt"
