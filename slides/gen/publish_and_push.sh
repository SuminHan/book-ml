#!/bin/bash
# Mirror finished decks into docs/ and push to GitHub main.
# Called by the scheduler after each deck; also safe to run by hand.
set -u
SLIDES="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$SLIDES/.." && pwd)"
cd "$REPO"
MSG="${1:-slides: progress}"

python3 "$SLIDES/gen/build_dashboard.py" >/dev/null 2>&1 || true

# copy every ready PDF into the Pages tree
mkdir -p docs/slides/kor
for p in "$SLIDES"/kor/*.pdf; do
  [ -f "$p" ] && cp -f "$p" "docs/slides/kor/$(basename "$p")"
done

git add -A slides docs/slides 2>/dev/null || true
git diff --cached --quiet && { echo "[push] nothing to commit"; exit 0; }

git -c user.name="KSA slide bot" -c user.email="suminhan@ksa.hs.kr" \
    commit -q -m "$MSG" || { echo "[push] commit failed"; exit 1; }

for try in 1 2 3; do
  git pull --rebase --autostash -q origin main 2>/dev/null || true
  if git push -q origin main 2>/dev/null; then
    echo "[push] ok ($MSG)"; exit 0
  fi
  echo "[push] retry $try ..."; sleep $((try*10))
done
echo "[push] FAILED after retries — commit is local"; exit 1
