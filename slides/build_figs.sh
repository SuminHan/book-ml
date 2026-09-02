#!/bin/bash
# Resolve every \includegraphics{NAME} in a .tex deck to a slide-ready PNG in
# figs/  (transparent backgrounds flattened to white, SVGs rasterised at 3x).
# Any NAME that cannot be resolved is NEUTRALISED in the .tex (the
# \includegraphics line is commented out) so a hallucinated filename never
# breaks the xelatex build.
#
#   bash build_figs.sh kor/ml1-week03.tex
#   bash build_figs.sh                     # base set only (ksa-logo)
set -u
cd "$(dirname "$0")"
SRC=../kor/src/images          # book images live in book-ml/kor/src/images
OUT=figs
mkdir -p "$OUT"

flatten_raster() { convert "$1" -background white -alpha remove -alpha off "$2"; }

resolve() {  # <NAME> -> 0 if figs/<name>.png now exists, else 1
  local name="${1%.png}"; name="${name%.svg}"
  [ -f "$OUT/$name.png" ] && return 0
  if [ -f "$SRC/$name.svg" ]; then
    rsvg-convert -b white -z 3 "$SRC/$name.svg" -o "$OUT/$name.png" 2>/dev/null \
      && { echo "  svg->png  $name.png"; return 0; }
  elif [ -f "$SRC/$name.png" ]; then
    flatten_raster "$SRC/$name.png" "$OUT/$name.png" 2>/dev/null \
      && { echo "  flatten   $name.png"; return 0; }
  fi
  echo "  MISSING   $name  (no $SRC/$name.{svg,png})"
  return 1
}

# KSA logo already cropped by hand into figs/ksa-logo.png; only rebuild from a
# raster asset if the cropped file is somehow gone.
if [ ! -f figs/ksa-logo.png ]; then
  for ext in png jpg jpeg; do
    [ -f "assets/ksa-logo.$ext" ] && { flatten_raster "assets/ksa-logo.$ext" figs/ksa-logo.png; break; }
  done
fi

[ $# -eq 0 ] && { echo "[build_figs] base set only."; exit 0; }

TEX="$1"
[ -f "$TEX" ] || { echo "no such file: $TEX"; exit 1; }
echo "[build_figs] resolving figures in $TEX"
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$TEX" \
  | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u \
  | while read -r fig; do
      [ "$fig" = "ksa-logo.png" ] && continue
      if ! resolve "$fig"; then
        # comment out every \includegraphics referencing this missing figure
        esc=$(printf '%s\n' "$fig" | sed 's/[.[\*^$/]/\\&/g')
        sed -i -E "s|^([[:space:]]*)(\\\\includegraphics(\\[[^]]*\\])?\\{$esc\\})|\1% [fig missing] \2|" "$TEX"
        echo "  -> neutralised \\includegraphics{$fig} in $(basename "$TEX")"
      fi
    done
echo "[build_figs] done -> $OUT/"
