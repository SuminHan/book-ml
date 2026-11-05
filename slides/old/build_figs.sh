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

resolve() {  # <NAME as it appears in \includegraphics{}> -> 0 if usable in figs/, else 1
  local orig="$1" name="${1%.png}"; name="${name%.svg}"
  # already sitting in figs/ under the exact name requested (any extension,
  # e.g. a .jpg copied straight from src/images -- LaTeX includes it as-is)
  [ -f "$OUT/$orig" ] && return 0
  [ -f "$OUT/$name.png" ] && return 0
  if [ -f "$SRC/$name.svg" ]; then
    rsvg-convert -b white -z 3 "$SRC/$name.svg" -o "$OUT/$name.png" 2>/dev/null \
      && { echo "  svg->png  $name.png"; return 0; }
  elif [ -f "$SRC/$orig" ]; then
    flatten_raster "$SRC/$orig" "$OUT/$name.png" 2>/dev/null \
      && { echo "  flatten   $name.png"; return 0; }
  elif [ -f "$SRC/$name.png" ]; then
    flatten_raster "$SRC/$name.png" "$OUT/$name.png" 2>/dev/null \
      && { echo "  flatten   $name.png"; return 0; }
  fi
  echo "  MISSING   $orig  (not in $OUT/, no $SRC/$orig or $SRC/$name.{svg,png})"
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

# Since the "덱마다 개별 폴더" reorg, each deck keeps its own figs/ next to
# its .tex (referenced first via \graphicspath{{figs/}{./}{../../../figs/}}
# in the theme) -- resolve into THAT, not the old shared slides/figs/ (which
# is now only a fallback for legacy shared assets like ksa-logo).
OUT="$(dirname "$TEX")/figs"
mkdir -p "$OUT"

# A migrated deck (per-page pages/pNN.tex, wrapper .tex is just \input lines)
# keeps its \includegraphics calls inside pages/, not in $TEX itself -- scan
# those too, and neutralise missing figures in whichever page file has them.
PAGES_DIR="$(dirname "$TEX")/pages"
SEARCH_FILES=("$TEX")
if [ -d "$PAGES_DIR" ]; then
  while IFS= read -r -d '' f; do SEARCH_FILES+=("$f"); done \
    < <(find "$PAGES_DIR" -maxdepth 1 -name '*.tex' -print0 | sort -z)
fi

echo "[build_figs] resolving figures in ${SEARCH_FILES[*]}"
grep -hoE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "${SEARCH_FILES[@]}" \
  | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u \
  | while read -r fig; do
      [ "$fig" = "ksa-logo.png" ] && continue
      if ! resolve "$fig"; then
        # comment out every \includegraphics referencing this missing figure,
        # in every file (wrapper or page) that references it. Write to a temp
        # file + mv instead of `sed -i` -- BSD sed's `-i` needs an explicit
        # (possibly empty) backup-suffix arg or it silently eats the next
        # flag, which breaks `-i -E` on macOS.
        esc=$(printf '%s\n' "$fig" | sed 's/[.[\*^$/]/\\&/g')
        for f in "${SEARCH_FILES[@]}"; do
          grep -q "includegraphics.*{$fig}" "$f" || continue
          sed -E "s|^([[:space:]]*)(\\\\includegraphics(\\[[^]]*\\])?\\{$esc\\})|\1% [fig missing] \2|" "$f" > "$f.tmp" \
            && mv "$f.tmp" "$f"
          echo "  -> neutralised \\includegraphics{$fig} in $(basename "$f")"
        done
      fi
    done
echo "[build_figs] done -> $OUT/"
