#!/bin/bash
# Auto-repair a small set of PROVEN-SAFE slips in a cq-generated deck .tex.
# Deliberately minimal: anything ambiguous (stray $, \mid in text, nested math)
# is left for the manual review pass so we never corrupt a good deck.
# Usage: bash fix_tex.sh <tex>
set -u
F="$1"
[ -f "$F" ] || { echo "fix_tex: no such file $F"; exit 1; }

perl -0777 -i -pe '
  # 1. "\*" star notation cq uses for optimal values (w^\*  \pi^\*  V^\* ...)
  s/\^\\\*/^{*}/g;
  s/(?<!\\)\\\*/*/g;

  # 1b. doubled backslash on a command right after / before a "$"  ($\\to$ -> $\to$)
  s/\$\\\\([a-zA-Z])/\$\\$1/g;
  s/\\\\([a-zA-Z]+)\$/\\$1\$/g;

  # 2. stray "\\" right after a list / block end  ->  "no line here to end"
  s/\\end\{(itemize|enumerate|block|columns)\}([ \t]*\r?\n?[ \t]*)\\\\[ \t]*/\\end{$1}\n\\vskip0.4em\n/g;

  # 3. "\vskip1em\\"  ->  "\vskip1em"
  s/(\\vskip[0-9.]+ ?e?m[ \t]*)\\\\/$1/g;

' "$F"

# 4. frame missing exactly one closing brace (cq forgets a "{\footnotesize"
#    closer before \end{frame}) -> add one "}" just before \end{frame}.
#    Only the +1 case; anything else is left for the review pass.
perl -0777 -i -pe '
  s#(\\begin\{frame\}.*?)(\n[ \t]*\\end\{frame\})# my($b,$t)=($1,$2); my $o=()=$b=~/(?<!\\)\{/g; my $c=()=$b=~/(?<!\\)\}/g; ($o-$c==1)?$b.chr(125).$t:$b.$t #ges;
' "$F"

echo "fix_tex: patched $(basename "$F")"
