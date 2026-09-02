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

  # 3b. "\end{frame>"  /  "\end{frame]"  ->  "\end{frame}"   (closing-brace typo)
  s/\\end\{frame[>\]\)]/\\end{frame}/g;
' "$F"

# 4. frame with brace balance +/-1  (cq forgets a "{\footnotesize" closer, or
#    writes ".}}" one brace too many).  +1 -> add "}" ; -1 -> drop last "}".
#    Any larger imbalance is left for the review pass.
perl -0777 -i -pe '
  s#(\\begin\{frame\}.*?)(\n[ \t]*\\end\{frame\})# my($b,$t)=($1,$2); my $o=()=$b=~/(?<!\\)\{/g; my $c=()=$b=~/(?<!\\)\}/g; my $d=$o-$c; ($d==1)?$b.chr(125).$t : ($d==-1 && $b=~s/(?<!\\)\}(\s*)$/$1/)?$b.$t : $b.$t #ges;
' "$F"

echo "fix_tex: patched $(basename "$F")"
