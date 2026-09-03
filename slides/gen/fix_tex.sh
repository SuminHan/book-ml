#!/bin/bash
# Auto-repair a small set of PROVEN-SAFE slips in a cq-generated deck .tex.
# Deliberately minimal: anything ambiguous (stray $, \mid in text, nested math)
# is left for the manual review pass so we never corrupt a good deck.
# Usage: bash fix_tex.sh <tex>
set -u
F="$1"
[ -f "$F" ] || { echo "fix_tex: no such file $F"; exit 1; }

perl -CSD -Mutf8 -0777 -i -pe '
  # 1. "\*" star notation cq uses for optimal values (w^\*  \pi^\*  V^\* ...)
  s/\^\\\*/^{*}/g;
  s/(?<!\\)\\\*/*/g;

  # 1b. doubled backslash on a command right after / before a "$"  ($\\to$ -> $\to$)
  s/\$\\\\([a-zA-Z])/\$\\$1/g;
  s/\\\\([a-zA-Z]+)\$/\\$1\$/g;

  # 1c. math connector glued to a Hangul char = text-mode error, fatal.
  #     "$G_t$\cdot할인율" -> "$G_t$$\cdot$할인율".  Requires a literal Hangul
  #     neighbour so real math ("$a \cdot b$", "\cdot \text{..}") is untouched.
  s/\\(cdot|times|div|Rightarrow|rightarrow|leftrightarrow|to|mapsto|leq|geq|approx|sim|cup|cap|equiv)(?=\s*[\x{AC00}-\x{D7A3}])/\$\\$1\$/g;
  s/(?<=[\x{AC00}-\x{D7A3}])(\s*)\\(cdot|times|div|Rightarrow|rightarrow|leftrightarrow|to|mapsto|leq|geq|approx|sim|cup|cap|equiv)(?![a-zA-Z])/$1\$\\$2\$/g;

  # 2. stray "\\" right after a list / block end  ->  "no line here to end"
  s/\\end\{(itemize|enumerate|block|columns)\}([ \t]*\r?\n?[ \t]*)\\\\[ \t]*/\\end{$1}\n\\vskip0.4em\n/g;

  # 3. "\vskip1em\\"  ->  "\vskip1em"
  s/(\\vskip[0-9.]+ ?e?m[ \t]*)\\\\/$1/g;

  # 3b. "\end{frame>"  /  "\end{frame]"  ->  "\end{frame}"   (closing-brace typo)
  s/\\end\{frame[>\]\)]/\\end{frame}/g;

  # 3c. math operators (\tanh ...) left bare in Korean body text -> plain word.
  #     A bare \tanh in text makes TeX enter math and never leave (fatal).
  #     Only fires when followed by a Hangul char or closing punctuation, so
  #     genuine "\tanh(z)" / "$\tanh z$" in math are untouched.
  s/(?<![\$\\])\\(tanh|sigmoid|softmax|relu|argmax|argmin|logit)(?=\s*[가-힣)\x{201d}"'"'"'\]])/$1/g;

  # 3c2. bare "_" / "#" / "%" / "&" inside \texttt{...} (non-verbatim) -> escaped.
  #      cq writes \texttt{env.action_space} with a raw underscore -> Missing $.
  1 while s/(\\texttt\{[^{}]*?)(?<![\\_])_(?![_])/${1}\\_/;
  1 while s/(\\texttt\{[^{}]*?)(?<!\\)([#%&])/${1}\\$2/;

  # 3d. "$\texttt{ ... }$"  ->  "\texttt{ ... }"   (\texttt is text-mode; wrapping
  #     it in $ breaks).  If the inner text has no $ island, also de-math a bare
  #     \times / \cdot to "*".  Valid "\texttt{a $\times$ b}" is left untouched.
  s#\$(\\texttt\{([^{}]*)\})\$# my($f,$in)=($1,$2); if($in!~/\$/){$in=~s/\\(times|cdot)/*/g; "\\texttt{$in}"} else {$f} #ge;
' "$F"

# 3e. \verb on a \begin{frame} line = \verb in the frametitle = fatal
#     ("moving argument" -> "TeX capacity exceeded").  Swap every one for \texttt.
perl -CSD -Mutf8 -i -pe '
  s/\\verb(\S)(.*?)\1/\\texttt{$2}/g if /^\s*\\begin\{frame\}/;
' "$F"

# 3f. frame whose body uses \verb or lstlisting but isn't [fragile] -> make it.
# 3g. tabular colspec narrower than its rows (cq wrote "ll" for a 3-col table)
#     -> widen the colspec with extra "l"s.  Widening never breaks a table.
python3 - "$F" <<'PYEOF'
import sys,re
p=sys.argv[1]; t=open(p,encoding='utf-8').read()

def frag(m):
    head,body=m.group(1),m.group(2)
    return (head+'[fragile]'+body) if re.search(
        r'\\verb[^a-zA-Z]|\\begin\{(lstlisting|verbatim|Verbatim|semiverbatim|minted)\}',
        body) else m.group(0)
t=re.sub(r'(\\begin\{frame\})(?!\[)(\{.*?\\end\{frame\})', frag, t, flags=re.S)

# 3h. \textbf{ / \kb{ / ... that is never closed before a structural token
#     (cq forgets the "}", a stray "}" much later "balances" it, and the
#     mangled argument silently eats every following frame).
FMT=('textbf','textit','emph','textsc','kb','kq')
STOP=re.compile(r'\\end\{(block|itemize|enumerate|frame|columns|column)\}'
                r'|\\begin\{(tabular|itemize|enumerate|block)\}'
                r'|\\\[|\\item(?![a-zA-Z])|\\vskip|\\bigskip|\\medskip')
def close_runaway_fmt(t):
    op=re.compile(r'\\(?:'+ '|'.join(FMT) +r')\{')
    out=[]; i=0; n=len(t)
    while True:
        m=op.search(t,i)
        if not m: out.append(t[i:]); break
        out.append(t[i:m.end()])          # emit up to and incl. "\textbf{"
        j=m.end(); depth=1
        while j<n and depth>0:
            c=t[j]
            if c=='\\':
                if STOP.match(t,j):        # structural token while still open
                    out.append(t[m.end():j] + '}'*depth)   # content + missing }s
                    break
                j+=2; continue
            depth += (c=='{') - (c=='}')
            j+=1
        else:
            out.append(t[m.end():j])       # closed normally (or EOF)
        i=j
    return ''.join(out)
t=close_runaway_fmt(t)

# 3i. bare "%" in body text is a LaTeX comment -> eats the rest of the line
#     (incl. closing braces) -> runaway.  cq means the percent sign: escape it.
#     Verbatim-ish blocks (real % comments / Python %) are left alone.
def esc_percent(t):
    parts=re.split(r'(\\begin\{(?:lstlisting|verbatim|Verbatim|semiverbatim)\}.*?'
                   r'\\end\{(?:lstlisting|verbatim|Verbatim|semiverbatim)\})',
                   t, flags=re.S)
    for i in range(0, len(parts), 2):          # even indices = outside verbatim
        # escape only an in-text "%" (has a non-space, non-backslash char before
        # it on the line); a line-initial "%" is a real comment -> leave it.
        parts[i]=re.sub(r'(?<=[^\s\\\n])(?<!\\)%', r'\\%', parts[i])
    return ''.join(parts)
t=esc_percent(t)

def ncols(spec):
    s=re.sub(r'@\{[^}]*\}','',spec); s=re.sub(r'[|>{}<]','',s)
    s=re.sub(r'p\s*\{[^}]*\}','p',s); s=re.sub(r'\*\{(\d+)\}\{([lcrp])\}',
             lambda mm:mm.group(2)*int(mm.group(1)),s)
    return len(re.findall(r'[lcrp]',s))

def widen(m):
    spec,body=m.group(1),m.group(2)
    have=ncols(spec)
    if have==0: return m.group(0)
    # bail on anything that makes cell-counting unreliable
    if '\\multicolumn' in body or '\\&' in body or '\\multirow' in body:
        return m.group(0)
    counts=[]
    for row in re.split(r'\\\\', body):
        if '&' not in row: continue
        if re.search(r'\\(top|mid|bottom)rule|\\hline|\\cmidrule', row): continue
        r=re.sub(r'\\[a-zA-Z]+\{[^{}]*\}', '', row)   # drop \cmd{...}
        r=re.sub(r'\$[^$]*\$', '', r)                 # drop $...$
        counts.append(r.count('&')+1)
    # only act when every data row agrees and it exceeds the spec by >=1
    if counts and len(set(counts))==1 and counts[0]>have:
        add='l'*(counts[0]-have)
        spec=(re.sub(r'@\{\}\s*$', add+'@{}', spec)
              if spec.rstrip().endswith('@{}') else spec+add)
    return '\\begin{tabular}{'+spec+'}'+body+'\\end{tabular}'
t=re.sub(r'\\begin\{tabular\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}(.*?)\\end\{tabular\}',
         widen, t, flags=re.S)
open(p,'w',encoding='utf-8').write(t)
PYEOF

# 4. frame with brace balance +/-1  (cq forgets a "{\footnotesize" closer, or
#    writes ".}}" one brace too many).  +1 -> add "}" ; -1 -> drop last "}".
#    Any larger imbalance is left for the review pass.
perl -0777 -i -pe '
  s#(\\begin\{frame\}.*?)(\n[ \t]*\\end\{frame\})# my($b,$t)=($1,$2); my $o=()=$b=~/(?<!\\)\{/g; my $c=()=$b=~/(?<!\\)\}/g; my $d=$o-$c; ($d==1)?$b.chr(125).$t : ($d==-1 && $b=~s/(?<!\\)\}(\s*)$/$1/)?$b.$t : $b.$t #ges;
' "$F"

echo "fix_tex: patched $(basename "$F")"
