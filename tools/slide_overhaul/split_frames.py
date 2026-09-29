"""프레임을 텍스트 단위로 쪼개고 LaTeX을 벗긴 평문을 만든다 (결정적, LLM 없음)."""
import re, pathlib
FIXED_ENVS = r'(?:tabular|lstlisting|verbatim|center|figure|tikzpicture|table|columns|block)'
SYM = {'approx':'≈','to':'→','times':'×','cdot':'·','alpha':'α','beta':'β','sigma':'σ','lambda':'λ',
       'kappa':'κ','theta':'θ','mu':'μ','partial':'∂','nabla':'∇','sum':'Σ','infty':'∞','leftarrow':'←',
       'Rightarrow':'⇒','rightarrow':'→','le':'≤','leq':'≤','ge':'≥','geq':'≥','sim':'~','ldots':'…',
       'varepsilon':'ε','epsilon':'ε','pi':'π','eta':'η','log':'log','max':'max','min':'min','ne':'≠'}
def arg(s, i):                      # s[i]=='{' -> (content, end)
    d = 0
    for j in range(i, len(s)):
        d += (s[j] == '{') - (s[j] == '}')
        if d == 0: return s[i+1:j], j+1
    return s[i+1:], len(s)
def plain(t):
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = t.replace("``", '"').replace("''", '"').replace('---', '—').replace('--', '–').replace('~', ' ')
    t = re.sub(r'\\(?:frac|tfrac)\{', r'\\FRAC{', t)
    out, i = [], 0
    while i < len(t):
        m = re.match(r'\\([a-zA-Z]+)\*?(\[[^\]]*\])?', t[i:])
        if t[i] == '\\' and m:
            name = m.group(1); i += m.end()
            if name == 'FRAC' or name in ('frac', 'tfrac'):
                a, i = arg(t, i-1) if t[i-1] == '{' else arg(t, i); b, i = arg(t, i) if i < len(t) and t[i] == '{' else ('', i)
                out.append(f'({plain(a)})/({plain(b)})'); continue
            if name in SYM: out.append(SYM[name]); continue
            if i < len(t) and t[i] == '{' and name not in ('vskip', 'hspace', 'includegraphics'):
                a, i = arg(t, i); out.append(plain(a)); continue      # \kb{x} \textbf{x} ... -> x
            if name in ('vskip','hspace','vspace'):
                mm = re.match(r'\s*\{?-?[0-9.]+\s*(?:em|cm|pt|ex|mm)\}?', t[i:]); i += mm.end() if mm else 0
            continue                                                   # \footnotesize 등 -> 제거
        if t[i] == '\\' and i+1 < len(t):
            c = t[i+1]; out.append(' ' if c in '\\,;! ' else c); i += 2; continue
        out.append(t[i] if t[i] not in '{}$' else ''); i += 1
    return re.sub(r'\s+', ' ', ''.join(out)).strip()

def outer_lists(b):
    """최상위 itemize/enumerate 구간 [(start,end)]"""
    spans, depth, st = [], 0, 0
    for mo in re.finditer(r'\\(begin|end)\{(?:itemize|enumerate)\}', b):
        if mo.group(1) == 'begin':
            if depth == 0: st = mo.start()
            depth += 1
        else:
            depth -= 1
            if depth == 0: spans.append((st, mo.end()))
    return spans

PARA = r'\\vskip\s*[0-9.]+\s*em|\n\s*\n|\x00F\d+\x00|\\(?:medskip|bigskip|smallskip)'

def split_frame(src):
    """-> {title, units[{kind,latex,text}] (원문 순서), has_visual, n_fixed}"""
    m = re.search(r'\\begin\{frame\}(?:\[[^\]]*\])?', src)
    if not m: return None
    i = m.end(); title = ''
    if i < len(src) and src[i] == '{': title, i = arg(src, i)
    body = src[i: src.rfind(r'\end{frame}')]
    visual = bool(re.search(r'\\includegraphics|\\begin\{(?:tabular|lstlisting|verbatim|tikzpicture)\}|\\\[|\$\$|\\begin\{(?:align|equation)', body))
    fixed = []
    def hold(mo): fixed.append(mo.group(0)); return f'\x00F{len(fixed)-1}\x00'
    b = re.sub(r'\\begin\{(tabular|lstlisting|verbatim|center|figure|tikzpicture|table|columns|block)\}.*?\\end\{\1\}', hold, body, flags=re.S)
    b = re.sub(r'\\\[.*?\\\]|\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}', hold, b, flags=re.S)
    units, pos = [], 0
    def prose(seg):
        for para in re.split(PARA, seg):
            if plain(para): units.append({'kind': 'prose', 'latex': para.strip()})
    for a, e in outer_lists(b):
        prose(b[pos:a]); pos = e
        inner = re.sub(r'\\(?:begin|end)\{(?:itemize|enumerate)\}(?:\[[^\]]*\])?', '', b[a:e])
        for it in re.split(r'\\item(?![a-zA-Z])', inner)[1:]:
            units.append({'kind': 'item', 'latex': it.strip()})
    prose(b[pos:])
    for u in units: u['text'] = plain(u['latex'])
    units = [u for u in units if u['text']]
    return {'title': plain(title), 'units': units, 'has_visual': visual, 'n_fixed': len(fixed), 'fixed': fixed}
