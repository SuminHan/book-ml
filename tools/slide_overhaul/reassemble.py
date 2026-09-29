"""라벨대로 프레임 LaTeX을 결정적으로 재조립: NOTE -> \\note{}, headline -> 화면 문구 교체."""
import re
from split_frames import split_frame
ESC = {'\\': r'\textbackslash{}', '%': r'\%', '&': r'\&', '_': r'\_', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}'}
def esc(s): return ''.join(ESC.get(c, c) for c in s)
def deverb(s):
    return re.sub(r'\\verb\*?(.)(.*?)\1', lambda m: r'\texttt{' + esc(m.group(2)) + '}', s)
def restore(s, fixed): return re.sub(r'\x00F(\d+)\x00', lambda m: fixed[int(m.group(1))], s)

def reassemble(src, result):
    """result: run_week 결과 한 프레임. -> (new_src, n_changed)"""
    f = split_frame(src); fixed = f['fixed']
    labels = result['units']
    assert [u['text'] for u in f['units']] == [u['text'] for u in labels], "unit mismatch"
    notes, out = [], src
    for u, l in zip(f['units'], labels):
        lat = restore(u['latex'], fixed)
        hl = headline_latex(l['headline'], lat) if (l['label'] == 'KEEP' and l.get('headline')) else None
        if l['label'] == 'NOTE' or hl:
            notes.append(deverb(lat))
            if l['label'] == 'NOTE':
                repl = ''
            else:                                   # KEEP + headline: 화면엔 짧은 문구
                repl = r'\textbf{' + hl + '}'
            if u['kind'] == 'item':
                pat = re.compile(r'\\item(?![a-zA-Z])\s*' + re.escape(lat))
                out, n = pat.subn((r'\\item ' + repl.replace('\\', '\\\\')) if repl else '', out, count=1)
            else:
                n = out.count(lat); out = out.replace(lat, repl, 1)
            if not n: raise ValueError("source span not found")
    # 비어버린 리스트 제거
    out = re.sub(r'\\begin\{(itemize|enumerate)\}(\[[^\]]*\])?\s*\\end\{\1\}', '', out)
    # 첫 항목이 노트로 빠져 바깥 목록이 곧바로 중첩 목록으로 시작하면 'missing \item' -> 빈 \item[] 삽입
    out = re.sub(r'(\\begin\{(?:itemize|enumerate)\}(?:\[[^\]]*\])?\s*)(\\begin\{(?:itemize|enumerate)\})', r'\1\\item[] \2', out)
    if notes:
        note = '\n\\note{\\begin{itemize}\n' + '\n'.join(r'\item ' + n for n in notes) + '\n\\end{itemize}}\n'
        k = out.rfind(r'\end{frame}'); out = out[:k] + note + out[k:]
    return out, len(notes)

def center_frame(src):
    """프레임 본문 전체를 가로 가운데 블록으로 감싼다 (목록 내부는 왼쪽 정렬 유지).
    세로는 beamer 기본(c)이 이미 가운데."""
    m = re.search(r'\\begin\{frame\}(?:\[[^\]]*\])?', src)
    if not m: return src
    i = m.end()
    if i < len(src) and src[i] == '{':
        from split_frames import arg
        _, i = arg(src, i)
    end = src.rfind(r'\end{frame}')
    k = src.find('\\note{', i); k = k if 0 <= k < end else end
    body = src[i:k]
    # block·columns 는 이미 전폭 + varwidth 안에서 무한재귀; 코드블록도 제외
    if not body.strip() or re.search(r'\\titlepage|varwidth|\\begin\{(?:block|alertblock|exampleblock|columns|lstlisting|verbatim)\}', body): return src
    return (src[:i] + '\n\\begin{center}\\begin{varwidth}{0.92\\textwidth}\\usebeamercolor[fg]{normal text}' + body.rstrip()
            + '\n\\end{varwidth}\\end{center}\n' + src[k:])

MATHY = re.compile(r'[_^\\]|[α-ωΑ-Ω∂∇Σ∞≈≤≥→←⇒×·∈∑∏√]')
def headline_latex(h, item_latex):
    """headline(평문) -> LaTeX. 원문 수식 조각을 복원하고, 수식 흔적이 남으면 None(=headline 버림)."""
    from split_frames import plain
    segs = re.findall(r'\$[^$]+\$|\\\(.+?\\\)', item_latex)
    cand = sorted({(plain(s), s) for s in segs if plain(s)}, key=lambda x: -len(x[0]))
    out, i = [], 0
    while i < len(h):
        for pl, lat in cand:
            if h.startswith(pl, i):
                before = h[i-1] if i else ' '; after = h[i+len(pl)] if i+len(pl) < len(h) else ' '
                if not (before.isascii() and before.isalnum()) and not (after.isascii() and after.isalnum()):
                    out.append(('M', lat)); i += len(pl); break
        else:
            out.append(('T', h[i])); i += 1
    text = ''.join(c for k, c in out if k == 'T')
    if MATHY.search(text): return None
    return ''.join(esc(c) if k == 'T' else c for k, c in out)
