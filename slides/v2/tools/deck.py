#!/usr/bin/env python3
"""slides/v2 덱 작업 도구. 덱 폴더(= <주제>.tex 가 있는 폴더)를 인자로 준다.

  deck.py build  DECK            빌드 + 오류/Overfull 요약(어느 프레임인지) + 쪽 수 + 발표자 노트 md 재생성
  deck.py where  DECK N [N..]    하단 번호 N/총 의 프레임 제목과 tex 줄 번호
  deck.py show   DECK N [N..]    하단 번호 N 쪽을 png 로 렌더 (경로 출력 -> Read 로 본다)
  deck.py sheet  DECK            전체 쪽을 12장씩 한 장으로 모은 미리보기 (전체 훑어보기용)
  deck.py notes  DECK            발표자 노트 md 만 재생성

쪽 번호: 사용자가 말하는 "8/24" 는 하단 번호다. 표지는 번호가 없으므로 PDF 쪽 = 하단 번호 + 1.
(\\pause 같은 오버레이를 쓰면 프레임과 쪽이 1:1 이 아니게 되니 v2 덱에서는 쓰지 않는다.)
"""
import os, re, subprocess, sys

VENV_PY = os.path.expanduser("~/Documents/manim_shorts/_env/v/bin/python")   # PIL/matplotlib 이 있는 파이썬
OUT = "/tmp/deck"


def deck_paths(d):
    d = os.path.abspath(d)
    topic = os.path.basename(d)
    tex = os.path.join(d, f"{topic}.tex")
    if not os.path.exists(tex):
        sys.exit(f"{tex} 가 없다 (폴더 이름 = 주제 = tex 이름이어야 한다)")
    return d, topic, tex, os.path.join(d, f"{topic}.pdf")


def frames(tex_src):
    """[(제목, 시작 줄, note 텍스트)]  0번 = 표지"""
    out = []
    for m in re.finditer(r"\\begin\{frame\}(\[[^\]]*\])?(\{(.*?)\})?", tex_src):
        line = tex_src.count("\n", 0, m.start()) + 1
        end = tex_src.find(r"\end{frame}", m.end())
        body = tex_src[m.end():end]
        n = re.search(r"\\note\{(.*)\}\s*$", body, re.S)
        out.append((m.group(3) or "(표지)", line, n.group(1).strip() if n else ""))
    return out


def clean_title(t):
    return re.sub(r"[`$\\{}]|textbf|kb|kq", "", t).strip()


def write_notes(d, topic, tex):
    fr = frames(open(tex, encoding="utf-8").read())
    md = [f"# 발표자 노트: {topic}", "", "번호는 슬라이드 하단 번호입니다 (표지 제외).", ""]
    for i, (t, _, note) in enumerate(fr[1:], 1):
        md += [f"## {i}. {clean_title(t)}", "", note or "(노트 없음)", ""]
    p = os.path.join(d, f"{topic}_speaker_notes.md")
    open(p, "w", encoding="utf-8").write("\n".join(md))
    return p, len(fr) - 1


def frame_of_line(fr, line):
    k = max((i for i, (_, l, _) in enumerate(fr) if l <= line), default=0)
    return k, clean_title(fr[k][0])


def cmd_build(d):
    d, topic, tex, pdf = deck_paths(d)
    r = subprocess.run(["tectonic", "-X", "compile", os.path.basename(tex)], cwd=d, capture_output=True, text=True)
    log = r.stdout + r.stderr
    fr = frames(open(tex, encoding="utf-8").read())
    print(f"build {'OK' if r.returncode == 0 else 'FAIL'} (exit={r.returncode})")
    for l in log.splitlines():
        if l.startswith("error") or l.startswith("!"):
            print("  " + l[:200])
    seen = set()
    for m in re.finditer(r"tex:(\d+): (Overfull \\[hv]box \([\d.]+pt too (?:wide|high)\))", log):
        k, t = frame_of_line(fr, int(m.group(1)))
        key = (k, m.group(2))
        if key not in seen:
            seen.add(key); print(f"  {m.group(2)}  -> {k}쪽 '{t}' (tex {m.group(1)}줄)")
    if r.returncode == 0:
        pages = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
        n = re.search(r"Pages:\s+(\d+)", pages).group(1)
        p, nf = write_notes(d, topic, tex)
        print(f"  {n}쪽 (표지 + {nf}) · 노트 {os.path.basename(p)} 갱신")
    sys.exit(r.returncode)


def cmd_where(d, nums):
    d, topic, tex, pdf = deck_paths(d)
    fr = frames(open(tex, encoding="utf-8").read())
    for n in nums:
        n = int(n)
        if 0 <= n < len(fr):
            print(f"{n}/{len(fr)-1}: '{clean_title(fr[n][0])}'  tex {fr[n][1]}줄")
        else:
            print(f"{n}: 없음 (프레임 {len(fr)-1}개)")


def cmd_show(d, nums):
    d, topic, tex, pdf = deck_paths(d)
    os.makedirs(OUT, exist_ok=True)
    for n in nums:
        p = int(n) + 1                      # 하단 번호 -> PDF 쪽
        pre = f"{OUT}/{topic}-{n}"
        subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", "80", "-png", "-singlefile", pdf, pre], check=True)
        print(pre + ".png")


def cmd_sheet(d):
    d, topic, tex, pdf = deck_paths(d)
    try:
        import PIL  # noqa: F401
    except ImportError:                     # 시스템 파이썬에 PIL 이 없으면 venv 로 다시 실행
        os.execv(VENV_PY, [VENV_PY, __file__, "sheet", d])
    from PIL import Image, ImageDraw
    import glob, shutil
    tmp = f"{OUT}/{topic}-sheet"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    subprocess.run(["pdftoppm", "-r", "40", "-png", pdf, f"{tmp}/p"], check=True)
    ims = [Image.open(f) for f in sorted(glob.glob(f"{tmp}/p-*.png"))]
    w, h = ims[0].size; cols = 4
    for part in range(0, len(ims), 12):
        sub = ims[part:part + 12]; rows = (len(sub) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), "white"); dr = ImageDraw.Draw(sheet)
        for k, im in enumerate(sub):
            x, y = (k % cols) * w, (k // cols) * h
            sheet.paste(im, (x, y)); dr.rectangle([x, y, x + w - 1, y + h - 1], outline="gray")
            dr.text((x + 4, y + 4), str(part + k), fill="red")        # 하단 번호 (0 = 표지)
        p = f"{OUT}/{topic}-sheet{part // 12}.png"; sheet.save(p); print(p)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    c, d, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    {"build": lambda: cmd_build(d), "where": lambda: cmd_where(d, rest), "show": lambda: cmd_show(d, rest),
     "sheet": lambda: cmd_sheet(d), "notes": lambda: print(*write_notes(*deck_paths(d)[:3]))}.get(c, lambda: sys.exit(__doc__))()
