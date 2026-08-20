#!/usr/bin/env python3
"""Split a book-ml chapter .md (with ## N.M block headers) into an overview
page + 3 sub-page files (dir/1.md, dir/2.md, dir/3.md), fixing relative
../images/ links (one extra directory deep) along the way.

Usage: python split_chapter.py <path/to/chapterNN.md>
Writes: <path/to/chapterNN.md> (rewritten as overview) and
        <path/to/chapterNN/1.md> <.../2.md> <.../3.md>
"""
import re
import sys
import os

def main(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")

    # Find title line (starts with "# Chapter")
    assert lines[0].startswith("# "), f"first line not a title: {lines[0]!r}"
    title_line = lines[0]

    # Find block header line indices: "## N.M ..."
    block_re = re.compile(r"^## (\d+)\.(\d+) (.+)$")
    block_starts = []
    for i, line in enumerate(lines):
        m = block_re.match(line)
        if m:
            block_starts.append((i, m.group(1), m.group(2), m.group(3)))

    assert len(block_starts) == 3, f"expected 3 blocks, found {len(block_starts)}: {block_starts}"

    # intro = lines[1:block_starts[0][0]], strip leading/trailing blank lines
    intro_lines = lines[1:block_starts[0][0]]
    while intro_lines and intro_lines[0].strip() == "":
        intro_lines.pop(0)
    while intro_lines and intro_lines[-1].strip() == "":
        intro_lines.pop()
    intro = "\n".join(intro_lines)

    # exercises marker: line "---" followed by "## 연습문제" (or similar) -- find last "---" before EOF that precedes a non-block ## heading
    # We detect the start of the "trailer" (exercises section) as the first "---" line at or after block_starts[2][0]
    trailer_start = None
    for i in range(block_starts[2][0], len(lines)):
        if lines[i].strip() == "---":
            trailer_start = i
            break

    def block_content(idx):
        start = block_starts[idx][0]
        end = block_starts[idx + 1][0] if idx + 1 < len(block_starts) else (trailer_start if trailer_start is not None else len(lines))
        chunk = lines[start:end]
        # strip trailing blank lines
        while chunk and chunk[-1].strip() == "":
            chunk.pop()
        return chunk

    blocks = []
    for bi in range(3):
        chunk = block_content(bi)
        # chunk[0] is "## N.M Title" -> convert to "# N.M Title"
        num1, num2, htext = block_starts[bi][1], block_starts[bi][2], block_starts[bi][3]
        chunk[0] = f"# {num1}.{num2} {htext}"
        content = "\n".join(chunk)
        blocks.append((num1, num2, htext, content))

    trailer = ""
    if trailer_start is not None:
        trailer_lines = lines[trailer_start:]
        while trailer_lines and trailer_lines[-1].strip() == "":
            trailer_lines.pop()
        trailer = "\n".join(trailer_lines)

    # fix relative image paths: ../images/ -> ../../images/ in block content + trailer
    def fix_paths(s):
        return s.replace("](../images/", "](../../images/")

    chapter_dir = os.path.splitext(path)[0]  # e.g. .../ml1/chapter02
    os.makedirs(chapter_dir, exist_ok=True)

    for bi, (num1, num2, htext, content) in enumerate(blocks):
        out = fix_paths(content)
        if bi == 2 and trailer:
            out = out + "\n\n" + fix_paths(trailer)
        out += "\n"
        outpath = os.path.join(chapter_dir, f"{bi+1}.md")
        with open(outpath, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print("wrote", outpath)

    # rewrite overview page
    chapter_basename = os.path.basename(chapter_dir)  # e.g. chapter02
    overview_lines = [title_line, "", intro, "", "이번 주는 세 개의 수업 블록으로 진행된다:", ""]
    for bi, (num1, num2, htext, content) in enumerate(blocks):
        overview_lines.append(f"- [{num1}.{num2} {htext}]({chapter_basename}/{bi+1}.md)")
    overview = "\n".join(overview_lines) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(overview)
    print("rewrote overview", path)


if __name__ == "__main__":
    main(sys.argv[1])
