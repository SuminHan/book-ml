"""학생용 노트북의 코드 칸마다 '예상 결과' 칸을 붙인다.

정답 노트북을 실제로 실행해 나온 출력(print · 표 · 그래프 수)을 그대로 옮기므로,
코드를 고쳐 make_lab.py 를 다시 돌리면 예상 결과도 함께 맞춰진다.

    from expected import add_expected
    add_expected(student_path, solution_path)
"""
import os
import re
import nbclient
import nbformat as nbf

MAX_LINES = 18


def _text_of(cell):
    """코드 칸 하나의 출력: (텍스트, 그래프 수). stderr(경고)는 뺀다."""
    texts, images = [], 0
    for o in cell.get("outputs", []):
        kind = o.get("output_type")
        if kind == "stream" and o.get("name") == "stdout":
            texts.append(o["text"].rstrip("\n"))
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                images += 1
            elif "text/plain" in data:
                texts.append(data["text/plain"].rstrip("\n"))
    text = "\n".join(t for t in texts if t.strip())
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        text = "\n".join(lines[:MAX_LINES] + [f"... ({len(lines) - MAX_LINES}줄 더)"])
    return text, images


def add_expected(student_path, solution_path, timeout=900):
    """정답 노트북을 실행하고, 학생용 노트북의 같은 코드 칸 아래에 '예상 결과' 마크다운 칸을 넣는다.
    돌려주는 값: 정답 노트북의 stdout 전체 (make_lab.py 가 검증용으로 출력)."""
    sol = nbf.read(solution_path, as_version=4)
    nbclient.NotebookClient(sol, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(solution_path)}}).execute()
    stu = nbf.read(student_path, as_version=4)
    assert len(stu.cells) == len(sol.cells), "학생용 · 정답 노트북의 칸 수가 다르다"

    cells, log = [], []
    for s_cell, a_cell in zip(stu.cells, sol.cells):
        cells.append(s_cell)
        if a_cell.cell_type != "code":
            continue
        text, images = _text_of(a_cell)
        log.append(text)
        if not text and not images:
            continue
        parts = ["**예상 결과**"]
        if text:
            parts.append("```\n" + text + "\n```")
        if images:
            parts.append(f"그래프 {images}개가 나온다." if images > 1 else "그래프가 하나 나온다.")
        if re.search(r"\d\.\d+ ?s\)", text):
            parts.append("걸린 시간(초)은 컴퓨터마다 다르다.")
        cells.append(nbf.v4.new_markdown_cell("\n\n".join(parts)))
    stu.cells = cells
    nbf.write(stu, student_path)
    return "\n".join(t for t in log if t)
