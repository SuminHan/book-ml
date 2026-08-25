#!/usr/bin/env python3
"""Sweep kor/src/**/*.md files that have literal, unrendered "**" surviving
into the built HTML (i.e. CommonMark failed to parse it as <strong>...</strong>)
and have cq fix the markdown source so it renders correctly, using judgment
per instance (this is NOT purely mechanical -- see prompt below).

Discovered failure modes (found by diffing built docs/kor/**/*.html against
kor/src/**/*.md on 2026-08-25):
  1. Closing "**" preceded by whitespace ("**word **rest") -- CommonMark
     never treats a closer preceded by whitespace as right-flanking.
  2. Closing "**" preceded by ASCII punctuation (quote, paren, ...) AND
     immediately followed by a non-space/non-punctuation character (very
     common in Korean, since particles attach directly with no space) --
     CommonMark's intraword-emphasis heuristic refuses to close it.
  3. Genuine authoring typos: a stray extra "**" left behind near a bolded
     phrase (e.g. "**갱신 **앞에서**" -- an orphaned "**" before "갱신" that
     doesn't belong at all). A blind regex fix is NOT safe here -- an
     earlier attempt at a purely mechanical fix corrupted 11 files by
     merging words together across list-item boundaries. This is why the
     task is delegated to cq per file, reading full context, rather than
     scripted.

Same rolling-pool + retry-round + hard-timeout pattern as
citation_orchestrator.py (reuses cq_run_once.py as-is).
"""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")

CONCURRENCY = 4  # lower than citation's 8 -- each task runs a full `mdbook
# build` to verify, and concurrent full-book builds writing into the same
# shared docs/kor/ output directory is a real (if low-probability) race;
# keep the pool small to reduce how often that overlaps.
TIMEOUT = 900  # 15 min
MAX_ATTEMPTS = 3
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

DONE_RE = re.compile(r'DONE:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)
FAILED_RE = re.compile(r'FAILED:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)

TEMPLATE = """너는 KSA Machine Learning 1/2 교재(book-ml 저장소, 현재 디렉토리)의 공동
저자로서, 이번엔 **`**볼드**` 마크다운이 실제로 안 렌더링되고 리터럴
"**"로 그대로 남는 버그를 고치는** 작업을 한다.

## 배경

mdBook(pulldown-cmark, CommonMark)이 `**text**`를 파싱할 때, 닫는 `**` 바로
앞이 공백이거나, 닫는 `**` 바로 앞이 문장부호(따옴표/괄호 등)이면서 바로
뒤가 공백도 문장부호도 아닌 일반 문자(한국어 조사가 붙는 경우 매우 흔함)면
emphasis가 안 닫히고 `**`가 텍스트 그대로 출력된다. 실제 저자가 남긴
오타(엉뚱한 위치에 `**`가 하나 더 붙어있는 경우 등)도 섞여 있어서, **기계적
정규식으로 일괄 치환하면 위험하다** -- 실제로 이전 시도에서 리스트 항목
경계를 잘못 처리해 11개 파일을 깨뜨린 적이 있다. 그래서 파일마다 문맥을
직접 읽고 판단해서 고쳐야 한다.

## 대상 파일

{path} (이 파일의 빌드된 HTML에서 코드블록 밖 리터럴 `**`가 대략 {count}개
발견됨 -- 정확한 개수/위치는 아래 방법으로 직접 재확인해라)

## 할 일

1. **먼저 빌드해서 실제 깨진 위치를 정확히 찾아라** (소스만 보고 추측하지
   마라 -- 코드블록 안의 `x**2` 같은 파이썬 거듭제곱은 정상이니 제외해야
   한다):
   ```bash
   cd /home/smhan/book-ml/kor && PATH="/home/smhan/miniconda3/bin:$PATH" mdbook build
   ```
   그 다음 이 파일에 대응하는 `docs/kor/.../*.html`을 Read하거나, 아래처럼
   Bash로 `<code>`/`<pre>` 밖의 리터럴 `**` 맥락을 뽑아봐라:
   ```bash
   python3 -c "
import re
text = open('docs/kor/.../대응파일.html', encoding='utf-8').read()
spans = [(m.start(), m.end()) for m in re.finditer(r'<(code|pre)\\b[^>]*>.*?</\\1>', text, re.DOTALL)]
def in_code(p): return any(s <= p < e for s, e in spans)
for m in re.finditer(r'\\*\\*', text):
    if not in_code(m.start()):
        print(text[max(0,m.start()-40):m.start()+40])
"
   ```
   (경로의 `.../대응파일.html`은 `{path}`에서 `kor/src/`를 `docs/kor/`로,
   `.md`를 `.html`로 바꾼 것이다.)

2. 찾은 각 위치를 `kor/src/`의 원본 마크다운에서 찾아서, **문맥을 읽고**
   무엇이 문제인지 판단해라. 대개 다음 중 하나다:
   - 닫는 `**` 바로 앞에 불필요한 공백이 있다 → 그 공백을 지운다
     (`내용 **` → `내용**`).
   - 닫는 `**` 바로 앞이 `"`/`)` 같은 문장부호이고 바로 뒤가 한글이다 →
     문장부호를 볼드 범위 밖으로 옮긴다 (`"내용"**뒷말` →
     `"내용**"뒷말` 처럼, 볼드가 끝나는 지점을 마지막 실질 문자 뒤로
     당긴다). 원래 강조하려던 범위가 최대한 유지되게 판단해라.
   - **저자의 오타로 엉뚱한 위치에 `**`가 하나 더 있다** (예: 문장 중간에
     `**단어 **다음단어**`처럼 첫 `**`가 뜬금없이 끼어든 경우) → 문맥상
     저자가 진짜 강조하려던 구간이 어디인지 판단해서, 불필요한 `**`를
     지워라. 확신이 안 서면 가장 보수적인 선택(의미가 안 바뀌는 쪽,
     즉 강조 범위를 넓히기보다 좁히는 쪽)을 해라.

3. **절대 하지 말 것**:
   - 코드블록(펜스드 ``` 또는 인라인 `` ` ``) 안의 `**`는 건드리지 마라
     (파이썬 거듭제곱 연산자 등 정상 코드).
   - 문장의 실제 내용/의미/단어를 바꾸지 마라 -- `**` 위치 조정 또는 삭제
     만 한다. 공백을 지울 때도 **단어 사이의 필요한 공백까지 지우면
     안 된다** (예: "갱신 앞에서" 두 단어 사이 공백은 유지해야 함 --
     `**`만 옮기거나 지워라, 원래 있던 단어 사이 공백은 건드리지 마라).
   - 이 절 저 절 다 돌아다니며 고치지 마라 -- 오직 `{path}` 하나만.
   - git commit 하지 마라.

4. 수정 후 **다시 빌드하고 재확인**해서, 이 파일에 대응하는 HTML에 코드블록
   밖 리터럴 `**`가 하나도 안 남았는지 확인해라 (1번 방법 재사용). 남아있으면
   계속 고쳐라.
5. `git diff {path}`로 최종 변경사항을 보고 - 의도한 것보다 훨씬 많은 줄이
   바뀌었거나, 단어가 붙어버렸거나 이상하면 (예: "갱신앞에서"처럼 공백이
   사라졌으면) 잘못된 것이니 되돌리고 다시 신중하게 고쳐라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <몇 건을 어떻게 고쳤는지 1~2문장으로, 재빌드 검증 결과 포함>

또는:
FAILED: <이유>
"""


def load_targets():
    """Live ground-truth rescan every round (rebuild + grep the actual HTML
    for literal ** outside code/pre) instead of a static target list -- a
    static list plus a DONE_RE that didn't match ("unknown" results never
    entered done_ids) caused round 2 to blindly requeue all 85 files even
    though ~69 were already genuinely fixed. This can't happen with a live
    rescan: a truly-fixed file just won't show up next round, full stop,
    regardless of how its own log got parsed."""
    subprocess.run(
        ["mdbook", "build"], cwd=REPO / "kor",
        env={**__import__("os").environ, "PATH": "/home/smhan/miniconda3/bin:" + __import__("os").environ.get("PATH", "")},
        capture_output=True,
    )
    code_span_re = re.compile(r'<(code|pre)\b[^>]*>.*?</\1>', re.DOTALL)
    bb_re = re.compile(r'\*\*')
    items = []
    for f in sorted((REPO / "docs" / "kor").glob("**/*.html")):
        if f.name == "print.html":
            continue
        text = f.read_text(encoding="utf-8")
        spans = [(m.start(), m.end()) for m in code_span_re.finditer(text)]
        def in_code(pos):
            return any(s <= pos < e for s, e in spans)
        count = sum(1 for m in bb_re.finditer(text) if not in_code(m.start()))
        if count == 0:
            continue
        rel_html = f.relative_to(REPO / "docs" / "kor")
        src = f"kor/src/{rel_html}".replace(".html", ".md")
        if not (REPO / src).exists():
            continue  # e.g. toc.html has no kor/src/toc.md -- generated from SUMMARY.md, not a real target
        task_id = "boldfix_" + src.replace("/", "_").replace(".md", "")
        items.append({"path": src, "count": count, "task_id": task_id})
    return items


def running_task_ids():
    try:
        out = subprocess.run(["pgrep", "-af", "cq_run_once.py"],
                              capture_output=True, text=True).stdout
    except Exception:
        return set()
    ids = set()
    for line in out.splitlines():
        for m in re.finditer(r'batch_(?:prompts|logs)/([\w]+)\.(?:txt|log)', line):
            ids.add(m.group(1))
    return ids


def log_info(task_id):
    log_path = LOGS_DIR / f"{task_id}.log"
    if not log_path.exists():
        return None, ""
    text = log_path.read_text(errors="ignore")
    m = DONE_RE.search(text)
    if m:
        return "done", m.group(1).strip()
    m = FAILED_RE.search(text)
    if m:
        return "failed", m.group(1).strip()
    if "=== TIMEOUT" in text:
        return "failed", "타임아웃"
    return "unknown", ""


def build_queue(retry_counts, done_ids):
    items = load_targets()
    running = running_task_ids()
    return [it for it in items
            if it["task_id"] not in done_ids
            and retry_counts.get(it["task_id"], 0) < MAX_ATTEMPTS
            and it["task_id"] not in running]


run_log = []
log_lock = threading.Lock()


def run_task(it):
    task_id = it["task_id"]
    prompt = TEMPLATE.format(path=it["path"], count=it["count"])
    prompt_path = PROMPTS_DIR / f"{task_id}.txt"
    log_path = LOGS_DIR / f"{task_id}.log"
    wrapper_path = LOGS_DIR / f"{task_id}.wrapper.log"
    prompt_path.write_text(prompt)

    t0 = time.time()
    with open(wrapper_path, "w") as wf:
        subprocess.run(
            ["python3", "tools/cq_run_once.py", str(prompt_path), str(log_path), str(TIMEOUT)],
            cwd=REPO, stdout=wf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    elapsed = time.time() - t0
    term, summary = log_info(task_id)
    result = term or "unknown"
    with log_lock:
        run_log.append({"task_id": task_id, "result": result, "elapsed": elapsed, "summary": summary or ""})
    print(f"[boldfix] finished {task_id} result={result} elapsed={elapsed:.0f}s :: {summary or ''}", flush=True)
    return result


def worker(q, retry_counts, done_ids):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        try:
            result = run_task(item)
            with log_lock:
                if result == "done":
                    retry_counts.pop(item["task_id"], None)
                    done_ids.add(item["task_id"])
                else:
                    retry_counts[item["task_id"]] = retry_counts.get(item["task_id"], 0) + 1
        except Exception as e:
            print(f"[boldfix] ERROR on {item['task_id']}: {e}", flush=True)
        q.task_done()


def run_round(round_num, retry_counts, done_ids):
    tasks = build_queue(retry_counts, done_ids)
    if not tasks:
        return False
    print(f"[boldfix] round {round_num}: {len(tasks)} file(s) queued (concurrency={CONCURRENCY})", flush=True)
    q = Queue()
    for it in tasks:
        q.put(it)
    threads = [threading.Thread(target=worker, args=(q, retry_counts, done_ids), daemon=True) for _ in range(CONCURRENCY)]
    for th in threads:
        th.start()
    q.join()
    for _ in threads:
        q.put(None)
    for th in threads:
        th.join()
    print(f"[boldfix] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = {}
    done_ids = set()
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts, done_ids)
        if not progressed:
            print("[boldfix] all files done or stuck. Stopping.", flush=True)
            break
        round_num += 1


if __name__ == "__main__":
    main()
