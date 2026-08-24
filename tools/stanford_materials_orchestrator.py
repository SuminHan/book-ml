#!/usr/bin/env python3
"""Organize a batch of local Stanford course PDFs (lecture slides/notes the
user already has on disk -- not fetched from the web, not committed to the
repo) into a small catalog: what each one covers, and which book-ml
chapters/sections it's most relevant to. One cq task per PDF.

Output: tools/stanford_materials/<slug>.json (metadata only -- the PDFs
themselves stay wherever the user's copy lives, outside the repo, same
"cite/link, don't redistribute the whole document" policy as the
Sutton&Barto textbook)."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")
sys.path.insert(0, str(REPO / "tools"))
from orchestrator import fix_math_delimiters  # noqa: E402 (unused, parity import)
import generate_progress_data as gpd  # noqa: E402

CONCURRENCY = 6
TIMEOUT = 900
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
OUT_DIR = REPO / "tools" / "stanford_materials"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# (slug, absolute PDF path, course label) -- the 11 files from the user's
# "unconfirmed batch" uploads, extracted to this local scratch dir.
SRC_DIR = "/tmp/claude-1002/-home-smhan/4fdcb316-9aa8-40bb-b9a3-9d42af7a1d1f/scratchpad/stanford_batches"
FILES = [
    ("cs230_lecture02", f"{SRC_DIR}/lecture02.pdf", "CS230"),
    ("cs230_lecture05", f"{SRC_DIR}/lecture05.pdf", "CS230"),
    ("cs230_lecture06_main", f"{SRC_DIR}/lecture06_main.pdf", "CS230"),
    ("cs230_lecture06_guest", f"{SRC_DIR}/lecture06_guest.pdf", "CS230"),
    ("cs230_lecture08", f"{SRC_DIR}/lecture08.pdf", "CS230"),
    ("cs230_lecture09", f"{SRC_DIR}/lecture09.pdf", "CS230"),
    ("cs230_lecture09_guest", f"{SRC_DIR}/lecture09_guest.pdf", "CS230"),
    ("cs230_lecture10", f"{SRC_DIR}/lecture10.pdf", "CS230"),
    ("cs230_c5m4_transformer", f"{SRC_DIR}/C5M4_transformer.pdf", "CS230"),
    ("cs224n_derivatives", f"{SRC_DIR}/derivatives.pdf", "CS224N"),
    ("cs224n_python_review", f"{SRC_DIR}/CS224N_Python_Review_Session_Slides.pdf", "CS224N"),
]

TEMPLATE = """너는 book-ml 저장소(현재 디렉토리)의 "외부 강의자료 카탈로그" 작업을 한다.
로컬에 있는 스탠포드 강의자료 PDF 1개를 분석해서, 이 책의 어느 챕터/절과
관련있는지 조사한다.

## 대상 파일

{path}
(강좌: {course})

## 할 일

1. **PDF 내용을 가볍게 확인해라.** 전체를 다 읽지 말고, 텍스트 추출로
   제목/목차/슬라이드 제목 정도만 파악하면 충분하다:
   ```python
   import pypdf
   r = pypdf.PdfReader("{path}")
   print(f"{{len(r.pages)}} pages")
   for i in range(min(len(r.pages), 15)):
       print(f"--- page {{i+1}} ---")
       print(r.pages[i].extract_text()[:300])
   ```
   (`/home/smhan/miniconda3/bin/python`로 실행 -- pypdf가 여기 설치돼 있다.)
   **Read 툴로 PDF를 직접 열지 마라** -- 슬라이드가 이미지 위주라 페이지당
   토큰을 많이 먹는다. 텍스트 추출만으로 주제를 파악해라. 그래도 애매하면
   앞 15페이지 정도만 더 보고 판단해라 (전체를 다 훑지 마라).
2. 이 강의가 다루는 **핵심 주제를 2~3문장으로 요약**해라.
3. `kor/src/SUMMARY.md`를 Read해서 이 책(ML1/ML2)의 전체 챕터 목록을 파악하고,
   이 강의 내용과 **가장 관련 있는 book-ml 챕터/절을 1~4개** 골라라 (책
   제목만 보고 대충 짐작하지 말고, 필요하면 후보 절의 본문 도입부 정도는
   Read해서 실제로 겹치는지 확인해라).
4. 결과를 다음 형식으로 저장해라: `tools/stanford_materials/{slug}.json`
   ```json
   {{
     "slug": "{slug}",
     "filename": "<파일명만, 예: lecture02.pdf>",
     "course": "{course}",
     "title": "<이 강의의 실제 제목/주제, 슬라이드에서 확인한 그대로>",
     "summary": "<2~3문장 요약>",
     "related_chapters": [
       {{"book": "ml1", "chapter": "09", "section": "2", "label": "9.2 순전파와 활성화 함수", "why": "<왜 관련있는지 한 문장>"}}
     ]
   }}
   ```
   (`section`이 특정 절이 아니라 챕터 전체와 관련 있으면 `section: null`로,
   `label`은 "9장 신경망 기초" 처럼.)

## 하지 말 것

- book-ml의 다른 파일(kor/src/*.md 등)은 건드리지 마라 -- 이건 카탈로그
  작성 작업이지, 본문에 뭘 추가하는 게 아니다.
- git commit 하지 마라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <이 강의 주제와 관련 챕터 몇 개를 찾았는지 1~2문장으로>

또는:
FAILED: <이유>
"""


run_log = []
log_lock = threading.Lock()


def run_task(slug, path, course):
    task_id = f"stanford_{slug}"
    prompt = TEMPLATE.format(path=path, course=course, slug=slug)
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
    term, summary = gpd.log_info(task_id)
    result = term or "unknown"
    has_meta = (OUT_DIR / f"{slug}.json").exists()
    with log_lock:
        run_log.append({"slug": slug, "result": result, "elapsed": elapsed, "summary": summary or "", "has_meta": has_meta})
    print(f"[stanford] finished {slug} result={result} elapsed={elapsed:.0f}s meta={'yes' if has_meta else 'no'} :: {summary or ''}", flush=True)


def worker(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        slug, path, course = item
        try:
            run_task(slug, path, course)
        except Exception as e:
            print(f"[stanford] ERROR on {slug}: {e}", flush=True)
        q.task_done()


def merge_catalog():
    catalog = []
    for meta_path in sorted(OUT_DIR.glob("*.json")):
        try:
            catalog.append(json.loads(meta_path.read_text()))
        except Exception as e:
            print(f"[stanford] WARNING: couldn't parse {meta_path}: {e}", flush=True)
    out_path = REPO / "tools" / "stanford_catalog.json"
    out_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
    print(f"[stanford] merged {len(catalog)} entries -> {out_path}", flush=True)


def main():
    running = gpd.running_task_ids()
    q = Queue()
    for slug, path, course in FILES:
        if (OUT_DIR / f"{slug}.json").exists():
            print(f"[stanford] skipping {slug} (already catalogued)", flush=True)
            continue
        if f"stanford_{slug}" in running:
            print(f"[stanford] skipping {slug} (already running)", flush=True)
            continue
        if not Path(path).exists():
            print(f"[stanford] WARNING: {path} not found, skipping {slug}", flush=True)
            continue
        q.put((slug, path, course))
    print(f"[stanford] {q.qsize()} file(s) queued (concurrency={CONCURRENCY})", flush=True)
    if q.empty():
        print("[stanford] nothing to do.", flush=True)
    else:
        threads = [threading.Thread(target=worker, args=(q,), daemon=True) for _ in range(CONCURRENCY)]
        for th in threads:
            th.start()
        q.join()
        for _ in threads:
            q.put(None)
        for th in threads:
            th.join()
    merge_catalog()


if __name__ == "__main__":
    main()
