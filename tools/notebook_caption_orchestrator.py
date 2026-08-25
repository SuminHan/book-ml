#!/usr/bin/env python3
"""Sweep the remaining notebooks/**/*.ipynb that still have Korean text in
matplotlib title/xlabel/ylabel/suptitle/legend/text/annotate calls, and have
cq (local Qwen via litellm) translate just those rendered-plot strings to
English -- Colab's default matplotlib has no CJK font, so Korean in a plot
renders as tofu/missing glyphs. Markdown prose and print() output stay
Korean (this is a Korean-language textbook); only text matplotlib actually
draws onto the figure gets translated.

Same rolling-pool + retry-round + hard-timeout pattern as
citation_orchestrator.py (reuses cq_run_once.py as-is for the actual
subprocess + timeout enforcement).
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

CONCURRENCY = 8  # shares the GPU/vLLM instance with everything else on the box
TIMEOUT = 900  # 15 min -- pure local editing + verification, no WebFetch round-trips
MAX_ATTEMPTS = 3
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

DONE_RE = re.compile(r'DONE:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)
FAILED_RE = re.compile(r'FAILED:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)

TEMPLATE = """너는 KSA Machine Learning 1/2 교재(book-ml 저장소, 현재 디렉토리)의 공동
저자로서, 이번엔 **Colab 노트북의 matplotlib 그래프 캡션을 한글에서 영어로
번역**하는 작업을 한다.

## 배경

이 노트북들은 Google Colab에서 실행되는데, Colab 기본 matplotlib에는 한글
(CJK) 폰트가 없어서 title/xlabel/ylabel/legend/annotate 등에 한글이 들어가면
글자가 깨져서(네모 박스, tofu) 안 보이는 경우가 있다. 본문/설명 텍스트
(마크다운 셀, print() 출력)는 한국어 교재이므로 그대로 두고, **오직
matplotlib이 이미지로 렌더링하는 텍스트만** 영어로 바꾼다.

## 대상 파일

{path}

## 할 일

1. 이 노트북 파일을 Read해서 (또는 Python으로 json 파싱해서) 코드 셀 안에서
   다음 패턴에 해당하는 한글 텍스트를 전부 찾아라:
   - `plt.title(...)`, `ax.set_title(...)`, `fig.suptitle(...)`,
     `plt.xlabel/ylabel(...)`, `ax.set_xlabel/set_ylabel(...)`
   - `.legend()`에 쓰이는 `label="..."` (같은 줄이든, 위에서
     `lab = "..."`처럼 변수에 담아놓고 나중에 `label=lab`으로 쓰는
     경우든 둘 다 찾아라)
   - `.text(...)`, `.annotate(...)`의 표시 문자열
2. 찾은 각 한글 문자열을 자연스러운 영어로 번역해라. **f-string의
   `{{...}}` 표현식(변수 삽입)이나 LaTeX 수식(`$...$`)은 그대로 두고, 그
   주변 한글 텍스트만** 번역한다. 숫자/수식/영어 약어(SVM, CNN, GDA,
   MDP, PPO 등)는 그대로 둔다. 길이가 아주 긴 서술형 제목도 자연스러운
   영어 문장으로 번역해라 (직역 대신 의미 전달 우선).
3. `.ipynb` 파일을 **Python json 모듈로 파싱해서 안전하게** 수정해라
   -- cell['source'] 리스트의 해당 줄이 예상한 원본 문자열과 정확히
   일치하는지 확인한 뒤에만 교체하는 작은 스크립트를 짜서 실행하는 걸
   추천한다 (원본과 다르면 그 줄은 건너뛰고 보고해라 -- 절대 짐작으로
   덮어쓰지 마라). 저장할 땐 `json.dump(nb, f, ensure_ascii=False,
   indent=1)` + 파일 끝 개행 하나로, 원본 포맷을 그대로 유지해라.
4. 수정 후 **반드시 검증**:
   - 각 코드 셀 소스를 `compile()`해서 문법 오류 없는지 확인.
   - 시간이 되면 `conda run -n bookml jupyter nbconvert --to notebook
     --execute --inplace {path}` 로 실제 실행까지 확인해라 (60초 넘게
     걸리는 무거운 셀이 있으면 스킵하고 compile 검증만으로 충분).
   - 수정 후 파일에 matplotlib 캡션 관련 한글이 더 안 남아있는지
     (1번의 패턴들로) 재검색.

## 하지 말 것

- 마크다운 셀, `print()` 출력 문자열, 코드 주석은 건드리지 마라
  (한국어 교재 본문이라 그대로 둬야 함).
- 코드 로직 자체(계산, 알고리즘, 데이터)는 절대 바꾸지 마라 -- 문자열
  리터럴만 바꾼다.
- 한글 폰트 fallback 설정 코드(`font.sans-serif` 관련)가 있으면 그건
  건드리지 마라 (해가 되지 않고, 다른 셀에 남아있을 수도 있는 한글에
  대한 안전망으로 그냥 둔다).
- git commit 하지 마라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <몇 개의 문자열을 번역했는지, compile/실행 검증 결과 1~2문장으로>

또는:
FAILED: <이유>
"""


def all_notebooks():
    kor_re = re.compile(r'[가-힣]')
    mpl_hint = re.compile(r'\.(set_)?(title|xlabel|ylabel|suptitle|legend|text|annotate)\s*\(')
    items = []
    for f in sorted((REPO / "notebooks").glob("**/*.ipynb")):
        rel = str(f.relative_to(REPO))
        try:
            nb = json.loads(f.read_text())
        except Exception:
            continue
        found = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for line in cell.get("source", []):
                if kor_re.search(line) and (mpl_hint.search(line) or "label=" in line):
                    found = True
                    break
            if found:
                break
        if found:
            task_id = "capt_" + rel.replace("/", "_").replace(".ipynb", "")
            items.append({"path": rel, "task_id": task_id})
    return items


def running_task_ids():
    running = set()
    for wf in LOGS_DIR.glob("*.wrapper.log"):
        log = LOGS_DIR / (wf.stem.replace(".wrapper", "") + ".log")
        if not log.exists():
            continue
        text = log.read_text(errors="ignore")
        if "=== cq run ended" not in text:
            running.add(wf.stem.replace(".wrapper", ""))
    return running


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


def build_queue(retry_counts):
    items = all_notebooks()
    running = running_task_ids()
    return [it for it in items
            if retry_counts.get(it["task_id"], 0) < MAX_ATTEMPTS
            and it["task_id"] not in running]


run_log = []
log_lock = threading.Lock()


def run_task(it):
    task_id = it["task_id"]
    prompt = TEMPLATE.format(path=it["path"])
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
    print(f"[nbcaption] finished {task_id} result={result} elapsed={elapsed:.0f}s :: {summary or ''}", flush=True)
    return result


def worker(q, retry_counts):
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
                else:
                    retry_counts[item["task_id"]] = retry_counts.get(item["task_id"], 0) + 1
        except Exception as e:
            print(f"[nbcaption] ERROR on {item['task_id']}: {e}", flush=True)
        q.task_done()


def run_round(round_num, retry_counts):
    tasks = build_queue(retry_counts)
    if not tasks:
        return False
    print(f"[nbcaption] round {round_num}: {len(tasks)} notebook(s) queued (concurrency={CONCURRENCY})", flush=True)
    q = Queue()
    for it in tasks:
        q.put(it)
    threads = [threading.Thread(target=worker, args=(q, retry_counts), daemon=True) for _ in range(CONCURRENCY)]
    for th in threads:
        th.start()
    q.join()
    for _ in threads:
        q.put(None)
    for th in threads:
        th.join()
    print(f"[nbcaption] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = {}
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts)
        if not progressed:
            print("[nbcaption] no notebooks left to attempt (all done or all stuck at "
                  f"{MAX_ATTEMPTS}+ failures). Stopping.", flush=True)
            break
        round_num += 1


if __name__ == "__main__":
    main()
