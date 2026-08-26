#!/usr/bin/env python3
"""Expand the [STUB: ...]-marked files left by the ML1 curriculum
restructuring (new Chapter 13 graph learning, Chapter 14.3 rewrite,
Chapter 4.4 GMM/EM+LDA, Chapter 15/17 renumbering) to full content,
matching the density/style of the rest of the book.

Each target file already has a real skeleton (not empty) plus one or
more [STUB: ...] markers describing exactly what's missing -- this is
lighter-weight than the original from-scratch orchestrator.py, so a
single shared prompt (with the file's own STUB text as the primary
instruction) is enough rather than separate opener/section templates.

Same rolling-pool + retry-round + hard-timeout pattern as the other
orchestrators in this file (reuses cq_run_once.py as-is).
"""
import re
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")

CONCURRENCY = 6
TIMEOUT = 3000  # 50 min -- these are real "expand to lecture density" tasks,
                # same order of magnitude as orchestrator.py's EXPAND_TIMEOUT (60 min)
MAX_ATTEMPTS = 3
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

DONE_RE = re.compile(r'DONE:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)
FAILED_RE = re.compile(r'FAILED:\s*(.+?)(?:\\n\\n|\.\.\.\[\d+chars\]|$)', re.MULTILINE)

TARGETS = [
    "kor/src/ml1/chapter04.md",
    "kor/src/ml1/chapter04/4.md",
    "kor/src/ml1/chapter13.md",
    "kor/src/ml1/chapter13/1.md",
    "kor/src/ml1/chapter13/2.md",
    "kor/src/ml1/chapter13/3.md",
    "kor/src/ml1/chapter13/4.md",
    "kor/src/ml1/chapter14.md",
    "kor/src/ml1/chapter14/3.md",
    "kor/src/ml1/chapter15.md",
    "kor/src/ml1/chapter15/2.md",
    "kor/src/ml1/chapter17.md",
]

TEMPLATE = """너는 KSA(한국과학영재학교) Machine Learning 1/2 교재(book-ml 저장소,
현재 디렉토리)의 공동 저자다. 이 저장소는 mdBook 기반이며, 각 챕터는
1.1/1.2/1.3처럼 번호가 매겨진 "절" 단위로 나뉘어 있고 절 하나 = 실제
수업 50분 분량이다.

## 배경: 지금 무슨 작업 중인가

방금 ML1 커리큘럼을 재구성했다 -- 새 13장(그래프 표현학습, AlphaFold
오프너 → Random Walk → Node2Vec → PageRank → 실전 확장), 14.3 절
교체(Transformer 임베딩 + RAG), 4.4 신설(GMM/EM/LDA, 15.1에서 이동),
15장 리넘버링(2절로), 구 13장(LLM)을 17장(부록)으로 이동. 이 과정에서
각 파일에 **뼈대 콘텐츠 + `[STUB: ...]` 마커**를 남겨뒀다 -- 마커
안에 정확히 뭘 채워야 하는지 적혀 있다.

## 이번 작업 대상

- 파일: {path}
- 이 파일을 Read해서 `[STUB: ...]` 마커를 찾아라 -- 거기 적힌 지시를
  **정확히** 따르되, 마커 밖의 기존 내용(이미 다른 절에서 옮겨온 실제
  콘텐츠일 수 있음)은 함부로 다시 쓰지 마라.
- 문맥 파악을 위해 같은 챕터의 다른 절들(예: `kor/src/ml1/chapter13/`
  안의 나머지 파일들, 또는 챕터 개요 `kor/src/ml1/chapter13.md`)을
  Read해서 실제 분량/문체 감각을 잡아라. 챕터 개요(`chapterNN.md`)면
  500~800단어, 본문 절이면 기존 책의 다른 절과 비슷하게(2500~4000단어
  상당, 이미 그 정도 있다면 STUB 지시 부분만 추가).

## 절대 원칙

- **기존 문체를 그대로 따른다** -- 친절하지만 밀도 높은 설명체, 수학
  표기 `\\\\( \\\\)`/`\\\\[ \\\\]` (마크다운 이스케이프 때문에 백슬래시
  1개가 아니라 **2개**를 써야 한다 -- 예: `\\\\(x\\_i\\\\)`), "왜?"를
  항상 설명, 필요하면 "자주 하는 실수"/"확인 문제"/"자주 묻는 질문"
  패턴.
- **작업이 끝나면 `[STUB: ...]` 마커 텍스트 자체를 지워라** -- 채워진
  자리에 지시문이 그대로 남아있으면 안 된다.
- 이미 있는 내용(다른 절에서 옮겨온 것)은 삭제하지 마라 -- STUB이
  요청한 것만 **추가**한다.
- **인용이 필요한 곳(예: DeepFRI, STGCN, AlphaFold, SceneDiffuser,
  RAG 논문)은 WebFetch로 실제 서지사항(저자/연도/저널·학회/arXiv
  번호)을 재검증해라** -- 특히 DeepFRI는 이전 세션에서 arXiv/bioRxiv/
  Semantic Scholar API가 일시적으로 막혀서 기억에 의존해 적어둔
  상태라고 파일에 표시돼 있으면, 반드시 다시 확인해라.
- 노트북이 필요한 절이면 `notebooks/ml1/` 안에 이미 최소 스텁
  노트북이 만들어져 있을 수 있다(Colab 배지 링크로 파일명 확인) --
  있으면 그 노트북을 채우고, 실제로 실행해서 에러 없이 도는지
  검증해라(`/home/smhan/miniconda3/envs/bookml/bin/python`로
  `jupyter nbconvert --to notebook --execute` 등).

## 절대 하지 말 것 (이번 세션에서 실제로 발견된 버그들 -- 다시 만들지 마라)

- **볼드(`**text**`)가 안 닫히는 경우**: 닫는 `**` 바로 앞이 `)`/`"` 같은
  문장부호이고 바로 뒤가 한글 조사(예: `는`/`가`/`이다`)면 CommonMark가
  강조를 못 닫는다. 문장부호를 볼드 범위 밖으로 빼라(`**단어(설명)**가`
  대신 `**단어(설명**)가`).
- **수식 서브스크립트 이스케이프**: `\\_`(백슬래시 1개)가 이 저장소의
  올바른 컨벤션이다 -- `\\\\_`(2개)로 쓰면 CommonMark가 밑줄을 이탤릭
  마커로 오해해서 수식이 깨진다.
- **행렬 줄바꿈**: `\begin{{bmatrix}}...\\\\...\end{{bmatrix}}`에서 행
  구분자는 소스에 백슬래시 **4개**(`\\\\\\\\`)가 필요하다(2개는
  마크다운 이스케이프에 씹혀서 1개만 남아 깨진다).
- 위 세 가지가 걱정되면, 작업 끝나고 아래 "검증"의 재빌드로 실제
  확인해라.
- **Edit 도구가 "String to replace not found" 를 반복하면 즉시 포기하고
  전환해라**: 이 저장소의 한글 본문은 em-dash(—)와 곡선따옴표(" " )를
  자주 쓰는데, 네가 옮겨적은 문자열이 실제 바이트와 미묘하게 달라
  Edit이 매칭에 실패하는 경우가 잦다. 같은 블록에 Edit을 2번 이상
  실패하면 더 시도하지 말고 바로 `python3 -c "..."`로 파일을 읽어
  (원본 텍스트를 눈으로 복사하지 말고 변수에 담아) 문자열
  치환/재작성하는 방식으로 전환해라 -- 매 실패마다 `od -c`로 바이트를
  다시 확인하며 재시도하는 것은 시간만 태운다.
- **수식 이스케이프 컨벤션을 이미 위에서 확정해줬다** -- 파일마다 다시
  `od -c`로 여러 번 재검증하지 말고(이미 위 규칙이 이 저장소 전체에
  일관되게 적용된다는 게 확인된 사실이다) 한 번만 믿고 써라. 재검증은
  최종 재빌드 결과(리터럴 `**`/깨진 수식 유무)로 충분하다.

## 하지 말 것

- 다른 챕터/절 파일은 건드리지 마라(같은 STUB 목록의 다른 파일도
  다른 cq 작업이 동시에 처리 중일 수 있다).
- git commit은 하지 마라.

## 검증

작업 끝나면 반드시:
```bash
cd /home/smhan/book-ml/kor && PATH="/home/smhan/miniconda3/bin:$PATH" mdbook build
```
으로 재빌드해서 에러 없는지, 그리고 이 파일에 대응하는
`docs/kor/.../*.html`에 코드블록 밖 리터럴 `**`가 안 남았는지 확인해라.

## 완료 조건

작업이 끝나면 마지막에 다음을 정확히 출력하라 (맨 마지막 줄에, 이
형식 그대로):

DONE: <이번에 채운 내용을 2~3문장으로 한국어 요약 -- 어떤 STUB
지시를 어떻게 반영했는지, 재빌드 검증 결과 포함.>

또는 실패 시:
FAILED: <이유>
"""


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


def has_stub(path):
    p = REPO / path
    if not p.exists():
        return False
    return "[STUB" in p.read_text(encoding="utf-8")


def load_targets():
    """Live rescan: a file drops off the list the moment its [STUB]
    marker is actually removed, regardless of how its own task's log
    got parsed -- same fix as bold_math_qa_orchestrator's live-rescan
    (a static list + a DONE_RE mismatch caused blind requeuing there)."""
    items = []
    for rel in TARGETS:
        if has_stub(rel):
            task_id = "restructure_" + rel.replace("/", "_").replace(".md", "")
            items.append({"path": rel, "task_id": task_id})
    return items


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
    print(f"[restructure] finished {task_id} result={result} elapsed={elapsed:.0f}s :: {summary or ''}", flush=True)
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
            print(f"[restructure] ERROR on {item['task_id']}: {e}", flush=True)
        q.task_done()


def run_round(round_num, retry_counts, done_ids):
    tasks = build_queue(retry_counts, done_ids)
    if not tasks:
        return False
    print(f"[restructure] round {round_num}: {len(tasks)} file(s) queued (concurrency={CONCURRENCY})", flush=True)
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
    print(f"[restructure] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = {}
    done_ids = set()
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts, done_ids)
        if not progressed:
            print("[restructure] all files done or stuck. Stopping.", flush=True)
            break
        round_num += 1


if __name__ == "__main__":
    main()
