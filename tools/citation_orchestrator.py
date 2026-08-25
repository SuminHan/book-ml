#!/usr/bin/env python3
"""Sweep all 128 kor/ files (96 sections + 32 chapter openers) and add
footnote citations for named algorithms/papers/historical claims, using
WebFetch against structured APIs (arXiv, Semantic Scholar) -- Anthropic's
server-side WebSearch tool 400s against the local Qwen backend (tool_choice
schema mismatch), but WebFetch (a plain client-side URL fetch) works fine
and was verified to correctly resolve e.g. PPO -> arXiv:1707.06347.

Same rolling-pool + retry-round pattern as orchestrator.py /
chapter_opener_orchestrator.py, reused via direct import.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")
sys.path.insert(0, str(REPO / "tools"))
import generate_progress_data as gpd  # noqa: E402
from orchestrator import fix_math_delimiters  # noqa: E402

CONCURRENCY = 8  # heavier per-task (real WebFetch round-trips) + be polite to arXiv/S2 APIs
TIMEOUT = 1500  # 25 min -- research + verification takes longer than pure writing
MAX_ATTEMPTS = 3
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

TEMPLATE = """너는 KSA Machine Learning 1/2 교재(book-ml 저장소, 현재 디렉토리)의 공동
저자로서, 이번엔 **레퍼런스 체크 및 각주(+가능하면 원 논문 그림) 추가** 작업을
한다.

## 대상 파일

{path} ("{label}")
(상대 이미지 경로 접두사: `{img_prefix}` -- 예: 절 파일이면 `../../images/`,
챕터 개요 파일이면 `../images/`. 이 파일 안에 있는 기존 `![...](...)` 링크를
보면 정확한 접두사를 바로 알 수 있다.)

## 먼저 확인: 미리 만들어둔 "레퍼런스 그림 라이브러리"

`tools/ref_library.json`을 Read해라. PPO/DQN/TRPO/GAE/A3C/RLHF/InstructGPT/
AlphaGo/Transformer/VAE/GAN/word2vec/ResNet/AlexNet/Dropout/BatchNorm 등
**16개 핵심 논문**은 이미 원 논문 그림을 캡쳐해서
`kor/src/images/ref_<slug>.png`에 저장해뒀고, 이 JSON에 각주 텍스트
(`footnote`)와 그림 설명(`figure_caption`)까지 다 준비되어 있다.

여기에 더해 다음도 라이브러리에 있다 (전부 그림 없이 `image_ok: false`,
텍스트 각주만):
- **Sutton & Barto 교과서** (슬러그 `suttonbarto`) -- 본문에 "Sutton과
  Barto의 표준 교과서"처럼 언급된 곳에.
- **David Silver UCL 강의** (슬러그 `silvercourse`).
- **스탠포드 강좌 4개** -- `cs229`(Machine Learning, ML1의 회귀/분류 절과
  잘 맞음), `cs230`(Deep Learning, ML1의 신경망 절), `cs224n`(NLP with Deep
  Learning, Transformer/Attention/RLHF 절), `cs234`(Reinforcement Learning,
  ML2 전체와 잘 맞음). 본문에 "이 절의 내용은 스탠포드 XX 강좌와도
  겹친다" 식의 명시적 언급이 없어도, **주제가 명확히 겹치는 절이면 "더
  깊이 보려면" 류의 참고 각주로 자연스럽게 달아도 좋다** (예: ML2의 정책
  경사 절 마지막 문단에 "이 주제를 더 다루는 자료: [^cs234]"). 단, 절당
  스탠포드 강좌 각주는 **최대 1개**로 제한해라 (책 전체가 스탠포드
  강좌 광고처럼 보이면 안 된다).

## 배경: 이 환경의 검색 툴 상황 (라이브러리에 없는 논문을 다뤄야 할 때만 해당)

- **WebSearch 툴은 작동하지 않는다** (백엔드 호환성 문제로 400 에러) -- 쓰지 마라.
- **WebFetch는 정상 작동한다.** arXiv API
  (`http://export.arxiv.org/api/query?search_query=all:<검색어>&max_results=3`)나
  Semantic Scholar API
  (`https://api.semanticscholar.org/graph/v1/paper/search?query=<검색어>&fields=title,authors,year,externalIds`)에
  직접 질의해서 정확한 저자·연도·arXiv ID를 확인해라.

## 할 일

1. 이 파일을 Read해서, **이름이 붙은 알고리즘/논문/역사적 사실**이 언급된
   곳을 찾아라. **이미 `[^...]` 각주가 붙어있는 곳은 건드리지 마라**(중복 방지).
2. 그 중 **원저자·원논문이 명확한 것 4~8개 정도**를 골라라 (문장 하나하나에
   달지 마라 -- 핵심 개념 몇 개면 충분. 각주 달 만한 게 없으면 0개도 괜찮다).
3. 고른 각 항목에 대해:
   - **`ref_library.json`에 슬러그가 있으면(예: "ppo", "dqn")** -- 이미 검증된
     `footnote`를 그대로 쓰고, **그림도 함께 삽입해라**: 그 개념이 언급되는
     문단 근처에
     `![<figure_caption>]({img_prefix}ref_<슬러그>.png)`
     형태로 넣어라. WebFetch/브라우저 작업 필요 없음 -- 이미 다 있다.
   - **라이브러리에 없으면** -- **1990년대 이전의 고전적 결과(예: Q-learning,
     REINFORCE, SARSA, TD학습, 통계학의 오래된 정리 등)는 그냥 건너뛰어라.**
     이런 것들은 arXiv/Semantic Scholar 시대 이전이라 API로 잘 안 잡히고,
     확인하려고 여러 API를 돌아가며 재시도하다 보면 시간을 전부 잡아먹는다
     (지난 시도에서 실제로 이것 때문에 타임아웃났다). **2000년대 이후 논문만,
     그것도 파일당 최대 2개까지만, WebFetch 총 5회 이내로** 다뤄라. 하나의
     API가 429(rate limit)를 반환하면 그 논문은 포기하고 다음으로 넘어가라
     (다른 API로 우회 재시도하지 마라). 텍스트 각주만 추가하고 그림은 만들지
     마라 (라이브러리 구축은 범위 밖).
   - 본문에서 처음 언급되는 자리에 `[^슬러그]` 각주 참조를 삽입하고, 파일
     맨 끝에 각주 정의(`[^슬러그]: <footnote 텍스트>`)를 추가해라 (기존
     각주 정의가 있으면 그 아래에 이어서).

## 하지 말 것

- 본문 설명/수식/예제 문장은 **절대 고치지 마라** -- 각주(+라이브러리에
  있는 경우 그림 한 장) 추가만 한다.
- 라이브러리에 없는 논문의 그림은 새로 캡쳐하지 마라 (텍스트 각주만).
- 노트북 작업은 범위 밖이다.
- WebSearch 쓰지 마라 (작동 안 함).
- git commit 하지 마라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <몇 개의 각주를 추가했는지, 그림도 넣었는지, 어떤 논문들인지 1~2문장으로>

또는:
FAILED: <이유>
"""


def all_files():
    sections, _ = gpd.build_kor()
    items = []
    for s in sections:
        book, ch, sec = s["book"], s["chapter"], s["section"]
        if s.get("kind") == "opener":
            path = f"kor/src/{book}/chapter{ch}.md"
            label = f"Chapter {int(ch)} 개요 -- {s['chapter_title']}"
            task_id = f"{book}_ch{ch}_opener_cite"
            img_prefix = "../images/"
        else:
            path = f"kor/src/{book}/chapter{ch}/{sec}.md"
            label = f"{int(ch)}.{sec} {s['title']}"
            task_id = f"{book}_ch{ch}_{sec}_cite"
            img_prefix = "../../images/"
        items.append({"path": path, "label": label, "task_id": task_id, "img_prefix": img_prefix})
    return items


CITED_MARKER = "batch_logs cite-done marker"  # unused, status comes from log_info()


def build_queue(retry_counts, done_ids):
    items = all_files()
    running = gpd.running_task_ids()
    return [it for it in items
            if it["task_id"] not in done_ids
            and retry_counts.get(it["task_id"], 0) < MAX_ATTEMPTS
            and it["task_id"] not in running]


run_log = []
log_lock = threading.Lock()


def run_task(it):
    task_id = it["task_id"]
    prompt = TEMPLATE.format(path=it["path"], label=it["label"], img_prefix=it["img_prefix"])
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
    fix_math_delimiters(REPO / it["path"])
    term, summary = gpd.log_info(task_id)
    result = term or "unknown"
    with log_lock:
        run_log.append({"task_id": task_id, "result": result, "elapsed": elapsed, "summary": summary or ""})
    print(f"[citation] finished {task_id} result={result} elapsed={elapsed:.0f}s :: {summary or ''}", flush=True)
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
                    done_ids.add(item["task_id"])  # never requeue a task that already succeeded
                else:
                    retry_counts[item["task_id"]] = retry_counts.get(item["task_id"], 0) + 1
        except Exception as e:
            print(f"[citation] ERROR on {item['task_id']}: {e}", flush=True)
        q.task_done()


def run_round(round_num, retry_counts, done_ids):
    tasks = build_queue(retry_counts, done_ids)
    if not tasks:
        return False
    print(f"[citation] round {round_num}: {len(tasks)} file(s) queued (concurrency={CONCURRENCY})", flush=True)
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
    print(f"[citation] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = {}
    done_ids = set()  # tasks that returned "done" -- fixes the old bug where a
    # successful task's retry_count got popped back to 0 and looked eligible
    # again next round, so the orchestrator just re-processed all 128 files
    # forever instead of stopping. (Found this after it had been looping for
    # ~24h since the previous day's resume.)
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts, done_ids)
        if not progressed:
            print("[citation] all files done or stuck. Stopping.", flush=True)
            break
        round_num += 1


if __name__ == "__main__":
    main()
