#!/usr/bin/env python3
"""Expand the 32 kor/ chapter-opener stubs (kor/src/{book}/chapter{NN}.md,
one per chapter -- distinct from the N.1/N.2/N.3 section files) from their
current ~70-110 word stub into a proper 500-800 word chapter opener:
hook (keep the existing one), why-this-order / connection to the previous
and next chapter, 3-4 learning objectives, and a one-line summary for each
of the chapter's 3 sections.

Reuses the same cq_run_once.py execution + fix_math_delimiters safety net
as orchestrator.py, but as a single simple rolling-pool round (no partial/
done-notebook status machine needed -- word count alone tells us whether a
chapter opener still needs expanding).
"""
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")
sys.path.insert(0, str(REPO / "tools"))
import generate_progress_data as gpd  # noqa: E402
from orchestrator import fix_math_delimiters  # noqa: E402

CONCURRENCY = 10
TIMEOUT = 1200  # 20 min -- much smaller task than a full section expand
DONE_WORD_THRESHOLD = 400  # stub is ~70-110 words; treat >=400 as already expanded
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

TEMPLATE = """너는 KSA(한국과학영재학교) Machine Learning 1/2 교재(book-ml 저장소, 현재
디렉토리)의 공동 저자다. 이 저장소는 mdBook 기반이고, 각 챕터는 chapter{ch}.md라는
"챕터 오프너"(개요) 파일 하나 + chapter{ch}/1.md, 2.md, 3.md라는 3개의 "절"
파일로 구성된다. 절들은 이미 각각 50분 수업 분량으로 완성돼 있다. 이번 작업은
**챕터 오프너 파일만** 확장하는 것이다.

## 대상

kor/src/{book}/chapter{ch}.md ("Chapter {ch_int}. {ch_title}")

이 챕터의 3개 절: {section_list}

문맥 참고용(수정 금지): 이전 챕터 개요 kor/src/{book}/chapter{prev_ch}.md{prev_note},
다음 챕터 개요 kor/src/{book}/chapter{next_ch}.md{next_note}, 그리고 이 챕터의 3개
절 본문(kor/src/{book}/chapter{ch}/1.md, 2.md, 3.md) -- 각 절이 실제로 뭘 다루는지
파악하려면 먼저 Read해봐라 (전체를 다 읽을 필요는 없고 도입부/구조만 봐도 충분하다).

## 목표

지금 이 파일은 짧은 후킹 문단 + 절 링크 목록뿐인 스텁이다. **500~800단어** 분량의
제대로 된 "챕터 오프너"로 확장하라. 다음 요소를 포함하되, 정해진 소제목 형식을
강제하지는 않는다 (자연스러운 산문으로 녹여도 되고, 소제목을 나눠도 된다):

1. **기존 후킹 문단은 유지**하거나 자연스럽게 다듬어서 도입부로 살려라 (완전히
   갈아엎지 마라 -- 이미 괜찮은 도입부다).
2. **왜 이 순서인가 / 이전·다음 챕터와의 연결.** {prev_next_guidance}
3. **이 챕터의 학습 목표 3~4개** (불릿 리스트 형태 권장 -- "이 챕터를 마치면
   ~할 수 있다" 같은 구체적 문장).
4. **각 절에 대한 한두 문장 요약.** 기존의 절 링크 목록
   (`- [{ch_int}.1 ...](chapter{ch}/1.md)` 형식)은 **그대로 유지**하되, 각 항목
   뒤에 그 절이 실제로 다루는 내용을 한두 문장으로 덧붙여라 (절 본문을 Read해서
   파악한 실제 내용 기반으로 -- 뭉뚱그리지 말고 구체적으로).

## 하지 말 것

- 절 본문 파일(1.md/2.md/3.md)은 이미 완성되어 있다 -- **절대 건드리지 마라.**
- 다른 챕터 파일은 건드리지 마라.
- 노트북/그림 작업은 이번 범위가 아니다 -- 하지 마라.
- git commit 하지 마라.
- 500~800단어를 크게 벗어나지 마라 (짧은 개요 페이지지, 절 본문이 아니다).

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <무엇을 추가했는지 1~2문장>

또는:
FAILED: <이유>
"""


def build_chapter_index():
    sections = gpd.parse_kor_summary()
    by_book_ch = defaultdict(list)
    for s in sections:
        by_book_ch[(s["book"], s["chapter"])].append(s)
    chapters = []
    for (book, ch), secs in by_book_ch.items():
        secs.sort(key=lambda s: s["section"])
        title = secs[0]["chapter_title"]
        chapters.append({"book": book, "chapter": ch, "title": title, "sections": secs})
    chapters.sort(key=lambda c: (c["book"], c["chapter"]))
    return chapters


def word_count(path: Path):
    return len(path.read_text(encoding="utf-8", errors="replace").split()) if path.exists() else 0


def build_queue():
    chapters = build_chapter_index()
    by_book = defaultdict(list)
    for c in chapters:
        by_book[c["book"]].append(c)
    for book in by_book:
        by_book[book].sort(key=lambda c: c["chapter"])

    queue = []
    for c in chapters:
        path = REPO / "kor" / "src" / c["book"] / f"chapter{c['chapter']}.md"
        wc = word_count(path)
        if wc >= DONE_WORD_THRESHOLD:
            continue
        siblings = by_book[c["book"]]
        idx = next(i for i, s in enumerate(siblings) if s["chapter"] == c["chapter"])
        prev_c = siblings[idx - 1] if idx > 0 else None
        next_c = siblings[idx + 1] if idx < len(siblings) - 1 else None
        c["prev"] = prev_c
        c["next"] = next_c
        c["word_count_before"] = wc
        queue.append(c)
    return queue


def make_prompt(c):
    section_list = ", ".join(f"{c['chapter']}.{s['section']} {s['title']}" for s in c["sections"])
    prev_c, next_c = c["prev"], c["next"]
    if prev_c:
        prev_ch, prev_note = prev_c["chapter"], f" (\"{prev_c['title']}\")"
        prev_txt = f"이전 챕터(\"{prev_c['title']}\")에서 다룬 내용과 어떻게 이어지는가"
    else:
        prev_ch, prev_note = c["chapter"], " (이 챕터가 이 책의 첫 챕터이므로 해당 없음)"
        prev_txt = "이 챕터가 책의 출발점이라는 점"
    if next_c:
        next_ch, next_note = next_c["chapter"], f" (\"{next_c['title']}\")"
        next_txt = f"다음 챕터(\"{next_c['title']}\")로 어떻게 이어지는가"
    else:
        next_ch, next_note = c["chapter"], " (이 챕터가 이 책의 마지막 챕터이므로 해당 없음)"
        next_txt = "이 챕터가 이 책(또는 이 블록)의 마무리라는 점"
    prev_next_guidance = f"{prev_txt}, 그리고 {next_txt}를 한두 문장씩 짚어라."
    return TEMPLATE.format(
        book=c["book"], ch=c["chapter"], ch_int=int(c["chapter"]), ch_title=c["title"],
        section_list=section_list, prev_ch=prev_ch, prev_note=prev_note,
        next_ch=next_ch, next_note=next_note, prev_next_guidance=prev_next_guidance,
    )


run_log = []
log_lock = threading.Lock()


def run_task(c):
    book, ch = c["book"], c["chapter"]
    task_id = f"{book}_ch{ch}_opener"
    prompt = make_prompt(c)
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
    path = REPO / "kor" / "src" / book / f"chapter{ch}.md"
    fix_math_delimiters(path)
    new_wc = word_count(path)
    term, summary = gpd.log_info(task_id)
    result = term or "unknown"
    with log_lock:
        run_log.append({"task_id": task_id, "result": result, "elapsed": elapsed,
                         "wc_before": c["word_count_before"], "wc_after": new_wc,
                         "summary": summary or ""})
    print(f"[chapter_opener] finished {task_id} result={result} elapsed={elapsed:.0f}s "
          f"words={c['word_count_before']}->{new_wc}", flush=True)


def worker(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        try:
            run_task(item)
        except Exception as e:
            print(f"[chapter_opener] ERROR on {item['book']}_ch{item['chapter']}: {e}", flush=True)
        q.task_done()


def run_round(round_num, retry_counts, max_attempts=3):
    tasks = build_queue()
    key = lambda c: f"{c['book']}_ch{c['chapter']}"
    stuck = [c for c in tasks if retry_counts.get(key(c), 0) >= max_attempts]
    tasks = [c for c in tasks if retry_counts.get(key(c), 0) < max_attempts]
    if stuck:
        print(f"[chapter_opener] round {round_num}: skipping {len(stuck)} stuck: "
              + ", ".join(key(c) for c in stuck), flush=True)
    if not tasks:
        return False
    print(f"[chapter_opener] round {round_num}: {len(tasks)} chapter(s) queued "
          f"(concurrency={CONCURRENCY})", flush=True)
    q = Queue()
    for c in tasks:
        q.put(c)
    threads = [threading.Thread(target=worker, args=(q,), daemon=True) for _ in range(CONCURRENCY)]
    for th in threads:
        th.start()
    q.join()
    for _ in threads:
        q.put(None)
    for th in threads:
        th.join()
    for r in run_log[-len(tasks):]:
        if r["wc_after"] >= DONE_WORD_THRESHOLD:
            retry_counts.pop(r["task_id"].replace("_opener", ""), None)
        else:
            k = r["task_id"].replace("_opener", "")
            retry_counts[k] = retry_counts.get(k, 0) + 1
    print(f"[chapter_opener] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = {}
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts)
        if not progressed:
            print("[chapter_opener] all chapters done or stuck. Stopping.", flush=True)
            break
        round_num += 1


if __name__ == "__main__":
    main()
