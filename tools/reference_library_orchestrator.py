#!/usr/bin/env python3
"""Build a one-time "reference figure library" for the ~20 most-cited named
algorithms/papers across kor/ (PPO, DQN, Transformer, VAE, ...), each as its
own cq task: find the correct paper (WebFetch against arXiv/Semantic
Scholar APIs -- WebSearch 400s on this backend), capture ONE illustrative
figure from it, and save both the image + a small metadata JSON.

Why a separate pass instead of doing this inline per-section: the same
~20 papers get cited across dozens of kor/ sections -- capturing each
paper's figure once here (kor/src/images/ref_<slug>.{png,json}) and having
citation_orchestrator.py's per-section pass just re-use whatever's already
in this library (cheap file check, no browser) avoids re-running a full
browser session 5-10x for the same PPO/DQN/Transformer diagram.

Each task writes its own kor/src/images/ref_<slug>.json (no shared-file
write race) -- merge_library() consolidates them into
tools/ref_library.json afterward for citation_orchestrator.py to read.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path("/home/smhan/book-ml")
sys.path.insert(0, str(REPO / "tools"))
from orchestrator import fix_math_delimiters  # noqa: E402 (unused here, kept for parity)
import generate_progress_data as gpd  # noqa: E402

CONCURRENCY = 6  # each task drives its own Chromium instance -- lighter than the LLM-only pool
TIMEOUT = 1500
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
IMAGES_DIR = REPO / "kor" / "src" / "images"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# (slug, search hint) -- picked from a frequency scan of kor/src/**/*.md for
# named algorithms/papers with a real, findable original paper (skipped
# GDA/나이브베이즈/LSTM/Q-learning/SARSA/REINFORCE/backprop -- classic
# results predating arXiv with no clean single-paper figure to cite).
TARGETS = [
    ("ppo", "Proximal Policy Optimization Algorithms Schulman 2017"),
    ("dqn", "Human-level control through deep reinforcement learning Mnih 2015 Nature DQN"),
    ("trpo", "Trust Region Policy Optimization Schulman 2015"),
    ("gae", "High-Dimensional Continuous Control Using Generalized Advantage Estimation Schulman 2015"),
    ("a3c", "Asynchronous Methods for Deep Reinforcement Learning Mnih 2016"),
    ("rlhf", "Deep reinforcement learning from human preferences Christiano 2017"),
    ("instructgpt", "Training language models to follow instructions with human feedback Ouyang 2022 InstructGPT"),
    ("alphago", "Mastering the game of Go with deep neural networks and tree search Silver 2016 AlphaGo"),
    ("transformer", "Attention Is All You Need Vaswani 2017 Transformer"),
    ("vae", "Auto-Encoding Variational Bayes Kingma Welling 2013 VAE"),
    ("gan", "Generative Adversarial Networks Goodfellow 2014"),
    ("word2vec", "Efficient Estimation of Word Representations in Vector Space Mikolov 2013 word2vec"),
    ("resnet", "Deep Residual Learning for Image Recognition He 2015 ResNet"),
    ("alexnet", "ImageNet Classification with Deep Convolutional Neural Networks Krizhevsky 2012 AlexNet NeurIPS"),
    ("dropout", "Dropout A Simple Way to Prevent Neural Networks from Overfitting Srivastava 2014"),
    ("batchnorm", "Batch Normalization Ioffe Szegedy 2015"),
]

TEMPLATE = """너는 book-ml 저장소(현재 디렉토리)의 "레퍼런스 그림 라이브러리"를 만드는
작업을 한다. 대상 논문 1개에 대해서만 작업한다.

## 대상

슬러그: {slug}
검색 힌트: {hint}

## 배경: 이 환경의 툴 상황

- **WebSearch는 작동하지 않는다** (백엔드 호환성 문제, 400 에러) -- 쓰지 마라.
- **WebFetch는 정상 작동한다** -- arXiv API(`http://export.arxiv.org/api/query?search_query=all:<검색어>`),
  Semantic Scholar API(`https://api.semanticscholar.org/graph/v1/paper/search?query=<검색어>&fields=title,authors,year,externalIds`)에
  직접 질의해서 정확한 저자/연도/arXiv ID(또는 학회·저널)를 확인해라.
- **Playwright MCP 브라우저 툴**(`mcp__playwright__*`)이 연결되어 있다.
  구글 검색은 이 서버 IP에서 reCAPTCHA에 막히니 **쓰지 마라** -- 대신 논문
  URL에 직접 navigate해라.

## 할 일

1. WebFetch로 정확한 논문(제목/저자/연도/arXiv ID 또는 학회)을 확인해라.
2. 그 논문에서 **핵심 아이디어를 보여주는 그림 1개**(주로 Figure 1이나
   아키텍처/구조 다이어그램)를 구하는데, 다음 순서로 시도해라:
   a. **먼저 (가벼움):** arXiv ID가 있으면
      `https://ar5iv.labs.arxiv.org/html/<ID>` 를 WebFetch나 `curl`로 확인해서
      `class="ltx_graphics"` 이미지 태그가 있는지 봐라. 있으면 그 이미지의
      `src` URL을 `curl -sL <url> -o {images_dir}/ref_{slug}.png` 로 직접
      다운로드해라 (스크린샷보다 훨씬 깨끗하다).
   b. **ar5iv에 그림이 없으면 (예: 오래된 논문, 변환 실패):** Playwright MCP로
      논문 PDF를 직접 열어라 (arXiv면 `https://arxiv.org/pdf/<ID>`, NeurIPS
      논문이면 `https://papers.nips.cc/paper_files/...` 형태로 검색해서 찾아라).
      Chromium이 PDF를 네이티브로 렌더링한다. 핵심 그림이 있는 페이지로
      이동해서(썸네일 클릭 등), `browser_run_code_unsafe`의
      `page.screenshot({{clip: {{x,y,width,height}}}})`로 그 그림 영역만
      정확히 크롭 캡쳐해라. **크롭할 때 그림의 라벨/텍스트가 잘리지 않도록
      여유를 충분히 둬라** (그림 전체가 잘림 없이 프레임 안에 들어와야 한다).
      캡쳐 후 `{images_dir}/ref_{slug}.png` 에 저장해라.
3. 이미지가 저장됐으면, **Read 툴로 딱 1번만** 읽어서 잘리지 않고 제대로
   캡쳐됐는지 육안 확인해라 (여러 번 반복 캡쳐하지 마라 -- 컨텍스트 낭비).
   확실히 별로면 크롭 좌표만 조정해서 한 번 더 시도하고, 그래도 안 되면
   포기하고 이미지 없이 FAILED로 끝내라 (억지로 계속 재시도하지 마라).
4. 메타데이터 파일을 작성해라: `{images_dir}/ref_{slug}.json`
   ```json
   {{
     "slug": "{slug}",
     "title": "<정확한 논문 제목>",
     "authors": "<저자, et al. 형태>",
     "year": <연도>,
     "venue": "<arXiv:XXXX.XXXXX 또는 학회/저널명>",
     "footnote": "<Author, A. et al. (연도). \\"제목.\\" arXiv:XXXX.XXXXX. 형태의 각주 텍스트>",
     "figure_caption": "<이 그림이 뭘 보여주는지 1문장, 원 논문 Figure 번호 포함>",
     "image_ok": true
   }}
   ```
   이미지 캡쳐에 실패했으면 `"image_ok": false`로 쓰고 image_path 필드는 빼라
   (메타데이터만 있어도 citation 작업에서 텍스트 각주로는 쓸 수 있다).

## 하지 말 것

- 다른 슬러그/논문 작업하지 마라 (이 슬러그 하나만).
- book-ml의 다른 파일(kor/src/*.md 등)은 건드리지 마라 -- 이 작업은
  라이브러리 구축 단계이지, 아직 각주를 본문에 삽입하는 단계가 아니다.
- git commit 하지 마라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <논문 확인 결과 + 이미지 캡쳐 성공 여부 1~2문장>

또는:
FAILED: <이유>
"""


def make_prompt(slug, hint):
    return TEMPLATE.format(slug=slug, hint=hint, images_dir="kor/src/images")


run_log = []
log_lock = threading.Lock()


def run_task(slug, hint):
    task_id = f"reflib_{slug}"
    prompt = make_prompt(slug, hint)
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
    meta_path = IMAGES_DIR / f"ref_{slug}.json"
    has_meta = meta_path.exists()
    with log_lock:
        run_log.append({"slug": slug, "result": result, "elapsed": elapsed,
                         "summary": summary or "", "has_meta": has_meta})
    print(f"[reflib] finished {slug} result={result} elapsed={elapsed:.0f}s "
          f"meta={'yes' if has_meta else 'no'} :: {summary or ''}", flush=True)


def worker(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        slug, hint = item
        try:
            run_task(slug, hint)
        except Exception as e:
            print(f"[reflib] ERROR on {slug}: {e}", flush=True)
        q.task_done()


def merge_library():
    lib = {}
    for meta_path in sorted(IMAGES_DIR.glob("ref_*.json")):
        try:
            data = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[reflib] WARNING: couldn't parse {meta_path}: {e}", flush=True)
            continue
        slug = data.get("slug") or meta_path.stem.replace("ref_", "")
        img_path = IMAGES_DIR / f"ref_{slug}.png"
        data["image_path"] = f"images/ref_{slug}.png" if (data.get("image_ok") and img_path.exists()) else None
        lib[slug] = data
    out_path = REPO / "tools" / "ref_library.json"
    out_path.write_text(json.dumps(lib, ensure_ascii=False, indent=2))
    print(f"[reflib] merged {len(lib)} entries -> {out_path}", flush=True)
    return lib


def main():
    q = Queue()
    for slug, hint in TARGETS:
        meta_path = IMAGES_DIR / f"ref_{slug}.json"
        if meta_path.exists():
            print(f"[reflib] skipping {slug} (already has metadata)", flush=True)
            continue
        q.put((slug, hint))
    if q.empty():
        print("[reflib] nothing to do -- all targets already have metadata.", flush=True)
    else:
        n = q.qsize()
        print(f"[reflib] {n} paper(s) queued (concurrency={CONCURRENCY})", flush=True)
        threads = [threading.Thread(target=worker, args=(q,), daemon=True) for _ in range(CONCURRENCY)]
        for th in threads:
            th.start()
        q.join()
        for _ in threads:
            q.put(None)
        for th in threads:
            th.join()
    merge_library()


if __name__ == "__main__":
    main()
