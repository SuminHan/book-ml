#!/usr/bin/env python3
"""Continuously drive cq (local Qwen) over every remaining kor/ section until
the whole book is done. Rolling worker pool (not fixed batches) so a slot
refills the moment any task finishes -- keeps the GPU consistently busy.

- "done" sections are skipped.
- "partial" sections (real content but missing notebook/DONE marker from an
  earlier timeout) get a lighter FINISH_TEMPLATE task.
- everything else gets the full EXPAND_TEMPLATE task.

Writes tools/RUN_STATUS.md after every task completion (mirrors the
RUN_STATUS.md convention already used for the Traffic-UAGCRNTF delegation on
this box). Intended to run for hours inside a detached tmux session.
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
import generate_progress_data as gpd  # noqa: E402

CONCURRENCY = 12
EXPAND_TIMEOUT = 3600   # 60 min -- full "expand this section" task
FINISH_TIMEOUT = 2400   # 40 min -- "just finish what's missing" task
                        # (was 1500s / 25min: 5/5 finish-tasks timed out in
                        # round 1 -- cq explores just as much for "finish"
                        # tasks as full expands, so give it real headroom)
PROMPTS_DIR = REPO / "tools" / "batch_prompts"
LOGS_DIR = REPO / "tools" / "batch_logs"
STATUS_FILE = REPO / "tools" / "RUN_STATUS.md"
PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

FIGURE_NOTE = """
## 그림 (가능하면 넣어라)

이 절에 개념을 시각화할 그림을 **최소 1개** 넣는 걸 목표로 하라 (억지로 욱여넣진
말고, 자연스러운 경우에). 어떤 도구를 쓸지는 내용에 따라 골라라:
- 함수 그래프, 데이터 분포, 결정 경계, 학습 곡선 등 **수치 시각화** -> matplotlib
  (노트북 안에서 만들고, 필요하면 `kor/src/images/`에도 SVG로 내보내 본문에 삽입)
- 신경망/트리/그래프 **구조도**(레이어 연결, 노드-엣지 구조) -> Graphviz 추천:
  `dot` 명령이 PATH에 있다(`dot -Tsvg in.dot -o out.svg`), 또는 Python
  `graphviz` 패키지(bookml 환경에 설치됨) 사용. 문법이 간단해서 빠르게 만들 수
  있다.
- 정말 정교한 다이어그램이 필요하면 TikZ(LaTeX)도 가능하다 -- `tectonic
  파일.tex`로 컴파일해서 PDF/SVG로 변환할 수 있다 (PATH에 있음). 웬만하면
  Graphviz로 충분할 것이다.

만든 그림은 `kor/src/images/` 안에 SVG로 저장하고(기존 파일명 컨벤션인
`chNN_설명.svg` 형식을 따라라 -- 예: `ch03_sigmoid_crossentropy.svg`),
본문에 `![대체텍스트](../../images/파일명.svg)` 형태로 삽입하라.

**절대 금지: 박스/화살표를 ┌└│▼ 같은 유니코드 문자나 +--+ 같은 기호로
그려서 \`\`\`코드블록\`\`\` 안에 넣는 "ASCII 아트"는 그림이 아니다.**
모노스페이스 폭 차이 때문에 렌더링하면 줄이 다 어긋나서 깨져 보인다.
구조도/흐름도가 필요하면 반드시 위의 Graphviz(또는 matplotlib/TikZ)로 실제
이미지 파일을 만들어서 `![]()`로 삽입해라. 코드블록에 박스 그림을 그리고
싶은 유혹이 들면 그 자리에 Graphviz `dot` 파일을 대신 써라.

**주의 (컨텍스트 예산):** 그림을 "검증"한다고 rasterize해서 `Read` 툴로 다시
읽어들이는(이미지로 보는) 짓을 반복하지 마라 -- PNG 하나 읽을 때마다 수만
토큰이 소모돼서 컨텍스트가 금방 바닥난다. 그림이 맞게 만들어졌는지는 **파일이
생성됐는지, SVG 안에 예상한 태그/텍스트가 있는지**(grep, 파일 크기, 텍스트
파싱) 정도로만 가볍게 확인하고 넘어가라. 이미지를 눈으로 보고 확인하고 싶으면
**최대 1번만** Read해라.
"""

EXPAND_TEMPLATE = """너는 KSA(한국과학영재학교) Machine Learning 1/2 교재(book-ml 저장소, 현재
디렉토리)의 공동 저자다. 이 저장소는 mdBook 기반이며, 각 챕터는 1.1/1.2/1.3처럼
번호가 매겨진 "절" 단위로 나뉘어 있고 절 하나 = 실제 수업 50분 분량이다.

## 이번 작업 대상

- 본문 파일: kor/src/{book}/chapter{ch}/{sec}.md ("{ch}.{sec} {sec_title}")
- 챕터 개요(문맥 참고용, 수정 금지): kor/src/{book}/chapter{ch}.md
- 같은 챕터의 다른 절(문맥 참고용, 수정 금지): kor/src/{book}/chapter{ch}/ 안의 나머지 파일들

## 목표

이 절은 지금 짧은 분량인데, **실제 수업 50분을 채우는 분량**으로 대폭 확장해야
한다. 대략 2500~4000단어 상당의 밀도(+ 코드)가 필요하니, 지금의 2.5~3배 이상으로
늘어나야 한다.
{capstone_note}
**절대 원칙: 기존 내용(수식, 표, 예시, FAQ)은 삭제하지 말고 유지한 채 확장한다.**
기존 문체(친절하지만 밀도 높은 설명체, 수학 표기 \\\\( \\\\)/\\\\[ \\\\], "왜?"를 항상
설명, 역사적 일화, "자주 하는 실수", "확인 문제", "자주 묻는 질문" 같은 섹션
패턴)를 그대로 따른다 -- 먼저 이 절과 같은 챕터의 다른 절들을 Read해서 실제
분량/문체 감각을 잡아라.

## 작업 순서 -- 반드시 지킬 것 (중요)

**먼저 써라, 검증은 짧게.** 숫자 예제 검증용 스크립트는 최대 1개만 작성해서
한 번에 필요한 숫자를 확인하고, 결과가 그럭저럭 괜찮으면 바로 채택하고 다음
단계로 넘어가라. 파라미터를 이것저것 바꿔가며 "가장 예쁜" 예제를 찾겠다고
여러 번 재실행하지 마라.

**순서 엄수: 본문(.md)을 완전히 다 쓰기 전에는 노트북을 시작하지 마라.**
시간이 부족해지면 노트북 대신 본문이 끝나 있어야 한다 -- 본문이 이번 절의
핵심 산출물이고, 노트북 없이 본문만 있어도 절반의 성공이지만 그 반대(노트북만
있고 본문이 그대로)는 사실상 실패다. 시간 배분 기준: 전체 예산의 약 2/3은
본문에, 나머지 1/3만 노트북에 써라.

## 실습 (Colab 스타일 Jupyter 노트북) -- 필수

- 새 파일 생성: notebooks/{book}/chapter{ch}_{sec}_<slug>.ipynb (slug는 절 내용에
  맞게 네가 정해라)
- 기존 컨벤션을 그대로 따를 것 -- **참고 노트북은
  `notebooks/ml2/chapter04_policy_evaluation.ipynb` 딱 1개만 Read해라** (마크다운
  셀 + Colab 배지 + numpy/matplotlib 중심, 실행 검증된 출력 포함). 다른 노트북을
  추가로 더 열어보며 "컨벤션이 진짜 맞는지" 재확인하지 마라 -- 하나 보고 바로
  따라 써라. `.ipynb` JSON의 `metadata.colab.provenance` 같은 지엽적 필드는
  신경 쓰지 마라.
- **중요: 노트북을 실제로 로컬에서 실행해서 에러 없이 끝까지 도는지 검증하라.**
  Python 인터프리터는 `/home/smhan/miniconda3/envs/bookml/bin/python` 사용
  (numpy/matplotlib/pandas/scikit-learn/gymnasium/torch(cpu)/graphviz 설치되어 있음).
  `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3`
  로 실행하거나, 로직을 .py로 추출해 먼저 `python`으로 돌려보고 결과를 노트북
  출력 셀에 반영해도 된다.
- **한글 폰트 컨벤션을 grep으로 찾아다니지 마라** -- 노트북 맨 앞 코드 셀에
  아래를 그대로 써라:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  from matplotlib import font_manager
  import matplotlib.pyplot as plt
  kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
  if kr: plt.rcParams["font.sans-serif"] = [kr[0]]
  plt.rcParams["axes.unicode_minus"] = False
  ```
  그림은 `IMG = "/home/smhan/book-ml/kor/src/images"`; `fig.savefig(IMG + "/파일명.svg", bbox_inches="tight")`로 저장.
- kor/src/{book}/chapter{ch}/{sec}.md 본문 상단(제목 바로 아래)에 Colab 배지 링크
  추가. 배지 마크다운 패턴은 저장소 내 다른 챕터(예: kor/src/ml1/chapter15.md)에서
  grep으로 실제 형식 확인해서 그대로 따라라.
{figure_note}
## 하지 말 것

- Beamer 슬라이드는 이번 작업 범위 아님 -- 만들지 마라.
- **notebooks/{book}/README.md는 건드리지 마라** -- 다른 절 작업과 동시에 돌고
  있어서 충돌 위험이 있다.
- 다른 챕터/절 파일은 건드리지 마라.
- git commit은 하지 마라.

## 완료 조건

작업이 끝나면 마지막에 다음을 정확히 출력하라 (맨 마지막 줄에, 이 형식 그대로):

DONE: <이번 절에 새로 추가한 핵심 내용을 2~3문장으로 한국어 요약 -- 어떤
개념/예제/실습/그림을 추가했는지 구체적으로.>

또는 실패 시:
FAILED: <이유>
"""

FINISH_TEMPLATE = """너는 KSA Machine Learning 1/2 교재(book-ml 저장소, 현재 디렉토리)의
공동 저자다. 이 절은 **이전 작업에서 이미 본문 확장이 대부분 끝났지만, 시간
제한(타임아웃)으로 마무리가 안 된 상태**다.

## 대상

- 본문 파일: kor/src/{book}/chapter{ch}/{sec}.md ("{ch}.{sec} {sec_title}")
- 현재 단어수: {word_count}

## 할 일 -- 20분 안에 끝내는 것이 목표다 (엄수)

1. 먼저 `kor/src/{book}/chapter{ch}/{sec}.md`를 Read해서 지금 상태를 파악하라.
   본문이 이미 2500단어+ 수준으로 확장되어 있다면 **다시 쓰지 말고 그대로 둬라.**
   (문장이 중간에 끊겨 있거나 명백히 미완성인 부분만 다듬어라.)
2. `notebooks/{book}/` 안에 이 절에 해당하는 노트북(`chapter{ch}_{sec}_*.ipynb`)이
   있는지 확인하라.
   - **없으면**: 새로 만들어라. **컨벤션 확인용으로 Read할 참고 노트북은
     `notebooks/ml2/chapter04_policy_evaluation.ipynb` 딱 1개면 충분하다 --
     "혹시 다른 노트북은 형식이 다를까" 싶어서 두 번째, 세 번째 노트북을 또
     Read하지 마라. 셀 구조(마크다운+Colab배지 → 코드 → ...), 폰트 설정(위
     참조), `IMG=".../images"` + savefig 정도만 따라하면 충분하고, `.ipynb`
     JSON의 `metadata.colab.provenance` 같은 지엽적 필드는 신경 쓰지 말고
     그냥 최소 형태로 둬도 된다 (mdbook 빌드/실행에 영향 없음).** 확인 다
     했으면 바로 Write로 노트북을 만들기 시작해라. numpy/matplotlib 중심,
     `/home/smhan/miniconda3/envs/bookml/bin/python`으로 실제 실행 검증까지
     끝내라.
   - **있는데 실행 검증이 안 되어 있으면**(출력 셀이 비어있거나 에러 흔적):
     `jupyter nbconvert --to notebook --execute --inplace`로 실행해서 채워라.
   - **있고 이미 실행 검증도 되어 있으면**: 건드리지 말고 3번으로 넘어가라.
3. Colab 배지가 본문 상단에 없으면 추가하라 (다른 챕터 grep해서 패턴 확인).

**절대 하지 마라 -- 지난 시도들이 이것 때문에 전부 타임아웃났다:**
- 책에 나온 숫자·실패율·비율을 **완벽히 재현**하려고 여러 시드/파라미터를
  재실행하며 파고들지 마라. "책 시나리오와 대략 비슷한 결과가 나오는가" 정도만
  확인하고, 안 맞아도(정확히 똑같지 않아도) **그냥 넘어가라** -- 이미 있는
  본문/노트북 숫자를 고치는 것 자체가 이번 작업 범위가 아니다.
- 기존 수식의 "미묘한 오류"를 찾겠다고 GDA/LDA 같은 걸 처음부터 다시
  유도하지 마라. 명백히 틀린 게 눈에 보이지 않는 한 그냥 둬라.
- 노트북/본문이 이미 그럴듯하면 재검증 없이 다음 단계로 넘어가라.
- **matplotlib 한글 폰트/스타일 컨벤션을 grep으로 찾아다니지 마라** -- 이미
  정해져 있다. 새 노트북의 맨 앞 코드 셀에 아래를 그대로 복붙해서 시작하라:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  from matplotlib import font_manager
  import matplotlib.pyplot as plt
  kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
  if kr: plt.rcParams["font.sans-serif"] = [kr[0]]
  plt.rcParams["axes.unicode_minus"] = False
  ```
  그림 저장은 `IMG = "/home/smhan/book-ml/kor/src/images"`; `fig.savefig(IMG + "/파일명.svg", bbox_inches="tight")`.
  이거면 충분하다 -- 다른 노트북이 정말 이거랑 다르게 하고 있는지 확인하러
  돌아다니지 마라.
{figure_note}
## 하지 말 것

git commit 하지 마라. notebooks/{book}/README.md 건드리지 마라. 다른 절 파일
건드리지 마라. Beamer 슬라이드 만들지 마라.

## 완료 조건

끝나면 마지막 줄에 정확히:

DONE: <이번에 실제로 마무리한 것을 2~3문장으로 요약 -- 노트북을 새로 만들었는지,
검증만 다시 돌렸는지, 본문을 얼마나 손봤는지.>

또는:
FAILED: <이유>
"""


def build_queue():
    sections, _ = gpd.build_kor()
    queue = []
    for s in sections:
        # "done" -- nothing to do. "in_progress" -- a cq_run_once.py process
        # is already alive for this section (detected via pgrep in
        # running_task_ids()); re-queuing it would spawn a second concurrent
        # writer on the same file. This matters most right after an
        # orchestrator restart that intentionally left the previous round's
        # workers running (start_new_session=True survivors) -- without this
        # skip, the fresh round's queue re-includes every section still
        # in flight and duplicate processes race on the same .md file.
        if s["status"] in ("done", "in_progress"):
            continue
        kind = "finish" if s["status"] == "partial" else "expand"
        queue.append((s, kind))
    return queue


def make_prompt(section, kind):
    book, ch, sec = section["book"], section["chapter"], section["section"]
    if kind == "expand":
        capstone_note = gpd_capstone_note(ch)
        return EXPAND_TEMPLATE.format(
            book=book, ch=ch, sec=sec, sec_title=section["title"],
            capstone_note=capstone_note, figure_note=FIGURE_NOTE,
        )
    else:
        return FINISH_TEMPLATE.format(
            book=book, ch=ch, sec=sec, sec_title=section["title"],
            word_count=section["word_count_after"], figure_note=FIGURE_NOTE,
        )


def gpd_capstone_note(ch):
    if ch not in ("08", "16"):
        return ""
    return """
**이 절은 팀 프로젝트 / Peer-Review / 총정리 성격의 캡스톤 절이다.** 위의
"50분 강의 분량"을 억지로 적용하지 말고, 이 절의 실제 성격(프로젝트 일정과
채점 기준, Peer-Review 체크리스트/루브릭, 또는 챕터 개념맵+복습 문제 등)에
맞게 확장하라. "실습" 요구사항도 자유롭게 재해석해도 된다.
"""


import re as _re

# CommonMark treats backslash + any of these ASCII punctuation chars as an
# escape and *strips the backslash* in the output. LaTeX uses backslash +
# punctuation for several real commands (\( \) \[ \] delimiters, but also
# spacing commands \, \; \! and \{ \} \| \% \_ \& \$ \#) -- every one of
# them needs doubling in the markdown source or the renderer never sees the
# backslash at all.
_SINGLE_BACKSLASH_MATH = _re.compile(r'(?<!\\)\\([()\[\]{}|,;!%&$#])')
# Bare `_` (subscript, e.g. A_t) inside a math span is valid LaTeX but reads
# as CommonMark emphasis syntax once outside \( \)/\[ \] -- opposite fix
# from the above: escape it (-> \_) so CommonMark's own escaping neutralizes
# it as an emphasis delimiter while still emitting a literal `_` for KaTeX.
_MATH_SPAN = _re.compile(r'\\\\\((.*?)\\\\\)|\\\\\[(.*?)\\\\\]', _re.DOTALL)
_BARE_UNDERSCORE = _re.compile(r'(?<!\\)_')
# A footnote reference immediately followed by `(` -- e.g. "DQN[^dqn](Ch09)"
# -- is CommonMark link syntax `[text](url)` in disguise: pulldown-cmark
# reads `[^dqn]` as literal link TEXT (not a footnote ref) and `(Ch09)` as
# the URL, rendering `<a href="Ch09">^dqn</a>` instead of a footnote. A
# single space between them is enough to stop the link-syntax match while
# leaving the footnote reference intact.
_FOOTNOTE_THEN_PAREN = _re.compile(r'(\[\^[a-zA-Z0-9_-]+\])\(')

# Same failure mode, different character: bare `*` (optimal-value notation,
# e.g. V^*, Q^*, \pi^*) inside a math span is valid LaTeX but CommonMark
# reads paired `*`s as emphasis -- it swallows the asterisks and wraps the
# enclosed text in <em>, which splits the \[ \) text node the client-side
# KaTeX auto-render walks, so the whole span silently fails to render (shows
# as raw \[ ... \] text instead of math). Escape it the same way as `_`.
_BARE_ASTERISK = _re.compile(r'(?<!\\)\*')


def fix_math_delimiters(path: Path):
    """Safety net: cq sometimes writes \\( \\) (single backslash) instead of
    the \\\\( \\\\) this repo's markdown source needs (a single backslash is
    CommonMark's own escape syntax and gets silently stripped by the
    renderer, breaking MathJax/KaTeX on the built page). Idempotent -- a
    no-op if the file already uses double backslashes."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    fixed, n = _SINGLE_BACKSLASH_MATH.subn(r'\\\\\1', text)
    if n:
        text = fixed
        print(f"[orchestrator] fixed {n} single-backslash math delimiter(s) in {path.relative_to(REPO)}", flush=True)

    text, fn_n = _FOOTNOTE_THEN_PAREN.subn(r'\1 (', text)
    if fn_n:
        print(f"[orchestrator] inserted space after {fn_n} footnote-then-paren "
              f"link-syntax collision(s) in {path.relative_to(REPO)}", flush=True)

    underscore_hits = [0]
    asterisk_hits = [0]

    def _escape_span(m):
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        inner, k = _BARE_UNDERSCORE.subn(r'\_', inner)
        underscore_hits[0] += k
        inner, k = _BARE_ASTERISK.subn(r'\*', inner)
        asterisk_hits[0] += k
        wrapper = ("\\\\(", "\\\\)") if m.group(1) is not None else ("\\\\[", "\\\\]")
        return wrapper[0] + inner + wrapper[1]

    text2 = _MATH_SPAN.sub(_escape_span, text)
    if underscore_hits[0]:
        print(f"[orchestrator] escaped {underscore_hits[0]} bare subscript underscore(s) in {path.relative_to(REPO)}", flush=True)
    if asterisk_hits[0]:
        print(f"[orchestrator] escaped {asterisk_hits[0]} bare optimal-value asterisk(s) in {path.relative_to(REPO)}", flush=True)
    if underscore_hits[0] or asterisk_hits[0]:
        text = text2

    if n or fn_n or underscore_hits[0] or asterisk_hits[0]:
        path.write_text(text, encoding="utf-8")

    fffd = text.count("�")
    if fffd:
        print(f"[orchestrator] WARNING: {fffd} U+FFFD replacement char(s) in "
              f"{path.relative_to(REPO)} -- likely corrupted output, needs a manual look", flush=True)


status_lock = threading.Lock()
run_log = []  # list of dicts for RUN_STATUS.md


def write_status():
    total = len(run_log)
    done = sum(1 for r in run_log if r["result"] == "done")
    failed = sum(1 for r in run_log if r["result"] == "failed")
    lines = [
        "# cq Orchestrator — 진행 상황",
        "",
        f"업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"동시성: {CONCURRENCY}",
        f"이번 실행에서 처리: {total}건 (완료 {done}, 실패 {failed})",
        "",
        "| 절 | 종류 | 결과 | 소요(s) | 요약 |",
        "|---|---|---|---|---|",
    ]
    for r in reversed(run_log[-40:]):
        lines.append(
            f"| {r['task_id']} | {r['kind']} | {r['result']} | {r['elapsed']:.0f} | {r['summary'][:80]} |"
        )
    STATUS_FILE.write_text("\n".join(lines) + "\n")


def run_task(section, kind):
    book, ch, sec = section["book"], section["chapter"], section["section"]
    task_id = f"{book}_ch{ch}_{sec}"
    prompt = make_prompt(section, kind)
    prompt_path = PROMPTS_DIR / f"{task_id}.txt"
    log_path = LOGS_DIR / f"{task_id}.log"
    wrapper_path = LOGS_DIR / f"{task_id}.wrapper.log"
    prompt_path.write_text(prompt)
    timeout = EXPAND_TIMEOUT if kind == "expand" else FINISH_TIMEOUT

    t0 = time.time()
    with open(wrapper_path, "w") as wf:
        proc = subprocess.run(
            ["python3", "tools/cq_run_once.py", str(prompt_path), str(log_path), str(timeout)],
            cwd=REPO, stdout=wf, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group/session: killing the
            # orchestrator (or one worker's subprocess.run call) must not
            # cascade a signal to this child -- it survives independently.
        )
    elapsed = time.time() - t0
    fix_math_delimiters(REPO / "kor" / "src" / book / f"chapter{ch}" / f"{sec}.md")

    term, summary = gpd.log_info(task_id)
    result = term or "unknown"
    with status_lock:
        run_log.append({
            "task_id": task_id, "kind": kind, "result": result,
            "elapsed": elapsed, "summary": summary or "",
        })
        write_status()
    print(f"[orchestrator] finished {task_id} ({kind}) result={result} elapsed={elapsed:.0f}s", flush=True)


def worker(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        section, kind = item
        try:
            run_task(section, kind)
        except Exception as e:
            print(f"[orchestrator] ERROR on {section}: {e}", flush=True)
        q.task_done()


RETRY_FILE = REPO / "tools" / "retry_counts.json"
MAX_ATTEMPTS = 4  # give up auto-retrying a section after this many rounds


def load_retry_counts():
    if RETRY_FILE.exists():
        return json.loads(RETRY_FILE.read_text())
    return {}


def save_retry_counts(counts):
    RETRY_FILE.write_text(json.dumps(counts, ensure_ascii=False, indent=2))


def run_round(round_num, retry_counts):
    tasks = build_queue()
    task_id_of = lambda s: f"{s['book']}_ch{s['chapter']}_{s['section']}"

    stuck = [t for t in tasks if retry_counts.get(task_id_of(t[0]), 0) >= MAX_ATTEMPTS]
    tasks = [t for t in tasks if retry_counts.get(task_id_of(t[0]), 0) < MAX_ATTEMPTS]
    if stuck:
        print(f"[orchestrator] round {round_num}: skipping {len(stuck)} section(s) that "
              f"failed {MAX_ATTEMPTS}+ times (needs human look): "
              + ", ".join(task_id_of(t[0]) for t in stuck), flush=True)

    if not tasks:
        return False  # nothing left to do (either all done, or all stuck)

    print(f"[orchestrator] round {round_num}: {len(tasks)} section(s) queued "
          f"(concurrency={CONCURRENCY})", flush=True)
    q = Queue()
    for t in tasks:
        q.put(t)
    threads = []
    for _ in range(CONCURRENCY):
        th = threading.Thread(target=worker, args=(q,), daemon=True)
        th.start()
        threads.append(th)
    q.join()
    for _ in threads:
        q.put(None)
    for th in threads:
        th.join()

    # update retry counts based on this round's outcomes
    for r in run_log[-len(tasks):]:
        if r["result"] == "done":
            retry_counts.pop(r["task_id"], None)
        else:
            retry_counts[r["task_id"]] = retry_counts.get(r["task_id"], 0) + 1
    save_retry_counts(retry_counts)
    print(f"[orchestrator] round {round_num} drained.", flush=True)
    return True


def main():
    retry_counts = load_retry_counts()
    round_num = 1
    while True:
        progressed = run_round(round_num, retry_counts)
        if not progressed:
            print("[orchestrator] no sections left to attempt (all done or all "
                  f"stuck at {MAX_ATTEMPTS}+ failures). Stopping.", flush=True)
            write_status()
            break
        round_num += 1
    write_status()


if __name__ == "__main__":
    main()
