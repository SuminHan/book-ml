너는 한국과학영재학교(KSA) 기계학습 강의 슬라이드 덱 하나에 **이미지를 넣어 week01 수준으로 끌어올리는** 작업을 한다.
결과물은 슬라이드 파일 수정 + 이미지 파일이다. 커밋하지 마라. 원본 저장소(/home/smhan/book-ml)는 절대 수정하지 마라(읽기만).

## 작업 대상
- 덱 폴더(여기서만 작업): {DECK_DIR}
- 메인 파일: {DECK_DIR}/{MAIN}  — `\input{pages/pNN}` 목록
- 페이지: {DECK_DIR}/pages/pNN.tex (프레임 1개씩). 같은 이름의 `.thin` `.orig` 파일은 **절대 건드리지 마라**(빌드 실패 시 되돌림용).
- 그림 폴더: {DECK_DIR}/figs/  (새 이미지는 전부 여기)
- 교재 원문(사실 확인용, 읽기 전용): /home/smhan/book-ml/kor/src/{COURSE}/chapter{NN}/  와 chapter{NN}.md
- 텍스트는 이미 발표용으로 얇아진 상태다. 각 프레임의 `\note{...}` 는 발표자 노트이니 그대로 둬라.

## 기준: week01 이 가진 것 (참고: /home/smhan/book-ml/slides/kor/ml1/week01/pages/ 의 p04 p05 p07 p08 p10 p12b)
1. **사진으로 이야기하는 슬라이드**: 글 없이 사진 + 한두 줄 캡션만 있는 `[plain]` 프레임. 인물(논문 저자, 개척자), 역사적 장면, 실제 데이터셋 샘플, 하드웨어·제품·실제 응용 화면. 캡션에는 연도·이름·수치 같은 **구체적 사실**.
   예) p05: 두 사진을 columns 로 나란히 + 각 캡션 `{\small Yann LeCun --- 2018 Turing Laureate}`
2. **"그림으로: ○○" 슬라이드**: 핵심 개념·모델마다 대표 그림 한 장 + 짧은 설명 (p12b 참고).
3. 분류·관계는 도식(마인드맵, 비교표 그림, 흐름도)으로.

## 할 일
1. 메인 파일과 페이지들을 훑어 덱의 흐름을 파악하라. {LABELS} 의 `needs_figure`/`figure_query` 는 힌트다.
2. **이미지 10~14개**를 반드시 넣어라(10개 미만이면 미완성으로 본다). 배분 목표:
   - 사진/실물 이미지 4~6개 (웹에서 찾기). 이 챕터 주제의 역사·인물·실제 사례. 출처는 자유(블로그·위키·기사·논문 모두 가능)지만
     워터마크 있는 스톡 미리보기와 저해상도(가로 600px 미만) 썸네일은 피하라.
   - 개념 도식 6~8개 (직접 생성). 분포, 결정경계, 학습곡선, 손실곡면, 알고리즘 비교, 수치 예제 시각화 등.
     교재에 있는 수치·예제를 그대로 그림으로 옮겨라. 이미 덱에 같은 그림이 있으면 중복하지 마라.
3. 넣는 방법 (둘 중 하나):
   - **새 프레임**: `pages/pNNg1.tex`, `pNNg2.tex` … 처럼 관련 페이지 번호 뒤에 `g숫자`를 붙인 새 파일을 만들고,
     메인 파일의 `\input{pages/pNN}` 바로 다음 줄에 `\input{pages/pNNg1}` 를 추가. 사진 슬라이드는 `\begin{frame}[plain]`.
   - **기존 프레임에 그림 추가**: 글만 남은 짧은 프레임이면 그림을 넣어 "그림으로" 슬라이드로 만든다.
     이때 그 프레임에 `\begin{center}\begin{varwidth}...` / `\end{varwidth}\end{center}` 래퍼가 있으면 **반드시 제거**하라
     (varwidth 안에 columns/block/그림이 들어가면 컴파일이 무한루프). `\note{...}` 는 보존.
4. 이미지 크기: 사진 단독 슬라이드는 `height=0.8\textheight` 안팎, 두 장이면 columns 로 각 `width=\linewidth`.
   그림+글 슬라이드는 그림 `height=5.5cm` 안팎. 넘치지 않게.
5. 도식 생성: 파이썬은 `{FIGPY}` 를 써라(matplotlib/numpy/sklearn 설치됨). 한글 폰트:
   `from matplotlib import font_manager as fm; fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'); plt.rcParams['font.family']='Noto Sans CJK JP'; plt.rcParams['axes.unicode_minus']=False`
   흰 배경, dpi 200, 글자 크게(12pt 이상), KSA 색 `#2E3192`(주색) `#D6CBB1`(보조) + 필요시 빨강 강조. 생성 스크립트는 `{DECK_DIR}/figs/_gen/` 에 남겨라.
   생성한 PNG 확인은 **Read 로 이미지를 열지 말고**(토큰 절약) 로컬 비전 모델에 물어라:
   `~/qwen-setup/venv/bin/python /tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/pipeline/vl_ask.py 그림.png "한국어 글자가 깨지거나 겹친 곳이 있는가? 축 라벨이 읽히는가? 한 문장으로."`
   VL 답이 애매할 때만 Read 로 직접 열어 봐라.
6. 웹 이미지: 다운로드 후 PNG/JPG 로 저장(webp·svg 는 변환: `rsvg-convert`, `convert` 사용 가능). 파일명은 영문 소문자.
   **모든 웹 이미지는 `{DECK_DIR}/figs/SOURCES.md` 에 한 줄씩 기록**: `파일명 | 원본 페이지 URL | 무엇인지`.
   받은 사진은 `~/qwen-setup/venv/bin/python /tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/pipeline/vl_ask.py 파일 "무엇이 찍혀 있는가? 워터마크나 잘림이 있는가? 한 문장."` 로 확인하라.
   이 모델은 **인물이 누구인지는 못 알아본다** — 인물 확인은 사진을 가져온 원본 페이지의 설명으로 하라.
   사실(연도·이름)이 캡션과 맞는지 원본 페이지·교재로 확인하라 — 지어내지 마라.
7. 컴파일 확인: `cd {DECK_DIR} && xelatex -interaction=nonstopmode -halt-on-error -jobname=slides-only {MAIN}` 를
   **에러 0으로** 통과할 때까지 네가 고쳐라 (`grep '^!' slides-only.log`). 처음부터 실패하는 기존 페이지가 있으면
   그 페이지만 `.thin` 내용으로 복사해 되돌려도 된다(보고에 적어라). Overfull vbox 로 잘리는 새 슬라이드가 없게 하라.

8. 시각 검수: 컴파일 후 `~/qwen-setup/venv/bin/python /tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/pipeline/vqa.py {DECK_DIR}/slides-only.pdf {DECK_DIR}/vqa.json` 을 돌려라.
   전 쪽을 비전 모델이 보고 잘림·겹침·과밀·빈 슬라이드를 표시한다. **네가 만들거나 고친 쪽**이 걸리면 고쳐라
   (그림 크기 줄이기, 프레임 나누기 등). 원래부터 있던 쪽의 문제는 고치지 말고 보고에만 쪽 번호를 적어라.

9. 넘침 검사: `python3 /tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/pipeline/overfull.py {DECK_DIR}/slides-only.log` — 슬라이드 아래로 넘친 쪽이 나온다(비전 모델은 이걸 잘 못 잡는다).
   네가 만든/고친 쪽이 있으면 그림 크기를 줄여 고쳐라. 원래 있던 쪽은 빌드 단계에서 자동 축소되니 두어라.
10. 수식: 새로 쓰는 캡션·문장의 수식·기호(x_j, α, X^T 등)는 **반드시 `$...$` 안에** 써라. 수식 밖에 `_` `^` 를 쓰지 마라.

11. 텍스트 프레임의 `\begin{center}\begin{varwidth}` 래퍼는 가운데 정렬용으로 **의도된 것**이다. 그림을 넣는 프레임에서만 제거하고 나머지는 두어라.

## 보고 (이것만, 25줄 이내)
- 추가한 이미지 목록: `페이지 | 사진/도식 | 한 줄 설명 | (웹이면) 출처 도메인`
- 수정한 기존 페이지 목록
- 컴파일 결과 (에러 0 / 쪽수)
- 못 한 것이 있으면 한 줄
