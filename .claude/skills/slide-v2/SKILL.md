---
name: slide-v2
description: "book-ml/slides/v2 의 차시별(50분) 수업 덱을 새로 만들거나 고치는 스킬. 한 덱 = tex 한 파일 + 그림 원본(make_figs.py) + 실습 노트북(lab/make_lab.py) + 발표자 노트. 도구 slides/v2/tools/deck.py 로 빌드·쪽 찾기·렌더. 트리거: 'v2 슬라이드', '차시별 슬라이드', '새 차시 만들어줘', 'weekNN <주제> 덱', slides/v2 아래 tex/ipynb 수정, '8/24 쪽 이거 고쳐' 처럼 하단 번호로 지적하는 흐름. (kor/ 주 단위 덱 수정은 slide-fix)"
version: "v1.0 (2026-10-04)"
---

# slide-v2 — 차시별 수업 덱 만들기·고치기

먼저 `slides/v2/README.md` 를 읽는다 (폴더 규칙, 도구 사용법, 진행 현황). 첫 덱 예시: `slides/v2/ml1/week03/naive_bayes/`.
새 차시는 이 예시 폴더를 복사해 시작한다 (`data.py`, `make_figs.py`, `lab/make_lab.py`, tex 머리말의 `\markpage`/`\pref` 매크로).

## 루프 (지적 1건당)
1. 사용자는 **하단 번호**("8/24")로 말한다. `deck.py where <덱> 8` 로 프레임·tex 줄을 찾는다. (PDF 쪽 = 하단 번호 + 1)
   "이거"·"그거"가 애매하면 IDE 선택 영역/열린 파일을 먼저 본다. 그래도 둘 이상으로 읽히면 짧게 묻는다
   (예: "제목 폰트" = 프레임 제목인지 표지 제목인지 → 실제로는 **표지**였다).
2. 고친다. 슬라이드 내용이 바뀌면 그 프레임의 `\note{}` 도 같이 고친다. 숫자를 바꾸면 `data.py`·그림·노트·노트북까지 전부.
3. `python3 slides/v2/tools/deck.py build <덱>` (노트 md 자동 갱신). Overfull 이 뜨면 **바로 고친다** (보고만 하지 않는다).
4. 바꾼 쪽만 `deck.py show <덱> N` 으로 렌더해 눈으로 확인. 큰 변경 뒤에는 `deck.py sheet` 로 전체 훑기.
5. 보고는 짧게: 무엇을 바꿨는지(하단 번호로) + build OK + 쪽 수 + 남은 선택지 1개.
   사용자가 "반영 안 됐다"고 하면 디스크의 파일을 먼저 확인하고, 맞으면 IDE 탭이 옛 상태라고 알려 준다.

## 사용자 취향 (이 세션에서 직접 지적받은 것)
- **한 덱 = tex 한 파일.** `pages/` 분할 금지. 폴더 `weekNN/<주제>/`, 파일명은 영어 + 주제 접두어.
  학생용 노트북 `<주제>.ipynb`, 정답 `<주제>_solution.ipynb`, 노트 `<주제>_speaker_notes.md`.
- **"확인 1" 같은 퀴즈 프레임 넣지 않는다** (짜치다). 문제풀이는 한 쪽에 모은다.
- 분량은 쇼츠 덱 수준: 50분 차시에 20~25쪽, 한 장 한 생각, 거의 매 장 그림이나 큰 수식. 텍스트만 몇 장짜리는 실패작.
- **표·예시의 개수는 정수.** "0.7통", "1 미만" 금지. 특정 결과값(99% 등)에 집착하지 말고 입력 확률을 바꿔 정수로 만든다.
- **계산을 보여 줄 때는 각 숫자가 무엇인지 대응시킨다.** 색(가능도 = ksablue, 사전확률 = ksared) + `\underbrace{}_{\text{posterior}}`,
  `\underbrace{분모}_{\text{evidence}}` + 아래에 "값 → 용어(영어/한글) → 뜻" 표. 한 쪽에 안 들어가면 다음 쪽으로 나눈다.
- **"이해가 안 간다"** = 화면에 수식만 있고 이유·구체 예가 없다는 신호. 채팅으로 설명 + 1000통 같은 직접 세기 예시로 답하고,
  슬라이드에도 한 줄/한 쪽을 넣을지 묻는다. 유도는 **한 쪽에 단계별로**(각 줄 오른쪽에 짧은 근거), 결과식은 **다음 쪽에 크게**.
- 여러 경우를 비교하는 그림/표는 경우를 빠짐없이 (예: '무료' / '당첨' / '무료'+'당첨').
- **수식 안 라벨은 영어**: spam, ham, `free', posterior/prior/likelihood/evidence, present. 본문 문장은 한글.
  한글 데이터 단어를 수식에서 영어로 쓰면 그 쪽에 대응을 한 줄 적는다 ("수식에서는 `무료'를 `free' 로 쓴다").
- 기호는 정확하게: 로그 식은 `= … + C` (C 가 무엇인지 적기), 로그에 `∝` 는 "더하는 상수 무시"라고 표시할 때만.
  나이브 가정이 들어가는 줄은 슬라이드 전체에서 `≈` 로 통일한다. 합·곱에는 한계 `\sum_{i=1}^{n}`. 앞쪽 근거를 쓸 땐 `\pref{이름}` 으로 "(p.N)" 자동 참조 (`\markpage{이름}` 을 그 프레임에).
- 슬라이드와 노트북은 **같은 모델**이어야 한다 (naive_bayes 는 베르누이: "메일의 60% 에 '무료'" = 있다/없다 확률,
  스무딩 (n+1)/(M+2)). 다항 (count+1)/(N+V) 와 섞지 않는다.
- 실습은 **실제 공개 데이터(영어)**, 첫 셀에서 내려받기. 빈칸은 `return None  # TODO: …` / `x = None  # TODO: …` (`___` 금지: 문법 오류).
  `make_lab.py` 가 정답 노트북을 실제 실행해 검증한다. 학생용에 정답이 새지 않았는지 grep 으로 확인.
- 크기 조정은 **한 단계씩**. 한 번에 두세 단계 키우면 "너무 크다"가 온다. "원래대로" = 정확히 이전 값으로.
  공용 테마는 건드리지 않고 덱 머리말에서만 (`\setbeamerfont{title}{size=\fontsize{24}{30}\selectfont}` 등).

## LaTeX 함정 (실제로 걸린 것)
- `\[\LARGE ... \]` 는 **아무 효과 없다** (수식 안 크기 명령은 무시됨). `{\LARGE\[ ... \]}` 처럼 **밖에** 둔다.
- 화면 폭을 넘는 큰 수식: `\makebox[\linewidth][c]{\fontsize{19}{24}\selectfont$\displaystyle ...$}`.
  `align*` 의 오른쪽 근거 글은 짧게 (Bayes / log / naive (p.N)) — 길면 잘린다.
- `\rightarrow`, `\Rightarrow`, `\times` 를 문장 안에 쓰면 `Missing $` → `$\rightarrow$`.
- 따옴표: tex 에서는 `` `무료' `` (여는 쪽 = 백틱). 노트북 마크다운 수식(MathJax)은 백틱이 그대로 보이니 ‘free’ (유니코드).
- `% --- N. 제목` 주석 번호는 믿지 말고 `deck.py where` 를 쓴다.
- macOS `sed -i ''` 에서 `\n` 치환은 안 된다 → Edit 도구나 python 으로.

## 환경
- 빌드 `tectonic` (VS Code LaTeX Workshop 도 `book-ml/.vscode/settings.json` 에서 tectonic 으로 설정됨, 저장 시 빌드).
- 그림·노트북 파이썬: `~/Documents/manim_shorts/_env/v/bin/python` (matplotlib, PIL, nbformat, nbclient, sklearn).
- 학생 환경 확인: conda `lab` (`~/miniconda3/envs/lab/bin/jupyter lab`). `conda init` 은 안 해 둠.
- **커밋**: 사용자가 그날 하라고 할 때만. 정답 노트북(`*_solution.ipynb`)은 `slides/v2/.gitignore` 로 막혀 있다 —
  옛 이름(`정답.ipynb` 등)으로 다시 생긴 정답 파일이 없는지 `git status --untracked-files=all slides/v2` 로 확인.
  `lessons/` 의 쇼츠·영상 작업물은 올리지 않는다. v2 는 `lessons/` 와 **별개**로 관리한다 (숫자를 맞추지 않는다).
- `lq` (로컬 Qwen) 는 사용자가 시킬 때만. 원본 슬라이드 요약·노트 초안 같은 대량 초안에만, 수치는 반드시 대조.
