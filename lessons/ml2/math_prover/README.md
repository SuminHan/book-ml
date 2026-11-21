<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# 리만가설과Lean

*Math AI and the proof checker Lean*

> AI가 증명했다는 리만 가설, 증명 검사기 Lean을 통과하면 믿어도 될까?

## 영상 내용

- 2026년 11월 20일, 앤트로픽의 수학자 레벤트 알포게(Levent Alpöge)는 Claude와 함께 리만 가설의 증명을 찾은 것 같다고 발표했어요. 증명은 사람이 읽는 논문이 아니라 Lean 코드(보도 기준 파일 약 17만 5천 개)로 먼저 공개됐어요
- 리만 가설: s = a + bi 같은 복소수를 넣는 제타 함수 ζ(s)에서, 값이 0이 되는 (자명하지 않은) 점이 모두 실수부(Real part) 1/2 위에 있다는 1859년의 추측
- Lean은 수식과 증명을 코드로 옮겨 적는 프로그램이에요 (Mathlib에는 리만 가설이 RiemannHypothesis라는 정의로 들어 있어요). 코드로 쓰면 한 줄 한 줄이 규칙에 맞는지 컴퓨터가 검사하는 프로그램이에요. 한 줄만 틀려도 통과하지 못하고, 최종 검사는 기본 규칙 몇 개만 가진 작은 커널(kernel)이 맡아요
- 그래서 수학 AI도 Lean으로 배워요. 구글 딥마인드 AlphaProof는 Lean이 통과시킨 증명만 골라 다시 배워 2024 국제수학올림피아드에서 42점 중 28점(은메달 수준)을 받았어요
- Lean을 통과해도 확인할 곳 ① 문장: Lean은 적힌 문장을 증명했는지만 봐요. 예를 들어 '모든 영점'을 '처음 영점 10개'로 잘못 옮겨 적으면, 이미 계산으로 확인된 쉬운 문장이라 Lean은 통과시키지만 리만 가설을 증명한 건 아니에요
- ② 공리: "이건 참이라고 치자"는 공리를 몰래 넣으면 뭐든 증명돼요. #print axioms로 확인하는데, 이번 증명은 표준 공리 3개(propext, Classical.choice, Quot.sound)만 썼다고 MIT 앤드루 서덜랜드가 밝혔어요
- ③ Lean 자체 버그: 최근에도 공리 없이 거짓(False)을 증명할 수 있는 커널 버그가 고쳐졌어요(Lean 4.32.2). 그래서 따로 만든 검사기로도 돌려요. 서덜랜드의 검증 페이지 기준 nanoda(27분), lean4lean, con-leche, leanchecker는 이미 통과했고, 문장 · 공리 · 커널을 한꺼번에 보는 Comparator는 문장 · 공리 확인을 통과한 채 커널 검사가 진행 중이에요 (2026년 11월 21일 12:55 UTC 기준)
- 전문가 분위기: 서덜랜드 "뭔가 이상할 가능성은 거의 0", 케빈 버자드 "Comparator 결과를 아직 기다리는 중", 브라이언 콘리 "인류 역사의 기념비적인 순간, 더 쉬운 증명도 나올 것". 아직 공식적으로 풀린 것으로 인정된 상태는 아니에요

**참고**: 출처: AFP (phys.org, 2026.11.20), Scientific American (2026.11.20), Interesting Engineering, Andrew Sutherland 검증 페이지 (math.mit.edu/~drew/grh), Lean Language Reference "Validating a Lean Proof", leanprover-community/riemann-hypothesis (verification/NOTES.md), Google DeepMind 블로그 (2024.7.25), Hubert et al., Nature (2025) AlphaProof. 화면의 Lean 코드 · axiom cheat · '처음 영점 10개만' 문장 · 증명 카드 8장 중 2장 통과는 설명용 예시예요. 이모지: Twemoji (CC BY 4.0) · 음성: Typecast (필재) · 배경음: Suno "Curious Groove" (직접 제작)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 알포게: Claude와 함께 리만 가설 증명 / Lean 통과 = 진짜 증명? | 11월 20일, 앤트로픽의 레벤트 알포게가 클로드와 함께 리만 가설을 증명한 것 같다고 발표했어요. 증명은 린이라는 프로그램의 코드로 공개됐죠. 그럼 린을 통과하면, 진짜 맞는 증명일까요? |
| 2 | s = a + bi (복소수) / 영점은 모두 실수부 1/2 위? | 리만 가설은 이 제타 함수에 관한 백육십칠 년 된 문제예요. 여기서 s는 실수부와 허수부가 있는 복소수예요. 제타 함수가 영이 되는 점이, 모두 실수부 이분의 일 위에 있다는 주장이죠. |
| 3 | Lean: 수식과 증명을 코드로 / 한 줄만 틀려도 통과 X | 린은 수식과 증명을 코드로 옮겨 적는 프로그램이에요. 이렇게 코드로 쓰면, 한 줄 한 줄이 규칙에 맞는지 컴퓨터가 검사해요. 한 줄만 틀려도 통과 못 해요. |
| 4 | 커널 Kernel: 기본 규칙 몇 개로 / 파일 약 17만 개를 끝까지 검사 | 검사는 커널이라는 작은 부분이 맡아요. 증명이 파일 십칠만 개로 아무리 길어도, 커널이 몇 가지 기본 규칙으로 끝까지 다시 따져요. |
| 5 | AlphaProof: 통과한 증명만 다시 학습 / IMO 2024 42점 중 28점 | 그래서 수학 AI도 린으로 배워요. 딥마인드의 알파프루프는 린이 통과시킨 증명만 골라 다시 배워서, 국제수학올림피아드 은메달 수준까지 갔어요. |
| 6 | ① 문장을 잘못 옮기면 / 쉬운 문장이 통과 ≠ 리만 가설 증명 | 그런데 린을 통과해도 틀릴 수 있는 곳이 세 군데 있어요. 첫째, 문장. 린은 적힌 문장을 증명했는지만 봐요. 예를 들어 모든 영점을, 처음 영점 열 개로 잘못 옮겨 적으면, 그건 이미 계산된 쉬운 문장이라 통과해요. 하지만 리만 가설을 증명한 건 아니죠. |
| 7 | ② 공리: 참이라고 치면 뭐든 증명 / 이번엔 기본 공리 3개만 | 둘째, 몰래 넣은 가정. 이건 그냥 참이라고 치자는 공리를 넣으면 뭐든 증명돼요. 그래서 쓴 공리를 출력해 보는데, 이번엔 기본 공리 세 개만 나왔대요. |
| 8 | ③ Lean 버그: 다른 검사기로도 확인 / nanoda 등 통과 · Comparator 진행 중 | 셋째, 린 자체의 버그. 최근에도 공리 없이 거짓을 증명하는 버그가 고쳐졌어요. 그래서 따로 만든 검사기로도 돌려 봐요. 이번 증명은 나노다 같은 다른 검사기들을 이미 통과했고, 전부 한꺼번에 보는 컴패러터 검사는 아직 진행 중이에요. |
| 9 | 전문가 분위기 / 기대 + 검증 결과 기다리는 중 | 전문가 분위기는 이래요. 엠아이티 서덜랜드는 이상할 가능성이 거의 없다고 했고, 케빈 버자드는 컴패러터 결과를 기다리자고 해요. 리만 가설 전문가 콘리는 기념비적인 순간이라면서도, 더 쉬운 증명이 나올 거라고 했어요. |
| 10 | 수학 AI와 Lean 한 장 정리 | 린은 한 줄씩 검사해서 수학 AI의 채점자가 됐어요. 하지만 문장, 공리, 린 자체까지 확인해야 진짜 증명이에요. |

## 출처

- 슬라이드: 없음 (쇼츠 전용). 사용자: '리만 가설이 풀렸다는 얘기 → 수학에 쓰이는 LLM 은 어떻게 학습했는지, ML 흐름 유지' (2026-11-21)
- 실험 없음. 숫자는 보도 · 논문 (data.py)
- 리만 가설 증명 주장 (2026-11-20, 아직 검증 중, 2026-11-21 기준)
  - AFP (phys.org/news/2026-11-anthropic-claude-math-greatest-riddle.html): Levent Alpöge (Anthropic) "I believe I have, working with Claude, discovered a proof of the Riemann hypothesis", 4쪽 노트, GRH 까지 주장, 본인이 증명의 수학을 아직 다 익히지 못했다고 말함, Lean 판만 우선 공개. 같은 날 Stanford Vakil 교수가 박사후연구원의 독립 증명 제출을 언급 (영상엔 안 씀)
  - Scientific American (2026-11-20, Clara Moskowitz): Andrew Sutherland (MIT) 독립 컴파일, "compiled and independently rechecked by the Lean kernel with only the three standard axioms", Kevin Buzzard "We still await a successful comparator run" · Lean 속 정리의 문장이 실제 정리와 같은지가 첫 질문
  - Interesting Engineering: 약 175,187 소스 파일 · 239MB ("reportedly"), GRH, 공식적으로는 아직 미해결
- 리만 가설: 1859 (2026 − 1859 = 167년). 영점 높이 14.1347 · 21.0220 · 25.0109 · 30.4249 · 32.9351 · 37.5862 · 40.9187 · 43.3271 · 48.0052 · 49.7738 (LMFDB). 계산 확인 처음 10^13 개: X. Gourdon (2004)
- AlphaProof: DeepMind 블로그 (2024-07-25) '약 100만 개 자연어 문제 → 형식 문장', AlphaZero RL, 'Each proof that was found and verified is used to reinforce', IMO 2024 28/42 (금 29), 한 문제는 몇 분 · 나머지는 최대 사흘, 대회는 4.5시간 × 2, 문제는 사람이 형식 언어로 옮김. Hubert et al., Nature (2025-11-12) "Olympiad-level formal mathematical reasoning with reinforcement learning"
- 설명용 예시: Lean 코드 ((a+b)^2 전개, 틀린 줄 a^2 + b^2), 증명 카드 8장 중 2장 통과, '처음 10조 개만 1/2 위' 문장 바꾸기. 영점 10개 뒤 흐르는 점은 '많다' 표시일 뿐 실제 위치 아님. 띠 위 '?' 점은 가상의 영점
- 이모지: Twemoji (CC BY 4.0) 🧐 ⏳ 🥈 🔍 · 음성: Typecast (필재) · 배경음: Suno "Pulse of News" (사용자 생성)

### v3 (2026-11-21, 사용자: 리만 가설 설명은 식 한 줄만, Lean · Lean 이 틀릴 가능성 · 전문가 분위기 위주)
- 수식: ζ(s) = Σ 1/n^s, ζ(s) = 0 ⇒ Re(s) = 1/2 (자명하지 않은 영점, 화면은 간단히 줄인 표기)
- Lean 신뢰 문제: Lean Language Reference 4.29.0 "Validating a Lean Proof" — 문장의 의미는 사람이 확인 ('No human error or misleading presentation of the theorem statement'), sorry · 사용자 공리는 #print axioms 로 확인 (sorryAx · 표준 외 공리), 표준 공리 propext · Quot.sound · Classical.choice, comparator = 문장 일치 + 허용 공리 + 커널 재검사 (샌드박스)
- 커널 버그: leanprover-community/riemann-hypothesis verification/NOTES.md — 이슈 #14576 'wrong-structure projections giving an axiom-free proof of False', 4.32.2 · 4.33.0 에서 고침. comparator.json 은 Lean paranoid · nanoda 두 커널 사용. 결과는 11-21 기준 '진행 중' (Buzzard 'We still await a successful comparator run')
- 전문가: Sutherland "Honestly the chance that anything is fishy here is approximately zero" (Scientific American), Buzzard (Lean Zulip, SciAm 인용), Conrey "It's a monumental moment in human history" · "there will be simpler ways to do it" (AFP)
- 예시: axiom cheat : RiemannHypothesis 코드, '처음 영점 10개만' 문장 바꾸기

### v4 (2026-11-21 저녁): Real(s) 표기 · 복소수 s = a + bi · ① 문장 예시 구체화 · nanoda 통과 반영 · 배경음 Curious Groove
- Sutherland 검증 페이지 math.mit.edu/~drew/grh/ ('Page last updated 2026-11-21 12:55 UTC'): nanoda accepts (27 min 42 s), lean4lean 175,187/175,187 pass, leanchecker pass, con-leche accepts, Lean default · paranoid running, lake comparator running ('statement comparison and axiom check already passed in one run; kernels in progress'). 검사한 GRH 는 Dirichlet L-함수까지만

## 파일

- 영상: `~/book-ml/videos/주제별/리만가설과Lean/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
