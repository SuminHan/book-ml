<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# 수학AI는어떻게배웠나

*How AlphaProof learned math*

> 수학 AI는 어떻게 수학을 풀까? 2024 국제수학올림피아드에서 학생 609명 중 5명만 푼 문제를 푼 구글 딥마인드 AlphaProof는 어떻게 배웠을까

## 영상 내용

- IMO 2024 6번: 모든 유리수 x, y 에 대해 f(x + f(y)) = f(x) + y 또는 f(f(x) + y) = x + f(y) 를 만족하는 유리수 함수 f 에서, f(r) + f(−r) 은 최대 몇 가지 값이 될 수 있을까? (답: 2가지)
- Lean 기본: theorem 이름 (변수) : 증명할 문장 := by 다음 줄부터 전술(tactic)을 한 줄씩 써요. 줄마다 '남은 목표'가 바뀌고, 목표가 하나도 안 남으면 통과예요
- ① 읽기: 먼저 공개된 코드와 수학 글 약 3,000억 토큰으로 다음 단어 맞히기를 배워요. 그다음 사람이 Lean으로 쓴 증명(Mathlib)에서 뽑은 '지금 상태 → 한 걸음' 약 30만 개를 따라 해요
- ② 문제 만들기: Lean으로 쓰인 문제는 너무 적어요 (사람이 직접 옮긴 문제 약 3,500개). 그래서 Gemini 기반 AI가 말로 쓴 수학 문제 약 100만 개를 Lean 문제 약 8,000만 개로 옮겼어요
- 잘못 옮긴 문제도 버리지 않아요. 원래 문제와 달라도 그 자체로 올바른 Lean 문제라서, 증명하거나 틀렸음을 보이는(반증) 연습 문제가 돼요
- ③ 강화학습: 8,000만 문제마다 증명 또는 반증을 시도하고, Lean이 검사해 통과한 것만 다시 배워요. TPU 약 8만 일 (예: 4,000개로 20일), 풀린 문제 비율이 꾸준히 늘어요
- ④ 시험 중 연습(Test-Time RL): 그래도 못 푼 어려운 문제는 쉽게 바꾸기 · 일반화 · 중간 단계 떼어 내기로 닮은 문제를 수십만 개 만들고, 그걸로 다시 강화학습을 해요
- 올림피아드 기출(formal-imo) 정답률: 문제당 2 TPU분 33.2% → 12 TPU시간 43.7% → 닮은 문제 연습 50 TPU일 53.9% → 500 TPU일 58.3%. 실제 IMO 2024에서도 1 · 2 · 6번을 문제마다 2~3일씩 이렇게 연습해서 풀었어요 (AlphaGeometry 2의 4번과 합쳐 42점 중 28점, 은메달 수준)

**참고**: 출처: Hubert et al., "Olympiad-level formal mathematical reasoning with reinforcement learning", Nature (2025.11.12) 본문 · 보충자료, Google DeepMind 블로그 (2024.7.25). 대회 문제는 전문가가 Lean으로 옮겨 줬어요. 화면의 a + b ≥ 2√(ab) 번역 · 1번 문제 변형 4개 · 타일 격자는 설명용 예시예요. 이모지: Twemoji (CC BY 4.0) · 음성: Typecast (필재) · 배경음: Suno "Curious Groove" (직접 제작)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 수학 AI는 어떻게 수학을 풀까? / IMO 2024 6번: 609명 중 5명만 푼 문제 | 수학 AI는 어떻게 수학을 푸는 걸까요? 이건 2024년 국제수학올림피아드 육 번, 학생 육백구 명 중 다섯 명만 완벽히 푼 함수 문제예요. 딥마인드의 알파프루프는 이 문제를 풀었어요. |
| 2 | Lean: theorem 이름 (변수) : 문장 := by / 증명 = 전술 한 줄씩, 목표가 다 사라지면 통과 | 린은 이렇게 생겼어요. 띠어럼 뒤에 정리 이름과 변수, 콜론 뒤가 증명할 문장이에요. 바이 다음 줄부터가 증명인데, 전술이라는 명령을 한 줄씩 써요. 줄마다 남은 목표가 바뀌고, 목표가 하나도 안 남으면 통과예요. |
| 3 | ① AI가 읽기: 코드 · 수학 글 3,000억 토큰 / 사람이 쓴 Lean 증명 한 걸음 30만 개 따라 하기 | AI는 코드와 수학 글을 삼천억 토큰 읽어요. 다음 단어 맞히기죠. 그다음 수학자들이 린으로 쓴 증명을 한 걸음씩 따라 해요. 이런 한 걸음이 삼십만 개예요. |
| 4 | ② 사람이 옮긴 Lean 문제는 3,500개뿐 / AI가 100만 개를 옮겨 Lean 문제 8천만 개 | 그런데 사람이 린으로 옮긴 문제는 삼천오백 개뿐이에요. 그래서 말로 쓴 수학 문제 백만 개를 AI가 린 문제로 옮겼어요. 그렇게 모은 린 문제가 팔천만 개예요. |
| 5 | 잘못 옮긴 문제도 버리지 않는다 / 틀렸음을 보이는 연습 문제 | 잘못 옮긴 문제는 버릴까요? 아니에요. 조건 하나를 빠뜨려도 그 자체로 멀쩡한 린 문제라서, 틀렸다고 보이는 연습이 돼요. |
| 6 | ③ AI가 써 보기 → Lean 검사 → 통과만 배움 / TPU 4,000개 × 20일 | 이제 강화학습이에요. AI가 팔천만 문제마다 증명이나 반증을 써 보고, 린이 통과시킨 것만 다시 배워요. 구글 AI 칩 사천 개로 이십 일 동안, 풀린 문제가 점점 늘어나요. |
| 7 | ④ 시험 중 연습: 닮은 문제 수십만 개 / 쉽게 바꾸기 · 일부만 떼어 내기 | 그래도 못 푼 어려운 문제는, 시험 중에 또 연습해요. 일 번 문제라면 알파가 정수일 때만, 엔이 이일 때만처럼 쉽게 바꾸거나 일부만 떼어 내서, 닮은 문제를 수십만 개 만들어요. 이걸로 다시 강화학습을 하면 원래 문제가 풀려요. |
| 8 | 기출 정답률 33% → 44% → 58% / 대회 1 · 2 · 6번: 문제마다 2~3일 연습 | 올림피아드 기출 문제로 보면, 문제당 이 분 찾으면 삼십삼 퍼센트, 열두 시간 찾아도 사십사 퍼센트인데, 닮은 문제로 연습하면 오십팔 퍼센트까지 올라요. 실제 대회의 일 번, 이 번, 육 번도 문제마다 이삼 일씩 이렇게 연습해서 풀었어요. |
| 9 | 읽기 → 문제 만들기 → Lean 채점 강화학습 / 어려우면 닮은 문제로 시험 중 연습 | 수학 AI는 읽고 따라 한 다음, 스스로 만든 문제를 린 채점으로 풀며 배웠어요. 어려운 문제 앞에서는 닮은 문제를 수십만 개 만들어 연습했고요. |

## 출처

- 슬라이드: 없음 (쇼츠 전용). 사용자: 리만가설과Lean 편 다음 편으로 '수학 AI 학습법' 선택 (2026-11-23)
- 실험 없음. 숫자는 Hubert et al., "Olympiad-level formal mathematical reasoning with reinforcement learning", Nature (2025-11-12), doi 10.1038/s41586-025-09833-y — 전문은 Europe PMC PMC12999475 (fullTextXML), 보충자료 41586_2025_9833_MOESM1_ESM.pdf
  - 사전학습 'approximately 300 billion tokens of publicly available code and mathematical texts', SFT 'approximately 300,000 state–tactic pairs ... from the human-authored proofs within the Mathlib library'
  - 자동 형식화 'approximately 1 million natural-language problems into a dataset of around 80 million formal Lean problems', 'each auto-formalized statement, regardless of its fidelity ... provides a valid formal problem that AlphaProof can attempt to prove or disprove', 사람이 옮긴 문제 'approximately 3,500'
  - main RL 'approximately 80,000 TPU days (... 4,000 TPUs utilized for 20 days)', 'Lean-verified outcomes—whether a proof, a disproof or a timeout—provide grounded feedback', 'proportion of problems successfully proved or disproved steadily increases'
  - TTRL: 변형 'simplification, generalization, lemma proposal, exploring analogies', 'hundreds of thousands of unique, syntactically valid Lean variants for each target', IMO 2024 P1 · P2 · P6 'Each of these solutions required 2–3 days of TTRL'
  - Table 1 formal-imo: 2 TPU minutes 33.2% · 12 TPU hours 43.7% · TTRL 50 TPU days 53.9% · TTRL 500 TPU days 58.3%
  - P6 'fully solved by only 5 of the 609 student contestants' (Extended Data Fig. 9). 대회 문제는 전문가가 Lean 으로 옮김
  - 42점 중 28점 (AlphaGeometry 2 의 P4 포함): Google DeepMind 블로그 2024-07-25
- 설명용 예시: 609개 점 중 금색 5개 위치, a + b ≥ 2√(ab) 번역과 조건 빠뜨린 번역 (a = b = −1 반례), Mathlib 한 걸음 예 (nlinarith), 타일 격자 색, 1번 문제 변형 4개 (논문 Extended Data Fig. 5 에 실제 P1 변형이 있으나 화면 문구는 직접 쓴 예시)
- 이모지: Twemoji (CC BY 4.0) 🥈 · 음성: Typecast (필재) · 배경음: Suno "Curious Groove"

### v2 (2026-11-23, 사용자: '6번을 여섯 번으로 읽음 · 푼 문제가 뭔지 살짝 · Lean 문법 기본')
- IMO 2024 P6 문장 (유리수 함수, 'aquaesulian'): f(x + f(y)) = f(x) + y 또는 f(f(x) + y) = x + f(y), f(r) + f(−r) 값의 가짓수 최대 c, 최소 c = 2. 논문 Extended Data Fig. 9 의 핵심 줄 f(x) = −x + 2⌈x⌉
- Lean 예시 (설명용): theorem two_mul_le (a b : ℝ) : 2*a*b ≤ a^2 + b^2 := by / have h := sq_nonneg (a - b) / nlinarith [h], 남은 목표 표시

## 파일

- 영상: `~/book-ml/videos/주제별/수학AI는어떻게배웠나/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
