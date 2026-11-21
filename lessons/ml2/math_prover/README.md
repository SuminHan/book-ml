<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# 리만가설과Lean

*Math AI and the proof checker Lean*

> 리만 가설 증명, AI가 찾았다는데 맞는지는 누가 확인할까? 수학 AI와 증명 검사기 Lean

## 영상 내용

- 2026년 11월 20일, 앤트로픽의 수학자 레벤트 알포게(Levent Alpöge)는 Claude와 함께 리만 가설의 증명을 찾은 것 같다고 발표했어요. 본인도 아직 증명을 다 이해하지 못했다고 밝혔어요
- 리만 가설은 제타 함수의 (자명하지 않은) 영점이 모두 실수부 1/2 위에 있다는 1859년의 추측이에요. 컴퓨터로 처음 10조 개를 확인했지만, 그건 증명이 아니에요
- 계산 문제는 답을 정답과 맞춰 보면 되지만, 증명엔 맞춰 볼 정답이 없어요. 그래서 증명을 코드로 쓰고, 한 줄 한 줄이 규칙에 맞는지 컴퓨터가 검사하는 Lean을 써요. 한 줄만 틀려도 통과하지 못해요
- 구글 딥마인드의 AlphaProof는 수학 문제 약 100만 개를 Lean 문장으로 옮기고, 증명을 시도해서 Lean이 통과시킨 증명만 다시 배웠어요 (AlphaZero 방식 강화학습)
- 그 결과 2024 국제수학올림피아드에서 42점 중 28점, 은메달 수준(금 29점)이었어요. 다만 사람은 4시간 반씩 두 번, AI는 문제에 따라 최대 사흘이 걸렸고 문제는 사람이 Lean으로 옮겨 줬어요
- 이번 리만 가설 증명도 Lean 코드(보도 기준 약 17만 5천 개 파일)로 공개됐고, MIT의 앤드루 서덜랜드가 따로 컴파일해 표준 공리 3개만으로 검사를 통과했다고 밝혔어요
- 남은 질문: Lean은 적힌 문장을 증명했는지만 봐요. 그 문장이 진짜 리만 가설과 같은지는 전문가들이 확인하고 있고, 아직 공식적으로 풀린 것으로 인정된 상태는 아니에요 (2026년 11월 21일 기준)

**참고**: 출처: AFP (phys.org, 2026.11.20), Scientific American (2026.11.20), Interesting Engineering (파일 수 · 용량 보도), Google DeepMind 블로그 (2024.7.25), Hubert et al., Nature (2025.11.12) AlphaProof 논문, 영점 계산 확인은 Gourdon (2004). 화면의 Lean 코드, 증명 카드 8장 중 2장 통과, '처음 10조 개만' 문장은 설명용 예시예요. 이모지: Twemoji (CC BY 4.0) · 음성: Typecast (필재) · 배경음: Suno "Pulse of News" (직접 제작)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 알포게: Claude와 함께 리만 가설 증명 / 이 증명, 누가 확인할까? | 11월 20일, 앤트로픽의 레벤트 알포게가 클로드와 함께 리만 가설을 증명한 것 같다고 발표했어요. 본인도 아직 다 이해하지 못했대요. 그럼 이 증명, 누가 확인할까요? |
| 2 | 영점은 모두 실수부 1/2 위에? / 10조 개 확인 ≠ 증명 | 리만 가설은 제타 함수의 영점이 모두 이 한 줄, 실수부 이분의 일 위에 있다는 주장이에요. 컴퓨터로 십조 개를 확인했지만, 그건 증명이 아니에요. |
| 3 | 계산 문제: 답 = 정답? / 증명: 맞춰 볼 정답이 없음 | 계산 문제는 답을 정답과 맞춰 보면 되지만, 증명엔 맞춰 볼 정답이 없어요. 사람이 한 줄씩 읽어야 해서 몇 년이 걸리기도 해요. |
| 4 | Lean 증명 검사기 / 한 줄이라도 틀리면 통과 X | 그래서 린을 써요. 증명을 코드로 쓰면, 한 줄 한 줄이 규칙에 맞는지 컴퓨터가 검사해요. 한 줄만 틀려도 통과 못 해요. |
| 5 | AlphaProof: 통과한 증명만 다시 학습 / 알파고(AlphaZero) 방식 | 딥마인드의 알파프루프는 이 검사기로 연습했어요. 문제 약 백만 개를 린으로 옮기고, 증명을 시도해서, 통과한 증명만 다시 배워요. 알파고처럼 혼자 연습한 거죠. |
| 6 | IMO 2024: 42점 중 28점 (은메달 수준) / 사람 4.5시간 × 2 · AI 최대 3일 | 그 결과 이천이십사 년 국제수학올림피아드에서 사십이 점 중 이십팔 점, 은메달 수준을 받았어요. 다만 사람은 네 시간 반씩 두 번, AI는 최대 사흘이 걸렸어요. |
| 7 | Lean 파일 약 17만 개 / MIT 서덜랜드: 검사 통과 | 이번 리만 가설 증명도 린 코드, 파일 약 십칠만 개로 공개됐어요. 엠아이티의 서덜랜드가 따로 돌려 보니, 검사를 통과했대요. |
| 8 | Lean 속 문장 = 진짜 리만 가설? / 전문가들이 확인 중 | 그런데 린은 적힌 문장을 증명했는지만 봐요. 문장을 살짝 바꾸면 더 쉬운 문제를 푼 셈이죠. 그래서 그 문장이 진짜 리만 가설과 같은지, 지금 전문가들이 따져 보고 있어요. |
| 9 | 수학 AI와 Lean 한 장 정리 | 수학 AI는 증명 검사기를 통과했는지로 배워요. 알파프루프는 그렇게 은메달 수준까지 왔고, 리만 가설 증명은 사람의 확인을 기다리고 있어요. |

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

## 파일

- 영상: `~/book-ml/videos/주제별/리만가설과Lean/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
