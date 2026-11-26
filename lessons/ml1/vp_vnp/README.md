<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# VP와VNP

*Valiant's Conjecture · VP ≠ VNP*

> 부호 하나 차이가 47년 난제가 된 이유, VP ≠ VNP(발리언트 추측): 2026년 11월 23일 Anthropic이 Claude가 쓴 증명을, 몇 시간 뒤 OpenAI도 같은 결론의 증명을 공개했어요. 왜 사람이 증명하기 어려웠고, 이번 증명은 무엇이 달랐는지 따라가 봤어요.

## 영상 내용

- 행렬식은 ad − bc, 퍼머넌트는 ad + bc. 50×50 행렬이면 행렬식은 약 12만 번, 퍼머넌트는 알려진 방법(1963 라이저)으로도 약 5.6경 번
- 1979 발리언트 추측: 퍼머넌트는 작은 계산 회로로는 안 된다 = VP ≠ VNP, P vs NP의 대수 버전
- 사람이 증명한 하한은 1983년 n² log n 근처에서 40년 넘게 제자리, 이번 주장은 어떤 다항식보다도 큰 하한
- 왜 어려울까: 1994 라즈보로프·루디치, 어려운 함수만 가진 성질을 검사하는 '자연스러운 증명'은 암호를 깨지 않는 한 통하지 않음
- 사람들은 25년 동안 대수기하(GCT)의 길을 팠는데, 이번 증명은 대수기하를 거의 쓰지 않고 압축 + 대각선 논법으로 끝냄. 아론슨은 이 조합으로 벽을 피한 것 같다고 봄
- 아직 확정 아님: Lean 72,932줄이 검사기 두 개를 통과했지만 사람 수학자 검증 전, P vs NP 자체를 푼 것도 아님

**참고**: 체스 → 스타크래프트는 이해를 돕는 비유예요. 2×2 숫자와 대각선 표는 예시, 계산 횟수는 상수를 뺀 대략값이에요. 출처: github.com/anthropics/formal-math (vp-vnp), github.com/openai/math, Scott Aaronson 블로그 Shtetl-Optimized (2026.11.24), Lance Fortnow Computational Complexity 블로그 (2026.11), Valiant (1979), Razborov · Rudich (1994). 이모지: Twemoji (CC BY 4.0) · 음성: Typecast (필재) · 배경음: Suno

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 부호 하나 차이, 47년 난제 / Anthropic · OpenAI 증명 공개 | 부호 하나 바꿨을 뿐인데, 사십칠 년 동안 아무도 못 푼 문제가 있어요. 지난 23일 앤트로픽이 클로드가 쓴 증명을 공개했고, 몇 시간 뒤 오픈에이아이도 같은 결론을 내놨어요. |
| 2 | 행렬식: ad − bc / 퍼머넌트: ad + bc | 행렬식은 곱을 더하고 빼요. 퍼머넌트는 그 빼기를 전부 더하기로 바꾼 거예요. |
| 3 | 50×50: 행렬식 약 12만 번 / 퍼머넌트 약 5.6경 번 | 그런데 오십 칸 행렬이면, 행렬식은 약 십이만 번 계산으로 끝나지만, 퍼머넌트는 알려진 방법으로도 약 오 점 육 경 번이에요. |
| 4 | 1979 발리언트 추측: VP ≠ VNP / P vs NP의 대수 버전 | 1979년 발리언트는 퍼머넌트는 작은 계산 회로로는 절대 안 된다고 추측했어요. 쉬운 쪽이 브이피, 퍼머넌트가 든 쪽이 브이엔피. 피 대 엔피의 대수 버전이에요. |
| 5 | 사람 기록: n² log n (1983) / 이번: 모든 다항식보다 큼 | 사람이 증명한 하한은 1983년 엔 제곱 로그 엔 근처에서 사십 년 넘게 멈춰 있었어요. 이번 증명은 어떤 다항식보다도 빨리 커지는 하한이에요. |
| 6 | 자연스러운 증명의 벽 / 1994 라즈보로프 · 루디치 | 왜 이렇게 어려울까요? 흔한 방법은 어려운 함수만 가진 성질을 찾아 검사하는 건데, 1994년 라즈보로프와 루디치는 이런 자연스러운 증명이 암호를 깨지 않는 한 통하지 않는다고 보였어요. |
| 7 | 사람: 25년 대수기하의 길 / 이번: 압축 + 대각선 논법 | 그래서 사람들은 이십오 년 동안 대수기하라는 무거운 길을 팠어요. 그런데 이번 증명은 대수기하를 거의 안 쓰고, 가장 오래된 기술인 대각선 논법으로 끝내요. |
| 8 | 대각선을 뒤집으면 목록 밖 / 아론슨: 압축과 합쳐 벽을 피한 듯 | 대각선 논법은 목록의 대각선만 뒤집어서 목록에 없는 걸 만드는 방법이에요. 혼자서는 피 대 엔피에 못 쓴다고 알려졌는데, 아론슨은 압축 단계와 합쳐 벽을 피한 것 같다고 봤어요. |
| 9 | 체스 → 스타크래프트 / 정답 있는 문제 → 길 모르는 연구 | 비유하면, 체스만 두던 AI가 갑자기 스타크래프트를 잘하게 된 느낌이에요. 정답이 있는 문제를 풀던 AI가, 아무도 길을 모르는 연구 문제에서 사람들이 덜 가던 길을 골랐어요. |
| 10 | Lean 7만 3천 줄 통과 / 사람 검증 전 · P vs NP는 그대로 | 아직 확정은 아니에요. 린 증명 칠만 삼천 줄은 검사기 두 개를 통과했지만, 사람 수학자는 아직 확인하지 않았어요. 피 대 엔피 자체를 푼 것도 아니고요. |
| 11 | 부호 하나 차이, 47년 난제 / 검증은 이제 시작 | 행렬식과 퍼머넌트는 부호 하나 차이. 흔한 증명은 벽에 막히고, 이번 증명은 압축과 대각선 논법으로 돌아갔어요. |

## 출처

- Anthropic: github.com/anthropics/formal-math/tree/main/vp-vnp (README: "VP ≠ VNP over every field", internal research Claude model, AI 가 논문·Lean 작성, "no human mathematician has checked the proof", Comparator + nanoda 재검사, Statement.lean 은 기계가 뜻을 확인 못 함). 커밋 2026-11-23 20:04 UTC, 병합 21:43 UTC (MACCUS). 논문 70쪽, Lean 72,932줄 · 324파일 (MACCUS 집계), Statement.lean 98줄. 하한 m^(c (log m)^(1/3)) — 상수 c 미정이라 횟수로 바꾸지 않음.
- OpenAI: github.com/openai/math preprints/VP-and-VNP-are-different-November-23-2026, 약 4시간 뒤 병합 (vibemathlab.com/guide/is-vp-vs-vnp-solved), exp(Ω((log n)^(5/4))) border circuits, 저장소에 11월 6일에 얻었다고 적음.
- 증명 방법: Aaronson, Shtetl-Optimized (scottaaronson.blog/?p=10362, 2026-11-24) "highly nontrivial 'compression' or 'shortening' … diagonalization … should let these arguments evade the natural proofs barrier", "from what little I understand so far" → 영상 '벽을 피한 듯', '(본인도 아직 조금밖에 모름)'. Fortnow, Computational Complexity blog (2026/11/vp-vnp): Strassen asymptotic spectrum 변형으로 두 경우 (rank argument / diagonalization), "uses essentially no algebraic geometry" → 영상 '대수기하 거의 없이'.
- 배경: Valiant 1979 (STOC) 퍼머넌트 VNP 완전 · 추측. 사람 하한 기록 약 m² log m (Baur–Strassen 1983, MACCUS). 자연스러운 증명 Razborov–Rudich 1994 (불 회로, 의사난수 함수가 있으면 natural proof 불가). 대수 쪽 비슷한 벽 논의 (Forbes–Shpilka–Volk 2017, Grochow–Kumar–Saks–Saraf 2017). 대각선 논법만으로는 P vs NP 불가 (상대화, Baker–Gill–Solovay 1975). GCT (Mulmuley–Sohoni 2001) → '25년'.
- 비용 숫자 (data.py assert): 50×50 행렬식 약 n³ = 125,000 ('약 12만', 상수 무시), 퍼머넌트 라이저 1963 약 n·2ⁿ = 5.6×10¹⁶ ('약 5.6경'). 2×2 [[1,2],[3,4]], 대각선 표 비트는 예시.
- 체스 → 스타크래프트: 강의에서 들은 비유 (영상에선 '비유하면' 으로만, 누구 말인지 밝히지 않음). AI 학습 데이터·과정은 공개되지 않아 '학습하지 않은 영역' 같은 말은 쓰지 않음.
- 상태 (2026-11-26): 두 증명 모두 사람 검증 전.
- 이모지: Twemoji (CC BY 4.0). 음성: Typecast 필재. 배경음: Suno "Curious Groove".

## 파일

- 영상: `~/book-ml/videos/주제별/VP와VNP/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
