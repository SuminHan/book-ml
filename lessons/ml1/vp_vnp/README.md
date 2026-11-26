<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# VP와VNP

*Valiant's Conjecture · VP ≠ VNP*

> 부호 하나 차이가 47년 난제가 된 이유, VP ≠ VNP(발리언트 추측): 2026년 11월 23일 Anthropic과 OpenAI가 각각 AI가 이 문제를 증명했다고 발표했어요. 아직 사람 수학자 확인 전이에요.

## 영상 내용

- 행렬식은 ad − bc, 퍼머넌트는 ad + bc. 빼기 하나를 더하기로 바꿨을 뿐
- 50×50 행렬이면 행렬식은 약 12만 번, 퍼머넌트는 알려진 방법(1963 라이저)으로도 약 5.6경 번
- 못 찾은 것과 없는 것은 달라요. 없다고 하려면 앞으로 나올 모든 방법까지 이겨야 해서 1979년부터 아무도 못 함
- 사람들이 쓰던 흔한 무기(어려운 문제만 가진 특징 찾기)는 쓰면 암호까지 깨지는 무기라 안 통한다는 게 1994년에 증명됨 (라즈보로프 · 루디치, natural proofs)
- 사람들은 25년 동안 아주 어려운 수학(기하학적 복잡도 이론)을 팠는데, 이번 증명은 1891년 칸토어의 대각선 논법 같은 옛 기술로 다른 길로 감
- 컴퓨터 검사기(Lean)는 통과, 사람 수학자 확인은 이제 시작. P vs NP의 사촌 문제이고 P vs NP 자체는 아직

**참고**: 체스 → 스타크래프트는 이해를 돕는 비유예요. 2×2 숫자는 예시, 계산 횟수는 상수를 뺀 대략값이에요. 출처: github.com/anthropics/formal-math (vp-vnp), github.com/openai/math, Scott Aaronson 블로그 Shtetl-Optimized (2026.11.24), Lance Fortnow Computational Complexity 블로그 (2026.11), Valiant (1979), Razborov · Rudich (1994). 이모지: Twemoji (CC BY 4.0) · 음성: Typecast (필재) · 배경음: Suno

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 부호 하나 차이, 47년 난제 / AI가 증명했다고 발표 | 빼기 하나를 더하기로 바꿨을 뿐인데, 사십칠 년 동안 아무도 못 푼 문제가 있어요. 앤트로픽과 오픈에이아이의 AI가 이걸 증명했다고 발표했어요. |
| 2 | 행렬식: ad − bc / 퍼머넌트: ad + bc | 왼쪽은 행렬식, 곱을 빼요. 오른쪽은 퍼머넌트, 전부 더해요. |
| 3 | 50×50: 행렬식 약 12만 번 / 퍼머넌트 약 5.6경 번 | 오십 칸 행렬이면, 행렬식은 약 십이만 번이면 끝나는데 퍼머넌트는 약 오 점 육 경 번이 걸려요. |
| 4 | 못 찾음 ≠ 없음 / 없다 = 모든 방법이 느리다 | 더 빠른 방법이 정말 없을까요? 못 찾은 것과 없는 것은 달라요. 없다고 하려면 앞으로 나올 모든 방법까지 이겨야 해요. 1979년부터 아무도 못 했어요. |
| 5 | 흔한 무기 = 쓰면 암호도 깨짐 / 1994년: 이걸로는 안 된다 | 사람들이 쓰던 흔한 무기는, 쓰면 암호까지 깨져 버리는 무기라서 안 통한다는 게 1994년에 증명됐어요. |
| 6 | 사람: 25년 어려운 수학 / AI: 백 년 넘은 옛 기술 | 그래서 사람들은 이십오 년 동안 아주 어려운 수학을 팠는데, AI는 백 년도 넘은 옛날 기술을 꺼내 와 다른 길로 갔어요. |
| 7 | 체스 → 스타크래프트 / 정답 있는 문제 → 길 모르는 문제 | 비유하면, 체스만 두던 AI가 갑자기 스타크래프트를 잘하게 된 느낌이에요. 정답이 있는 문제를 풀던 AI가, 아무도 길을 모르는 문제에서 새 길을 찾은 거예요. |
| 8 | 검사기 통과 · 사람 확인 전 / P vs NP의 사촌 문제 | 컴퓨터 검사기는 통과했지만, 사람 수학자의 확인은 이제 시작이에요. 피 대 엔피도 아직이고요, 이건 그 사촌 문제예요. |
| 9 | 부호 하나 차이, 47년 난제 / 검증은 이제 시작 | 부호 하나 차이가 계산을 이렇게 바꿔요. 없다는 증명은 어렵고, AI는 옛 기술로 새 길을 냈어요. |

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
