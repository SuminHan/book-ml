# 출처
- 내용: ~/book-ml/slides/kor/ml1/week04/pages/p60.tex (닭이 먼저냐 달걀이 먼저냐: 잠재변수 순환), p65.tex (E-step 책임값), p66.tex (M-step 가중 평균), p67.tex (em_step 코드 1차원), p69.tex (우도가 매 반복 줄지 않음), p71.tex (손으로 한 라운드, 정중앙 점은 반반)
- 데이터: **예시**(손으로 만든 키, cm). 초6 12명 [142…159] + 중3 12명 [158…176], 실제 통계 아님
- 계산: 1차원 GMM(K=2) EM, 초기 평균 150·155, 표준편차 6·6, 비율 0.5·0.5, 40번 반복. src/data.py assert: 처음 E 단계 153cm → 중3 52%, M 1번 → 152·162, 40번 → 150·165 (진짜 평균 151·167), 로그 우도 단조 증가, γ>0.5 로 나누면 24명 중 22명 일치, 157cm(실제 초6) → 중3 60%
- 이모지: Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
