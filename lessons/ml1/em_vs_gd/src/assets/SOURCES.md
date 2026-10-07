# 출처
- 데이터: ml1/em_algorithm 의 예시 키 명단 24명(손으로 만든 키). src/data.py 가 그 data.py 를 경로로 불러옴
- 단순화: 초6 무리 평균 150·표준편차 5 고정, 중3 무리 표준편차 5, 비율 반반 → 중3 평균 μ 하나만 학습(시작 152cm)
- 계산(assert): 점수 꼭대기 166.4cm · 경사상승 학습률 0.3 → 8걸음 161.5 · 학습률 6 → 185.1로 튄 뒤 162~172 오락가락, 점수 계속 하락 · EM 152 → 161.1 → 164.9 → 166.0, 점수 매번 상승 · EM 한 걸음 = (σ²/N)·dℓ/dμ 를 걸음마다 검증, 자동 학습률 1.65 → 2.0
- 관계식: EM 은 preconditioned 경사상승 (L. Xu & M. I. Jordan, "On Convergence Properties of the EM Algorithm for Gaussian Mixtures", Neural Computation 8(1), 1996)
- 수식 SVG: src/make_figs.py (matplotlib mathtext)
