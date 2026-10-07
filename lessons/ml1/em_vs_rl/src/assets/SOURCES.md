# 출처
- EM 곡선: ml1/em_translation (번역편 v3) IBM Model 1 의 로그 우도, 50번 반복 (-53 → -35.5, 한 번도 안 떨어짐). src/data.py 가 번역편 data.py 를 경로로 불러와 계산
- 강화학습 곡선: **예시** 사탕 기계 5대(평균 2·4·6·3·5개, 잡음 표준편차 2), softmax 정책 + REINFORCE(평균 기준선, 학습률 0.15), 라운드마다 20번, 40라운드, seed 0. assert: 평균 사탕 4.2 → 6.6, 39번 중 20번 하락, 6개 기계 선택 확률 84%
- EM 으로 강화학습: P. Dayan & G. E. Hinton, "Using Expectation-Maximization for Reinforcement Learning", Neural Computation 9(2), 1997 · A. Abdolmaleki et al. (DeepMind), "Maximum a Posteriori Policy Optimisation (MPO)", ICLR 2018
- EM 일반: week04 p60·p65–p66·p69 (우도가 매 반복 줄지 않음)
- 이모지: Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0)
