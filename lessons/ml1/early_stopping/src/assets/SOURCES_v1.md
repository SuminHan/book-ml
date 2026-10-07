# 출처
- 내용: ~/book-ml/slides/kor/ml1/week07/pages/p42.tex (검증 성능이 나빠지기 시작하는 시점에서 early stopping), p51.tex (early stopping 필수, 검증 곡선 정점), p59–p60.tex (n_estimators 를 early stopping 으로 결정)
- 속도–제동거리 데이터: 설명용 예시. d = v²/(2μg), μ=0.7 + 측정 잡음(표준편차 4m, seed 4), 학습 12개 · 검증 12개
- 모델: sklearn GradientBoostingRegressor(깊이 2, learning_rate 0.1, 200번), staged_predict 로 매 학습 횟수의 오차(평균 절대 오차) 계산. src/data.py 에서 최저 23번째 2.4m, 200번째 학습 0.0m · 검증 4.4m, patience 10 → 33번째 멈춤 assert
- 이모지(🔒 🛑 ⏱️): Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
