# 출처
- 내용: ~/book-ml/slides/kor/ml1/week07/pages/p42.tex (검증 성능이 나빠지기 시작하는 시점에서 early stopping), p51.tex (early stopping 필수, 검증 곡선 정점), p59–p60.tex (n_estimators 를 early stopping 으로 결정)
- 데이터: scikit-learn `load_digits` (UCI Optical Recognition of Handwritten Digits, 8×8 그림 1797장). 학습 500장 · 검증 500장 (seed 0). 학습 답의 20%(100장)를 일부러 다른 숫자로 잘못 적음(라벨 잡음), 검증 정답은 그대로
- 모델: sklearn MLPClassifier(은닉 256, adam lr 0.001, random_state 0), partial_fit 으로 300번. 결과는 assets/digits_run2.npz 캐시(검증 예측 PRED_VA 포함), src/data.py assert: 검증 최고 31번째 92%, 300번째 학습 97% · 검증 83%, 틀린 정답 외움 2% → 86%, patience 10 → 41번째 멈춤
- 화면 예시 그림(v4): 숨긴 검증 그림 VA[325·115·212·483] = 실제 1·0·6·4 → 31번째 모두 맞힘, 300번째 2·8·5·1. 31번째엔 맞혔는데 300번째엔 틀린 검증 그림 59장. 라벨 잡음은 화면에 안 보이고 캡션에만 밝힘
- v3(실제 정답/잘못 적힌 답 표): src/shorts/lines_v3.json · shorts_v3.py, 영상 _v3
- 이모지(🔒 🛑): Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
- v2(화면 "정답표" 표기판): src/shorts/lines_v2.json · shorts_v2.py, 영상 _v2
- v1(제동거리 + GBDT 판): src/data_v1.py · src/shorts/lines_v1.json · shorts_v1.py · assets/SOURCES_v1.md, 영상 _v1
