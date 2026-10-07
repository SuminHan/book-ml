<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# Early Stopping 조기 종료

*Early Stopping · 검증 데이터 · Patience*

> Early Stopping(조기 종료)은 숨겨 둔 검증 데이터의 정확도가 가장 높을 때 학습을 멈추는 방법이에요. 오래 학습한다고 좋은 모델이 되지는 않아요.

## 영상 내용

- 손글씨 숫자 그림 500장으로 학습하고, 다른 500장은 학습에 쓰지 않고 숨겨 둬요 = 검증 데이터(Validation Data)
- 학습할수록 정확도가 올라요. 31번째 학습에서 숨긴 그림을 92% 맞혀요
- 300번째까지 계속하면 학습 그림 정확도는 97%. 그런데 31번째엔 맞히던 숨긴 그림을 하나둘 틀리기 시작해요 (1을 2로, 0을 8로 … 59장)
- 검증 정확도는 92%에서 83%로 떨어져요 → 학습 그림만 외운 과대적합(Overfitting)
- 그래서 검증 정확도가 가장 높은 31번째에서 멈춰요 = Early Stopping
- 최고점은 지나 봐야 알아요. 10번 더 학습해도 나아지지 않으면(41번째) 멈추고, 가장 좋았던 31번째 모델로 되돌려요. 기다리는 횟수 = 참을성(Patience)

**참고**: 실제 데이터와 실제 계산: scikit-learn 손글씨 숫자 데이터(8×8, load_digits), 작은 신경망(MLP)을 300번 학습. 과대적합이 잘 보이도록 학습 그림 20%의 답을 일부러 바꿔 둔 실험이에요.

**참고**: 이모지: Twemoji (CC BY 4.0)

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 오래 학습할수록 더 좋을까? / 멈출 때를 알아야 한다 | AI는 오래 학습할수록 좋아질까요? 멈출 때를 알아야 해요. |
| 2 | 손글씨 숫자 500장으로 학습 / 숨긴 500장 = 검증 데이터 | 손글씨 숫자 500장으로 학습하고, 다른 500장은 숨겨 둬요. 이게 검증 데이터. |
| 3 | 학습할수록 정확도가 오른다 / 31번째: 숨긴 그림 92% 맞힘 | 학습할수록 정확도가 올라요. 31번째엔 숨긴 그림을 92퍼센트 맞혀요. |
| 4 | 300번째: 학습 정확도 97% / 그런데 맞히던 그림을 틀린다 | 계속하면 학습 정확도는 97퍼센트. 그런데 맞히던 그림을 틀리기 시작해요. 1을 2로, 0을 8로. |
| 5 | 검증 정확도 92% → 83% / 학습 그림만 외운 과대적합 | 검증 정확도는 92에서 83퍼센트로 떨어져요. 학습 그림만 외운 과대적합. |
| 6 | 검증 정확도가 가장 높을 때 멈춘다 / Early Stopping = 조기 종료 | 그래서 검증 정확도가 가장 높은 31번째에서 멈춰요. 얼리 스토핑, 조기 종료. |
| 7 | 10번 더 안 나아지면 멈추고 / 31번째로 되돌리기 = Patience | 최고점은 지나 봐야 알죠. 10번 더 안 나아지면 멈추고 31번째로 되돌려요. 이게 참을성, 페이션스. |
| 8 | 오래 학습은 답이 아니다 / 최고점에서, 참을성 있게 멈추기 | 오래 학습은 답이 아니다. 최고점에서, 참을성 있게 멈추기. |

## 출처

- 내용: ~/book-ml/slides/kor/ml1/week07/pages/p42.tex (검증 성능이 나빠지기 시작하는 시점에서 early stopping), p51.tex (early stopping 필수, 검증 곡선 정점), p59–p60.tex (n_estimators 를 early stopping 으로 결정)
- 데이터: scikit-learn `load_digits` (UCI Optical Recognition of Handwritten Digits, 8×8 그림 1797장). 학습 500장 · 검증 500장 (seed 0). 학습 답의 20%(100장)를 일부러 다른 숫자로 잘못 적음(라벨 잡음), 검증 정답은 그대로
- 모델: sklearn MLPClassifier(은닉 256, adam lr 0.001, random_state 0), partial_fit 으로 300번. 결과는 assets/digits_run2.npz 캐시(검증 예측 PRED_VA 포함), src/data.py assert: 검증 최고 31번째 92%, 300번째 학습 97% · 검증 83%, 틀린 정답 외움 2% → 86%, patience 10 → 41번째 멈춤
- 화면 예시 그림(v4): 숨긴 검증 그림 VA[325·115·212·483] = 실제 1·0·6·4 → 31번째 모두 맞힘, 300번째 2·8·5·1. 31번째엔 맞혔는데 300번째엔 틀린 검증 그림 59장. 라벨 잡음은 화면에 안 보이고 캡션에만 밝힘
- v3(실제 정답/잘못 적힌 답 표): src/shorts/lines_v3.json · shorts_v3.py, 영상 _v3
- 이모지(🔒 🛑): Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
- v2(화면 "정답표" 표기판): src/shorts/lines_v2.json · shorts_v2.py, 영상 _v2
- v1(제동거리 + GBDT 판): src/data_v1.py · src/shorts/lines_v1.json · shorts_v1.py · assets/SOURCES_v1.md, 영상 _v1

## 파일

- 영상: `~/book-ml/videos/주제별/EarlyStopping조기종료/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
