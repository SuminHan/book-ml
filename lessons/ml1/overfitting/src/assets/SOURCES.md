# 출처
- 내용: ~/book-ml/slides/kor/ml1/week01/pages/p74–p76.tex (과대적합: 훈련 R²=1 vs 테스트 마이너스, 직선=과소적합), week04/pages/p11.tex (과적합·과소적합), p34b.tex·p34bg1.tex (k-겹 교차검증, 5 fold)
- 속도–제동거리 데이터: 설명용 예시. 진짜 규칙 = 등가속 제동의 물리 법칙 d = v²/(2μg), μ=0.7(마른 아스팔트 가정) + 측정 잡음(표준편차 3m, seed 21). 모델은 1·2·9차 다항식, 오차 = 평균 절대 오차. src/data.py 에서 측정 5/2/0 · 새 데이터 6/3/9 · 5겹 교차 검증 7/3/121 m, 속도 2배 → 4배 assert
- v2(음수·오차 계산 없는 제동거리판)는 src/shorts/lines_v2.json · shorts_v2.py, v1(공부 시간–점수 예시)은 src/data_study.py · src/shorts_study/ 에 보존
- 이모지(🚗 🔒): Twemoji (jdecked/twemoji, 그래픽 CC BY 4.0) — lessons/_tools/assets/emoji/ 캐시, mbase.emoji()
