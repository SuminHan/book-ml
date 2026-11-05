# 문제 정형화와 평가지표 (1주 2차시)

- 쇼츠 없음. 1주 1차시 course_intro 다음, 3차시 data_handling (데이터 직접 다루기) 앞
- 구성: 문제 = 칸 채우기 (한 행 · 입력 X · 출력 y · 평가지표 · 기준선) → 와인 두 문제 → MAE · RMSE → 기준선 (항공 승객: kNN 74 vs 지난달 그대로 44) → 시간 순서로 나누기 → IBM AML (정확도 99.90% 의 함정, 정밀도 · 재현율, 거래 0.10% vs 계좌 0.68%) → 넷플릭스 프라이즈 (RMSE 10% 개선, 숨긴 최근 별점으로 시험, 우승 모델을 그대로 쓰지 않은 이유)
- 상태: 덱 21쪽 (표지 + 20) · 실습: 와인 · 항공 승객 · IBM AML (Hugging Face 사본, 로그인 없이) · 넷플릭스 프라이즈 축소판 (MovieLens)
- 숫자: `exp_formulation.py` → `assets/formulation.json`, `exp_aml.py` → `assets/aml.json` (덱 머리말 \aml... 와 `data.py` assert 로 맞춘다)
- 관련: 데이터 가게 smhanlab.com/book-ml/datashop/ (같은 '문제 정하기' 를 산업 데이터 32개로)
