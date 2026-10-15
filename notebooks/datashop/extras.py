"""데이터 페이지의 평가지표 수식 (KaTeX) 과 관련 데이터 링크. 링크는 2026-10-15 에 열리는지 확인했다."""
from urllib.parse import quote

F = {
    "acc": ("정확도 (accuracy)", r"\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\hat y_i = y_i]", "맞힌 비율. 한쪽이 아주 드물면 '전부 흔한 쪽' 만으로도 높게 나온다"),
    "prec": ("정밀도 (precision)", r"\text{Precision} = \frac{TP}{TP + FP}", "'양성' 이라고 한 것 중 진짜 양성의 비율 (헛짚지 않았나)"),
    "rec": ("재현율 (recall)", r"\text{Recall} = \frac{TP}{TP + FN}", "진짜 양성 중 잡아낸 비율 (놓치지 않았나)"),
    "auc": ("ROC-AUC", r"\text{AUC} = P\big(s(x^{+}) > s(x^{-})\big)", "양성 하나 · 음성 하나를 뽑았을 때 양성 점수가 더 높을 확률. 0.5 = 동전 던지기"),
    "mae": ("MAE", r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat y_i \rvert", "평균적으로 얼마나 틀렸나 (y 와 같은 단위)"),
    "rmse": ("RMSE", r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2}", "크게 틀린 것에 더 큰 벌을 준다"),
    "same5": ("추천 5개 중 같은 종류 비율", r"\frac{1}{5}\sum_{j=1}^{5} \mathbb{1}[\text{type}(r_j) = \text{type}(q)]", "q = 고른 상품, r_j = 추천한 j번째 상품"),
    "logloss": ("로그 손실 (log loss)", r"-\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]", "확률을 얼마나 잘 맞혔나 (작을수록 좋다)"),
    "ade": ("평균 위치 오차 (ADE)", r"\text{ADE} = \frac{1}{T}\sum_{t=1}^{T} \lVert \hat{\mathbf p}_t - \mathbf p_t \rVert_2", "앞으로 T 걸음 동안 예측 위치와 실제 위치의 거리 평균 (미터)"),
    "success": ("성공률", r"\text{Success} = \frac{\text{성공한 시도}}{\text{전체 시도}}", "로봇이 실제로 블록을 자리에 넣은 비율"),
}

METRICS = {
    "fashion_commerce": ["same5"], "shop_conversion": ["prec", "rec", "acc"], "movie_ratings": ["rmse", "mae"], "korean_reviews": ["acc"],
    "card_fraud": ["prec", "rec", "acc"], "aml_transactions": ["prec", "rec", "acc"], "stock_index": ["acc", "mae"], "support_routing": ["acc"],
    "hotel_booking": ["prec", "rec", "mae"], "link_prediction": ["auc"], "bike_demand": ["mae"], "real_estate": ["mae"],
    "earthquakes": ["prec", "rec", "auc"], "semiconductor_yield": ["prec", "rec", "auc"], "predictive_maintenance": ["prec", "rec"],
    "steel_defects": ["acc", "rec"], "bearing_vibration": ["acc"], "form_understanding": ["acc", "rec"], "ai_usecases": [],
    "point_cloud": ["acc"], "driving": ["ade"], "robot": ["success", "mae"], "medical": ["rec", "auc"], "satellite": ["acc"],
    "air_quality": ["mae", "rec"], "electricity": ["mae", "rmse"], "football": ["logloss"], "chess": ["acc", "logloss"], "speech": ["acc"],
    "drug": ["auc"], "protein": [], "crop": ["acc", "rec"],
}

# 관련 데이터 (이름, 주소, 한 줄) + 영어 검색어 (Hugging Face 데이터셋 검색)
RELATED = {
    "fashion_commerce": ([("H&M Personalized Fashion Recommendations (Kaggle)", "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations", "고객 · 구매 기록이 있는 진짜 추천 문제 (로그인 필요)"),
                          ("DeepFashion", "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html", "옷 사진 80만 장, 속성 · 위치 표시"),
                          ("Fashion-MNIST", "https://github.com/zalandoresearch/fashion-mnist", "28×28 흑백 옷 사진 7만 장, 10종류")], "fashion"),
    "shop_conversion": ([("UCI Online Retail II", "https://archive.ics.uci.edu/dataset/502/online+retail+ii", "온라인 쇼핑몰 2년치 주문, 고객생애가치 문제"),
                         ("Amazon Reviews 2023", "https://amazon-reviews-2023.github.io/", "분야별 상품 리뷰 · 별점 (Beauty 포함)")], "ecommerce"),
    "movie_ratings": ([("MovieLens (전체 크기들)", "https://grouplens.org/datasets/movielens/", "10만 · 100만 · 2천500만 별점 판"),
                       ("Netflix Prize (Kaggle)", "https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data", "2006년 넷플릭스 추천 대회 데이터 (로그인 필요)")], "movie recommendation"),
    "korean_reviews": ([("Hugging Face e9t/nsmc", "https://huggingface.co/datasets/e9t/nsmc", "같은 데이터의 Hugging Face 판"),
                        ("KLUE", "https://klue-benchmark.com/", "한국어 이해 벤치마크 (뉴스 주제 분류 · 문장 관계 등 8가지)")], "korean sentiment"),
    "card_fraud": ([("PaySim (Kaggle)", "https://www.kaggle.com/datasets/ealaxi/paysim1", "모바일 송금 사기 시뮬레이션 (보이스피싱과 비슷한 송금 사기 모양)"),
                    ("IEEE-CIS Fraud Detection (Kaggle)", "https://www.kaggle.com/competitions/ieee-fraud-detection", "온라인 결제 사기 대회, 열 400개 이상")], "fraud detection"),
    "aml_transactions": ([("IBM AML 원본 (Kaggle)", "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml", "HI · LI × Small · Medium · Large 6가지 크기"),
                          ("Elliptic (Kaggle)", "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set", "비트코인 거래 그래프의 불법 거래 표시")], "anti money laundering"),
    "stock_index": ([("FRED (미국 연준 경제 데이터)", "https://fred.stlouisfed.org/", "지수 · 금리 · 환율 시계열"),
                     ("KRX 정보데이터시스템", "http://data.krx.co.kr/", "한국거래소 지수 · 종목 통계")], "stock time series"),
    "support_routing": ([("CLINC150 (Hugging Face)", "https://huggingface.co/datasets/clinc/clinc_oos", "의도 150가지 + '모르는 질문' 구분"),
                         ("Bitext 고객 상담 (Hugging Face)", "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset", "질문 · 상담원 답변 쌍, 의도 27가지")], "intent classification"),
    "hotel_booking": ([("Inside Airbnb", "https://insideairbnb.com/get-the-data/", "도시별 숙소 · 날짜별 가격 달력"),
                       ("Expedia Hotel Recommendations (Kaggle)", "https://www.kaggle.com/competitions/expedia-hotel-recommendations", "검색 → 예약할 호텔 묶음 맞히기")], "hotel booking"),
    "link_prediction": ([("SNAP 데이터 모음", "https://snap.stanford.edu/data/", "SNS · 인용 · 도로 등 연결망 수백 개"),
                         ("OGB 링크 예측 (ogbl)", "https://ogb.stanford.edu/docs/linkprop/", "연구 공동 저자 · 약물 상호작용 · 인용 연결 예측 벤치마크")], "link prediction"),
    "bike_demand": ([("서울 열린데이터광장", "https://data.seoul.go.kr/", "따릉이 대여소별 대여 이력 (대여소 × 시간)"),
                     ("Citi Bike 이용 기록 (뉴욕)", "https://citibikenyc.com/system-data", "달마다 공개되는 이용 기록")], "bike sharing"),
    "real_estate": ([("국토교통부 실거래가 공개시스템", "https://rt.molit.go.kr/", "한국 아파트 · 주택 실거래 가격"),
                     ("House Prices (Ames, Kaggle)", "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques", "집 한 채마다 79가지 특성 → 가격")], "house prices"),
    "earthquakes": ([("USGS 지진 검색", "https://earthquake.usgs.gov/earthquakes/search/", "기간 · 지역 · 규모를 골라 직접 받기"),
                     ("STEAD 지진파 데이터", "https://github.com/smousavi05/STEAD", "지진파 파형 120만 개 (지진 · 잡음 구분, 조기 경보 연구)")], "earthquake"),
    "semiconductor_yield": ([("WM-811K 웨이퍼 맵 (Kaggle)", "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map", "웨이퍼 불량 무늬 사진 81만 장, 9종류"),
                             ("Bosch Production Line (Kaggle)", "https://www.kaggle.com/competitions/bosch-production-line-performance", "생산 라인 측정값 · 불량, 아주 드문 불량")], "semiconductor manufacturing"),
    "predictive_maintenance": ([("NASA PCoE 데이터 저장소", "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/", "터보팬 엔진 수명 (C-MAPSS) · 베어링 · 배터리 열화 데이터")], "predictive maintenance"),
    "steel_defects": ([("Severstal Steel Defect (Kaggle)", "https://www.kaggle.com/competitions/severstal-steel-defect-detection", "강판 사진에서 결함 위치 찾기"),
                       ("MVTec AD", "https://www.mvtec.com/company/research/datasets/mvtec-ad", "공장 부품 15종의 이상 (결함) 사진")], "surface defect"),
    "bearing_vibration": ([("NASA PCoE (IMS 베어링 수명 시험)", "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/", "베어링이 망가질 때까지 계속 잰 진동")], "bearing vibration"),
    "form_understanding": ([("CORD 영수증 (Hugging Face)", "https://huggingface.co/datasets/naver-clova-ix/cord-v2", "영수증 사진 → 항목 · 금액"),
                            ("CUAD 계약서 (Hugging Face)", "https://huggingface.co/datasets/theatticusproject/cuad-qa", "계약서 510건, 조항 41종"),
                            ("DocVQA", "https://www.docvqa.org/", "문서 사진에 대한 질문 · 답")], "document understanding"),
    "ai_usecases": ([("O*NET OnLine", "https://www.onetonline.org/", "미국 직업 · 업무 설명 데이터베이스"),
                     ("AI Incident Database", "https://incidentdatabase.ai/", "실제 AI 사고 사례 모음 (거버넌스)"),
                     ("Stanford AI Index", "https://hai.stanford.edu/ai-index", "해마다 AI 연구 · 산업 통계")], "ai usage"),
}

SEARCH = {"point_cloud": "point cloud", "driving": "autonomous driving", "robot": "robot manipulation", "medical": "chest x-ray",
          "satellite": "satellite land cover", "air_quality": "air quality", "electricity": "electricity load", "football": "soccer events",
          "chess": "chess games", "speech": "speech commands", "drug": "molecule toxicity", "protein": "protein structure", "crop": "plant disease"}


def hf_search(slug):
    q = RELATED[slug][1] if slug in RELATED else SEARCH.get(slug, slug)
    return f"https://huggingface.co/datasets?search={quote(q)}"
