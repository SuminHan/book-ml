"""ML1 1차시 (강의 소개) 공유 숫자. 슬라이드 · 그림 · 노트가 같은 숫자를 쓴다."""

# 정확도 90%의 함정 (쇼츠 lessons/ml1/precision_recall 과 같은 숫자): 100명 중 환자 10명
N_PEOPLE = 100
N_SICK = 10
# AI가 12명을 환자로 골랐다: 맞힘 8 · 헛짚음 4 · 놓침 2
TP, FP, FN = 8, 4, 2
TN = N_PEOPLE - TP - FP - FN            # 86
ALL_HEALTHY_ACC = (N_PEOPLE - N_SICK) / N_PEOPLE   # '모두 건강'이라고 찍어도 0.90
RECALL = TP / (TP + FN)                 # 8/10 = 0.80
PRECISION = TP / (TP + FP)              # 8/12 = 0.67
assert (TN, TP + FP) == (86, 12)


# 역사 연표: (연도, 사건, 종류)  idea = 아이디어(과학) / engine = 공학(계산·데이터·도구) / boom = 터진 순간
TIMELINE = [
    (1943, "인공 뉴런\n(McCulloch-Pitts)", "idea"),
    (1958, "퍼셉트론", "idea"),
    (1986, "역전파", "idea"),
    (1998, "LeNet (CNN)", "idea"),
    (2005, "크라우드소싱\n(Mechanical Turk)", "engine"),
    (2007, "GPU 프로그래밍\n(CUDA)", "engine"),
    (2009, "ImageNet\n(라벨 140만 장)", "engine"),
    (2012, "AlexNet", "boom"),
    (2016, "AlphaGo vs 이세돌", "boom"),
    (2022, "ChatGPT", "boom"),
    (2024, "노벨 물리학상·화학상", "boom"),
]

# 이번 학기 17주
ROADMAP = [
    (1, "머신러닝이라는\n문제"), (2, "회귀"), (3, "베이즈·GDA"), (4, "kNN·k-means\n·EM"),
    (5, "SVM"), (6, "정규화·\n교차검증"), (7, "트리·GBDT"), (8, "프로젝트 1\n+ 중간"),
    (9, "신경망·\n역전파"), (10, "CNN"), (11, "RNN·LSTM"), (12, "어텐션"),
    (13, "그래프"), (14, "임베딩·RAG"), (15, "VAE·GAN·\nDiffusion"), (16, "프로젝트 2\n+ 총정리"),
    (17, "LLM"),
]

# EM 그림용 1차원 데이터 (두 무리)
EM_SEED = 3
EM_GROUPS = [(-2.0, 0.8, 120), (2.5, 1.1, 180)]   # (평균, 표준편차, 개수)
EM_ITERS = 12
