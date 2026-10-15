"""데이터 가게 (Data Shop): 회사 문제 영역별 공개 데이터 노트북 + 웹 페이지 원본.
   ~/Documents/manim_shorts/_env/v/bin/python make_shop.py          # 노트북 7개를 만들고 실행해 출력까지 저장 + docs/datashop/index.html
   ~/Documents/manim_shorts/_env/v/bin/python make_shop.py --no-run # 실행 없이 만들기만
노트북 = notebooks/datashop/<번호>_<slug>.ipynb (번호는 SECTIONS 순서대로 자동), 웹 = docs/datashop/index.html (smhanlab.com/book-ml/datashop/).
1주 2차시 '문제 정형화와 평가지표' 의 다섯 칸 (한 행 · 입력 X · 출력 y · 평가지표 · 기준선) 을 회사 문제로 연습한다.
주제는 산업 분야별로 '그 분야의 회사들이 실제로 푸는 문제' 로 잡았다 (특정 회사 이름은 쓰지 않는다). 데이터는 그와 닮은 공개 데이터.
데이터는 모두 로그인 없이 받는다. 코드 칸은 영어, 설명은 한국어."""
import html, json, os, sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
REL = os.path.relpath(HERE, ROOT)
COLAB = "https://colab.research.google.com/github/SuminHan/book-ml/blob/main/" + REL + "/{}.ipynb"
GITHUB = "https://github.com/SuminHan/book-ml/blob/main/" + REL + "/{}.ipynb"

FIVE_MD = """### 나의 다섯 칸

아래 표를 직접 채워 보자. 위의 예시와 다르게 정해도 된다.

| 칸 | 내가 정한 것 |
|---|---|
| ① 한 행은 무엇? | |
| ② 입력 X (예측할 때 알 수 있는 것만) | |
| ③ 출력 y | |
| ④ 평가지표 | |
| ⑤ 기준선 | |"""

COMMON_IMPORTS = """import io, os, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def download(url, path):
    \"\"\"Download url to path once (skip if it already exists)\"\"\"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ML course data shop)"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    return path"""


def five_table(rows):
    return "| 칸 | 이 노트북의 예시 |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


SHOP = [
# ------------------------------------------------------------------ 패션 커머스
dict(slug="fashion_commerce", title="패션 커머스: 비슷한 옷 추천", sector="커머스 · 콘텐츠",
     keywords=["개인화 추천", "수요예측 · 재고관리", "고객생애가치"],
     teaser="옷 사진 2만 2천 장. 비슷한 옷을 골라 줄 수 있을까?",
     data="Fashion Product Images (Small)", data_src="Kaggle 'Fashion Product Images' 의 축소판, Hugging Face ashraq/fashion-product-images-small (첫 묶음 22,036장)",
     size="약 140MB",
     five=[("① 한 행", "상품 하나 (사진 60×80 + 성별 · 종류 · 색 · 계절)"), ("② 입력 X", "상품 사진"),
           ("③ 출력 y", "비슷한 상품 5개 (추천)"), ("④ 평가지표", "추천 5개 중 같은 종류(articleType) 비율"),
           ("⑤ 기준선", "무작위로 5개 고르기")],
     cells=[
("md", """## 입장: 데이터 받기

상품 사진과 정보가 들어 있는 parquet 파일 하나 (약 140MB). 처음엔 1분쯤 걸린다."""),
("code", """URL = "https://huggingface.co/api/datasets/ashraq/fashion-product-images-small/parquet/default/train/0.parquet"
df = pd.read_parquet(download(URL, "data/fashion_0.parquet"))
print(df.shape)
df.drop(columns="image").head()"""),
("md", """## 열어 보기

한 행 = 상품 하나. 어떤 종류가 많을까? 사진도 몇 장 보자."""),
("code", """from PIL import Image
print(df["masterCategory"].value_counts(), "\\n")
print(df["articleType"].value_counts().head(10))
fig, axs = plt.subplots(1, 8, figsize=(14, 2.6))
for ax, (_, row) in zip(axs, df.sample(8, random_state=0).iterrows()):
    ax.imshow(Image.open(io.BytesIO(row["image"]["bytes"]))); ax.set_title(row["articleType"], fontsize=9); ax.axis("off")
plt.show()"""),
("md", """## 문제 하나 골라 보기: '이 옷과 비슷한 옷' 추천

사진을 32×32 흑백으로 줄여 숫자 1,024개로 만들고, 숫자가 가장 가까운 상품 5개를 '비슷한 옷' 으로 추천한다.
채점: 추천 5개 중 **같은 종류 (articleType)** 가 몇 개인가. 기준선: 무작위 5개."""),
("code", """from sklearn.neighbors import NearestNeighbors

def to_vector(image_bytes, size=32):
    \"\"\"Image bytes -> grayscale size x size, flattened to 0~1\"\"\"
    im = Image.open(io.BytesIO(image_bytes)).convert("L").resize((size, size))
    return np.asarray(im, dtype=np.float32).ravel() / 255

items = df[df["masterCategory"] == "Apparel"].reset_index(drop=True)
V = np.stack([to_vector(b["bytes"]) for b in items["image"]])
types = items["articleType"].astype(str).to_numpy()
print("apparel items:", V.shape)

def same_type_rate(neighbors, types):
    \"\"\"Share of recommended items that have the same articleType as the query\"\"\"
    return np.mean(types[neighbors] == types[:, None])

nn = NearestNeighbors(n_neighbors=6).fit(V)
neighbors = nn.kneighbors(V, return_distance=False)[:, 1:]          # drop the item itself
rng = np.random.default_rng(0)
random5 = rng.integers(0, len(items), size=(len(items), 5))
print(f"random 5           : {same_type_rate(random5, types):.0%} same type")
print(f"nearest 5 by pixels: {same_type_rate(neighbors, types):.0%} same type")"""),
("code", """fig, axs = plt.subplots(3, 6, figsize=(11, 6))
for r, q in enumerate([0, 100, 2000]):
    for c, i in enumerate([q] + list(neighbors[q])):
        axs[r, c].imshow(Image.open(io.BytesIO(items["image"][i]["bytes"]))); axs[r, c].axis("off")
        axs[r, c].set_title(("query: " if c == 0 else "") + types[i], fontsize=8)
plt.tight_layout(); plt.show()"""),
("md", """## 다른 문제로 바꿔 보기

- **수요예측 · 재고**: 이 데이터에는 판매량이 없다. **데이터에 없는 y 는 맞힐 수 없다.** 판매 기록이 있는 데이터 (예: Kaggle 'H&M Personalized Fashion Recommendations', 로그인 필요) 가 필요하다.
- **개인화 추천**: 여기서는 '사진이 비슷한 옷' 이다. 고객마다 다른 추천을 하려면 한 행이 '고객 × 상품' 인 구매 기록이 있어야 한다.
- **옷 사진 생성**: StyleGAN 으로 옷 입은 사람을 만든 연구 (StyleGAN-Human) 는 전신 사진 23만 장 SHHQ 를 썼다 (신청 필요)."""),
]),
# ------------------------------------------------------------------
dict(slug="card_fraud", title="카드 이상거래 탐지", sector="금융 · 핀테크",
     keywords=["컴플라이언스", "이상거래탐지 · 자금세탁방지", "신뢰성 · 설명가능성"],
     teaser="카드 거래 28만 건 중 사기는 0.17%. 정확도 99.8% 는 좋은 점수일까?",
     data="Credit Card Fraud Detection (ULB)", data_src="유럽 카드 소지자의 2013년 9월 이틀치 거래, OpenML 1597 (원본 Kaggle mlg-ulb). 성분은 개인정보 때문에 PCA 로 바꾼 V1~V28. OpenML 판에는 시각 열이 없지만 행이 시간 순서대로 놓여 있다",
     size="약 150MB",
     five=[("① 한 행", "카드 거래 한 건"), ("② 입력 X", "금액 · V1~V28"), ("③ 출력 y", "사기인가 (Class 0/1)"),
           ("④ 평가지표", "정밀도 · 재현율 (정확도는 함정)"), ("⑤ 기준선", "전부 '정상'")],
     cells=[
("md", """## 입장: 데이터 받기

OpenML 에서 받는다 (약 150MB, 처음엔 1~2분)."""),
("code", """from sklearn.datasets import fetch_openml
cc = fetch_openml(data_id=1597, as_frame=True, parser="auto", data_home="data")
df = cc.frame
df["Class"] = df["Class"].astype(int)
print(df.shape)
df.head()"""),
("md", """## 열어 보기

사기 (Class = 1) 는 몇 건일까? 금액은 어떻게 퍼져 있을까? 사기와 정상의 금액이 다를까?"""),
("code", """print(df["Class"].value_counts())
print(f"fraud rate: {df['Class'].mean():.3%}")
print(df.groupby("Class")["Amount"].describe().round(1))
plt.figure(figsize=(7, 3))
for c, name in [(0, "normal"), (1, "fraud")]:
    plt.hist(np.log10(df.loc[df["Class"] == c, "Amount"] + 1), bins=50, density=True, alpha=0.6, label=name)
plt.xlabel("log10(amount + 1)"); plt.legend(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 거래 하나 → 사기인가

**시간 순서대로** 나눈다: 앞 70% 행으로 학습, 뒤 30% 로 시험 (행이 시간 순서대로 놓여 있다. 미래의 사기를 잡는 게 목적이니까). 기준선 = 전부 '정상'."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

cut = int(len(df) * 0.7)
train, test = df.iloc[:cut], df.iloc[cut:]
X_cols = [c for c in df.columns if c != "Class"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(train[X_cols], train["Class"])

y = test["Class"].values
for name, pred in [("always normal", np.zeros(len(y), dtype=int)), ("logistic regression", model.predict(test[X_cols]))]:
    p, r = precision_recall(y, pred)
    print(f"{name:20s} accuracy {np.mean(pred == y):.3%}  precision {p:.0%}  recall {r:.0%}")"""),
("md", """**관찰**: 전부 '정상' 도 정확도가 99.8% 를 넘는다. 사기를 하나도 못 잡는데도. 그래서 이 문제는 정밀도 · 재현율로 채점한다.

## 다른 문제로 바꿔 보기

- **자금세탁방지**: 한 행이 '거래' 가 아니라 '계좌' 라면? IBM AML 데이터 (Kaggle, 1주 2차시 실습) 로 해 볼 수 있다.
- **설명가능성**: V1~V28 은 개인정보 때문에 원래 뜻을 지운 숫자다. 이 모델이 '왜' 사기라고 했는지 고객에게 설명할 수 있을까? (7주차 SHAP)
- **컴플라이언스**: 놓친 사기 한 건과, 정상 고객의 카드를 잘못 막은 한 건. 어느 손해가 더 큰가에 따라 지표가 달라진다."""),
]),
# ------------------------------------------------------------------
dict(slug="support_routing", title="고객 문의 자동 분류", sector="플랫폼 · 서비스",
     keywords=["대화형 인공지능", "고객응대 자동화", "상담 보조 · 문의 분산처리"],
     teaser="은행 고객 문의 1만 3천 개, 담당 77곳. 문의를 맞는 팀에 보낼 수 있을까?",
     data="Banking77", data_src="PolyAI (Casanueva et al., 2020), 온라인 은행 고객 문의 13,083개를 의도 77가지로 나눈 영어 데이터 (GitHub PolyAI-LDN/task-specific-datasets)",
     size="약 1MB",
     five=[("① 한 행", "고객 문의 한 문장"), ("② 입력 X", "문장"), ("③ 출력 y", "의도 77가지 중 하나 (보낼 팀)"),
           ("④ 평가지표", "정확도 + 의도별 정확도"), ("⑤ 기준선", "가장 흔한 의도로 보내기")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
train = pd.read_csv(download(BASE + "train.csv", "data/banking77_train.csv"))
test = pd.read_csv(download(BASE + "test.csv", "data/banking77_test.csv"))
print(train.shape, test.shape)
train.sample(5, random_state=0)"""),
("md", """## 열어 보기

의도가 몇 가지, 각각 몇 문장씩일까? 문장 길이는?"""),
("code", """counts = train["category"].value_counts()
print("intents:", len(counts), "| most:", counts.index[0], counts.iloc[0], "| fewest:", counts.index[-1], counts.iloc[-1])
words = train["text"].str.split().str.len()
print("words per query: min", words.min(), "median", int(words.median()), "max", words.max())
for cat in ["card_arrival", "lost_or_stolen_card", "exchange_rate"]:
    print(f"\\n[{cat}]", train.loc[train["category"] == cat, "text"].head(3).tolist())"""),
("md", """## 문제 하나 골라 보기: 문의 → 보낼 팀 (77가지 분류)

문장을 단어 빈도 (TF-IDF) 벡터로 바꾸고 로지스틱회귀로 분류한다. 기준선 = 늘 가장 흔한 의도."""),
("code", """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

most_common = train["category"].value_counts().idxmax()
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000)).fit(train["text"], train["category"])
pred = model.predict(test["text"])
print(f"baseline (always '{most_common}'): accuracy {np.mean(test['category'] == most_common):.1%}")
print(f"TF-IDF + logistic regression : accuracy {np.mean(test['category'] == pred):.1%}")

per_intent = pd.Series(test["category"].values == pred).groupby(test["category"].values).mean().sort_values()
print("\\nhardest intents:\\n", per_intent.head(5).round(2))"""),
("code", """wrong = pd.DataFrame({"true": test["category"].values, "pred": pred})
wrong = wrong[wrong["true"] != wrong["pred"]]
pairs = wrong.value_counts().head(5)
print("most confused (true -> predicted):")
for (t, p), n in pairs.items():
    print(f"  {t:35s} -> {p:35s} {n}")"""),
("md", """## 다른 문제로 바꿔 보기

- **문의 분산처리**: 77개 의도를 실제 팀 5~6개로 묶으면 문제가 쉬워질까? 묶는 방법은 누가 정하나?
- **상담 보조**: '맞는 팀' 대신 '상담원에게 보여 줄 비슷한 과거 문의 3개' 를 출력으로 정하면 지표는?
- **고객응대 자동화**: 자신 없는 문의 (예측 확률이 낮은 것) 만 사람에게 넘기면, 자동 처리율과 정확도는 어떻게 바뀔까?"""),
]),
# ------------------------------------------------------------------
dict(slug="shop_conversion", title="쇼핑몰 구매 전환 예측", sector="커머스 · 콘텐츠",
     keywords=["브랜드 · 콘텐츠 마케팅", "직접판매 전환율 최적화", "해외직구 커머스"],
     teaser="쇼핑몰 방문 1만 2천 건. 이 방문자는 살까, 그냥 나갈까?",
     data="Online Shoppers Purchasing Intention", data_src="UCI (Sakar et al., 2019), 온라인 쇼핑몰 방문 세션 12,330건 (1년, 사용자당 1세션)",
     size="약 1MB",
     five=[("① 한 행", "방문 세션 하나"), ("② 입력 X", "본 페이지 수 · 머문 시간 · 이탈률 · 월 · 방문자 유형 …"),
           ("③ 출력 y", "구매했나 (Revenue)"), ("④ 평가지표", "정밀도 · 재현율"), ("⑤ 기준선", "아무도 안 산다")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
with zipfile.ZipFile(download(URL, "data/shoppers.zip")) as z:
    df = pd.read_csv(z.open("online_shoppers_intention.csv"))
print(df.shape)
df.head()"""),
("md", """## 열어 보기

구매한 방문은 몇 %? 달마다 다를까?"""),
("code", """print(f"conversion rate: {df['Revenue'].mean():.1%}")
by_month = df.groupby("Month")["Revenue"].agg(["mean", "size"]).sort_values("size", ascending=False)
print(by_month.round(3))
print(df.groupby("VisitorType")["Revenue"].mean().round(3))"""),
("md", """## 문제 하나 골라 보기: 방문 → 살까?

기준선 = 아무도 안 산다. 모델은 그레이디언트 부스팅 (7주차). 그리고 **`PageValues` 를 넣을 때와 뺄 때** 를 비교한다.
`PageValues` 는 '이 방문자가 본 페이지들이 과거 거래에서 얼마짜리였나' 를 구글 애널리틱스가 계산한 값이다. 방문이 **끝나기 전에** 알 수 있는 값일까?"""),
("code", """from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

X = pd.get_dummies(df.drop(columns="Revenue"), columns=["Month", "VisitorType"]).astype(float)
y = df["Revenue"].astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

p, r = precision_recall(yte, np.zeros(len(yte), dtype=int))
print(f"{'nobody buys':26s} accuracy {np.mean(yte == 0):.1%}  precision {p:.0%}  recall {r:.0%}")
for name, cols in [("with PageValues", X.columns), ("without PageValues", [c for c in X.columns if c != "PageValues"])]:
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr[cols], ytr)
    pred = m.predict(Xte[cols]); p, r = precision_recall(yte, pred)
    print(f"{name:26s} accuracy {np.mean(yte == pred):.1%}  precision {p:.0%}  recall {r:.0%}")"""),
("md", """**관찰**: `PageValues` 하나가 성능을 크게 바꾼다. 방문 도중에 '살지' 를 맞혀 쿠폰을 띄우는 게 목적이라면, 그 순간에 이 값을 알 수 있는지부터 확인해야 한다.

## 다른 문제로 바꿔 보기

- **전환율 최적화**: '살까?' 대신 '쿠폰을 보여 주면 살 사람' 을 찾고 싶다면 y 가 달라진다 (쿠폰을 보여 준 실험 데이터가 필요).
- **콘텐츠 마케팅**: 한 행을 '방문' 이 아니라 '광고 · 콘텐츠 하나' 로 정하면?
- **해외직구**: 리뷰 데이터 (예: Amazon Reviews 2023 의 Beauty 분야, McAuley Lab) 로 '재구매할 상품' 을 맞힐 수도 있다."""),
]),
# ------------------------------------------------------------------
dict(slug="hotel_booking", title="여행 예약: 취소와 가격", sector="플랫폼 · 서비스",
     keywords=["실시간 가격책정", "여행일정 개인화", "수요-공급 매칭"],
     teaser="포르투갈 호텔 예약 12만 건. 이 예약은 취소될까? 1박에 얼마가 맞을까?",
     data="Hotel Booking Demand", data_src="Antonio, Almeida & Nunes (2019), Data in Brief. 포르투갈 리조트 호텔 · 도시 호텔 2곳의 2015~2017 예약 119,390건 (TidyTuesday 2020-02-11)",
     size="약 17MB",
     five=[("① 한 행", "예약 한 건"), ("② 입력 X", "예약 시점에 아는 것: 며칠 전 예약 · 인원 · 박 수 · 고객 유형 …"),
           ("③ 출력 y", "취소할까 (분류) / 1박 가격 adr (회귀)"), ("④ 평가지표", "정밀도 · 재현율 / MAE (유로)"),
           ("⑤ 기준선", "아무도 취소 안 함 / 같은 달 평균 가격")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
df = pd.read_csv(download(URL, "data/hotels.csv"))
print(df.shape)
df.head(3).T.head(20)"""),
("md", """## 열어 보기

취소율, 1박 가격 (`adr`, 유로) 의 분포, 달마다 가격."""),
("code", """print(f"cancellation rate: {df['is_canceled'].mean():.1%}")
print(df["adr"].describe().round(1))
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
price = df[df["is_canceled"] == 0].groupby(["arrival_date_month", "hotel"])["adr"].mean().unstack().reindex(months)
price.plot(figsize=(9, 3), marker="o"); plt.ylabel("mean adr (EUR)"); plt.xticks(range(12), [m[:3] for m in months]); plt.show()"""),
("md", """## 문제 1: 예약 → 취소할까 (분류)

**함정 하나**: `reservation_status` 열이 있다. 값이 `Canceled` 면 취소된 것이다. 이걸 입력에 넣으면?
예약하는 순간에는 알 수 없는 값 = **답이 새는 입력 (누수, leakage)**. 넣을 때와 뺄 때를 비교한다. 시험은 **시간 순서** 로 2017년 예약."""),
("code", """from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

known = ["hotel", "lead_time", "arrival_date_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
         "babies", "meal", "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
         "booking_changes", "deposit_type", "customer_type", "total_of_special_requests", "required_car_parking_spaces"]
leaky = known + ["reservation_status"]
train, test = df[df["arrival_date_year"] < 2017], df[df["arrival_date_year"] == 2017]

def fit_cancel(cols, train, test):
    \"\"\"Gradient boosting on the given columns; returns predictions for test\"\"\"
    enc = pd.get_dummies(pd.concat([train[cols], test[cols]]), dtype=float)
    m = HistGradientBoostingClassifier(random_state=0).fit(enc.iloc[:len(train)], train["is_canceled"])
    return m.predict(enc.iloc[len(train):])

y = test["is_canceled"].values
for name, pred in [("nobody cancels", np.zeros(len(y), dtype=int)), ("known at booking", fit_cancel(known, train, test)),
                   ("+ reservation_status", fit_cancel(leaky, train, test))]:
    p, r = precision_recall(y, pred)
    print(f"{name:22s} accuracy {np.mean(pred == y):.1%}  precision {p:.0%}  recall {r:.0%}")"""),
("md", """## 문제 2: 같은 데이터, 예약 → 1박 가격 (회귀)

취소되지 않은 예약만. 기준선 = 학습 기간의 '같은 호텔 · 같은 달' 평균 가격."""),
("code", """ok_train, ok_test = train[train["is_canceled"] == 0], test[test["is_canceled"] == 0]
month_mean = ok_train.groupby(["hotel", "arrival_date_month"])["adr"].mean()
base = ok_test.set_index(["hotel", "arrival_date_month"]).index.map(month_mean).values
enc = pd.get_dummies(pd.concat([ok_train[known], ok_test[known]]), dtype=float)
reg = HistGradientBoostingRegressor(random_state=0).fit(enc.iloc[:len(ok_train)], ok_train["adr"])
pred = reg.predict(enc.iloc[len(ok_train):])
print(f"baseline (hotel x month mean): MAE {np.nanmean(np.abs(base - ok_test['adr'])):.1f} EUR")
print(f"gradient boosting            : MAE {np.mean(np.abs(pred - ok_test['adr'])):.1f} EUR")"""),
("md", """**관찰**: 2017년 가격이 2015~2016년보다 올랐다면, 과거 평균은 미래 가격을 낮게 잡는다 (1주 3차시 항공 승객과 같은 이야기).

## 다른 문제로 바꿔 보기

- **실시간 가격책정**: '가격' 을 맞히는 것과 '얼마를 받아야 이익이 최대인가' 는 다른 문제다. 가격을 바꿨을 때 예약이 어떻게 변하는지는 이 데이터만으로 알 수 있을까?
- **수요-공급 매칭**: 한 행을 '예약' 이 아니라 '호텔 × 날짜' 로 바꾸면 '그날 방이 몇 개 찰까' 를 맞히는 문제가 된다.
- **여행일정 개인화**: 고객 한 명의 과거 예약이 여러 개 있어야 한다. 이 데이터에는 고객 번호가 없다."""),
]),
# ------------------------------------------------------------------
dict(slug="form_understanding", title="서식 문서 이해", sector="전문 서비스",
     keywords=["감사 자동화", "문서지능 처리", "리스크 · 내부통제"],
     teaser="스캔한 서식 199장, 글 덩어리 9천여 개. 어느 것이 '질문' 이고 어느 것이 '답' 일까?",
     data="FUNSD", data_src="Jaume, Ekenel & Thiran (2019), Form Understanding in Noisy Scanned Documents. 스캔 서식 199장 (학습 149 · 시험 50), 글 덩어리마다 header · question · answer · other 표시",
     size="약 17MB",
     five=[("① 한 행", "서식 속 글 덩어리 하나 (상자 + 글자)"), ("② 입력 X", "위치 · 크기 · 글자 (콜론으로 끝나나, 숫자인가 …)"),
           ("③ 출력 y", "header / question / answer / other"), ("④ 평가지표", "정확도 + 종류별 재현율"),
           ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """import json
URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
zf = zipfile.ZipFile(download(URL, "data/funsd.zip"))
names = [n for n in zf.namelist() if n.endswith(".json") and "__MACOSX" not in n]
print(len(names), "annotated forms;", names[:2])"""),
("md", """## 열어 보기

서식 한 장 = 사진 (png) + 글 덩어리 목록 (json). 한 장을 그려 보자. 색 = 종류."""),
("code", """from PIL import Image
COLORS = {"header": "purple", "question": "blue", "answer": "green", "other": "gray"}
name = names[0]
form = json.loads(zf.read(name))["form"]
img = Image.open(io.BytesIO(zf.read(name.replace("annotations", "images").replace(".json", ".png"))))
plt.figure(figsize=(6, 8)); plt.imshow(img, cmap="gray")
for block in form:
    x0, y0, x1, y1 = block["box"]
    plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=COLORS[block["label"]], lw=1.2))
plt.axis("off"); plt.title(" / ".join(f"{k}={v}" for k, v in COLORS.items()), fontsize=9); plt.show()
print([(b["label"], b["text"]) for b in form[:6]])"""),
("md", """## 문제 하나 골라 보기: 글 덩어리 → 종류

한 행 = 글 덩어리 하나. 사람이 서식을 볼 때 쓰는 단서를 숫자로 만든다: 위치, 크기, 글자 수, 콜론(`:`) 으로 끝나나, 숫자가 많은가, 대문자 비율. 학습 149장 · 시험 50장은 데이터가 정해 둔 대로 나눈다."""),
("code", """from sklearn.ensemble import RandomForestClassifier

def block_features(block, width, height):
    \"\"\"Simple layout + text clues for one text block\"\"\"
    x0, y0, x1, y1 = block["box"]; t = block["text"]
    return dict(x=x0 / width, y=y0 / height, w=(x1 - x0) / width, h=(y1 - y0) / height, n_chars=len(t),
                ends_colon=float(t.strip().endswith(":")), digits=sum(c.isdigit() for c in t) / max(len(t), 1),
                upper=sum(c.isupper() for c in t) / max(sum(c.isalpha() for c in t), 1))

rows = []
for n in names:
    img = Image.open(io.BytesIO(zf.read(n.replace("annotations", "images").replace(".json", ".png"))))
    for b in json.loads(zf.read(n))["form"]:
        if b["text"].strip():
            rows.append({**block_features(b, *img.size), "label": b["label"], "split": "test" if "testing" in n else "train"})
blocks = pd.DataFrame(rows)
print(blocks.groupby("split")["label"].value_counts().unstack())

tr, te = blocks[blocks["split"] == "train"], blocks[blocks["split"] == "test"]
feats = [c for c in blocks.columns if c not in ("label", "split")]
most_common = tr["label"].value_counts().idxmax()
m = RandomForestClassifier(300, random_state=0).fit(tr[feats], tr["label"])
pred = m.predict(te[feats])
print(f"\\nbaseline (always '{most_common}'): accuracy {np.mean(te['label'] == most_common):.1%}")
print(f"random forest on 8 clues       : accuracy {np.mean(te['label'] == pred):.1%}")
print("recall per label:", pd.Series(te["label"].values == pred).groupby(te["label"].values).mean().round(2).to_dict())"""),
("md", """## 다른 문제로 바꿔 보기

- **문서지능**: 종류를 맞힌 다음, 어느 '질문' 과 어느 '답' 이 짝인지 맞히는 문제 (FUNSD 의 `linking`) 로 넘어갈 수 있다. 출력이 '짝 목록' 이면 지표는?
- **감사 자동화**: 영수증 (SROIE), 계약서 (CUAD: 계약서 510건, 조항 41종) 처럼 서식이 정해진 문서라면 한 행 · 출력이 어떻게 바뀔까?
- **내부통제**: '틀린 서식' 을 찾는 문제라면 y 는 무엇이고, 그 정답은 누가 매기나?"""),
]),
# ------------------------------------------------------------------
dict(slug="ai_usecases", title="AI 활용사례 찾기", sector="전문 서비스",
     keywords=["인공지능 전환 전략", "활용사례 발굴 · 검증", "인공지능 거버넌스"],
     teaser="사람들은 AI 를 어떤 일에 쓰고 있을까? 업무 1만 9천 가지로 찾는 AI 활용사례.",
     data="Anthropic Economic Index (2025-02)", data_src="Anthropic (2025), Claude.ai 대화를 미국 O*NET 업무 목록에 대응시킨 공개 데이터 (Hugging Face Anthropic/EconomicIndex). O*NET 업무 설명 19,530개 포함",
     size="약 4MB",
     five=[("① 한 행", "업무 하나 (O*NET task) 또는 직업 하나"), ("② 입력 X", "업무 설명 · 직업"),
           ("③ 출력 y", "AI 대화 중 이 업무의 비율 (%)"), ("④ 평가지표", "순위가 맞나 (상위 몇 개가 실제로 겹치나)"),
           ("⑤ 기준선", "그 직업의 종사자 수 순서")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """BASE = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_02_10/"
usage = pd.read_csv(download(BASE + "onet_task_mappings.csv", "data/onet_task_mappings.csv"))
tasks = pd.read_csv(download(BASE + "onet_task_statements.csv", "data/onet_task_statements.csv"))
print(usage.shape, tasks.shape)
print(usage.sort_values("pct", ascending=False).head(5).to_string())"""),
("md", """## 열어 보기

`pct` = 전체 AI 대화 중 이 업무로 분류된 비율 (%). 업무 하나는 여러 직업에 들어 있다. 업무 설명 표와 붙여 **직업별** 로 모아 보자."""),
("code", """tasks["task_name"] = tasks["Task"].str.lower().str.strip()
merged = tasks.merge(usage, on="task_name", how="left").fillna({"pct": 0})
# a task shared by several occupations: split its share equally
merged["pct_split"] = merged["pct"] / merged.groupby("task_name")["Title"].transform("count")
by_job = merged.groupby("Title")["pct_split"].sum().sort_values(ascending=False)
print(f"tasks with any AI use: {(usage['pct'] > 0).sum()} / {len(tasks['task_name'].unique())}")
print("\\ntop 10 occupations by share of AI conversations (%):")
print(by_job.head(10).round(2).to_string())"""),
("code", """top = by_job.head(15)[::-1]
plt.figure(figsize=(8, 5)); plt.barh(top.index, top.values); plt.xlabel("% of AI conversations")
plt.title("Where is AI used? (Anthropic Economic Index, Feb 2025)", fontsize=10); plt.tight_layout(); plt.show()"""),
("md", """## 문제 정형화 연습: '활용사례 발굴' 을 다섯 칸으로

이 노트북은 모델을 학습하지 않는다. 대신 컨설팅 질문을 데이터 문제로 바꿔 보는 연습이다.

- '어느 부서부터 AI 를 도입할까?' → 한 행 = 부서의 업무, y = ? (지금 쓰는 비율? 절약되는 시간? 실수의 비용?)
- 이 데이터의 y 는 '사람들이 실제로 많이 쓰는 곳' 이다. '써야 하는 곳' 이나 '쓰면 이익이 큰 곳' 과 같을까?
- **거버넌스**: 많이 쓰이는 업무일수록 실수했을 때의 위험도 크다. '위험' 을 숫자로 정한다면 무엇을 세야 할까? (참고: AI Incident Database)
- **검증**: 우리 회사에서 고른 활용사례가 맞았는지는 무엇으로 확인할까? 도입 전 · 후로 무엇을 재야 할까?"""),
]),
# ------------------------------------------------------------------
dict(slug="movie_ratings", title="영화 · 동영상 추천", sector="커머스 · 콘텐츠",
     keywords=["개인화 추천", "평점 예측", "콜드 스타트"],
     teaser="610명이 영화 9천 편에 남긴 별점 10만 개. 아직 안 본 영화에 몇 점을 줄까?",
     data="MovieLens (ml-latest-small)", data_src="GroupLens Research, University of Minnesota (Harper & Konstan, 2015). 610명 · 9,742편 · 별점 100,836개 (0.5~5점)",
     size="약 1MB",
     five=[("① 한 행", "(사용자, 영화) 쌍 하나"), ("② 입력 X", "사용자 번호 · 영화 번호 (+ 장르)"),
           ("③ 출력 y", "별점 (0.5~5)"), ("④ 평가지표", "RMSE (별 몇 개만큼 틀리나)"), ("⑤ 기준선", "전체 평균 별점")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
z = zipfile.ZipFile(download(URL, "data/ml-latest-small.zip"))
ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
print(ratings.shape, movies.shape)
ratings.merge(movies, on="movieId").head()"""),
("md", """## 열어 보기

사용자 × 영화 표로 보면 얼마나 비어 있을까? 영화마다 받은 별점 수는?"""),
("code", """n_users, n_movies = ratings["userId"].nunique(), ratings["movieId"].nunique()
print(f"users {n_users}, movies {n_movies}, ratings {len(ratings)}")
print(f"filled share of the user x movie table: {len(ratings) / (n_users * n_movies):.2%}")
per_movie = ratings.groupby("movieId").size()
print("ratings per movie: median", int(per_movie.median()), "| movies with only 1 rating:", (per_movie == 1).sum())
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
axs[0].hist(ratings["rating"], bins=np.arange(0.25, 5.5, 0.5)); axs[0].set_xlabel("stars")
axs[1].hist(np.log10(per_movie), bins=40); axs[1].set_xlabel("log10(ratings per movie)")
plt.show()"""),
("md", """## 문제 하나 골라 보기: (사용자, 영화) → 별점

기준선 = 전체 평균. 조금 나은 방법 = **평균 + 사용자 치우침 + 영화 치우침** (후하게 주는 사람, 다들 좋아하는 영화). 시험은 무작위 20%."""),
("code", """from sklearn.model_selection import train_test_split

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def fit_biases(train, reg=5.0):
    \"\"\"Global mean + user bias + movie bias, shrunk toward 0 for users/movies with few ratings\"\"\"
    mu = train["rating"].mean()
    r = train.assign(res=train["rating"] - mu)
    b_movie = r.groupby("movieId")["res"].agg(lambda v: v.sum() / (len(v) + reg))
    r = r.assign(res=r["res"] - r["movieId"].map(b_movie))
    b_user = r.groupby("userId")["res"].agg(lambda v: v.sum() / (len(v) + reg))
    return mu, b_user, b_movie

def predict_biases(model, data):
    mu, b_user, b_movie = model
    return (mu + data["userId"].map(b_user).fillna(0) + data["movieId"].map(b_movie).fillna(0)).clip(0.5, 5)

train, test = train_test_split(ratings, test_size=0.2, random_state=0)
model = fit_biases(train)
print(f"baseline (global mean {train['rating'].mean():.2f}): RMSE {rmse(test['rating'], np.full(len(test), train['rating'].mean())):.3f}")
print(f"mean + user bias + movie bias : RMSE {rmse(test['rating'], predict_biases(model, test)):.3f}")
print(f"movies in test never seen in train: {(~test['movieId'].isin(train['movieId'])).sum()}")"""),
("md", """## 다른 문제로 바꿔 보기

- **추천은 순위 문제**: 실제 서비스는 별점이 아니라 '첫 화면에 보여 줄 10편' 을 고른다. y 를 '다음에 볼 영화', 지표를 '상위 10편 안에 들었나 (hit@10)' 로 바꾸면?
- **콜드 스타트**: 시험에 처음 보는 영화가 있다. 별점이 없는 새 영화는 무엇으로 추천할까? (`movies.csv` 의 장르)
- **시간 순서**: 무작위가 아니라 사용자마다 마지막 20% 를 시험으로 두면 점수가 어떻게 바뀔까?"""),
]),
# ------------------------------------------------------------------
dict(slug="korean_reviews", title="한국어 리뷰 감성 분석", sector="커머스 · 콘텐츠",
     keywords=["리뷰 분석", "브랜드 모니터링", "한국어 자연어처리"],
     teaser="한국어 영화 리뷰 20만 개. 이 리뷰는 칭찬일까, 불평일까?",
     data="NSMC (Naver Sentiment Movie Corpus)", data_src="Lucy Park (2015), github.com/e9t/nsmc. 한국어 영화 리뷰 20만 개 (학습 15만 · 시험 5만), 긍정 · 부정 반반",
     size="약 20MB",
     five=[("① 한 행", "리뷰 한 문장"), ("② 입력 X", "리뷰 글"), ("③ 출력 y", "긍정 1 / 부정 0"),
           ("④ 평가지표", "정확도 (반반이라 쓸 만하다)"), ("⑤ 기준선", "늘 '긍정' (50%)")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """BASE = "https://raw.githubusercontent.com/e9t/nsmc/master/"
train = pd.read_csv(download(BASE + "ratings_train.txt", "data/nsmc_train.txt"), sep="\\t").dropna()
test = pd.read_csv(download(BASE + "ratings_test.txt", "data/nsmc_test.txt"), sep="\\t").dropna()
print(train.shape, test.shape)
train.head()"""),
("md", """## 열어 보기

긍정 비율, 리뷰 길이, 그리고 긍정 · 부정 예시."""),
("code", """print(f"positive share: train {train['label'].mean():.1%}, test {test['label'].mean():.1%}")
chars = train["document"].str.len()
print("characters per review: median", int(chars.median()), "max", chars.max())
for lab in [1, 0]:
    print("\\nlabel", lab, train.loc[train["label"] == lab, "document"].sample(4, random_state=1).tolist())"""),
("md", """## 문제 하나 골라 보기: 리뷰 → 긍정 / 부정

한국어는 띄어쓰기 단위로 자르면 '재밌다', '재밌어요', '재밌는' 이 다 다른 단어가 된다. 그래서 **글자 1~3개 묶음** 을 센다 (형태소 분석기 없이). 시간을 줄이려고 학습은 5만 개만."""),
("code", """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

small = train.sample(50000, random_state=0)
model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=3, max_features=200000),
                      LogisticRegression(max_iter=1000)).fit(small["document"], small["label"])
pred = model.predict(test["document"])
print(f"baseline (always positive): accuracy {test['label'].mean():.1%}")
print(f"char 1-3 grams + logistic : accuracy {np.mean(pred == test['label']):.1%}")
for s in ["진짜 재밌어요 강추", "시간 아깝다", "배우는 좋은데 스토리가 별로", "안 보면 후회함"]:
    print(f"  {s:20s} -> P(positive) = {model.predict_proba([s])[0, 1]:.2f}")"""),
("md", """## 다른 문제로 바꿔 보기

- **브랜드 모니터링**: 리뷰 하나가 아니라 '하루 동안의 리뷰' 를 한 행으로 하면? y = 부정 리뷰 비율이 갑자기 늘었나 (경보).
- **별점**: 긍정 / 부정 두 칸 대신 별 1~10 개를 맞히면 회귀가 된다. 지표는?
- '배우는 좋은데 스토리가 별로' 처럼 섞인 리뷰는 무엇이 정답일까? 정답을 매긴 사람마다 다를 수 있다."""),
]),
# ------------------------------------------------------------------
dict(slug="link_prediction", title="연결 예측: 사람 · 친구 추천", sector="플랫폼 · 서비스",
     keywords=["링크 예측 (Link Prediction)", "친구 · 연결 추천", "그래프 데이터"],
     teaser="연구기관 1,005명이 주고받은 이메일 연결망. 아직 연결 안 된 두 사람, 곧 연락하게 될까?",
     data="email-Eu-core", data_src="SNAP, Stanford (Leskovec, Kleinberg & Faloutsos, 2007; Yin et al., 2017). 유럽 한 연구기관 구성원 1,005명 사이의 이메일 연결 25,571개 + 소속 부서 42개",
     size="약 0.1MB",
     five=[("① 한 행", "두 사람의 쌍"), ("② 입력 X", "지금까지의 연결망 (공통 지인 수, 같은 부서인가 …)"),
           ("③ 출력 y", "둘이 연결되는가"), ("④ 평가지표", "ROC-AUC (진짜 연결을 더 높게 매겼나)"),
           ("⑤ 기준선", "무작위 점수 (AUC 0.5)")],
     cells=[
("md", """## 입장: 데이터 받기

한 줄 = '이 사람이 저 사람에게 메일을 보낸 적 있다'. 방향은 무시하고 '연결' 로 본다."""),
("code", """import gzip
edges = pd.read_csv(download("https://snap.stanford.edu/data/email-Eu-core.txt.gz", "data/email-Eu-core.txt.gz"), sep=" ", header=None, names=["a", "b"])
dept = pd.read_csv(download("https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz", "data/email-dept.txt.gz"), sep=" ", header=None, names=["person", "dept"])
edges = edges[edges["a"] != edges["b"]]
pairs = {tuple(sorted(p)) for p in edges[["a", "b"]].itertuples(index=False)}
n = dept["person"].max() + 1
print("people:", n, " undirected links:", len(pairs), " departments:", dept["dept"].nunique())"""),
("md", """## 열어 보기

사람마다 연결 수 (차수) 는? 몇 명에게 몰려 있을까?"""),
("code", """degree = np.zeros(n, dtype=int)
for a, b in pairs:
    degree[a] += 1; degree[b] += 1
print("links per person: median", int(np.median(degree)), "max", degree.max(), "| people with 0 links:", (degree == 0).sum())
print(f"share of all possible pairs that are linked: {len(pairs) / (n * (n - 1) / 2):.2%}")
plt.figure(figsize=(6, 3)); plt.hist(degree, bins=50); plt.xlabel("links per person"); plt.ylabel("people"); plt.show()"""),
("md", """## 문제 하나 골라 보기: 두 사람 → 연결될까

연결 10% 를 **숨겨서** 시험 문제로 쓴다 (= 미래에 생길 연결). 같은 수만큼 연결 안 된 쌍도 섞는다. 남은 90% 연결망만 보고 점수를 매긴다:
- **공통 지인 수**: 둘 다 아는 사람이 많을수록 연결될 것 같다
- **같은 부서인가**
- 기준선: 무작위 점수"""),
("code", """from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
all_pairs = np.array(sorted(pairs))
hidden_idx = rng.choice(len(all_pairs), size=len(all_pairs) // 10, replace=False)
hidden = all_pairs[hidden_idx]
known = {tuple(p) for k, p in enumerate(all_pairs) if k not in set(hidden_idx)}
neighbors = [set() for _ in range(n)]
for a, b in known:
    neighbors[a].add(b); neighbors[b].add(a)

non_links = []
while len(non_links) < len(hidden):
    a, b = sorted(rng.integers(0, n, 2))
    if a != b and (a, b) not in pairs:
        non_links.append((a, b))
test_pairs = np.vstack([hidden, np.array(non_links)])
y = np.r_[np.ones(len(hidden)), np.zeros(len(non_links))]

dept_of = dept.set_index("person")["dept"].to_dict()
common = np.array([len(neighbors[a] & neighbors[b]) for a, b in test_pairs])
same_dept = np.array([dept_of[a] == dept_of[b] for a, b in test_pairs], dtype=float)
print(f"random score           : AUC {roc_auc_score(y, rng.random(len(y))):.2f}")
print(f"same department        : AUC {roc_auc_score(y, same_dept):.2f}")
print(f"number of common friends: AUC {roc_auc_score(y, common):.2f}")"""),
("md", """**ROC-AUC 읽는 법**: 진짜 연결 하나와 연결 아닌 쌍 하나를 무작위로 뽑았을 때, 진짜 쪽 점수가 더 높을 확률. 0.5 = 동전 던지기, 1 = 완벽.

## 다른 문제로 바꿔 보기

- **친구 추천**: 실제 서비스는 '이 쌍이 연결될까' 가 아니라 '이 사람에게 보여 줄 5명' 을 고른다. 한 행 = 사람, 지표 = 추천 5명 중 실제로 연결된 수.
- **시간**: 이 데이터의 시간 정보가 있는 판 (email-Eu-core-temporal) 을 쓰면 '과거 연결로 미래 연결 맞히기' 를 정직하게 할 수 있다.
- 그래프 신경망 (13주차 GCN) 은 공통 지인 같은 규칙을 직접 정하지 않고 배운다."""),
]),
# ------------------------------------------------------------------
dict(slug="bike_demand", title="공유 자전거 수요 예측", sector="플랫폼 · 서비스",
     keywords=["수요예측", "모빌리티 · 공유 서비스", "재배치 운영"],
     teaser="서울 공유 자전거 1년치 시간별 대여 수. 내일 아침 8시엔 몇 대가 필요할까?",
     data="Seoul Bike Sharing Demand", data_src="UCI (Sathishkumar V E & Cho, 2020), 서울 공공자전거 2017.12~2018.11 시간별 대여 수 8,760개 + 날씨",
     size="약 0.6MB",
     five=[("① 한 행", "한 시간"), ("② 입력 X", "시각 · 요일 · 기온 · 습도 · 비 · 휴일 …"), ("③ 출력 y", "그 시간의 대여 수"),
           ("④ 평가지표", "MAE (대)"), ("⑤ 기준선", "학습 기간의 '같은 시각' 평균")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
with zipfile.ZipFile(download(URL, "data/seoul_bike.zip")) as z:
    df = pd.read_csv(z.open("SeoulBikeData.csv"), encoding="latin1")
df.columns = ["date", "count", "hour", "temp", "humidity", "wind", "visibility", "dew_point", "solar", "rain", "snow", "season", "holiday", "functioning"]
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
print(df.shape, df["date"].min().date(), "~", df["date"].max().date())
df.head()"""),
("md", """## 열어 보기

하루 중 언제 많이 빌릴까? 출근 · 퇴근 시간이 보일까?"""),
("code", """df["weekend"] = df["date"].dt.dayofweek >= 5
by_hour = df[df["functioning"] == "Yes"].groupby(["hour", "weekend"])["count"].mean().unstack()
by_hour.columns = ["weekday", "weekend"]
by_hour.plot(figsize=(8, 3), marker="o"); plt.ylabel("bikes rented per hour"); plt.show()
print("hours when the service was off:", (df["functioning"] == "No").sum(), "(count is always", df.loc[df["functioning"] == "No", "count"].max(), ")")"""),
("md", """## 문제 하나 골라 보기: 한 시간 → 대여 수

**시간 순서** 로 나눈다: 2018년 9월 전까지 학습, 9~11월 시험. 기준선 = 학습 기간의 같은 시각 (평일/주말 따로) 평균."""),
("code", """from sklearn.ensemble import HistGradientBoostingRegressor

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

data = df[df["functioning"] == "Yes"].copy()
data["dow"] = data["date"].dt.dayofweek
train, test = data[data["date"] < "2018-09-01"], data[data["date"] >= "2018-09-01"]
hour_mean = train.groupby(["hour", "weekend"])["count"].mean()
base = test.set_index(["hour", "weekend"]).index.map(hour_mean).values
feats = ["hour", "dow", "temp", "humidity", "wind", "visibility", "solar", "rain", "snow"]
m = HistGradientBoostingRegressor(random_state=0).fit(train[feats], train["count"])
print(f"baseline (same hour mean): MAE {mae(test['count'], base):.0f} bikes")
print(f"gradient boosting        : MAE {mae(test['count'], m.predict(test[feats])):.0f} bikes")"""),
("md", """**생각해 보기**: 내일 아침 8시를 예측할 때 '내일 8시의 실제 기온 · 강수량' 을 알 수 있을까? 실제 서비스에서는 **일기예보** 를 입력으로 써야 한다. 이 노트북의 점수는 그래서 실제보다 좋게 나온다.

## 다른 문제로 바꿔 보기

- **재배치 운영**: 서울 전체가 아니라 '대여소 × 시간' 을 한 행으로 하면 '어느 대여소에 자전거를 옮겨 둘까' 문제가 된다.
- 비 오는 날을 많이 틀리는 게 더 손해라면 지표를 어떻게 바꿀까?"""),
]),
# ------------------------------------------------------------------
dict(slug="semiconductor_yield", title="반도체 공정 불량 예측", sector="제조",
     keywords=["수율 · 불량 예측", "센서 데이터", "공정 이상 탐지"],
     teaser="반도체 생산 라인 센서 590개, 제품 1,567개. 센서만 보고 불량을 미리 알 수 있을까?",
     data="SECOM", data_src="UCI (McCann & Johnston, 2008), 반도체 제조 공정 센서 590개 × 1,567개 제품, 불량 104개. 센서 이름은 공개되지 않음",
     size="약 2MB",
     five=[("① 한 행", "제품 (웨이퍼) 하나"), ("② 입력 X", "센서 값 590개 (빈칸 많음)"), ("③ 출력 y", "최종 검사 불량인가"),
           ("④ 평가지표", "재현율 · 정밀도 (불량을 놓치지 않기)"), ("⑤ 기준선", "전부 '양품'")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
with zipfile.ZipFile(download(URL, "data/secom.zip")) as z:
    X = pd.read_csv(z.open("secom.data"), sep=" ", header=None)
    labels = pd.read_csv(z.open("secom_labels.data"), sep=" ", header=None, names=["label", "time"])
X.columns = [f"s{i}" for i in range(X.shape[1])]
y = (labels["label"] == 1).astype(int).values
time = pd.to_datetime(labels["time"], format="%d/%m/%Y %H:%M:%S")
print(X.shape, f"fail rate {y.mean():.1%}", time.min(), "~", time.max())"""),
("md", """## 열어 보기: 공장 데이터는 지저분하다

빈칸은 얼마나? 값이 하나뿐인 (쓸모없는) 센서는? (1주 3차시 '데이터 직접 다루기' 와 같은 점검)"""),
("code", """missing = X.isna().mean()
print(f"cells missing: {X.isna().mean().mean():.1%}")
print("sensors more than half missing:", (missing > 0.5).sum())
print("sensors with a single value:", (X.nunique() <= 1).sum())
print(X.describe().T[["min", "max", "mean"]].sample(8, random_state=0).round(2))"""),
("md", """## 문제 하나 골라 보기: 제품 하나 → 불량일까

**시간 순서** 로 앞 70% 학습, 뒤 30% 시험. 청소: 반 이상 빈 센서 · 값이 하나뿐인 센서는 버리고, 남은 빈칸은 학습 데이터의 중앙값으로. 불량이 드물어서 모델에 '불량 쪽을 더 중요하게' (`class_weight="balanced"`) 를 준다."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

order = np.argsort(time.values); cut = int(len(order) * 0.7)
tr, te = order[:cut], order[cut:]
keep = [c for c in X.columns if X[c].iloc[tr].isna().mean() <= 0.5 and X[c].iloc[tr].nunique() > 1]
med = X[keep].iloc[tr].median()
A, B = X[keep].iloc[tr].fillna(med), X[keep].iloc[te].fillna(med)
print("sensors kept:", len(keep), "| test fails:", y[te].sum(), "of", len(te))

from sklearn.metrics import roc_auc_score
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000)).fit(A, y[tr])
for name, pred in [("all good", np.zeros(len(te), dtype=int)), ("logistic (balanced)", m.predict(B))]:
    p, r = precision_recall(y[te], pred)
    print(f"{name:20s} accuracy {np.mean(pred == y[te]):.1%}  precision {p:.0%}  recall {r:.0%}")

# A factory can re-inspect only some wafers: flag the 20% the model finds most suspicious
score = m.predict_proba(B)[:, 1]
flag = (score >= np.quantile(score, 0.8)).astype(int)
print(f"\\nre-inspect the top 20%: caught {np.sum(flag & y[te])} of {y[te].sum()} fails "
      f"(re-inspecting a random 20% would catch about {0.2 * y[te].sum():.0f})")
print(f"ROC-AUC {roc_auc_score(y[te], score):.2f}  (0.5 = coin flip)")"""),
("md", """**관찰**: 센서가 590개나 있어도 이 모델은 불량을 거의 못 맞힌다 (무작위로 고른 것과 비슷하다). 현실 데이터에서 흔한 결말이다. 기준선과 비교했기 때문에 '모델이 쓸모 있다' 고 착각하지 않을 수 있다.
다음에 할 일은 모델을 키우는 것보다 **문제를 다시 정하는 것** 일 때가 많다: 불량을 더 잘 설명하는 공정 단계의 데이터를 받거나, 엔지니어와 함께 의심 센서를 고르거나, '불량' 의 정의를 나누는 것.
그리고 공장에서는 '놓친 불량 하나' 와 '괜히 다시 검사한 양품 하나' 의 비용을 비교해 재검사 비율을 정한다.

## 다른 문제로 바꿔 보기

- **원인 찾기**: '불량일까' 대신 '불량과 가장 관련 있는 센서 10개' 를 출력으로 정하면? 지표는 무엇일까? (엔지니어가 확인해 줘야 한다)
- **이상 탐지**: 불량 라벨이 없다면? 평소와 다른 제품을 찾는 문제로 바꿀 수 있다 (4주차 k-means · 가우시안)."""),
]),
]

# 학생이 직접 확인해 볼 논문 · 출처 (DOI · arXiv 는 2026-10-15 에 제목까지 확인)
PAPERS = {
    "fashion_commerce": [("Fu et al. (2022) StyleGAN-Human: A Data-Centric Odyssey of Human Generation, ECCV", "https://arxiv.org/abs/2204.11823"),
                         ("데이터 원본: Fashion Product Images (Kaggle, P. Aggarwal)", "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")],
    "card_fraud": [("Dal Pozzolo et al. (2015) Calibrating Probability with Undersampling for Unbalanced Classification, IEEE SSCI", "https://doi.org/10.1109/SSCI.2015.33")],
    "support_routing": [("Casanueva et al. (2020) Efficient Intent Detection with Dual Sentence Encoders (Banking77)", "https://arxiv.org/abs/2003.04807")],
    "shop_conversion": [("Sakar et al. (2019) Real-time prediction of online shoppers' purchasing intention using MLP and LSTM, Neural Computing and Applications", "https://doi.org/10.1007/s00521-018-3523-0")],
    "hotel_booking": [("Antonio, Almeida & Nunes (2019) Hotel booking demand datasets, Data in Brief", "https://doi.org/10.1016/j.dib.2018.11.126")],
    "form_understanding": [("Jaume, Ekenel & Thiran (2019) FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents", "https://arxiv.org/abs/1905.13538")],
    "ai_usecases": [("Handa et al. (2025) Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations", "https://arxiv.org/abs/2503.04761")],
    "movie_ratings": [("Harper & Konstan (2015) The MovieLens Datasets: History and Context, ACM TiiS", "https://doi.org/10.1145/2827872")],
    "korean_reviews": [("NSMC 저장소 (Lucy Park, 2015): 데이터 설명 · 만든 방법", "https://github.com/e9t/nsmc")],
    "link_prediction": [("Liben-Nowell & Kleinberg (2007) The link-prediction problem for social networks, JASIST", "https://doi.org/10.1002/asi.20591"),
                        ("Yin, Benson, Leskovec & Gleich (2017) Local Higher-Order Graph Clustering, KDD (email-Eu-core)", "https://doi.org/10.1145/3097983.3098069")],
    "bike_demand": [("Sathishkumar V E, Park & Cho (2020) Using data mining techniques for bike sharing demand prediction in metropolitan city, Computer Communications", "https://doi.org/10.1016/j.comcom.2020.02.007")],
    "semiconductor_yield": [("UCI SECOM 데이터 설명 (McCann & Johnston, 2008)", "https://archive.ics.uci.edu/dataset/179/secom")],
}

SECTIONS = ["커머스 · 콘텐츠", "금융 · 핀테크", "플랫폼 · 서비스", "제조", "전문 서비스"]
assert set(PAPERS) == {it["slug"] for it in SHOP}
SHOP.sort(key=lambda it: SECTIONS.index(it["sector"]))          # 진열 순서 = 분야 순서 (같은 분야는 적은 순서)
for k, it in enumerate(SHOP):
    it["num"] = f"{k + 1:02d}"
    it["file"] = f"{it['num']}_{it['slug']}"


def notebook(item):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    kw = " · ".join(item["keywords"])
    head = f"""# 현실의 데이터 {item['num']}: {item['title']}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB.format(item['file'])})

**분야**: {item['sector']} &nbsp;|&nbsp; **이 분야에서 먼저 보는 문제**: {kw}

> {item['teaser']}

**데이터**: {item['data']} ({item['size']}). {item['data_src']}.

인턴 · 산학협력 과제에서 처음 받게 될 데이터와 닮은 공개 데이터다. 데이터를 열어 보고, 아래 예시처럼 문제 하나를 정해 기준선과 비교한 뒤, 맨 끝의 **나의 다섯 칸** 을 채운다 (1주 2차시 '문제 정형화와 평가지표').

{five_table(item['five'])}

**논문 · 출처** (직접 열어 보자: 이 데이터를 누가, 왜, 어떻게 만들었나):
""" + "\n".join(f"- [{t}]({u})" for t, u in PAPERS[item["slug"]])
    nb.cells.append(nbf.v4.new_markdown_cell(head))
    nb.cells.append(nbf.v4.new_code_cell(COMMON_IMPORTS))
    for kind, src in item["cells"]:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells.append(nbf.v4.new_markdown_cell(FIVE_MD))
    return nb


def page(items):
    """docs/datashop/index.html: 처음엔 진열대 (분야 · 주제 · 한 줄 질문만), 누르면 데이터 · 다섯 칸 · Colab"""
    data = [dict(num=i["num"], slug=i["slug"], title=i["title"], sector=i["sector"], keywords=i["keywords"], teaser=i["teaser"],
                 data=i["data"], src=i["data_src"], size=i["size"], five=i["five"], papers=PAPERS[i["slug"]],
                 colab=COLAB.format(i["file"]), github=GITHUB.format(i["file"])) for i in items]
    tpl = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read()
    return tpl.replace("/*__SHOP_DATA__*/[]", json.dumps(data, ensure_ascii=False)).replace("/*__SECTIONS__*/[]", json.dumps(SECTIONS, ensure_ascii=False))


if __name__ == "__main__":
    os.chdir(HERE)
    run = "--no-run" not in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]          # 이름 일부를 주면 그 노트북만
    for item in SHOP:
        if only and not any(o in item["slug"] for o in only):
            continue
        nb = notebook(item)
        path = os.path.join(HERE, item["file"] + ".ipynb")
        if run:
            import nbclient
            nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
            outs = [o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream"]
            print(f"== {item['file']}\n" + "\n".join(t.strip() for t in outs if t.strip())[-1500:])
        nbf.write(nb, path)
    out = os.path.join(ROOT, "docs", "datashop", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(page(SHOP))
    print("wrote", out)
