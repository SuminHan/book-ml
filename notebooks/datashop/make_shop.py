"""데이터 가게 (Data Shop): 회사 문제 영역별 공개 데이터 노트북 + 웹 페이지 원본.
   ~/Documents/manim_shorts/_env/v/bin/python make_shop.py          # 노트북 7개를 만들고 실행해 출력까지 저장 + docs/datashop/index.html
   ~/Documents/manim_shorts/_env/v/bin/python make_shop.py --no-run # 실행 없이 만들기만
노트북 = notebooks/datashop/<번호>_<slug>.ipynb (번호는 SECTIONS 순서대로 자동), 웹 = docs/datashop/ (smhanlab.com/book-ml/datashop/): 목록 + 데이터마다 페이지 (previews.py 그림 · 그래프, site_pages.py 틀).
   --pages : 노트북은 그대로 두고 웹 페이지만 다시 만든다 (결과 줄은 저장된 노트북 출력에서 읽는다)
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
dict(slug="card_fraud", title="카드 이상거래 탐지 (보이스피싱 · 자금세탁 등)", sector="금융 · 핀테크",
     keywords=["이상거래탐지 (FDS)", "보이스피싱 · 송금 사기", "자금세탁방지 · 설명가능성"],
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

- **보이스피싱**: 피해자가 직접 송금하는 사기라 카드 도용과 모양이 다르다. 한 행 = 송금 한 건, 입력에 '처음 보내는 계좌인가 · 평소보다 큰 금액인가 · 이체 직후 현금 인출이 있었나' 같은 **맥락** 이 필요하다. 이 데이터에는 그런 열이 없다 (송금 사기를 시뮬레이션한 PaySim 데이터가 있다, Kaggle).
- **자금세탁방지**: 한 행이 '거래' 가 아니라 '계좌' 라면? 이 가게의 '자금세탁 거래 탐지' (IBM AML) 노트북에서 해 볼 수 있다.
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
dict(slug="document_types", title="문서 종류 자동 분류", sector="전문 서비스", keywords=["문서 자동 분류", "업무 자동화 (RPA)", "이미지 분류"],
     teaser="스캔한 문서 1,200장. 편지일까, 영수증일까, 이력서일까?",
     data="RVL-CDIP (종류별 100장 판)", data_src="Harley et al. (2015). 스캔한 업무 문서 흑백 사진 16종류 (편지 · 양식 · 이메일 · 손글씨 · 광고 · 보고서 · 논문 · 사양서 · 파일 폴더 · 뉴스 · 예산 · 송장 · 발표 자료 · 설문 · 이력서 · 메모). 원본 40만 장 중 종류마다 75장씩 고른 판 (Hugging Face jordyvl/rvl_cdip_100_examples_per_class, 학습 800 · 시험 400)",
     size="약 135MB",
     five=[("① 한 행", "문서 사진 한 장"), ("② 입력 X", "줄인 흑백 사진 (48×64) + 줄 · 칸마다 잉크 양"), ("③ 출력 y", "16가지 문서 종류"),
           ("④ 평가지표", "정확도 + 종류별 재현율"), ("⑤ 기준선", "가장 흔한 종류 (모두 같은 수라 1/16)")],
     cells=[
("md", """## 입장: 데이터 받기

약 135MB. 처음엔 1~2분."""),
("code", """from PIL import Image
NAMES = ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification",
         "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
BASE = "https://huggingface.co/api/datasets/jordyvl/rvl_cdip_100_examples_per_class/parquet/default/"
train = pd.read_parquet(download(BASE + "train/0.parquet", "data/rvlcdip100_train.parquet"))
test = pd.read_parquet(download(BASE + "test/0.parquet", "data/rvlcdip100_test.parquet"))
print(train.shape, test.shape)
print(pd.Series([NAMES[i] for i in train["label"]]).value_counts().to_dict())"""),
("md", """## 열어 보기

종류마다 한 장씩. 글자를 읽지 않아도 **모양 (배치)** 만으로 구분되는 종류가 있을까?"""),
("code", """fig, axs = plt.subplots(2, 8, figsize=(16, 5))
for k, ax in enumerate(axs.ravel()):
    i = np.where(train["label"].values == k)[0][0]
    ax.imshow(Image.open(io.BytesIO(train["image"][i]["bytes"])), cmap="gray"); ax.set_title(NAMES[k], fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 문서 사진 → 종류 (16가지)

글자는 읽지 않는다. 사진을 48×64 로 줄이고, 줄마다 · 칸마다 잉크 양 (어디에 글이 몰려 있나) 을 더해 랜덤 포레스트. 데이터가 나눠 둔 학습 · 시험을 그대로 쓴다."""),
("code", """from sklearn.ensemble import RandomForestClassifier

def layout_features(image_bytes):
    \"\"\"Small grayscale picture + ink per row / column of one scanned page\"\"\"
    g = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("L").resize((48, 64)), dtype=np.float32) / 255
    ink = 1 - g
    return np.r_[g.ravel(), ink.mean(1), ink.mean(0), ink.mean(), (ink > 0.5).mean()]

Xtr = np.stack([layout_features(b["bytes"]) for b in train["image"]])
Xte = np.stack([layout_features(b["bytes"]) for b in test["image"]])
m = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(Xtr, train["label"])
pred = m.predict(Xte); y = test["label"].values
print(f"baseline (one class): accuracy {1 / 16:.1%}")
print(f"random forest on page layout: accuracy {np.mean(pred == y):.1%}")
per = pd.Series(pred == y).groupby([NAMES[i] for i in y]).mean().sort_values()
print("hardest:", per.head(4).round(2).to_dict()); print("easiest:", per.tail(4).round(2).to_dict())"""),
("md", """**관찰**: 글자를 하나도 읽지 않고 모양만 봐도 기준선 (6%) 보다 훨씬 잘 맞힌다. 손글씨 · 이메일 · 파일 폴더처럼 모양이 뚜렷한 것은 쉽고, 발표 자료 · 양식 · 송장 · 편지는 서로 많이 헷갈린다. 이런 것은 글자를 읽어야 (OCR) 구분된다.

## 다른 문제로 바꿔 보기

- **업무 자동화**: 회사로 들어오는 문서를 종류별로 나눠 담당 부서로 보내기. 틀리면 어떤 손해가 큰가 (송장을 광고로 분류하면?)
- **글자도 쓰면**: OCR 로 글자를 뽑아 단어 빈도를 더하면 정확도가 얼마나 오를까?
- **더 어려운 문제로**: 종류를 맞힌 뒤 송장에서 '금액 · 날짜 · 업체' 칸을 찾아 읽기 (문서 이해)."""),
]),
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
dict(slug="taxi_demand", title="뉴욕 택시 수요 예측 (시공간)", sector="도시 · 공간", keywords=["시공간 (spatiotemporal) 수요예측", "모빌리티 · 배차", "지도 데이터"],
     teaser="뉴욕 택시 한 달 300만 번. 다음 한 시간, 어느 구역에서 택시를 몇 번 부를까?",
     data="NYC TLC Yellow Taxi Trip Records · 2024년 1월", data_src="뉴욕시 택시 · 리무진 위원회 (TLC) 공개 데이터: 노란 택시 승차 기록 약 296만 건 (승차 · 하차 시각, 구역 번호, 거리, 요금) + 택시 구역 263개 지도 (shapefile)",
     size="약 50MB",
     five=[("① 한 행", "구역 × 한 시간"), ("② 입력 X", "그 구역의 지난 1 · 2시간, 어제 · 지난주 같은 시각 승차 수 + 이웃 구역 승차 수 + 시각 · 요일"),
           ("③ 출력 y", "다음 한 시간 승차 수"), ("④ 평가지표", "MAE (건)"), ("⑤ 기준선", "지난 한 시간 그대로 · 지난주 같은 시각")],
     cells=[
("md", """## 입장: 데이터 받기

승차 기록 (parquet, 약 50MB) + 택시 구역 지도 (shapefile)."""),
("code", """BASE = "https://d37ci6vzurychx.cloudfront.net/"
trips = pd.read_parquet(download(BASE + "trip-data/yellow_tripdata_2024-01.parquet", "data/yellow_2024_01.parquet"),
                        columns=["tpep_pickup_datetime", "PULocationID", "DOLocationID", "trip_distance", "total_amount"])
lookup = pd.read_csv(download(BASE + "misc/taxi_zone_lookup.csv", "data/taxi_zone_lookup.csv"))
print(trips.shape, trips["tpep_pickup_datetime"].min(), "~", trips["tpep_pickup_datetime"].max())
trips.sample(5, random_state=0)"""),
("md", """**관찰**: 2002년 · 2024년 2월 같은 날짜가 섞여 있다 (기계 시계 오류, 1주 3차시 '이상한 값'). 1월만 남긴다."""),
("code", """trips = trips[(trips["tpep_pickup_datetime"] >= "2024-01-01") & (trips["tpep_pickup_datetime"] < "2024-02-01")]
print(len(trips), "trips in January")
print(trips.merge(lookup, left_on="PULocationID", right_on="LocationID")["Zone"].value_counts().head(8))"""),
("md", """## 지도 파일 열어 보기

shapefile 도 결국 숫자 묶음이다 (`.shp` = 구역마다 꼭짓점 좌표, `.dbf` = 구역 이름 표). 라이브러리 없이 직접 읽어 보자."""),
("code", """import struct
from matplotlib.collections import PolyCollection

def read_shp(b):
    \"\"\"Polygons of a .shp file: list (one per record) of point arrays (one per part)\"\"\"
    shapes, pos = [], 100
    while pos < len(b):
        _, length = struct.unpack(">ii", b[pos:pos + 8]); rec = b[pos + 8:pos + 8 + length * 2]; pos += 8 + length * 2
        if struct.unpack("<i", rec[:4])[0] == 0:
            shapes.append([]); continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npts]
        pts = np.frombuffer(rec[44 + 4 * nparts:44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(-1, 2)
        shapes.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return shapes

def read_dbf(b):
    \"\"\"Attribute table of a .dbf file as a DataFrame\"\"\"
    nrec, hlen, rlen = struct.unpack("<IHH", b[4:12]); fields, p = [], 32
    while b[p] != 0x0D:
        fields.append((b[p:p + 11].split(b"\\0")[0].decode(), b[p + 16])); p += 32
    rows = []
    for r in range(nrec):
        rec, o, row = b[hlen + r * rlen + 1:hlen + (r + 1) * rlen], 0, {}
        for name, flen in fields:
            row[name] = rec[o:o + flen].decode("latin1").strip(); o += flen
        rows.append(row)
    return pd.DataFrame(rows)

with zipfile.ZipFile(download(BASE + "misc/taxi_zones.zip", "data/taxi_zones.zip")) as z:
    shapes = read_shp(z.read("taxi_zones/taxi_zones.shp")); zones = read_dbf(z.read("taxi_zones/taxi_zones.dbf"))
zones["LocationID"] = zones["LocationID"].astype(int)
print(len(shapes), "polygons;", zones[["LocationID", "zone", "borough"]].head(3).to_dict("records"))

def draw_zones(ax, values, shapes, zones, cmap="YlOrRd"):
    \"\"\"Colour each taxi zone by values[LocationID]\"\"\"
    polys, vals = [], []
    for s, zid in zip(shapes, zones["LocationID"]):
        for part in s:
            polys.append(part); vals.append(values.get(zid, 0))
    pc = PolyCollection(polys, array=np.log1p(np.array(vals, dtype=float)), cmap=cmap, edgecolor="white", linewidth=0.2)
    ax.add_collection(pc); ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    return pc

fig, ax = plt.subplots(figsize=(7, 7))
pc = draw_zones(ax, trips["PULocationID"].value_counts().to_dict(), shapes, zones)
plt.colorbar(pc, ax=ax, shrink=0.6, label="log(1 + pickups in January)"); plt.show()"""),
("md", """## 시공간 표로 바꾸기: 시간 × 구역

한 행 = (시간, 구역). 744시간 × 263구역. 맨해튼 한가운데와 공항 (JFK · 라과디아) 이 가장 많다."""),
("code", """trips["hour"] = trips["tpep_pickup_datetime"].dt.floor("h")
counts = trips.groupby(["hour", "PULocationID"]).size().unstack(fill_value=0).reindex(columns=range(1, 264), fill_value=0)
print(counts.shape)
top = counts.sum().sort_values(ascending=False).head(4).index
counts[top].iloc[:24 * 7].plot(figsize=(11, 3), lw=1, title="pickups per hour, first week (top 4 zones)")
plt.legend([lookup.set_index("LocationID").loc[z, "Zone"] for z in top], fontsize=8); plt.show()"""),
("md", """## 문제 하나 골라 보기: (구역, 시간) → 다음 한 시간 승차 수

**시간 순서**: 1월 25일 전 학습, 그 뒤 시험. 입력에 **이웃 구역** (지도에서 가까운 5개) 의 지난 한 시간 승차 수를 넣는다 = 공간 정보.
기준선: 지난 한 시간 그대로, 지난주 같은 시각."""),
("code", """from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

centre = pd.DataFrame([np.concatenate(s).mean(0) if s else [np.nan, np.nan] for s in shapes], columns=["x", "y"])
centre["LocationID"] = zones["LocationID"].values
centre = centre.groupby("LocationID").mean().dropna()
_, nb = cKDTree(centre.values).query(centre.values, k=6)
neighbours = {z: [centre.index[j] for j in row[1:]] for z, row in zip(centre.index, nb)}

def make_rows(counts, neighbours):
    \"\"\"(hour, zone) rows with past counts of the zone and of its neighbours\"\"\"
    f = pd.DataFrame({f"lag{k}": counts.shift(k).stack() for k in (1, 2, 24, 168)})
    f["neigh1"] = pd.DataFrame({z: counts[neighbours[z]].sum(1) if z in neighbours else 0 for z in counts.columns}).shift(1).stack()
    f["y"] = counts.stack()
    f = f.dropna().reset_index(); f.columns = ["hour", "zone"] + list(f.columns[2:])
    f["hod"], f["dow"] = f["hour"].dt.hour, f["hour"].dt.dayofweek
    return f

rows = make_rows(counts, neighbours)
tr, te = rows[rows["hour"] < "2024-01-25"].copy(), rows[rows["hour"] >= "2024-01-25"].copy()
zone_mean = tr.groupby("zone")["y"].mean(); tr["zmean"] = tr["zone"].map(zone_mean); te["zmean"] = te["zone"].map(zone_mean)
feats = ["lag1", "lag2", "lag24", "lag168", "neigh1", "hod", "dow", "zmean"]
m = HistGradientBoostingRegressor(random_state=0, max_iter=300).fit(tr[feats], tr["y"])
print(f"test rows {len(te):,}, mean pickups per zone-hour {te['y'].mean():.1f}")
for name, pred in [("last hour", te["lag1"]), ("same hour last week", te["lag168"]), ("gradient boosting", m.predict(te[feats]))]:
    print(f"{name:20s} MAE {np.mean(np.abs(pred - te['y'])):.2f} pickups")"""),
("md", """**관찰**: '지난 한 시간 그대로' 보다 '지난주 같은 시각' 이 낫다 (출퇴근 · 요일 패턴). 모델은 둘을 섞고 이웃 구역까지 봐서 더 낫다. 실제 배차 회사는 날씨 · 행사 (경기 · 공연) 까지 입력에 넣는다.

## 다른 문제로 바꿔 보기

- **배차 · 요금**: '수요' 만이 아니라 '공급 (빈 택시)' 도 구역 × 시간으로 맞히면, 어디로 택시를 보낼지 · 할증을 얼마나 붙일지 정할 수 있다.
- **도착 시간 (ETA)**: 한 행 = 운행 한 번, 입력 = 출발 · 도착 구역 · 시각, 출력 = 걸린 분 (`tpep_dropoff_datetime − tpep_pickup_datetime`).
- **그래프로**: 구역을 점, 이웃을 선으로 두면 13주차 그래프 신경망 문제가 된다."""),
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
# ------------------------------------------------------------------
dict(slug="predictive_maintenance", title="설비 고장 예측 (예지보전)", sector="제조",
     keywords=["예지보전", "설비 고장 예측", "다운타임 줄이기"],
     teaser="밀링 설비 가동 기록 1만 건, 고장은 3.4%. 온도 · 회전 속도 · 공구 마모로 고장을 미리 알 수 있을까?",
     data="AI4I 2020 Predictive Maintenance", data_src="UCI (Matzka, 2020). 실제 밀링 설비를 본떠 만든 합성 데이터 10,000건: 공기 · 공정 온도, 회전 속도, 토크, 공구 마모 시간, 고장 여부와 고장 종류 5가지",
     size="약 0.5MB",
     five=[("① 한 행", "가동 기록 한 건"), ("② 입력 X", "제품 등급 · 온도 · 회전 속도 · 토크 · 공구 마모"), ("③ 출력 y", "고장인가 (Machine failure)"),
           ("④ 평가지표", "정밀도 · 재현율"), ("⑤ 기준선", "늘 '정상'")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
with zipfile.ZipFile(download(URL, "data/ai4i2020.zip")) as z:
    df = pd.read_csv(z.open("ai4i2020.csv"))
print(df.shape)
df.head()"""),
("md", """## 열어 보기

고장은 몇 %? 그리고 `TWF`, `HDF`, `PWF`, `OSF`, `RNF` 는 무엇일까? (공구 마모 · 열 방출 · 전력 · 과부하 · 무작위 고장, 즉 **고장의 종류**)"""),
("code", """print(f"failure rate: {df['Machine failure'].mean():.1%}")
modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print(df[modes].sum().rename("count"))
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, col in zip(axs, ["Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"]):
    for f, name in [(0, "ok"), (1, "failure")]:
        ax.hist(df.loc[df["Machine failure"] == f, col], bins=40, density=True, alpha=0.6, label=name)
    ax.set_xlabel(col); ax.legend()
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 가동 기록 → 고장일까

**함정**: 고장 종류 열 (`TWF` …) 은 고장이 **난 뒤에** 적히는 값이다. 입력에 넣으면 답을 보고 맞히는 셈이다. 넣을 때와 뺄 때를 비교한다."""),
("code", """from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

sensors = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
X = pd.get_dummies(df[sensors + ["Type"] + modes], columns=["Type"], dtype=float)
y = df["Machine failure"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

p, r = precision_recall(yte, np.zeros(len(yte), dtype=int))
print(f"{'always ok':24s} accuracy {np.mean(yte == 0):.1%}  precision {p:.0%}  recall {r:.0%}")
for name, cols in [("sensors only", [c for c in X.columns if c not in modes]), ("+ failure-mode columns", list(X.columns))]:
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr[cols], ytr)
    pred = m.predict(Xte[cols]); p, r = precision_recall(yte, pred)
    print(f"{name:24s} accuracy {np.mean(yte == pred):.1%}  precision {p:.0%}  recall {r:.0%}")"""),
("md", """## 다른 문제로 바꿔 보기

- **예지보전의 진짜 질문**: '지금 고장인가' 가 아니라 '**앞으로 몇 시간 안에** 고장 날까' 다. 그러려면 한 행이 '설비 × 시각' 이고, 미래의 고장을 y 로 붙여야 한다. 이 데이터는 시간 순서가 없는 합성 데이터라 그걸 할 수 없다.
- **고장 종류 맞히기**: y 를 '고장 종류 5가지 중 무엇' 으로 바꾸면? 정비 팀이 무엇을 준비할지 정할 수 있다.
- 이 데이터는 **합성** 이다. 실제 공장 데이터 (예: 반도체 SECOM) 보다 훨씬 깔끔하고 잘 맞는다는 점도 기억하자."""),
]),
# ------------------------------------------------------------------
dict(slug="steel_defects", title="강판 표면 결함 분류", sector="제조",
     keywords=["품질 검사", "결함 종류 분류", "검사 자동화"],
     teaser="철강 판 1,941장, 결함 7종류. 결함 모양을 잰 숫자 27개로 어떤 결함인지 맞힐 수 있을까?",
     data="Steel Plates Faults", data_src="UCI, Semeion 연구소 (Buscema, Terzi & Tastle, 2010). 강판 결함 1,941개, 결함 영역의 위치 · 면적 · 둘레 · 밝기 · 강판 종류 · 두께 등 27개, 결함 7종류",
     size="약 0.1MB",
     five=[("① 한 행", "결함 하나"), ("② 입력 X", "결함 영역의 크기 · 모양 · 밝기 · 강판 종류 · 두께 (27개)"),
           ("③ 출력 y", "결함 종류 7가지"), ("④ 평가지표", "정확도 + 종류별 재현율"), ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기

숫자 표 (`Faults.NNA`) 와 열 이름 (`Faults27x7_var`) 이 따로 들어 있다. 마지막 7개 열이 결함 종류 (하나만 1)."""),
("code", """URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
with zipfile.ZipFile(download(URL, "data/steel_faults.zip")) as z:
    names = z.read("Faults27x7_var").decode().split()
    df = pd.read_csv(z.open("Faults.NNA"), sep=r"\\s+", header=None, names=names)
faults = names[-7:]
df["fault"] = df[faults].idxmax(axis=1)
print(df.shape)
print(df["fault"].value_counts())"""),
("md", """## 열어 보기

결함 종류마다 크기 (`Pixels_Areas`) 와 밝기 (`Sum_of_Luminosity`) 가 다를까?"""),
("code", """print(df.groupby("fault")[["Pixels_Areas", "Steel_Plate_Thickness", "Maximum_of_Luminosity"]].median().round(0))
plt.figure(figsize=(7, 4))
for f in df["fault"].unique():
    s = df[df["fault"] == f]
    plt.scatter(np.log10(s["Pixels_Areas"]), s["Maximum_of_Luminosity"], s=8, alpha=0.6, label=f)
plt.xlabel("log10(defect area in pixels)"); plt.ylabel("max luminosity"); plt.legend(fontsize=8); plt.show()"""),
("md", """## 문제 하나 골라 보기: 결함 → 종류 (7가지 분류)

기준선 = 가장 흔한 종류. 모델 = 랜덤 포레스트 (7주차). 무엇을 많이 틀릴까?"""),
("code", """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = df[names[:-7]], df["fault"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
most_common = ytr.value_counts().idxmax()
m = RandomForestClassifier(500, random_state=0).fit(Xtr, ytr)
pred = m.predict(Xte)
print(f"baseline (always '{most_common}'): accuracy {np.mean(yte == most_common):.1%}")
print(f"random forest                   : accuracy {np.mean(yte == pred):.1%}")
print("recall per fault:", pd.Series(yte.values == pred).groupby(yte.values).mean().round(2).to_dict())"""),
("md", """**관찰**: 'Pastry' 가 가장 많이 틀린다. 'Other_Faults' (기타 결함) 는 가장 많지만, '기타' 라는 라벨은 사람이 분류하기 애매했던 것들의 모음이라 경계가 흐리다. 라벨을 어떻게 나눴는지도 문제 정의의 일부다.

## 다른 문제로 바꿔 보기

- **검사 자동화**: 현장에서는 이 27개 숫자가 아니라 **카메라 사진** 이 들어온다. 한 행 = 사진, y = 결함 위치 (상자) + 종류로 바꾸면? (10주차 CNN)
- '정상 강판' 이 이 데이터에는 없다. 결함 여부부터 가려야 한다면 어떤 데이터가 더 필요할까?
- 결함 종류마다 비용이 다르다면 (예: 긁힘은 재가공, 얼룩은 폐기) 지표를 어떻게 바꿀까?"""),
]),
# ------------------------------------------------------------------
dict(slug="bearing_vibration", title="베어링 진동으로 고장 진단", sector="제조",
     keywords=["진동 · 센서 신호", "고장 진단", "상태 기반 정비"],
     teaser="모터 베어링에 붙인 진동 센서, 1초에 1만 2천 번. 소리만 듣고 어디가 망가졌는지 알 수 있을까?",
     data="CWRU Bearing Data", data_src="Case Western Reserve University Bearing Data Center. 모터 베어링 진동 (12 kHz) 녹음: 정상 · 내륜 · 볼 · 외륜 결함 (지름 0.007인치, 부하 0) 4개 파일",
     size="약 13MB",
     five=[("① 한 행", "진동 조각 하나 (1,024 샘플 ≈ 0.085초)"), ("② 입력 X", "조각의 통계 (RMS · 첨도 · 최댓값 …) 또는 파형 그대로"),
           ("③ 출력 y", "정상 / 내륜 / 볼 / 외륜"), ("④ 평가지표", "정확도 (네 종류 조각 수가 비슷하다)"), ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기

`.mat` (MATLAB) 파일 4개. 각 파일 = 같은 모터를 한 상태로 10초 정도 돌린 진동 녹음. `DE` = 구동 쪽 (drive end) 센서."""),
("code", """from scipy.io import loadmat
FILES = {"normal": 97, "inner race": 105, "ball": 118, "outer race": 130}
signals = {}
for state, num in FILES.items():
    mat = loadmat(download(f"https://engineering.case.edu/sites/default/files/{num}.mat", f"data/cwru_{num}.mat"))
    key = [k for k in mat if k.endswith("DE_time")][0]
    signals[state] = mat[key].ravel()
    print(f"{state:11s} {len(signals[state]):7d} samples = {len(signals[state]) / 12000:.1f} s")"""),
("md", """## 열어 보기

같은 0.05초를 네 상태로 그려 보자. 눈으로 구분이 될까?"""),
("code", """fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=True)
t = np.arange(600) / 12000
for ax, (state, s) in zip(axs, signals.items()):
    ax.plot(t, s[:600], lw=0.8); ax.set_ylabel(state, rotation=0, ha="right")
axs[-1].set_xlabel("seconds"); plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 진동 조각 → 상태

녹음을 1,024 샘플씩 잘라 한 행으로 만든다. 조각마다 통계 6개: RMS (세기), 최댓값, 첨도 (뾰족한 충격이 있나), 왜도, 파고율 (최댓값 ÷ RMS), 영점 교차 수.
**나누기 주의**: 같은 녹음의 이웃한 조각은 거의 같다. 무작위로 섞으면 시험이 너무 쉬워진다. 그래서 녹음마다 **앞 70% 시간 → 학습, 뒤 30% → 시험**."""),
("code", """from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier

def chunk_features(x):
    \"\"\"Six simple statistics of one vibration chunk\"\"\"
    rms = np.sqrt(np.mean(x ** 2))
    return [rms, np.max(np.abs(x)), kurtosis(x), skew(x), np.max(np.abs(x)) / rms, np.sum(np.diff(np.sign(x)) != 0)]

def make_rows(signals, size=1024, train_share=0.7):
    rows = []
    for state, s in signals.items():
        n = len(s) // size
        for k in range(n):
            rows.append(chunk_features(s[k * size:(k + 1) * size]) + [state, "train" if k < n * train_share else "test"])
    return pd.DataFrame(rows, columns=["rms", "peak", "kurtosis", "skew", "crest", "zero_cross", "state", "split"])

data = make_rows(signals)
print(data.groupby(["split", "state"]).size().unstack())
tr, te = data[data["split"] == "train"], data[data["split"] == "test"]
feats = ["rms", "peak", "kurtosis", "skew", "crest", "zero_cross"]
most_common = tr["state"].value_counts().idxmax()
m = RandomForestClassifier(300, random_state=0).fit(tr[feats], tr["state"])
print(f"\\nbaseline (always '{most_common}'): accuracy {np.mean(te['state'] == most_common):.1%}")
print(f"random forest on 6 statistics : accuracy {np.mean(te['state'] == m.predict(te[feats])):.1%}")
print(data.groupby("state")[["rms", "kurtosis"]].mean().round(2))"""),
("md", """**관찰**: 실험실에서 한 모터로 잰 깨끗한 데이터라 아주 잘 맞는다. 현실에서는 다른 모터, 다른 부하, 공장 소음이 섞인다. **학습한 설비와 다른 설비로 시험** 하면 점수가 어떻게 될까? (이 데이터의 부하 1~3 마력 파일로 해 볼 수 있다: Smith & Randall 2015 참고)

## 다른 문제로 바꿔 보기

- **상태 기반 정비**: '어디가 망가졌나' 대신 '결함이 얼마나 큰가 (0.007 / 0.014 / 0.021 인치)' 를 맞히면 회귀 문제가 된다.
- 결함 데이터가 없는 새 설비라면? 정상 진동만으로 '평소와 다르다' 를 찾는 이상 탐지로 바꾼다."""),
]),
# ------------------------------------------------------------------
dict(slug="aml_transactions", title="자금세탁 거래 탐지", sector="금융 · 핀테크",
     keywords=["자금세탁방지 (AML)", "거래 모니터링", "계좌 단위 조사"],
     teaser="은행 거래 508만 건 중 세탁은 0.1%. 거래를 볼까, 계좌를 볼까?",
     data="IBM Transactions for Anti Money Laundering (HI-Small)", data_src="IBM (Altman et al., NeurIPS 2023). 사람 · 회사 · 은행이 있는 가상 세계를 시뮬레이션한 합성 거래 5,078,345건, 세탁 표시 포함 (Kaggle 원본, 여기서는 Hugging Face 사본으로 로그인 없이 받는다)",
     size="약 90MB (압축)",
     five=[("① 한 행", "거래 한 건 (또는 계좌 하나)"), ("② 입력 X", "시각 · 보낸/받는 은행 · 계좌 · 금액 · 통화 · 결제 형식"),
           ("③ 출력 y", "세탁인가 (Is Laundering)"), ("④ 평가지표", "정밀도 · 재현율"), ("⑤ 기준선", "전부 '정상'")],
     cells=[
("md", """## 입장: 데이터 받기

압축 파일 약 90MB, 풀면 약 480MB. 처음엔 몇 분 걸린다 (Colab 메모리로 충분하다)."""),
("code", """URL = "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip"
with zipfile.ZipFile(download(URL, "data/HI-Small_Trans.csv.zip")) as z:
    df = pd.read_csv(z.open("HI-Small_Trans.csv"))
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
print(df.shape, df["Timestamp"].min(), "~", df["Timestamp"].max())
df.sample(5, random_state=0)"""),
("md", """## 열어 보기

세탁 비율, 결제 형식 · 통화별 세탁 비율. 날짜마다 거래 수는?"""),
("code", """y = df["Is Laundering"].values
print(f"transactions {len(df):,}, laundering {y.sum():,} ({y.mean():.3%})")
print(df.groupby("Payment Format")["Is Laundering"].agg(["mean", "sum", "size"]).sort_values("mean", ascending=False))
daily = df.groupby(df["Timestamp"].dt.date)["Is Laundering"].agg(["size", "sum"])
daily.plot(subplots=True, figsize=(9, 4), title=["transactions per day", "laundering per day"]); plt.tight_layout(); plt.show()"""),
("md", """## 문제 1: 한 행 = 거래 한 건

기준선 = 전부 '정상'. 규칙 하나 = 세탁 비율이 가장 높은 결제 형식이면 '세탁'."""),
("code", """def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

fmt = df.groupby("Payment Format")["Is Laundering"].mean().idxmax()
for name, pred in [("always normal", np.zeros(len(y), dtype=int)), (f"rule: {fmt}", (df["Payment Format"] == fmt).values.astype(int))]:
    p, r = precision_recall(y, pred)
    print(f"{name:16s} accuracy {np.mean(pred == y):.3%}  precision {p:.1%}  recall {r:.1%}")"""),
("md", """## 문제 2: 같은 파일, 한 행 = 보낸 계좌 하나

은행은 거래 하나하나를 막기보다 수상한 **계좌** 를 조사한다. 세탁 거래를 한 번이라도 보냈으면 1."""),
("code", """sender = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
acct = df.groupby(sender).agg(n_tx=("Is Laundering", "size"), y=("Is Laundering", "max"),
                              total=("Amount Paid", "sum"), n_formats=("Payment Format", "nunique"))
print(f"accounts {len(acct):,}, laundering accounts {acct['y'].sum():,} ({acct['y'].mean():.2%})")
print(f"rate: transaction level {y.mean():.3%}  ->  account level {acct['y'].mean():.2%}")
print(acct.groupby("y")[["n_tx", "n_formats"]].median())"""),
("md", """**관찰**: 같은 파일인데 한 행을 무엇으로 정하느냐에 따라 세탁 비율이 달라진다. 논문마다 '세탁 비율' 이 다르게 적힌 이유다.

## 다른 문제로 바꿔 보기

- **시간 순서**: 앞 날짜로 학습, 뒤 날짜로 시험해야 한다. 같은 계좌의 거래가 학습 · 시험 양쪽에 있으면?
- **그래프**: 세탁은 '돈이 계좌 여러 개를 돌아 다시 모이는' 모양 (원형, 부채꼴) 으로 나타난다. 한 행을 '거래' 가 아니라 '연결망의 모양' 으로 보면 그래프 신경망 (13주차) 문제가 된다. 원 논문이 이렇게 했다."""),
]),
# ------------------------------------------------------------------
dict(slug="real_estate", title="부동산: 집값 예측", sector="도시 · 공간",
     keywords=["부동산 시세 추정", "공간 데이터", "자동 가치 평가 (AVM)"],
     teaser="캘리포니아 2만 640개 동네의 소득 · 방 수 · 위치. 이 동네 집값은 얼마일까?",
     data="California Housing", data_src="Pace & Barry (1997), 1990년 미국 인구조사의 캘리포니아 블록 그룹 20,640개: 소득 중앙값 · 집 나이 · 방 수 · 인구 · 위도 · 경도 → 집값 중앙값 (10만 달러 단위, 5.0 에서 잘림). scikit-learn 이 내려받는다",
     size="약 0.5MB",
     five=[("① 한 행", "동네 (블록 그룹) 하나"), ("② 입력 X", "소득 · 집 나이 · 방 수 · 인구 · 위도 · 경도"), ("③ 출력 y", "집값 중앙값 (10만 달러)"),
           ("④ 평가지표", "MAE (10만 달러)"), ("⑤ 기준선", "전체 평균 집값")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """from sklearn.datasets import fetch_california_housing
house = fetch_california_housing(as_frame=True, data_home="data")
df = house.frame
print(df.shape)
df.sample(5, random_state=0)"""),
("md", """## 열어 보기: 지도로 그리기

위도 · 경도로 점을 찍고 색 = 집값. 해안가 (샌프란시스코 · 로스앤젤레스) 가 비싸다. 집값 5.0 에 몰린 점들은 무엇일까?"""),
("code", """fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
sc = axs[0].scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], s=2, cmap="viridis")
plt.colorbar(sc, ax=axs[0], label="median house value ($100k)"); axs[0].set_xlabel("longitude"); axs[0].set_ylabel("latitude")
axs[1].hist(df["MedHouseVal"], bins=50); axs[1].set_xlabel("median house value ($100k)")
plt.tight_layout(); plt.show()
print("blocks at the cap 5.0:", (df["MedHouseVal"] >= 5.0).sum())"""),
("md", """## 문제 하나 골라 보기: 동네 → 집값 (회귀)

**나누는 방법 두 가지** 를 비교한다.
- 무작위로 80 / 20
- **지역으로**: 경도 −121 보다 서쪽 (북부 해안) 을 시험으로. 모델이 '처음 보는 지역' 에서도 맞힐까?"""),
("code", """from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
splits = {"random 80/20": train_test_split(X.index, test_size=0.2, random_state=0),
          "by region (west of -121 = test)": (X.index[X["Longitude"] >= -121], X.index[X["Longitude"] < -121])}
for name, (tr, te) in splits.items():
    m = HistGradientBoostingRegressor(random_state=0).fit(X.loc[tr], y.loc[tr])
    print(f"{name:32s} baseline MAE {mae(y.loc[te], np.full(len(te), y.loc[tr].mean())):.2f}   model MAE {mae(y.loc[te], m.predict(X.loc[te])):.2f}  (x $100k)")"""),
("md", """**관찰**: 무작위로 나누면 이웃 동네가 학습에 있어서 쉽다. 처음 보는 지역에서는 오차가 커진다. 부동산 · 지도 데이터는 **어떻게 나누느냐** 가 점수를 바꾼다.

## 다른 문제로 바꿔 보기

- 한국 아파트라면? 국토교통부 실거래가 (API 신청 필요): 한 행 = 거래 한 건, y = 거래 가격, 입력 = 면적 · 층 · 건축 연도 · 위치 · **거래 날짜**. 시간 순서로 나눠야 한다.
- 집값 5.0 에서 잘린 동네는 '5.0 이상' 이라는 뜻이다. 이걸 그대로 정답으로 쓰면 무엇이 문제일까?"""),
]),
# ------------------------------------------------------------------
dict(slug="earthquakes", title="지진: 시공간 데이터", sector="도시 · 공간",
     keywords=["시공간 (spatiotemporal) 데이터", "재난 · 안전", "위험 지도"],
     teaser="2023~2024년 전 세계 규모 4.5 이상 지진 1만 4천 개. 다음 주, 어느 지역에서 지진이 날까?",
     data="USGS Earthquake Catalog (ComCat)", data_src="미국 지질조사국 (USGS) ANSS 통합 지진 목록, FDSN API 로 2023-01-01 ~ 2024-12-31 규모 4.5 이상 14,048개: 시각 · 위도 · 경도 · 깊이 · 규모",
     size="약 3MB",
     five=[("① 한 행", "격자 칸 (위도 10° × 경도 10°) × 한 주"), ("② 입력 X", "그 칸의 지난 지진 수 (지난주 · 지난 4주 · 지난해)"),
           ("③ 출력 y", "다음 주에 규모 4.5 이상 지진이 있나"), ("④ 평가지표", "정밀도 · 재현율, ROC-AUC"), ("⑤ 기준선", "'지난주에 있었으면 이번 주도'")],
     cells=[
("md", """## 입장: 데이터 받기

USGS 의 공개 API 로 바로 받는다 (2년, 규모 4.5 이상)."""),
("code", """URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2023-01-01&endtime=2025-01-01"
       "&minmagnitude=4.5&orderby=time-asc")
df = pd.read_csv(download(URL, "data/usgs_2023_2025.csv"), parse_dates=["time"])
print(df.shape)
df[["time", "latitude", "longitude", "depth", "mag", "place"]].sample(5, random_state=0)"""),
("md", """## 열어 보기: 지도 위의 점 (시간 = 색)

지구의 판 경계 (불의 고리) 가 보인다. 한 달씩 넘겨 보면 점이 어떻게 움직일까? (웹 페이지의 움직이는 그림)"""),
("code", """fig, axs = plt.subplots(1, 2, figsize=(13, 4))
t = (df["time"] - df["time"].min()).dt.days
sc = axs[0].scatter(df["longitude"], df["latitude"], c=t, s=(df["mag"] - 4) ** 2 * 6, cmap="viridis", alpha=0.6)
plt.colorbar(sc, ax=axs[0], label="days since 2023-01-01"); axs[0].set_xlabel("longitude"); axs[0].set_ylabel("latitude")
axs[1].hist(df["mag"], bins=np.arange(4.5, 8.2, 0.1)); axs[1].set_yscale("log"); axs[1].set_xlabel("magnitude")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: (격자 칸, 주) → 다음 주에 지진이 있나

지구를 10° × 10° 칸으로 나누고, 칸 × 주마다 지진 수를 센다. **2023년으로 학습, 2024년으로 시험** (시간 순서).
- 기준선 1: 전부 '없음'
- 기준선 2: 지난주에 있었으면 '있음'
- 점수: 2023년에 그 칸에서 지진이 난 주의 비율 (자주 나는 곳일수록 높게)"""),
("code", """from sklearn.metrics import roc_auc_score

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

df["cell"] = ((df["latitude"] + 90) // 10).astype(int) * 36 + ((df["longitude"] + 180) // 10).astype(int)
df["week"] = ((df["time"] - pd.Timestamp("2023-01-02", tz="UTC")).dt.days // 7).clip(lower=0)
counts = df.groupby(["cell", "week"]).size().unstack(fill_value=0).reindex(columns=range(df["week"].max() + 1), fill_value=0)
has = (counts > 0).astype(int)
print("cells with at least one quake:", len(has), "| weeks:", has.shape[1])

train_weeks, test_weeks = range(1, 52), range(53, has.shape[1])
rate_2023 = has[list(range(52))].mean(axis=1)                     # share of 2023 weeks with a quake
y_test = np.concatenate([has[w].values for w in test_weeks])
last_week = np.concatenate([has[w - 1].values for w in test_weeks])
score = np.concatenate([rate_2023.values for w in test_weeks])
for name, pred in [("always none", np.zeros_like(y_test)), ("same as last week", last_week), ("2023 rate > 0.5", (score > 0.5).astype(int))]:
    p, r = precision_recall(y_test, pred)
    print(f"{name:18s} accuracy {np.mean(pred == y_test):.1%}  precision {p:.0%}  recall {r:.0%}")
print(f"2023 rate as a score: ROC-AUC {roc_auc_score(y_test, score):.2f}")"""),
("md", """**관찰**: '어디서 자주 나는가' 는 꽤 맞힌다. 하지만 이건 **언제** 큰 지진이 날지를 맞히는 게 아니다. 큰 지진의 시각을 미리 맞히는 방법은 아직 없다. 문제를 무엇으로 정했는지 정확히 말해야 하는 이유다.

## 다른 문제로 바꿔 보기

- **조기 경보**: 한 행 = 지진파가 처음 잡힌 몇 초, y = 몇 초 뒤 큰 흔들림이 올 지역. 실제 지진 조기 경보가 이 문제다 (지진파 데이터 STEAD 등).
- 격자를 1° 로 줄이면? 한 행이 늘고 '있음' 은 더 드물어진다. 지표는 어떻게 바뀔까?
- 다른 시공간 데이터: 교통 속도 (도로 × 5분), 미세먼지 (측정소 × 시간), 공유 자전거 (대여소 × 시간)."""),
]),
# ------------------------------------------------------------------
dict(slug="stock_index", title="주가 지수: 내일 오를까?", sector="금융 · 핀테크",
     keywords=["시계열 예측", "퀀트 · 투자", "리스크 관리"],
     teaser="코스피 10년치 매일 종가. 오늘까지의 움직임으로 내일 오를지 맞힐 수 있을까?",
     data="KOSPI 일별 지수 (^KS11)", data_src="Yahoo Finance 의 공개 차트 API 로 받은 코스피 지수 10년치 일별 시가 · 고가 · 저가 · 종가 · 거래량 (데이터 제공 조건은 Yahoo 이용 약관을 따른다)",
     size="약 0.3MB",
     five=[("① 한 행", "거래일 하루"), ("② 입력 X", "지난 5일의 수익률 (그날까지 알 수 있는 것만)"), ("③ 출력 y", "다음 거래일에 오르나 (분류) / 수익률 (회귀)"),
           ("④ 평가지표", "정확도 · MAE"), ("⑤ 기준선", "늘 '오른다' / 늘 0%")],
     cells=[
("md", """## 입장: 데이터 받기

Yahoo Finance 의 공개 차트 주소에서 JSON 으로 받는다. 받기에 실패하면 잠시 뒤 다시 실행한다."""),
("code", """import json
URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EKS11?range=10y&interval=1d"
raw = json.load(open(download(URL, "data/yahoo_KS11.json")))["chart"]["result"][0]
q = raw["indicators"]["quote"][0]
df = pd.DataFrame({"date": pd.to_datetime(raw["timestamp"], unit="s").normalize(), "close": q["close"], "volume": q["volume"]}).dropna()
df = df.drop_duplicates("date").set_index("date")
print(df.shape, df.index.min().date(), "~", df.index.max().date())
df.sample(5, random_state=0).sort_index()"""),
("md", """## 열어 보기

종가와 **하루 수익률** (오늘 종가 ÷ 어제 종가 − 1). 수익률은 0 근처에서 위아래로 흔들린다."""),
("code", """df["ret"] = df["close"].pct_change()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
df["close"].plot(ax=axs[0], title="KOSPI close")
axs[1].hist(df["ret"].dropna() * 100, bins=80); axs[1].set_xlabel("daily return (%)")
plt.tight_layout(); plt.show()
print(f"up days: {(df['ret'] > 0).mean():.1%}")"""),
("md", """## 문제 하나 골라 보기: 지난 5일 → 내일 오를까

**시간 순서** 로 나눈다: 마지막 2년이 시험. 입력은 **그날까지 알 수 있는 것만** (내일 값을 쓰면 누수).
기준선: 늘 '오른다' (학습 기간에 오른 날이 더 많았다면)."""),
("code", """from sklearn.linear_model import LogisticRegression

lags = pd.concat({f"ret_lag{k}": df["ret"].shift(k - 1) for k in range(1, 6)}, axis=1)      # lag1 = today's return
data = lags.assign(up_next=(df["ret"].shift(-1) > 0).astype(int), ret_next=df["ret"].shift(-1)).dropna()
cut = data.index.max() - pd.DateOffset(years=2)
tr, te = data[data.index <= cut], data[data.index > cut]
feats = list(lags.columns)

m = LogisticRegression().fit(tr[feats], tr["up_next"])
always_up = int(tr["up_next"].mean() > 0.5)
same_as_today = (te["ret_lag1"] > 0).astype(int)
for name, pred in [(f"always {'up' if always_up else 'down'}", np.full(len(te), always_up)), ("same as today", same_as_today),
                   ("logistic on 5 days", m.predict(te[feats]))]:
    print(f"{name:20s} accuracy {np.mean(pred == te['up_next']):.1%}")
print(f"regression baseline (predict 0%): MAE {np.mean(np.abs(te['ret_next'])) * 100:.2f}%p")"""),
("md", """**관찰**: 지난 5일을 본 모델은 '늘 오른다' 기준선을 넘지 못한다 (모델이 사실상 늘 '오른다' 고 답한다). 그 기준선의 점수도 시험 기간 2년 동안 지수가 많이 올라서 높게 나온 것이라, 다른 기간이면 달라진다. 누구나 쓸 수 있는 과거 가격만으로는 내일을 맞히기 어렵다 (효율적 시장 가설). 기준선과 비교하지 않으면 '정확도 58%' 를 대단한 결과로 착각하기 쉽다.

## 다른 문제로 바꿔 보기

- **리스크 관리**: '오를까' 대신 '내일 얼마나 크게 흔들릴까 (변동성)' 는 꽤 잘 맞힌다. 큰 하루 뒤에는 큰 하루가 오기 쉽다.
- 다른 지수: `%5EIXIC` (나스닥 종합) 로 바꿔 보자. 결과가 비슷할까?
- **주의**: 시험 기간을 여러 번 바꿔 가며 가장 좋은 결과만 고르면, 그것도 미래를 엿본 셈이다."""),
]),
# ------------------------------------------------------------------
dict(slug="point_cloud", title="3D 포인트 클라우드", sector="이미지 · 소리 · 3D", keywords=["3D 비전", "LiDAR · 점 구름", "물체 인식"],
     teaser="점 2,048개만 보고 의자인지 비행기인지 맞힐 수 있을까?",
     data="ModelNet40 (점 2,048개 판, 시험 묶음 2,468개)", data_src="Princeton (Wu et al., 2015). CAD 물체 40종류의 표면에서 점 2,048개를 뽑은 판 (Hugging Face jxie/modelnet40-2048). 여기서는 시험 묶음 2,468개만 받아 다시 나눈다",
     size="약 60MB",
     five=[("① 한 행", "물체 하나 (점 2,048개 × x, y, z)"), ("② 입력 X", "점 구름에서 뽑은 모양 숫자 (퍼진 방향 · 크기 · 거리 분포)"), ("③ 출력 y", "40종류 중 무엇"),
           ("④ 평가지표", "정확도"), ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://huggingface.co/api/datasets/jxie/modelnet40-2048/parquet/default/test/0.parquet"
df = pd.read_parquet(download(URL, "data/modelnet40_test.parquet"))
NAMES = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", "curtain", "desk", "door",
         "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano",
         "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]
points = np.stack([np.stack(p) for p in df["inputs"]]).astype(np.float32)      # (objects, 2048, 3)
labels = df["label"].values
print(points.shape, "classes:", len(np.unique(labels)))
print(pd.Series([NAMES[i] for i in labels]).value_counts().head(8))"""),
("md", """## 열어 보기: 점 구름을 3D 로

점 2,048개가 물체의 겉면 모양을 이룬다. 몇 개를 그려 보자."""),
("code", """fig = plt.figure(figsize=(14, 3.2))
for k, i in enumerate([0, 300, 700, 1100, 1500, 2000]):
    ax = fig.add_subplot(1, 6, k + 1, projection="3d")
    p = points[i]; ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=1, c=p[:, 1], cmap="viridis")
    ax.set_title(NAMES[labels[i]], fontsize=10); ax.set_axis_off()
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 점 구름 → 종류 (40가지)

점의 순서는 의미가 없다 (같은 물체도 점 순서는 마음대로). 그래서 **순서와 상관없는 숫자** 를 뽑는다:
- 가운데로 옮기고 크기를 맞춘 뒤 (1주 3차시 '정규화'), 퍼진 방향 3개의 크기 (공분산 고윳값)
- 가로 · 세로 · 높이 길이
- 점 두 개 사이 거리의 분포 (D2 모양 분포), 중심에서의 거리 분포

그리고 랜덤 포레스트. 기준선 = 가장 흔한 종류."""),
("code", """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def shape_features(p, rng):
    \"\"\"Order-free numbers describing one point cloud (2048 x 3)\"\"\"
    p = p - p.mean(0); p = p / np.abs(p).max()
    eig = np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1]
    ext = np.sort(p.max(0) - p.min(0))[::-1]
    i, j = rng.integers(0, len(p), (2, 3000))
    d2 = np.histogram(np.linalg.norm(p[i] - p[j], axis=1), bins=16, range=(0, 2), density=True)[0]
    rad = np.histogram(np.linalg.norm(p, axis=1), bins=16, range=(0, 1.8), density=True)[0]
    hgt = np.histogram(p[:, 1], bins=8, range=(-1, 1), density=True)[0]
    return np.r_[eig / eig.sum(), ext, d2, rad, hgt]

rng = np.random.default_rng(0)
X = np.stack([shape_features(p, rng) for p in points])
Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, stratify=labels, random_state=0)
m = RandomForestClassifier(500, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pred = m.predict(Xte)
most = np.bincount(ytr).argmax()
print(f"baseline (always {NAMES[most]}): accuracy {np.mean(yte == most):.1%}")
print(f"random forest on shape numbers : accuracy {np.mean(yte == pred):.1%}")
per = pd.Series(yte == pred).groupby([NAMES[i] for i in yte]).mean().sort_values()
print("hardest:", per.head(5).round(2).to_dict())
print("easiest:", per.tail(5).round(2).to_dict())"""),
("md", """**관찰**: 손으로 만든 모양 숫자만으로도 72% 쯤 맞힌다. 기타 · 자동차 · 노트북처럼 모양이 뚜렷한 것은 거의 다 맞히고, xbox · 컵 · 계단처럼 **데이터가 20개뿐인 종류** 는 거의 못 맞힌다. 종류마다 데이터 수가 다르면 정확도 하나로는 이런 차이가 가려진다. 점 구름을 그대로 받는 신경망 (PointNet 등) 은 90% 안팎까지 간다.

## 다른 문제로 바꿔 보기

- **자율주행 · 로봇의 LiDAR**: 한 행이 '물체 하나' 가 아니라 '장면 전체 (수십만 점)' 가 되면, 출력은 '점마다 무엇인가 (분할)' 나 '물체마다 상자 (검출)' 가 된다.
- 점을 2,048개가 아니라 256개만 쓰면? 센서가 싸질수록 점이 줄어든다."""),
]),
# ------------------------------------------------------------------
dict(slug="medical", title="흉부 X-ray 폐렴 찾기", sector="이미지 · 소리 · 3D", keywords=["의료 영상", "진단 보조", "놓치면 안 되는 문제"],
     teaser="X-ray 사진 한 장에서 폐렴을 찾을 수 있을까?",
     data="MedMNIST · PneumoniaMNIST", data_src="Yang et al. (2023). 소아 흉부 X-ray 5,856장을 28×28 로 줄인 판 (학습 4,708 · 검증 524 · 시험 624), 정상 / 폐렴. Hugging Face 사본에서 받는다",
     size="약 4MB",
     five=[("① 한 행", "X-ray 사진 한 장 (28×28)"), ("② 입력 X", "흑백 픽셀 784개"), ("③ 출력 y", "정상 0 / 폐렴 1"),
           ("④ 평가지표", "재현율 (놓친 환자) · ROC-AUC"), ("⑤ 기준선", "가장 흔한 쪽 (폐렴)")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/main/data/pneumoniamnist.npz"
z = np.load(download(URL, "data/pneumoniamnist.npz"))
Xtr, ytr = z["train_images"], z["train_labels"].ravel()
Xte, yte = z["test_images"], z["test_labels"].ravel()
print(Xtr.shape, Xte.shape, Xtr.dtype, Xtr.min(), Xtr.max())
print(f"pneumonia share: train {ytr.mean():.1%}, test {yte.mean():.1%}")"""),
("md", """## 열어 보기

정상 · 폐렴 사진을 몇 장씩. 28×28 로 줄였는데도 차이가 보일까? 두 무리의 '평균 사진' 도 그려 보자."""),
("code", """fig, axs = plt.subplots(2, 7, figsize=(12, 3.8))
for r, lab in enumerate([0, 1]):
    idx = np.where(ytr == lab)[0][:6]
    for c, i in enumerate(idx):
        axs[r, c].imshow(Xtr[i], cmap="gray"); axs[r, c].axis("off")
    axs[r, 6].imshow(Xtr[ytr == lab].mean(0), cmap="gray"); axs[r, 6].axis("off"); axs[r, 6].set_title("mean", fontsize=9)
    axs[r, 0].set_title(["normal", "pneumonia"][lab], fontsize=10, loc="left")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 사진 → 폐렴일까

픽셀 784개를 ÷255 로 맞추고 (1주 3차시) 로지스틱회귀. 기준선 = 전부 '폐렴'. 의료에서는 **놓친 환자** (재현율) 가 특히 중요하다."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

A, B = Xtr.reshape(len(Xtr), -1) / 255, Xte.reshape(len(Xte), -1) / 255
m = LogisticRegression(C=0.1, max_iter=3000).fit(A, ytr)
score = m.predict_proba(B)[:, 1]
for name, pred in [("always pneumonia", np.ones(len(yte), dtype=int)), ("logistic (threshold 0.5)", (score > 0.5).astype(int)),
                   ("logistic (threshold 0.9)", (score > 0.9).astype(int))]:
    p, r = precision_recall(yte, pred)
    print(f"{name:26s} accuracy {np.mean(pred == yte):.1%}  precision {p:.0%}  recall {r:.0%}")
print(f"ROC-AUC {roc_auc_score(yte, score):.2f}")"""),
("md", """**관찰**: 기준을 올리면 '폐렴' 이라고 하는 사진이 줄어 정밀도는 오르고 재현율은 떨어진다. 병원에서는 놓친 환자의 비용이 커서 보통 재현율을 높게 잡는다.

## 다른 문제로 바꿔 보기

- 같은 사람의 사진이 학습 · 시험 양쪽에 있으면? 의료 데이터는 **환자 단위** 로 나눠야 한다.
- 사진 크기를 28×28 에서 224×224 로 키우면 (MedMNIST+) 무엇이 좋아지고 무엇이 비싸질까?
- 모델이 사진의 '병원 표시 글자' 나 '촬영 기계' 차이를 보고 맞히는 건 아닐까? (지름길 학습)"""),
]),
# ------------------------------------------------------------------
dict(slug="satellite", title="위성 사진 토지 분류", sector="이미지 · 소리 · 3D", keywords=["위성 영상", "토지 이용 지도", "환경 감시"],
     teaser="위성 사진 한 칸이 숲일까, 밭일까, 도시일까?",
     data="EuroSAT (RGB, 시험 묶음 5,400장)", data_src="Helber et al. (2019). 유럽우주국 Sentinel-2 위성 사진 64×64, 10종류 (밭 · 숲 · 풀밭 · 고속도로 · 산업 · 목초지 · 과수원 · 주거지 · 강 · 바다/호수). Hugging Face timm/eurosat-rgb 의 시험 묶음",
     size="약 18MB",
     five=[("① 한 행", "위성 사진 한 칸 (64×64)"), ("② 입력 X", "색 분포 (채널별 평균 · 히스토그램)"), ("③ 출력 y", "10가지 토지 종류"),
           ("④ 평가지표", "정확도 + 종류별 재현율"), ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """from PIL import Image
URL = "https://huggingface.co/api/datasets/timm/eurosat-rgb/parquet/default/test/0.parquet"
df = pd.read_parquet(download(URL, "data/eurosat_test.parquet"))
NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
imgs = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["image"]])
labels = df["label"].values
print(imgs.shape, imgs.dtype)
print(pd.Series([NAMES[i] for i in labels]).value_counts())"""),
("md", """## 열어 보기"""),
("code", """fig, axs = plt.subplots(2, 10, figsize=(15, 3.4))
for c in range(10):
    idx = np.where(labels == c)[0][:2]
    for r, i in enumerate(idx):
        axs[r, c].imshow(imgs[i]); axs[r, c].axis("off")
    axs[0, c].set_title(NAMES[c][:12], fontsize=8)
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 사진 → 토지 종류

색만으로 얼마나 맞힐까? 채널별 평균 · 표준편차 + 채널별 히스토그램 (8칸) 을 숫자로 만들어 랜덤 포레스트. 70% 학습 · 30% 시험."""),
("code", """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def color_features(img):
    \"\"\"Per-channel mean, std and 8-bin histogram of one RGB image\"\"\"
    x = img.reshape(-1, 3).astype(np.float32) / 255
    hist = [np.histogram(x[:, k], bins=8, range=(0, 1), density=True)[0] for k in range(3)]
    return np.r_[x.mean(0), x.std(0), np.concatenate(hist)]

X = np.stack([color_features(im) for im in imgs])
Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, stratify=labels, random_state=0)
m = RandomForestClassifier(400, random_state=0, n_jobs=-1).fit(Xtr, ytr)
pred = m.predict(Xte)
most = np.bincount(ytr).argmax()
print(f"baseline (always {NAMES[most]}): accuracy {np.mean(yte == most):.1%}")
print(f"random forest on colors : accuracy {np.mean(yte == pred):.1%}")
print("recall per class:", pd.Series(yte == pred).groupby([NAMES[i] for i in yte]).mean().round(2).to_dict())"""),
("md", """**관찰**: 숲 · 바다처럼 색이 뚜렷한 곳은 색만으로 잘 맞힌다. 고속도로 · 강처럼 '모양 (긴 선)' 이 중요한 곳은 색만으로는 어렵다. 모양까지 보려면 CNN (10주차).

## 다른 문제로 바꿔 보기

- **변화 감지**: 같은 장소의 작년 사진과 올해 사진 → 새 건물이 생겼나? 한 행 = (장소, 두 시점).
- 위성은 사람 눈에 안 보이는 파장 (근적외선 등 13개 대역) 도 찍는다. 식물은 근적외선에서 밝다. 입력에 이걸 넣으면?
- 나라가 다른 곳 (학습: 유럽, 시험: 한국) 에서도 맞힐까?"""),
]),
# ------------------------------------------------------------------
dict(slug="crop", title="작물 병해 진단", sector="이미지 · 소리 · 3D", keywords=["농업 AI", "이미지 분류", "스마트팜"],
     teaser="잎 사진 한 장으로 병에 걸렸는지 알 수 있을까?",
     data="PlantVillage (작은 판 1,900장)", data_src="Hughes & Salathé (2015) PlantVillage 의 작은 판 (Hugging Face geraldmc/plantvillage-tiny): 잎 사진 1,900장, 작물 14종 × 병 = 38종류, 학습 · 시험 나눔 포함",
     size="약 31MB",
     five=[("① 한 행", "잎 사진 한 장"), ("② 입력 X", "색 분포 (채널별 히스토그램)"), ("③ 출력 y", "건강 / 병 (또는 38종류)"),
           ("④ 평가지표", "정확도 · 병 재현율"), ("⑤ 기준선", "가장 흔한 쪽")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """from PIL import Image
URL = "https://huggingface.co/api/datasets/geraldmc/plantvillage-tiny/parquet/default/train/0.parquet"
df = pd.read_parquet(download(URL, "data/plantvillage_tiny.parquet"))
print(df.shape)
print(df.drop(columns="image").sample(5, random_state=0))
print(df["split"].value_counts().to_dict())"""),
("md", """## 열어 보기

작물 (host) 과 병 (disease) 이 따로 적혀 있다. 건강한 잎과 병든 잎을 같이 보자."""),
("code", """df["healthy"] = df["disease"].str.contains("healthy").astype(int)
print("hosts:", df["host"].nunique(), "| classes:", df["class_label"].nunique(), f"| healthy share {df['healthy'].mean():.1%}")
fig, axs = plt.subplots(2, 6, figsize=(12, 4.2))
for r, h in enumerate([1, 0]):
    for c, (_, row) in enumerate(df[df["healthy"] == h].sample(6, random_state=r).iterrows()):
        axs[r, c].imshow(Image.open(io.BytesIO(row["image"]["bytes"]))); axs[r, c].axis("off")
        axs[r, c].set_title(row["class_label"].replace("___", "\\n")[:28], fontsize=7)
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 잎 → 병에 걸렸나

색 히스토그램 (채널별 16칸) 으로 랜덤 포레스트. 데이터에 들어 있는 `split` 대로 나눈다. 기준선 = 가장 흔한 쪽."""),
("code", """from sklearn.ensemble import RandomForestClassifier

def leaf_features(image_bytes):
    \"\"\"Per-channel 16-bin colour histogram + mean/std of one leaf photo (resized to 64x64)\"\"\"
    x = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64)), dtype=np.float32).reshape(-1, 3) / 255
    hist = [np.histogram(x[:, k], bins=16, range=(0, 1), density=True)[0] for k in range(3)]
    return np.r_[x.mean(0), x.std(0), np.concatenate(hist)]

X = np.stack([leaf_features(b["bytes"]) for b in df["image"]])
tr, te = (df["split"] == "train").values, (df["split"] != "train").values
for target in ["healthy", "class_idx"]:
    y = df[target].values
    m = RandomForestClassifier(400, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    pred = m.predict(X[te]); most = pd.Series(y[tr]).value_counts().idxmax()
    print(f"[{target}] baseline {np.mean(y[te] == most):.1%}  ->  random forest {np.mean(y[te] == pred):.1%}")"""),
("md", """**관찰**: '건강 / 병' 두 가지는 색만으로도 꽤 맞히지만 38종류는 어렵다. 이 데이터는 실험실에서 잎을 떼어 같은 배경에 찍은 사진이라, 밭에서 찍은 사진에서는 성능이 크게 떨어진다고 알려져 있다.

## 다른 문제로 바꿔 보기

- `leaf_grouped` 열: 같은 잎을 여러 번 찍은 사진이 있다면, 같은 잎이 학습 · 시험 양쪽에 들어가지 않게 나눠야 한다.
- 밭 전체를 드론으로 찍으면? 한 행 = 밭의 한 칸, 출력 = 병이 퍼진 비율."""),
]),
# ------------------------------------------------------------------
dict(slug="speech", title="음성 명령 인식", sector="이미지 · 소리 · 3D", keywords=["음성 AI", "키워드 인식", "온디바이스"],
     teaser="1초짜리 소리. 'yes' 일까 'no' 일까?",
     data="Speech Commands v0.02 (시험 묶음 4,890개)", data_src="Warden (2018), Google. 1초짜리 녹음, 단어 10개 + 침묵 + '모르는 단어' = 12종류 (CC BY 4.0). 시험 묶음만 받는다 (약 110MB)",
     size="약 110MB",
     five=[("① 한 행", "1초짜리 녹음 하나"), ("② 입력 X", "멜 스펙트로그램 (시간 × 주파수 에너지, dB)"), ("③ 출력 y", "12종류 (yes, no, up, … , 침묵, 모름)"),
           ("④ 평가지표", "정확도"), ("⑤ 기준선", "가장 흔한 종류")],
     cells=[
("md", """## 입장: 데이터 받기

압축 파일 약 110MB. 처음엔 1~2분."""),
("code", """import tarfile
from scipy.io import wavfile
URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
tar = tarfile.open(download(URL, "data/speech_commands_test.tar.gz"))
waves, labels, speakers = [], [], []
for member in tar.getmembers():
    if member.name.endswith(".wav"):
        rate, w = wavfile.read(io.BytesIO(tar.extractfile(member).read()))
        w = np.pad(w, (0, max(0, 16000 - len(w))))[:16000].astype(np.float32) / 32768
        waves.append(w); labels.append(member.name.split("/")[-2]); speakers.append(member.name.split("/")[-1].split("_")[0])
waves, labels, speakers = np.stack(waves), np.array(labels), np.array(speakers)
print(waves.shape, "rate", rate, "| speakers", len(set(speakers)))
print(pd.Series(labels).value_counts().to_dict())"""),
("md", """## 열어 보기: 파형과 멜 스펙트로그램

소리를 짧은 조각마다 주파수별 에너지로 바꾸고, 사람 귀처럼 낮은 주파수를 촘촘하게 나눈 것이 멜 스펙트로그램이다 (1주 3차시: dB 로 본다)."""),
("code", """def mel_filterbank(n_mels=32, n_fft=512, rate=16000):
    \"\"\"Triangular mel filters (n_mels x n_fft//2+1)\"\"\"
    mel = lambda f: 2595 * np.log10(1 + f / 700); inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(0), mel(rate / 2), n_mels + 2)); bins = np.floor((n_fft + 1) * pts / rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1); fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb

FB = mel_filterbank()
def log_mel(w, fb, n_fft=512, hop=256):
    \"\"\"Log-mel spectrogram (n_mels x frames) of a 1-second wave\"\"\"
    frames = np.stack([w[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(w) - n_fft, hop)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(fb @ power.T + 1e-10)

fig, axs = plt.subplots(2, 4, figsize=(13, 4.5))
for c, word in enumerate(["yes", "no", "up", "_silence_"]):
    i = np.where(labels == word)[0][0]
    axs[0, c].plot(waves[i], lw=0.5); axs[0, c].set_title(word); axs[0, c].set_ylim(-1, 1)
    axs[1, c].imshow(log_mel(waves[i], FB), origin="lower", aspect="auto", cmap="magma")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 녹음 → 단어

멜 스펙트로그램을 시간 방향으로 8토막 내 평균 낸 숫자 (32 × 8 = 256개) 로 로지스틱회귀.
**나누는 방법 주의**: 한 사람이 여러 단어를 녹음했다. 같은 사람이 학습 · 시험 양쪽에 있으면 목소리를 외워서 맞힐 수 있다. **사람 단위** 로 나눈다."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def summarize(spec, n_seg=8):
    return np.concatenate([seg.mean(1) for seg in np.array_split(spec, n_seg, axis=1)])

X = np.stack([summarize(log_mel(w, FB)) for w in waves])
tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=0).split(X, labels, groups=speakers))
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X[tr], labels[tr])
pred = m.predict(X[te]); most = pd.Series(labels[tr]).value_counts().idxmax()
print(f"baseline (always '{most}'): accuracy {np.mean(labels[te] == most):.1%}")
print(f"logistic on log-mel summary: accuracy {np.mean(labels[te] == pred):.1%}")
print("recall per word:", pd.Series(labels[te] == pred).groupby(labels[te]).mean().round(2).to_dict())"""),
("md", """**관찰**: 침묵은 거의 다 맞히지만 '모르는 단어' 는 열에 하나 남짓만 맞힌다. 스마트 스피커처럼 실제로 쓰려면 모르는 말을 아는 단어로 잘못 알아듣지 않는 것이 더 중요하다 (잘못 깨어남). 소리를 그대로 받는 CNN 은 90% 넘게 맞힌다.

## 다른 문제로 바꿔 보기

- **깨우는 말 (wake word)**: 하루 종일 듣다가 '헤이 ○○' 에만 반응. 한 행 = 1초 조각, 지표 = 놓친 비율 + 한 시간에 잘못 깨어난 횟수.
- 사람 단위가 아니라 무작위로 나누면 정확도가 얼마나 오를까? 해 보고, 왜 그게 거짓 점수인지 말해 보자."""),
]),
# ------------------------------------------------------------------
dict(slug="air_quality", title="미세먼지 예보", sector="도시 · 공간", keywords=["환경 · 기상 서비스", "시계열 예측", "경보"],
     teaser="내일 이 시각 초미세먼지 (PM2.5) 는 얼마일까?",
     data="UCI Beijing PM2.5", data_src="Liang et al. (2015). 베이징 미국 대사관 측정소의 2010~2014 시간별 PM2.5 (㎍/㎥) + 이슬점 · 기온 · 기압 · 풍향 · 풍속 · 눈 · 비 (43,824시간)",
     size="약 2MB",
     five=[("① 한 행", "한 시간"), ("② 입력 X", "지금 PM2.5 · 지난 몇 시간 · 기온 · 바람 · 습도 · 달 · 시각"), ("③ 출력 y", "24시간 뒤 PM2.5 (㎍/㎥)"),
           ("④ 평가지표", "MAE, '매우 나쁨' (76 이상) 재현율"), ("⑤ 기준선", "지금 값 그대로")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip"
with zipfile.ZipFile(download(URL, "data/beijing_pm25.zip")) as z:
    df = pd.read_csv(z.open(z.namelist()[0]))
df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.set_index("time").rename(columns={"pm2.5": "pm25"})
print(df.shape, df.index.min(), "~", df.index.max())
print(f"missing PM2.5: {df['pm25'].isna().mean():.1%}")
df.sample(5, random_state=0)"""),
("md", """## 열어 보기

달마다 · 시각마다 평균. 겨울 난방철에 높을까?"""),
("code", """fig, axs = plt.subplots(1, 3, figsize=(14, 3.2))
df["pm25"].resample("D").mean().plot(ax=axs[0], lw=0.6, title="daily mean PM2.5")
df.groupby("month")["pm25"].mean().plot(kind="bar", ax=axs[1], title="by month")
df.groupby("hour")["pm25"].mean().plot(ax=axs[2], marker="o", title="by hour of day")
plt.tight_layout(); plt.show()
print(f"hours at 'very bad' (>= 76): {(df['pm25'] >= 76).mean():.1%}")"""),
("md", """## 문제 하나 골라 보기: 지금까지 → 24시간 뒤 PM2.5

**시간 순서** 로 나눈다: 2010~2013 학습, 2014 시험. 입력은 '지금' 알 수 있는 것만. 기준선 = 지금 값 그대로."""),
("code", """from sklearn.ensemble import HistGradientBoostingRegressor

def recall_at(y_true, y_pred, th=76):
    hit = y_true >= th
    return np.mean(y_pred[hit] >= th)

d = df.copy()
for k in (1, 2, 3, 24):
    d[f"pm_lag{k}"] = d["pm25"].shift(k)
d["target"] = d["pm25"].shift(-24)
d = pd.get_dummies(d, columns=["cbwd"], dtype=float).dropna(subset=["pm25", "target"])
feats = [c for c in d.columns if c.startswith(("pm_", "cbwd_"))] + ["pm25", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "month", "hour"]
tr, te = d[d["year"] < 2014], d[d["year"] == 2014]
m = HistGradientBoostingRegressor(random_state=0).fit(tr[feats], tr["target"])
for name, pred in [("same as now", te["pm25"].values), ("train mean", np.full(len(te), tr["target"].mean())), ("gradient boosting", m.predict(te[feats]))]:
    print(f"{name:18s} MAE {np.mean(np.abs(pred - te['target'])):.1f}  recall of 'very bad' {recall_at(te['target'].values, pred):.0%}")"""),
("md", """**관찰**: 하루 뒤는 지금 값만으로는 잘 맞지 않는다. 바람이 바뀌면 하루 만에 공기가 바뀌기 때문이다. 실제 예보는 **내일의 기상 예보** (바람 · 비) 를 입력으로 쓴다.

## 다른 문제로 바꿔 보기

- 한국 데이터: 에어코리아 측정소 (공공데이터포털 API 신청). 한 행 = 측정소 × 시간이면 이웃 측정소 · 중국 쪽 바람이 입력이 된다 (시공간 문제).
- '농도' 대신 '내일 마스크가 필요한가 (나쁨 이상)' 를 맞히면 분류가 된다. 지표는?"""),
]),
# ------------------------------------------------------------------
dict(slug="electricity", title="전력 수요 예측", sector="도시 · 공간", keywords=["에너지 · 전력", "시계열 예측", "스마트 그리드"],
     teaser="다음 한 시간, 이 집은 전기를 얼마나 쓸까?",
     data="UCI Individual Household Electric Power Consumption", data_src="Hebrail & Berard (2012). 프랑스 한 가정의 2006-12 ~ 2010-11 1분 간격 전력 사용량 약 207만 줄 (전체 · 부엌 · 세탁실 · 냉난방 계량기)",
     size="약 20MB (압축)",
     five=[("① 한 행", "한 시간"), ("② 입력 X", "지난 몇 시간 · 어제 같은 시각 · 지난주 같은 시각 · 시각 · 요일"), ("③ 출력 y", "다음 한 시간 평균 사용량 (kW)"),
           ("④ 평가지표", "MAE · RMSE"), ("⑤ 기준선", "지난 한 시간 그대로 · 어제 같은 시각")],
     cells=[
("md", """## 입장: 데이터 받기

1분 간격 207만 줄. 읽는 데 30초쯤 걸린다. 빈칸은 '?' 로 적혀 있다 (1주 3차시 '숫자인 척하는 결측치' 와 비슷)."""),
("code", """URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
with zipfile.ZipFile(download(URL, "data/household_power.zip")) as z:
    raw = pd.read_csv(z.open("household_power_consumption.txt"), sep=";", na_values="?", low_memory=False)
raw["time"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
raw = raw.set_index("time").drop(columns=["Date", "Time"])
print(raw.shape, raw.index.min(), "~", raw.index.max())
print(f"missing minutes: {raw['Global_active_power'].isna().mean():.2%}")
raw.sample(5, random_state=0)"""),
("md", """## 열어 보기: 한 시간 단위로

1분 값을 한 시간 평균으로 모은다 (kW). 하루 중 언제, 1년 중 언제 많이 쓸까?"""),
("code", """hourly = raw["Global_active_power"].resample("h").mean().to_frame("kw")
print(hourly.shape)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.2))
hourly["kw"].resample("W").mean().plot(ax=axs[0], title="weekly mean (kW)")
hourly.groupby(hourly.index.hour)["kw"].mean().plot(ax=axs[1], marker="o", title="by hour of day")
hourly.groupby(hourly.index.dayofweek)["kw"].mean().plot(kind="bar", ax=axs[2], title="by weekday (0 = Mon)")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 지금까지 → 다음 한 시간

**시간 순서**: 2009년까지 학습, 2010년 시험. 기준선 3가지와 모델 비교."""),
("code", """from sklearn.ensemble import HistGradientBoostingRegressor

h = hourly.copy()
for k in (1, 2, 3, 24, 168):
    h[f"lag{k}"] = h["kw"].shift(k)
h["yday"] = h["kw"].shift(23)                     # the target hour, one day earlier
h["hour"], h["dow"] = h.index.hour, h.index.dayofweek
h["target"] = h["kw"].shift(-1)
h = h.dropna()
feats = ["kw", "lag1", "lag2", "lag3", "lag24", "lag168", "hour", "dow"]
tr, te = h[h.index.year < 2010], h[h.index.year == 2010]
m = HistGradientBoostingRegressor(random_state=0).fit(tr[feats], tr["target"])
y = te["target"].values
for name, pred in [("last hour", te["kw"].values), ("same hour yesterday", te["yday"].values),
                   ("gradient boosting", m.predict(te[feats]))]:
    err = pred - y
    print(f"{name:20s} MAE {np.mean(np.abs(err)):.3f} kW   RMSE {np.sqrt(np.mean(err ** 2)):.3f} kW")"""),
("md", """**관찰**: '지난 한 시간 그대로' 가 꽤 강한 기준선이다. 모델은 시각 · 요일 패턴을 더해 조금 더 낫다. 한 집의 사용량은 갑자기 켜는 기기 (오븐, 세탁기) 때문에 들쭉날쭉해서 크게 줄이기 어렵다. 수많은 집을 합친 전력망 전체 수요는 훨씬 잘 맞는다.

## 다른 문제로 바꿔 보기

- **이상 탐지**: 평소와 다르게 많이 쓰는 시간 (전기 새는 곳, 고장) 찾기.
- **기기 분리 (NILM)**: 전체 사용량만 보고 부엌 · 세탁실 · 냉난방이 각각 얼마 썼는지 맞히기. 이 데이터에 정답 계량기 (Sub_metering_1~3) 가 있다."""),
]),
# ------------------------------------------------------------------
dict(slug="football", title="축구: 이 슛은 골일까?", sector="스포츠 · 게임", keywords=["스포츠 분석", "기대 득점 (xG)", "확률 예측"],
     teaser="슛 위치와 각도만 보고 골이 될 확률은? (기대 득점 xG)",
     data="StatsBomb Open Data · 2022 카타르 월드컵 64경기", data_src="StatsBomb 이 공개한 경기 이벤트 데이터 (패스 · 슛 · 태클마다 위치 · 시각 · 선수). 경기마다 JSON 파일 하나 (모두 약 190MB)",
     size="약 190MB (경기 64개)",
     five=[("① 한 행", "슛 한 번"), ("② 입력 X", "골대까지 거리 · 각도 · 몸의 부위 (발/머리) · 슛 종류 · 압박"), ("③ 출력 y", "골인가 (확률)"),
           ("④ 평가지표", "로그 손실 (확률이 맞나)"), ("⑤ 기준선", "모든 슛에 전체 골 비율")],
     cells=[
("md", """## 입장: 데이터 받기

경기 목록 → 경기마다 이벤트 파일 (64개, 처음엔 2~3분)."""),
("code", """import json
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
matches = json.load(open(download(BASE + "matches/43/106.json", "data/statsbomb_matches_43_106.json")))
matches = sorted(matches, key=lambda m: m["match_date"])
print(len(matches), "matches,", matches[0]["match_date"], "~", matches[-1]["match_date"])

rows = []
for k, m in enumerate(matches):
    events = json.load(open(download(BASE + f"events/{m['match_id']}.json", f"data/sb_events_{m['match_id']}.json")))
    for e in events:
        if e["type"]["name"] == "Shot":
            s = e["shot"]
            rows.append(dict(match=k, x=e["location"][0], y=e["location"][1], body=s["body_part"]["name"], kind=s["type"]["name"],
                             pressure=int(e.get("under_pressure", False)), first_time=int(s.get("first_time", False)),
                             goal=int(s["outcome"]["name"] == "Goal"), sb_xg=s["statsbomb_xg"], team=e["team"]["name"]))
shots = pd.DataFrame(rows)
print(shots.shape, f"goal rate {shots['goal'].mean():.1%}")
shots.sample(5, random_state=0)"""),
("md", """## 열어 보기: 경기장 위의 슛

경기장은 120 × 80 (야드), 골대는 x = 120, y = 36 ~ 44. 골 (주황) 은 어디서 나왔을까?"""),
("code", """fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color="gray"); ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], color="gray")
ax.scatter(shots.loc[shots.goal == 0, "x"], shots.loc[shots.goal == 0, "y"], s=10, alpha=0.4, label="no goal")
ax.scatter(shots.loc[shots.goal == 1, "x"], shots.loc[shots.goal == 1, "y"], s=30, color="orangered", label="goal")
ax.plot([120, 120], [36, 44], color="black", lw=4); ax.set_xlim(60, 122); ax.set_aspect("equal"); ax.legend(); plt.show()
print(shots.groupby("kind")["goal"].agg(["mean", "size"]).round(2))"""),
("md", """## 문제 하나 골라 보기: 슛 → 골 확률

거리와 각도 (골대 두 기둥 사이로 보이는 각) 를 계산해 로지스틱회귀. **경기 단위 · 날짜 순서** 로 앞 48경기 학습, 뒤 16경기 (토너먼트) 시험.
StatsBomb 도 자기 모델의 확률 (`sb_xg`) 을 데이터에 넣어 두었다. 같이 비교한다."""),
("code", """from sklearn.linear_model import LogisticRegression

def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def shot_features(shots):
    dx, dy = 120 - shots["x"], shots["y"] - 40
    angle = np.abs(np.arctan2(4 * dx, dx ** 2 + dy ** 2 - 16))          # angle between the two posts
    return pd.DataFrame({"dist": np.hypot(dx, dy), "angle": angle, "head": (shots["body"] == "Head").astype(int),
                         "penalty": (shots["kind"] == "Penalty").astype(int), "free_kick": (shots["kind"] == "Free Kick").astype(int),
                         "pressure": shots["pressure"], "first_time": shots["first_time"]})

X = shot_features(shots); y = shots["goal"].values
tr, te = shots["match"] < 48, shots["match"] >= 48
m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
p = m.predict_proba(X[te])[:, 1]
print(f"test shots {te.sum()}, goals {y[te].sum()}")
print(f"baseline (goal rate {y[tr].mean():.3f} for every shot): log loss {log_loss(y[te], np.full(te.sum(), y[tr].mean())):.3f}")
print(f"our logistic (7 numbers)                : log loss {log_loss(y[te], p):.3f}")
print(f"StatsBomb's own xG                      : log loss {log_loss(y[te], shots.loc[te, 'sb_xg'].values):.3f}")
print(pd.Series(m.coef_[0], index=X.columns).round(2).to_dict())"""),
("md", """**관찰**: 거리 · 각도 몇 개만으로도 기준선보다 확실히 낫고, 업체의 xG 모델 (수비수 · 골키퍼 위치까지 쓰는) 에 꽤 가깝다. 머리로 찬 슛은 같은 거리에서도 골 확률이 낮다.

## 다른 문제로 바꿔 보기

- **선수 평가**: 슛 하나가 아니라 '선수 × 경기' 를 한 행으로 하면? 실제 골 수 − 기대 골 (xG 합) = 운이 좋았나, 잘 찼나.
- **패스의 가치**: 한 행 = 패스, y = 이 패스 뒤 10초 안에 골이 났나 (VAEP, 아래 논문)."""),
]),
# ------------------------------------------------------------------
dict(slug="chess", title="체스: 누가 이길까?", sector="스포츠 · 게임", keywords=["게임 데이터", "승패 예측", "실력 측정 (레이팅)"],
     teaser="두 사람의 레이팅만 보고 누가 이길지 알 수 있을까?",
     data="Lichess 공개 대국 · 2013년 1월", data_src="온라인 체스 Lichess 의 모든 대국 기록 (CC0) 중 2013년 1월 121,332판 (Hugging Face Lichess/standard-chess-games): 결과 · 두 사람 레이팅 · 시간 규칙 · 수 기록",
     size="약 37MB",
     five=[("① 한 행", "대국 하나"), ("② 입력 X", "백 · 흑 레이팅, 레이팅 차, 시간 규칙"), ("③ 출력 y", "백 승 / 무승부 / 흑 승"),
           ("④ 평가지표", "정확도 · 로그 손실"), ("⑤ 기준선", "레이팅이 높은 쪽이 이긴다")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year=2013/month=01/train-00000-of-00001.parquet"
df = pd.read_parquet(download(URL, "data/lichess_2013_01.parquet"), columns=["Result", "WhiteElo", "BlackElo", "TimeControl", "Termination", "Opening", "movetext"])
print(df.shape)
print(df["Result"].value_counts().to_dict())
df.drop(columns="movetext").sample(5, random_state=0)"""),
("md", """## 열어 보기

레이팅 차 (백 − 흑) 에 따라 백이 이긴 비율은? 레이팅 (Elo) 은 원래 '차이가 400 이면 높은 쪽이 약 91% 이긴다' 가 되도록 만든 숫자다."""),
("code", """df = df[df["Result"].isin(["1-0", "0-1", "1/2-1/2"])].copy()
df["WhiteElo"] = pd.to_numeric(df["WhiteElo"], errors="coerce"); df["BlackElo"] = pd.to_numeric(df["BlackElo"], errors="coerce")
df["diff"] = df["WhiteElo"] - df["BlackElo"]
df["base_sec"] = pd.to_numeric(df["TimeControl"].str.split("+").str[0], errors="coerce")
bins = pd.cut(df["diff"], [-1000, -300, -200, -100, -50, 0, 50, 100, 200, 300, 1000])
win = df.groupby(bins, observed=True)["Result"].apply(lambda r: (r == "1-0").mean() + 0.5 * (r == "1/2-1/2").mean())
elo = 1 / (1 + 10 ** (-np.array([(-650), -250, -150, -75, -25, 25, 75, 150, 250, 650]) / 400))
plt.figure(figsize=(8, 3.5)); plt.plot(range(len(win)), win.values, "o-", label="actual (white score)"); plt.plot(range(len(win)), elo, "s--", label="Elo formula")
plt.xticks(range(len(win)), [str(i) for i in win.index], rotation=45, fontsize=8); plt.ylabel("white score"); plt.legend(); plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 대국 → 결과 (3가지)

**날짜 순서** (데이터가 시간 순서로 놓여 있다) 로 앞 80% 학습, 뒤 20% 시험.
- 기준선 1: 늘 '백 승' (백이 먼저 두니까)
- 기준선 2: 레이팅이 높은 쪽이 이긴다
- 모델: 레이팅 · 차 · 시간 규칙으로 로지스틱회귀 (확률까지)"""),
("code", """from sklearn.linear_model import LogisticRegression

d = df.dropna(subset=["base_sec", "diff", "WhiteElo", "BlackElo"])
X = np.c_[d["diff"], (d["WhiteElo"] + d["BlackElo"]) / 2, np.log1p(d["base_sec"])]
y = d["Result"].map({"1-0": 0, "1/2-1/2": 1, "0-1": 2}).values
cut = int(len(d) * 0.8); tr, te = slice(0, cut), slice(cut, None)
m = LogisticRegression(max_iter=2000).fit(X[tr] / [100, 1000, 1], y[tr])
prob = m.predict_proba(X[te] / [100, 1000, 1])
higher = np.where(X[te][:, 0] >= 0, 0, 2)
print(f"always white wins     : accuracy {np.mean(y[te] == 0):.1%}")
print(f"higher rating wins    : accuracy {np.mean(y[te] == higher):.1%}")
print(f"logistic (3 numbers)  : accuracy {np.mean(y[te] == prob.argmax(1)):.1%}, "
      f"log loss {-np.mean(np.log(prob[np.arange(len(prob)), y[te]] + 1e-9)):.3f}")
print(f"draw rate in test: {np.mean(y[te] == 1):.1%}")"""),
("md", """**관찰**: 레이팅 차만으로도 동전 던지기보다 훨씬 잘 맞힌다. 무승부는 드물어서 거의 맞히지 못한다. 그런데 `movetext` (둔 수 전체) 나 `Termination` (어떻게 끝났나) 을 입력에 넣으면? **경기가 끝나야 알 수 있는 값** 이라 누수다.

## 다른 문제로 바꿔 보기

- **수 예측**: 한 행 = 한 수, 입력 = 지금 판, 출력 = 사람이 둘 다음 수. 레이팅마다 다르게 두는 '사람 같은 AI' (Maia, 아래 논문).
- **부정행위 탐지**: 사람의 레이팅에 비해 너무 엔진 같은 수만 두는 계정 찾기."""),
]),
# ------------------------------------------------------------------
dict(slug="cyber_security", title="네트워크 침입 탐지", sector="플랫폼 · 서비스", keywords=["보안관제", "침입 탐지 (IDS)", "처음 보는 공격"],
     teaser="네트워크 연결 15만 건. 처음 보는 공격도 잡아낼 수 있을까?",
     data="NSL-KDD", data_src="Tavallaee et al. (2009). KDD Cup 99 를 정리한 침입 탐지 데이터: 연결 한 건마다 41개 숫자 (길이 · 프로토콜 · 서비스 · 바이트 수 · 오류 비율 …) + 정상 / 공격 이름. 학습 125,973건 · 시험 22,544건 (시험에만 있는 공격 포함)",
     size="약 22MB",
     five=[("① 한 행", "네트워크 연결 한 건"), ("② 입력 X", "41개 숫자 · 범주 (프로토콜 · 서비스 · 바이트 수 · 실패 비율 …)"), ("③ 출력 y", "정상 / 공격"),
           ("④ 평가지표", "재현율 (놓친 공격) · 정밀도 (헛경보)"), ("⑤ 기준선", "전부 '정상'")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"]
BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
train = pd.read_csv(download(BASE + "KDDTrain+.txt", "data/nslkdd_train.txt"), header=None, names=COLS)
test = pd.read_csv(download(BASE + "KDDTest+.txt", "data/nslkdd_test.txt"), header=None, names=COLS)
print(train.shape, test.shape)
train.sample(5, random_state=0).iloc[:, [0, 1, 2, 3, 4, 5, 22, 41]]"""),
("md", """## 열어 보기

공격 종류와 개수. 그리고 **시험에만 있는 공격** 은 몇 가지일까?"""),
("code", """print("train:", train["attack"].value_counts().head(8).to_dict())
new_attacks = sorted(set(test["attack"]) - set(train["attack"]))
print(f"attack types only in test: {len(new_attacks)} ->", new_attacks[:10])
print(f"test rows with a new attack type: {test['attack'].isin(new_attacks).mean():.1%}")
train["y"] = (train["attack"] != "normal").astype(int); test["y"] = (test["attack"] != "normal").astype(int)
print(f"attack share: train {train['y'].mean():.1%}, test {test['y'].mean():.1%}")"""),
("md", """## 문제 하나 골라 보기: 연결 → 공격일까

같은 모델 (랜덤 포레스트) 을 두 가지로 채점한다.
1. 학습 데이터를 무작위로 나눈 시험 (같은 공격들)
2. 공식 시험 데이터 (처음 보는 공격 포함)"""),
("code", """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0)); fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

both = pd.get_dummies(pd.concat([train, test])[COLS[:-2]], columns=["protocol_type", "service", "flag"], dtype=float)
Xtr_all, Xte_off = both.iloc[:len(train)], both.iloc[len(train):]
Xa, Xb, ya, yb = train_test_split(Xtr_all, train["y"].values, test_size=0.3, random_state=0)
m = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(Xa, ya)
for name, X, y in [("random split of train", Xb, yb), ("official test (new attacks)", Xte_off, test["y"].values)]:
    pred = m.predict(X); p, r = precision_recall(y, pred)
    print(f"{name:28s} accuracy {np.mean(pred == y):.1%}  precision {p:.1%}  recall {r:.1%}")
pred = m.predict(Xte_off); new = test["attack"].isin(new_attacks).values
print(f"recall on attack types never seen in training: {np.mean(pred[new] == 1):.1%}")"""),
("md", """**관찰**: 같은 공격끼리 나눈 시험에서는 거의 100% 이지만, 처음 보는 공격이 섞이면 공격을 많이 놓친다. 보안에서는 **공격자가 계속 바뀐다**. 시험 데이터를 '미래의 공격' 처럼 만들어야 진짜 점수가 나온다.

## 다른 문제로 바꿔 보기

- **이상 탐지**: 공격 라벨 없이 정상 연결만 배워서 '평소와 다른 연결' 을 찾으면? (4주차 가우시안 · k-means)
- 보안관제 요원이 하루에 볼 수 있는 경보가 100개뿐이라면? 지표를 '상위 100개 중 진짜 공격 수' 로 바꾼다."""),
]),
# ------------------------------------------------------------------
dict(slug="telco_churn", title="통신사 고객 이탈 예측", sector="플랫폼 · 서비스", keywords=["고객 이탈 (churn)", "구독 서비스", "리텐션 마케팅"],
     teaser="통신사 가입자 7천 명. 다음 달 떠날 사람을 미리 알 수 있을까?",
     data="IBM Telco Customer Churn", data_src="IBM 샘플 데이터: 가상의 통신사 가입자 7,043명. 가입 기간 · 요금제 · 부가 서비스 · 결제 방법 · 월 요금 · 지난달 해지 여부",
     size="약 1MB",
     five=[("① 한 행", "가입자 한 명"), ("② 입력 X", "가입 기간 · 계약 종류 · 인터넷 · 부가 서비스 · 결제 방법 · 월 요금"), ("③ 출력 y", "해지했나 (Churn)"),
           ("④ 평가지표", "재현율 · 정밀도 (붙잡을 사람 고르기)"), ("⑤ 기준선", "아무도 안 떠난다")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(download(URL, "data/telco_churn.csv"))
print(df.shape)
df.sample(5, random_state=0)"""),
("md", """## 열어 보기

`TotalCharges` 는 숫자인데 글자로 읽혔다. 왜일까? (공백으로 적힌 빈칸, 1주 3차시) 그리고 계약 종류마다 해지율은?"""),
("code", """print(df.dtypes["TotalCharges"], "| blank TotalCharges:", (df["TotalCharges"].str.strip() == "").sum())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["y"] = (df["Churn"] == "Yes").astype(int)
print(f"churn rate {df['y'].mean():.1%}")
print(df.groupby("Contract")["y"].mean().round(3).to_dict())
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
for lab, name in [(0, "stayed"), (1, "left")]:
    axs[0].hist(df.loc[df.y == lab, "tenure"], bins=36, alpha=0.6, density=True, label=name)
    axs[1].hist(df.loc[df.y == lab, "MonthlyCharges"], bins=36, alpha=0.6, density=True, label=name)
axs[0].set_xlabel("tenure (months)"); axs[1].set_xlabel("monthly charges"); axs[0].legend(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 가입자 → 떠날까

기준선 = 아무도 안 떠난다. 모델 = 그레이디언트 부스팅. 마케팅 팀이 **할인 쿠폰을 보낼 상위 500명** 을 고른다면, 그중 진짜 떠날 사람은 몇 명일까?"""),
("code", """from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = pd.get_dummies(df.drop(columns=["customerID", "Churn", "y"]), dtype=float)
Xtr, Xte, ytr, yte = train_test_split(X, df["y"].values, test_size=0.3, stratify=df["y"], random_state=0)
m = HistGradientBoostingClassifier(random_state=0, max_iter=200, learning_rate=0.05).fit(Xtr, ytr)
score = m.predict_proba(Xte)[:, 1]
print(f"nobody leaves: accuracy {np.mean(yte == 0):.1%}")
print(f"model: accuracy {np.mean((score > 0.5) == yte):.1%}, ROC-AUC {roc_auc_score(yte, score):.2f}")
top = np.argsort(-score)[:500]
print(f"top 500 by score: {yte[top].sum()} will leave ({yte[top].mean():.0%})  vs  random 500: about {500 * yte.mean():.0f}")"""),
("md", """**관찰**: 월 단위 계약 · 짧은 가입 기간 · 높은 요금이 이탈과 함께 간다. 실제로는 '떠날 사람' 보다 **'쿠폰을 주면 남을 사람'** 을 찾아야 돈이 된다 (어차피 떠날 사람, 어차피 남을 사람에게 주는 쿠폰은 낭비).

## 다른 문제로 바꿔 보기

- **언제 떠날까**: '떠났나' 대신 '몇 달 뒤 떠날까' 를 맞히면 생존 분석이 된다 (tenure 를 쓴다).
- 쿠폰 효과를 알려면 어떤 데이터가 더 필요할까? (쿠폰을 무작위로 준 실험)"""),
]),
# ------------------------------------------------------------------
dict(slug="credit_default", title="대출 상환 · 신용평가", sector="금융 · 핀테크", keywords=["신용평가 (CSS)", "대출 심사", "공정성 · 설명가능성"],
     teaser="카드 고객 3만 명. 다음 달 연체할 사람을 미리 알 수 있을까? 성별 · 나이를 써도 될까?",
     data="UCI Default of Credit Card Clients", data_src="Yeh & Lien (2009). 대만 카드 고객 30,000명의 2005년 4~9월 기록: 한도 · 성별 · 학력 · 혼인 · 나이 · 6달 상환 상태 · 청구액 · 납부액 → 다음 달 연체 여부 (Hugging Face 의 CSV 사본)",
     size="약 4MB",
     five=[("① 한 행", "카드 고객 한 명"), ("② 입력 X", "한도 · 지난 6달 상환 상태 · 청구액 · 납부액 (+ 성별 · 나이 · 학력?)"), ("③ 출력 y", "다음 달 연체할까"),
           ("④ 평가지표", "ROC-AUC · 재현율, 집단별 차이"), ("⑤ 기준선", "아무도 연체 안 한다")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """URL = "https://huggingface.co/datasets/Lancer73/uci-credit-card-default/resolve/main/UCI_Credit_Card.csv"
df = pd.read_csv(download(URL, "data/uci_credit_card.csv")).rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
print(df.shape, f"default rate {df['default'].mean():.1%}")
df.sample(5, random_state=0)"""),
("md", """## 열어 보기

`PAY_1` = 지난달 상환 상태 (−1 = 제때, 1 = 한 달 밀림, 2 = 두 달 …). 이게 가장 강한 단서일까? 성별 (1 = 남, 2 = 여) 마다 연체율은?"""),
("code", """print(df.groupby("PAY_1")["default"].agg(["mean", "size"]).round(3).T)
print(df.groupby("SEX")["default"].mean().round(3).to_dict(), "(1 = male, 2 = female)")
print(df.groupby(pd.cut(df["AGE"], [20, 30, 40, 50, 80]), observed=True)["default"].mean().round(3).to_dict())"""),
("md", """## 문제 하나 골라 보기: 고객 → 다음 달 연체할까

같은 모델을 두 가지로: 성별 · 나이 · 혼인 · 학력을 **넣을 때와 뺄 때**. 점수가 얼마나 다를까? 그리고 뺀 모델이 남녀에게 각각 얼마나 '연체 위험' 을 주는지 비교한다."""),
("code", """from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sensitive = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
features = [c for c in df.columns if c not in ("ID", "default")]
tr, te = train_test_split(df, test_size=0.3, stratify=df["default"], random_state=0)
print(f"nobody defaults: accuracy {np.mean(te['default'] == 0):.1%}")
for name, cols in [("all features", features), ("without sex/age/marriage/education", [c for c in features if c not in sensitive])]:
    m = HistGradientBoostingClassifier(random_state=0, max_iter=200, learning_rate=0.05).fit(tr[cols], tr["default"])
    s = m.predict_proba(te[cols])[:, 1]
    flag = s > 0.5
    by_sex = {k: f"{flag[(te['SEX'] == k).values].mean():.1%}" for k in (1, 2)}
    print(f"{name:36s} ROC-AUC {roc_auc_score(te['default'], s):.3f}  recall {flag[te['default'].values == 1].mean():.1%}  flagged by sex {by_sex}")"""),
("md", """**관찰**: 성별 · 나이를 빼도 점수가 거의 줄지 않는다. 상환 기록이 훨씬 강한 단서이기 때문이다. 그래도 뺀 모델이 집단마다 다르게 '위험' 을 줄 수 있다 (다른 열이 성별을 대신 알려 줄 때). 실제 대출 심사에서는 법 · 규제로 쓸 수 없는 정보가 정해져 있고, 거절 이유를 설명할 수 있어야 한다.

## 다른 문제로 바꿔 보기

- **한도 정하기**: '연체할까' 대신 '얼마까지 빌려줄까' 는 회귀 문제 + 손익 계산이다.
- 거절당한 사람은 데이터에 없다 (승인된 사람만 연체 기록이 생긴다). 이 치우침을 어떻게 다룰까?"""),
]),
# ------------------------------------------------------------------
dict(slug="ad_ctr", title="광고 클릭률 (CTR) 예측", sector="커머스 · 콘텐츠", keywords=["광고 랭킹", "클릭률 (CTR)", "아주 큰 데이터"],
     teaser="모바일 광고 노출 220만 번. 이 광고는 눌릴까?",
     data="Avazu CTR (원본 필드, 한 조각)", data_src="Avazu 모바일 광고 대회 (Kaggle 2014) 의 노출 기록 4,000만 줄 중 한 조각 220만 줄 (Hugging Face reczoo/Avazu_x4). 시각 · 광고 위치 · 사이트/앱 범주 · 기기 종류 등 (대부분 익명 해시)",
     size="약 105MB",
     five=[("① 한 행", "광고 노출 한 번"), ("② 입력 X", "시각 · 광고 위치 · 사이트/앱 범주 · 기기 · 광고 크기 …"), ("③ 출력 y", "눌렀나 (click)"),
           ("④ 평가지표", "로그 손실 · ROC-AUC"), ("⑤ 기준선", "모든 노출에 평균 클릭률")],
     cells=[
("md", """## 입장: 데이터 받기

약 105MB. 처음엔 1~2분. 220만 줄 중 50만 줄만 쓴다."""),
("code", """URL = "https://huggingface.co/api/datasets/reczoo/Avazu_x4/parquet/default/train/0.parquet"
df = pd.read_parquet(download(URL, "data/avazu_x4_0.parquet")).sample(500000, random_state=0)
df["day"] = df["hour"] // 100 % 100; df["hod"] = df["hour"] % 100
print(df.shape, f"CTR {df['click'].mean():.1%}", "| days:", sorted(df["day"].unique()))
df.sample(5, random_state=0).T.head(16)"""),
("md", """## 열어 보기

대부분의 열이 해시 (뜻을 알 수 없는 문자열) 다. 개인정보 · 영업 비밀 때문이다. 그래도 시각 · 광고 위치별 클릭률은 볼 수 있다."""),
("code", """fig, axs = plt.subplots(1, 3, figsize=(14, 3.2))
df.groupby("hod")["click"].mean().plot(ax=axs[0], marker="o", title="CTR by hour of day")
df.groupby("banner_pos")["click"].agg(["mean", "size"]).query("size > 500")["mean"].plot(kind="bar", ax=axs[1], title="CTR by banner position")
df["site_category"].value_counts().head(10).plot(kind="bar", ax=axs[2], title="top site categories (hashed)")
plt.tight_layout(); plt.show()
print("distinct values:", {c: df[c].nunique() for c in ["site_id", "app_id", "device_model", "device_ip"]})"""),
("md", """## 문제 하나 골라 보기: 노출 → 클릭할까 (확률)

**날짜 순서**: 마지막 날 시험, 그 전 날들 학습. 범주마다 원-핫 (드문 값은 '기타' 로 묶는다) + 로지스틱회귀. 광고는 순서 (어떤 광고를 먼저 보여 줄까) 와 확률 (얼마를 받을까) 이 둘 다 중요해서 로그 손실과 AUC 를 같이 본다."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

cats = ["C1", "banner_pos", "site_id", "site_category", "app_id", "app_category", "device_model", "device_type", "device_conn_type",
        "C14", "C15", "C16", "C18", "C19", "C21", "hod"]
last = df["day"].max(); tr, te = df[df["day"] < last], df[df["day"] == last]
enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100).fit(tr[cats].astype(str))
m = LogisticRegression(C=0.3, max_iter=1000).fit(enc.transform(tr[cats].astype(str)), tr["click"])
p = m.predict_proba(enc.transform(te[cats].astype(str)))[:, 1]
base = np.full(len(te), tr["click"].mean())
print(f"train {len(tr):,} rows, test (day {last}) {len(te):,} rows")
print(f"baseline (CTR {tr['click'].mean():.3f}): log loss {log_loss(te['click'].values, base):.4f}, ROC-AUC 0.5")
print(f"one-hot logistic          : log loss {log_loss(te['click'].values, p):.4f}, ROC-AUC {roc_auc_score(te['click'], p):.3f}")"""),
("md", """**관찰**: 로그 손실이 조금 준 것도 실제 광고에서는 큰돈이다 (노출이 하루 수십억 번). 해시 열은 뜻을 몰라도 '자주 나오는 값' 이면 그대로 단서가 된다.

## 다른 문제로 바꿔 보기

- 같은 사람이 같은 광고를 열 번 봤다면? '피로도' 를 입력에 넣으려면 한 행이 '사람 × 광고 × 시각' 이어야 한다.
- 클릭이 아니라 **구매 (전환)** 를 맞히면 양성이 훨씬 드물어진다. 지표는?"""),
]),
# ------------------------------------------------------------------
dict(slug="hiring_match", title="채용: 이력서 · 공고 매칭", sector="전문 서비스", keywords=["HR 테크", "텍스트 매칭", "공정성"],
     teaser="이력서와 채용 공고 한 쌍. 이 사람은 이 자리에 맞을까?",
     data="Resume - Job Description Fit", data_src="Hugging Face cnamuangtoun/resume-job-description-fit: 영어 이력서 글 + 채용 공고 글 쌍 약 8천 개, 맞음 (Good Fit) · 가능 (Potential Fit) · 안 맞음 (No Fit) 세 단계",
     size="약 12MB",
     five=[("① 한 행", "(이력서, 공고) 한 쌍"), ("② 입력 X", "이력서 글 · 공고 글 (단어 빈도, 두 글의 겹침)"), ("③ 출력 y", "맞음 / 가능 / 안 맞음"),
           ("④ 평가지표", "정확도 + 단계별 재현율"), ("⑤ 기준선", "가장 흔한 단계")],
     cells=[
("md", """## 입장: 데이터 받기"""),
("code", """BASE = "https://huggingface.co/api/datasets/cnamuangtoun/resume-job-description-fit/parquet/default/"
train = pd.read_parquet(download(BASE + "train/0.parquet", "data/resume_fit_train.parquet"))
test = pd.read_parquet(download(BASE + "test/0.parquet", "data/resume_fit_test.parquet"))
print(train.shape, test.shape)
print(train["label"].value_counts().to_dict())
print("\\nRESUME:", train["resume_text"].iloc[0][:400], "\\n\\nJOB:", train["job_description_text"].iloc[0][:400])"""),
("md", """## 열어 보기

글의 길이, 그리고 **두 글에 같이 나오는 단어의 비율** 이 '맞음' 일수록 높을까?"""),
("code", """import re
def words(t):
    return set(re.findall(r"[a-z][a-z+#.]{1,}", t.lower()))

def overlap(r, j):
    jw = words(j); return len(words(r) & jw) / max(len(jw), 1)

train["overlap"] = [overlap(r, j) for r, j in zip(train["resume_text"], train["job_description_text"])]
print(train.groupby("label")["overlap"].describe()[["mean", "50%"]].round(3))
print("resume words: median", int(train["resume_text"].str.split().str.len().median()))"""),
("md", """## 문제 하나 골라 보기: (이력서, 공고) → 맞음 / 가능 / 안 맞음

이력서 · 공고를 각각 단어 빈도 (TF-IDF) 로 바꾸고, 두 글의 코사인 유사도를 더해 로지스틱회귀. 데이터가 나눠 둔 학습 · 시험을 그대로 쓴다."""),
("code", """from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
vec.fit(pd.concat([train["resume_text"], train["job_description_text"]]))

def pair_features(d, vec):
    R, J = vec.transform(d["resume_text"]), vec.transform(d["job_description_text"])
    cos = np.asarray(R.multiply(J).sum(1))
    return hstack([R, J, csr_matrix(cos * 10)]).tocsr()

m = LogisticRegression(C=2, max_iter=3000).fit(pair_features(train, vec), train["label"])
pred = m.predict(pair_features(test, vec))
most = train["label"].value_counts().idxmax()
print(f"baseline (always '{most}'): accuracy {np.mean(test['label'] == most):.1%}")
print(f"TF-IDF pair + logistic : accuracy {np.mean(test['label'] == pred):.1%}")
print("recall per label:", pd.Series(test["label"].values == pred).groupby(test["label"].values).mean().round(2).to_dict())"""),
("md", """**관찰**: 단어가 겹치는지만으로는 '맞음' 을 잘 가려내지 못한다. 같은 일을 다른 말로 쓰기도 하고 (예: ML ↔ machine learning), '가능' 은 사람이 매겨도 애매하다. 문장 임베딩 (14주차) 을 쓰면 나아진다.

## 다른 문제로 바꿔 보기

- **공정성**: 이력서의 이름 · 학교 · 성별을 보고 모델이 판단한다면? 실제로 이 문제로 채용 AI 를 접은 회사가 있다. 무엇을 입력에서 빼야 할까, 빼면 충분할까?
- **검색 문제로**: 공고 하나에 지원자 1,000명. 출력 = 면접 볼 10명의 순서. 지표 = 상위 10명 중 '맞음' 의 수."""),
]),
# ------------------------------------------------------------------
dict(slug="traffic_speed", title="도로 교통 속도 예측 (내비)", sector="도시 · 공간", keywords=["내비게이션 · 도착시간", "시공간 그래프", "교통 관제"],
     teaser="로스앤젤레스 고속도로 센서 207개. 15분 뒤 이 구간은 막힐까?",
     data="METR-LA (시험 기간)", data_src="Li et al. (2018). 로스앤젤레스 고속도로 루프 센서 207개의 5분 간격 평균 속도 (mph), 2012년 3~6월. 센서 위치 · 도로 거리로 만든 연결 그래프 포함 (Hugging Face witgaw/METR-LA, 시험 기간 2012-06-04 ~ 06-27 만 받는다)",
     size="약 43MB",
     five=[("① 한 행", "센서 × 5분"), ("② 입력 X", "그 센서의 지난 1시간 속도 + 이웃 센서 속도 + 시각"), ("③ 출력 y", "15분 뒤 속도 (mph)"),
           ("④ 평가지표", "MAE (mph), 막힘 (< 40 mph) 재현율"), ("⑤ 기준선", "지금 속도 그대로")],
     cells=[
("md", """## 입장: 데이터 받기

데이터가 이미 '지난 12칸 (1시간) → 다음 12칸' 창으로 잘려 있다. 필요한 열만 읽는다. 속도 0 은 '센서 고장 · 빈칸' 이다 (1주 3차시)."""),
("code", """BASE = "https://huggingface.co/datasets/witgaw/METR-LA/resolve/main/"
cols = ["node_id", "t0_timestamp", "x_t+0_d1"] + [f"x_t-{k}_d0" for k in range(11, 0, -1)] + ["x_t+0_d0", "y_t+3_d0"]
df = pd.read_parquet(download(BASE + "test.parquet", "data/metrla_test.parquet"), columns=cols)
loc = pd.read_csv(download(BASE + "sensor_graph/sensor_locations.csv", "data/metrla_locations.csv"))
df["time"] = pd.to_datetime(df["t0_timestamp"])
print(df.shape, df["time"].min(), "~", df["time"].max(), "| sensors", df["node_id"].nunique())
print(f"zero (missing) speeds now: {(df['x_t+0_d0'] == 0).mean():.1%}")
loc.head(3)"""),
("md", """## 열어 보기: 지도와 하루의 속도

센서 위치 (위도 · 경도) 와, 한 센서의 며칠치 속도. 출퇴근 시간에 푹 꺼질까?"""),
("code", """fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
mean_speed = df[df["x_t+0_d0"] > 0].groupby("node_id")["x_t+0_d0"].mean()
axs[0].scatter(loc.iloc[:, -1], loc.iloc[:, -2], c=mean_speed.reindex(range(len(loc))).values, cmap="RdYlGn", s=18)
axs[0].set_title("sensors (colour = mean speed)"); axs[0].set_xlabel("longitude"); axs[0].set_ylabel("latitude")
one = df[df["node_id"] == 5].set_index("time")["x_t+0_d0"].replace(0, np.nan)
one[:"2012-06-08"].plot(ax=axs[1], title="sensor 5, one week (mph)")
plt.tight_layout(); plt.show()"""),
("md", """## 문제 하나 골라 보기: 지금까지 → 15분 뒤 속도

**시간 순서**: 앞 70% 기간 학습, 뒤 30% 시험. 빈칸 (0) 인 행은 뺀다.
- 기준선: 지금 속도 그대로
- 모델 1: 그 센서의 지난 1시간 속도 12개 + 시각 → 선형회귀
- 모델 2: 모델 1 + **연결 그래프의 이웃 센서들 평균 속도** (13주차 그래프 신경망의 아이디어)"""),
("code", """from sklearn.linear_model import LinearRegression

adj = np.load(download(BASE + "sensor_graph/adj_mx.npy", "data/metrla_adj.npy"), allow_pickle=True)
adj = adj[2] if adj.dtype == object else adj                  # some versions store (ids, id_map, matrix)
W = np.asarray(adj, dtype=float); np.fill_diagonal(W, 0); W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
now = df.pivot(index="time", columns="node_id", values="x_t+0_d0").replace(0, np.nan)
neigh = pd.DataFrame(np.nan_to_num(now.values) @ W.T / np.maximum((~now.isna()).values @ (W.T > 0), 1), index=now.index, columns=now.columns)
df["neigh"] = neigh.stack().reindex(pd.MultiIndex.from_arrays([df["time"], df["node_id"]])).values

lags = [f"x_t-{k}_d0" for k in range(11, 0, -1)] + ["x_t+0_d0"]
d = df[(df[lags + ["y_t+3_d0"]] > 0).all(axis=1)].copy()
cut = d["time"].quantile(0.7); tr, te = d[d["time"] <= cut], d[d["time"] > cut]
y = te["y_t+3_d0"].values; jam = y < 40
for name, cols_ in [("now", None), ("own past hour + time", lags + ["x_t+0_d1"]), ("+ neighbours", lags + ["x_t+0_d1", "neigh"])]:
    pred = te["x_t+0_d0"].values if cols_ is None else LinearRegression().fit(tr[cols_], tr["y_t+3_d0"]).predict(te[cols_])
    print(f"{name:22s} MAE {np.mean(np.abs(pred - y)):.2f} mph   jam (<40) recall {np.mean(pred[jam] < 40):.0%}")"""),
("md", """**관찰**: 15분 뒤는 '지금 그대로' 가 아주 강한 기준선이다. 지난 한 시간을 보면 조금 낫고, 이웃 센서의 단순 평균을 더해도 거의 차이가 없다. 15분은 너무 가까워서 자기 과거로 충분하기 때문이다. 목표를 1시간 뒤 (`y_t+12_d0`) 로 바꾸고 다시 해 보자. 그래프 신경망 (13주차, DCRNN) 은 '어느 이웃이 얼마나 중요한지' 까지 배운다.

## 다른 문제로 바꿔 보기

- **도착 시간 (ETA)**: 한 행 = 출발지 → 도착지 길 하나, 출력 = 걸리는 분. 길 위 센서들의 예측 속도를 이어 붙인다.
- 사고 · 공사 같은 '이벤트' 는 과거 속도에 없다. 무엇을 입력에 더해야 할까?"""),
]),
# ------------------------------------------------------------------
dict(slug="corporate_card", title="법인카드 지출 관리", sector="금융 · 핀테크", keywords=["지출 관리 (spend management)", "비용 자동 분류", "이상 지출 찾기"],
     teaser="공공기관 법인카드 결제 44만 건. 가맹점 이름만 보고 무슨 비용인지 자동으로 나눌 수 있을까?",
     data="Oklahoma Purchase Card (P-card) FY2014", data_src="미국 오클라호마 주정부 공개 데이터: 주 정부 기관 · 대학 124곳의 2013-07 ~ 2014-06 법인카드 결제 442,458건 (기관 · 적요 · 금액 · 가맹점 · 날짜 · 가맹점 업종 코드 MCC). 카드 소지자 이름 열은 노트북에서 바로 지운다",
     size="약 80MB",
     five=[("① 한 행", "법인카드 결제 한 건"), ("② 입력 X", "가맹점 이름 (글자)"), ("③ 출력 y", "비용 항목 (사무용품 · 출장 · 장비 · 통신 …)"),
           ("④ 평가지표", "정확도 + 항목별 재현율"), ("⑤ 기준선", "가장 흔한 항목")],
     cells=[
("md", """## 입장: 데이터 받기

약 80MB. 사람 이름 (카드 소지자) 은 분석에 필요 없으니 받자마자 지운다 (개인정보 최소화)."""),
("code", """URL = ("https://data.ok.gov/dataset/5f1e52fa-670d-4bb6-9072-4801e8c9ba17/resource/506a16ef-fcb6-4b03-8122-57cec56eb1b9"
       "/download/data-purchase-card-pcard-fiscal-year-2014.csv")
df = pd.read_csv(download(URL, "data/ok_pcard_2014.csv"), low_memory=False)
df = df.drop(columns=["Cardholder Last Name", "Cardholder First Initial"])
df = df.rename(columns={"Agency Name": "agency", "Description": "memo", "Amount": "amount", "Vendor": "vendor",
                        "Transaction Date": "date", "Merchant Category Code (MCC)": "mcc"})
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %I:%M:%S %p")
print(df.shape, df["date"].min().date(), "~", df["date"].max().date())
df[["agency", "memo", "amount", "vendor", "date", "mcc"]].sample(6, random_state=0)"""),
("md", """## 열어 보기

금액 분포 (음수 = 환불), 업종 코드 (MCC) 가 몇 가지일까? 435가지는 너무 잘다. 회계 담당자가 쓰는 **비용 항목 10개 정도** 로 묶어 보자 (규칙은 업종 이름의 단어)."""),
("code", """print(df["amount"].describe().round(1).to_dict())
print(f"refunds (negative): {(df['amount'] < 0).mean():.1%} | distinct MCC: {df['mcc'].nunique()}")
RULES = [("travel", ["AIRLINE", "AIR CARRIER", "HOTEL", "MOTEL", "LODGING", "TRAVEL AGENC", "CAR RENTAL", "MARRIOTT", "HILTON", "HOLIDAY INN", "SHERATON", "HYATT", "BEST WESTERN", "TAXI", "PARKING"]),
         ("office & books", ["STATIONERY", "OFFICE", "BOOK", "PRINTING", "NEWSPAPER", "PERIODICAL"]),
         ("food", ["GROCERY", "RESTAURANT", "EATING", "FAST FOOD", "CATERER", "BAKER", "DAIRY", "CANDY"]),
         ("medical & lab", ["MEDICAL", "DENTAL", "DRUG", "OPHTHALMIC", "HOSPITAL", "LABORATOR"]),
         ("IT & electronics", ["COMPUTER", "SOFTWARE", "ELECTRONIC", "TELECOMMUNICATION", "CABLE", "TELEPHONE"]),
         ("tools & facilities", ["HARDWARE", "INDUSTRIAL", "PLUMBING", "ELECTRICAL", "HOME SUPPLY", "LUMBER", "PAINT", "EQUIPMENT", "AUTOMOTIVE", "TIRE"]),
         ("shipping & services", ["COURIER", "POSTAL", "BUSINESS SERVICE", "PROFESSIONAL SERVICE", "ADVERTISING", "GOVERNMENT SERVICE", "CLEANING"]),
         ("retail & catalog", ["RETAIL", "CATALOG", "DIRECT MARKET", "DIRCT MARKET", "DEPARTMENT STORE", "DISCOUNT STORE", "WHOLESALE"])]

def expense_group(mcc):
    m = str(mcc).upper()
    for name, words in RULES:
        if any(w in m for w in words):
            return name
    return "other"

df["group"] = df["mcc"].map(expense_group)
print(df["group"].value_counts(normalize=True).round(3).to_dict())"""),
("md", """## 문제 하나 골라 보기: 가맹점 이름 → 비용 항목

지출 관리 서비스는 결제가 들어오면 **비용 항목을 자동으로** 붙여 준다. 가맹점 이름의 글자 조각 (2~4글자) 으로 로지스틱회귀.
**시간 순서**: 2014년 3월까지 학습, 4~6월 시험 (실제로는 미래의 결제를 분류하니까). 시간을 줄이려고 학습은 15만 건만."""),
("code", """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

tr = df[df["date"] < "2014-04-01"].sample(150000, random_state=0)
te = df[df["date"] >= "2014-04-01"]
model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=200000),
                      LogisticRegression(C=3, max_iter=2000)).fit(tr["vendor"].astype(str), tr["group"])
pred = model.predict(te["vendor"].astype(str))
most = tr["group"].value_counts().idxmax()
print(f"baseline (always '{most}'): accuracy {np.mean(te['group'] == most):.1%}")
print(f"vendor-name character n-grams: accuracy {np.mean(te['group'] == pred):.1%}")
print("recall per group:", pd.Series(te["group"].values == pred).groupby(te["group"].values).mean().round(2).to_dict())
new_vendor = ~te["vendor"].isin(set(tr["vendor"]))
print(f"vendors never seen in training: {new_vendor.mean():.0%} of test, accuracy on them {np.mean(te['group'][new_vendor] == pred[new_vendor]):.1%}")"""),
("md", """### 이상 지출 찾기: 규칙으로 먼저

라벨이 없는 문제다. 감사 담당자가 먼저 보는 규칙 몇 가지를 세어 보자.
- 주말 결제
- 같은 카드 소지자 · 같은 가맹점 · 같은 날 여러 번 (한도를 피하려는 **쪼개기 결제** 의심) — 여기서는 이름을 지웠으니 '같은 기관 · 같은 가맹점 · 같은 날 · 같은 금액' 으로 센다
- 정확히 딱 떨어지는 큰 금액 (예: 5,000.00)"""),
("code", """df["weekend"] = df["date"].dt.dayofweek >= 5
dup = df.groupby(["agency", "vendor", "date", "amount"]).size()
print(f"weekend payments: {df['weekend'].mean():.1%}")
print(f"(agency, vendor, day, amount) repeated 3+ times: {(dup >= 3).sum():,} groups, {dup[dup >= 3].sum():,} payments")
print(f"round amounts >= 1,000: {((df['amount'] >= 1000) & (df['amount'] % 100 == 0)).sum():,}")
print(df[df["weekend"]].groupby("group").size().sort_values(ascending=False).head(5).to_dict())"""),
("md", """**관찰**: 가맹점 이름만으로도 비용 항목을 꽤 맞히지만, **처음 보는 가맹점** 에서는 떨어진다. 실제 서비스는 가맹점 업종 코드 · 금액 · 적요 · 영수증 사진까지 함께 쓴다. 규칙으로 찾은 '의심 결제' 는 대부분 정상이다 (주말 출장 등). 그래서 이상 탐지의 지표는 '담당자가 하루에 검토할 수 있는 건수 중 진짜 문제의 비율' 이 된다.

## 다른 문제로 바꿔 보기

- **예산 예측**: 한 행 = 기관 × 달, 출력 = 다음 달 카드 지출 총액.
- **영수증 읽기**: 영수증 사진 → 가맹점 · 금액 · 날짜 (문서 이해 노트북과 연결)."""),
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
    "document_types": [("Harley, Ufkes & Derpanis (2015) Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval, ICDAR (RVL-CDIP)", "https://arxiv.org/abs/1502.07058")],
    "ai_usecases": [("Handa et al. (2025) Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations", "https://arxiv.org/abs/2503.04761")],
    "movie_ratings": [("Harper & Konstan (2015) The MovieLens Datasets: History and Context, ACM TiiS", "https://doi.org/10.1145/2827872")],
    "korean_reviews": [("NSMC 저장소 (Lucy Park, 2015): 데이터 설명 · 만든 방법", "https://github.com/e9t/nsmc")],
    "link_prediction": [("Liben-Nowell & Kleinberg (2007) The link-prediction problem for social networks, JASIST", "https://doi.org/10.1002/asi.20591"),
                        ("Yin, Benson, Leskovec & Gleich (2017) Local Higher-Order Graph Clustering, KDD (email-Eu-core)", "https://doi.org/10.1145/3097983.3098069")],
    "taxi_demand": [("Yao et al. (2018) Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction, AAAI", "https://arxiv.org/abs/1802.08714"),
                    ("NYC TLC Trip Record Data (데이터 원본 · 설명)", "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")],
    "predictive_maintenance": [("Matzka (2020) Explainable Artificial Intelligence for Predictive Maintenance Applications, AI4I", "https://doi.org/10.1109/AI4I49448.2020.00023")],
    "steel_defects": [("UCI Steel Plates Faults 데이터 설명 (Semeion Research Center)", "https://archive.ics.uci.edu/dataset/198/steel+plates+faults")],
    "bearing_vibration": [("Smith & Randall (2015) Rolling element bearing diagnostics using the CWRU data: A benchmark study, MSSP", "https://doi.org/10.1016/j.ymssp.2015.04.021"),
                          ("CWRU Bearing Data Center (데이터 원본 · 실험 장치 설명)", "https://engineering.case.edu/bearingdatacenter")],
    "aml_transactions": [("Altman et al. (2023) Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS Datasets and Benchmarks", "https://arxiv.org/abs/2306.16424")],
    "real_estate": [("Pace & Barry (1997) Sparse spatial autoregressions, Statistics & Probability Letters", "https://doi.org/10.1016/S0167-7152(96)00140-X")],
    "earthquakes": [("U.S. Geological Survey (2017) Advanced National Seismic System (ANSS) Comprehensive Catalog", "https://doi.org/10.5066/F7MS3QZH")],
    "stock_index": [("Fama (1970) Efficient Capital Markets: A Review of Theory and Empirical Work, Journal of Finance", "https://doi.org/10.2307/2325486")],
    "point_cloud": [('Wu et al. (2015) 3D ShapeNets: A Deep Representation for Volumetric Shapes, CVPR', 'https://arxiv.org/abs/1406.5670')],
    "medical": [('Yang et al. (2023) MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification, Scientific Data', 'https://arxiv.org/abs/2110.14795')],
    "satellite": [('Helber et al. (2019) EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification, IEEE JSTARS', 'https://arxiv.org/abs/1709.00029')],
    "air_quality": [("Liang et al. (2015) Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating, Proc. R. Soc. A", 'https://doi.org/10.1098/rspa.2015.0257')],
    "electricity": [('UCI 데이터 설명 (Hebrail & Berard, 2012)', 'https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption')],
    "football": [('Decroos et al. (2019) Actions Speak Louder Than Goals: Valuing Player Actions in Soccer, KDD', 'https://arxiv.org/abs/1802.07127')],
    "chess": [('McIlroy-Young et al. (2020) Aligning Superhuman AI with Human Behavior: Chess as a Model System (Maia), KDD', 'https://arxiv.org/abs/2006.01855')],
    "speech": [('Warden (2018) Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition', 'https://arxiv.org/abs/1804.03209')],
    "crop": [('Hughes & Salathé (2015) An open access repository of images on plant health to enable the development of mobile disease diagnostics', 'https://arxiv.org/abs/1511.08060')],
    "cyber_security": [("Tavallaee et al. (2009) A detailed analysis of the KDD CUP 99 data set, IEEE CISDA (NSL-KDD)", "https://doi.org/10.1109/CISDA.2009.5356528")],
    "telco_churn": [("IBM 샘플 데이터 저장소 (Telco Customer Churn)", "https://github.com/IBM/telco-customer-churn-on-icp4d")],
    "ad_ctr": [("McMahan et al. (2013) Ad Click Prediction: a View from the Trenches, KDD", "https://doi.org/10.1145/2487575.2488200")],
    "credit_default": [("Yeh & Lien (2009) The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients, Expert Systems with Applications", "https://doi.org/10.1016/j.eswa.2007.12.020")],
    "hiring_match": [("Hugging Face 데이터 설명 (resume-job-description-fit)", "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit")],
    "traffic_speed": [("Li et al. (2018) Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR (METR-LA)", "https://arxiv.org/abs/1707.01926")],
    "corporate_card": [("Oklahoma Open Data: Purchase Card (PCard) Fiscal Year 2014", "https://data.ok.gov/dataset/purchase-card-pcard-fiscal-year-2014")],
    "semiconductor_yield": [("UCI SECOM 데이터 설명 (McCann & Johnston, 2008)", "https://archive.ics.uci.edu/dataset/179/secom")],
}

# 이 데이터에 관심 있을 곳: (업종, 직무). 특정 회사 이름은 쓰지 않는다
FIT = {
    "fashion_commerce": ("온라인 패션 · 이커머스 플랫폼, 리테일 · 패션 브랜드", "추천 시스템 엔지니어, 비전 AI 엔지니어, 상품 · MD 데이터 분석가"),
    "shop_conversion": ("이커머스 · D2C 브랜드 (뷰티 등), 광고 · 마케팅 테크", "그로스 · 마케팅 데이터 분석가, CRM 데이터 사이언티스트"),
    "movie_ratings": ("OTT · 동영상 스트리밍, 음악 · 웹툰 · 콘텐츠 플랫폼", "추천 · 랭킹 엔지니어, 개인화 ML 엔지니어"),
    "korean_reviews": ("포털 · 검색, 커머스 리뷰 서비스, 브랜드 모니터링 · 마케팅 에이전시", "NLP 엔지니어, VOC (고객의 소리) 데이터 분석가"),
    "card_fraud": ("카드사 · 은행 · 증권, 간편결제 · 핀테크, 금융보안", "FDS (이상거래탐지) 분석가, 리스크 모델러, 자금세탁방지 (AML) 컴플라이언스"),
    "support_routing": ("고객 상담 · CX 소프트웨어, 콜센터, 금융 · 통신 고객센터", "대화형 AI · LLM 엔지니어, NLP 엔지니어, CX 데이터 분석가"),
    "hotel_booking": ("여행 · 숙박 플랫폼, 항공 · 호텔, 모빌리티", "가격 (레비뉴) 매니지먼트 분석가, 수요예측 데이터 사이언티스트"),
    "link_prediction": ("SNS · 메신저, 커리어 네트워크, 커머스 (함께 산 상품), 금융 (거래 연결망)", "그래프 ML 엔지니어, 추천 엔지니어"),
    "taxi_demand": ("모빌리티 · 택시 · 배차 플랫폼, 배달 · 물류, 지도 · 내비게이션", "시공간 수요예측 데이터 사이언티스트, 배차 · 운영 최적화 엔지니어"),
    "aml_transactions": ("은행 · 증권 · 카드사, 핀테크 · 가상자산 거래소, 금융 감독 · 규제", "자금세탁방지 (AML) 분석가, 그래프 ML 엔지니어, 컴플라이언스"),
    "real_estate": ("부동산 플랫폼 · 프롭테크, 은행 (담보 평가), 건설 · 도시계획", "부동산 데이터 분석가, 공간 데이터 사이언티스트, 자동 가치 평가 (AVM) 모델러"),
    "earthquakes": ("재난 · 안전 (기상 · 지질 기관), 보험 · 재보험, 건설 · 인프라", "시공간 데이터 사이언티스트, 재난 위험 모델러, 지구물리 데이터 분석가"),
    "stock_index": ("증권 · 자산운용, 퀀트 · 핀테크 투자, 금융 데이터 서비스", "퀀트 리서처, 리스크 매니저, 금융 데이터 엔지니어"),
    "point_cloud": ('자율주행 · 로봇, 3D 스캐너 · 디지털 트윈, 건설 · 측량', '3D 비전 엔지니어, LiDAR 인식 엔지니어'),
    "medical": ('의료 AI · 병원, 의료기기, 보험 (심사)', '의료 영상 AI 엔지니어, 임상 데이터 사이언티스트'),
    "satellite": ('우주 · 위성 서비스, 농업 · 환경, 지도 · 도시계획', '원격 탐사 데이터 사이언티스트, 지리공간 (GIS) 엔지니어'),
    "air_quality": ('기상 · 환경 기관, 지자체, 공기청정 · 가전', '환경 데이터 분석가, 시계열 예측 데이터 사이언티스트'),
    "electricity": ('전력 · 에너지 공기업, 스마트홈 · 가전, 에너지 거래 플랫폼', '에너지 데이터 분석가, 수요예측 데이터 사이언티스트'),
    "football": ('프로 구단 · 스포츠 데이터 회사, 스포츠 중계 · 미디어, 게임 (스포츠)', '스포츠 데이터 분석가, 게임 데이터 사이언티스트'),
    "chess": ('게임 회사, e스포츠 · 매칭 시스템, 교육용 AI', '게임 데이터 사이언티스트, 추천 · 매칭 엔지니어'),
    "speech": ('AI 스피커 · 가전, 자동차 (음성 제어), 통신사', '음성 인식 엔지니어, 온디바이스 AI 엔지니어'),
    "crop": ('스마트팜 · 농업 기술, 종자 · 비료, 공공 연구 (농촌진흥청 등)', '농업 AI 엔지니어, 비전 AI 엔지니어'),
    "cyber_security": ("보안 · 통신 · 클라우드, 금융 · 공공 보안관제, 보안 솔루션", "보안관제 (SOC) 분석가, 침입 탐지 · 위협 탐지 엔지니어"),
    "telco_churn": ("통신사 · 구독 서비스 (OTT · 음원 · SaaS), 보험 · 카드", "CRM · 고객 분석가, 리텐션 마케팅 데이터 사이언티스트"),
    "ad_ctr": ("포털 · 검색 광고, 커머스 광고, 광고 플랫폼 (애드테크)", "광고 추천 · 랭킹 엔지니어, 그로스 데이터 사이언티스트"),
    "credit_default": ("은행 · 카드 · 인터넷전문은행, 핀테크 대출, 신용평가사", "신용 리스크 모델러, 금융 데이터 사이언티스트"),
    "hiring_match": ("채용 플랫폼 · HR 테크, 대기업 인사팀, 헤드헌팅", "HR 데이터 분석가, 검색 · 매칭 (NLP) 엔지니어"),
    "traffic_speed": ("지도 · 내비게이션, 모빌리티 · 물류, 지자체 교통 관제", "시공간 데이터 사이언티스트, 경로 · 도착시간 예측 엔지니어"),
    "corporate_card": ("지출 관리 · 법인카드 핀테크, 회계 · 감사, 기업 재무팀 · ERP", "핀테크 데이터 사이언티스트, 재무 · 감사 데이터 분석가"),
    "semiconductor_yield": ("반도체 · 디스플레이 제조, 배터리 · 소재, 스마트팩토리 솔루션", "공정 데이터 엔지니어, 수율 · 품질 분석가, 제조 AI 엔지니어"),
    "predictive_maintenance": ("중공업 · 조선 · 자동차, 발전 · 플랜트, 산업 IoT 솔루션", "예지보전 데이터 사이언티스트, 설비 · 신뢰성 엔지니어"),
    "steel_defects": ("철강 · 금속, 자동차 부품, 비전 검사 장비", "비전 검사 AI 엔지니어, 품질 엔지니어"),
    "bearing_vibration": ("모터 · 기계 · 풍력, 철도 · 항공 정비, 산업 센서 · 진단 솔루션", "신호처리 · 진단 엔지니어, 예지보전 엔지니어"),
    "document_types": ("회계 · 감사 법인, 은행 · 보험 (서류 심사), 문서 AI · RPA 솔루션", "문서 AI 엔지니어, 업무 자동화 (RPA) 엔지니어"),
    "ai_usecases": ("AI 전환 (AX) 컨설팅, 기업 디지털 전략 · 혁신 조직, 공공 정책 연구", "AI 전략 컨설턴트, 데이터 기반 기획 · 정책 분석가"),
}

SECTIONS = ["커머스 · 콘텐츠", "금융 · 핀테크", "플랫폼 · 서비스", "도시 · 공간", "제조", "이미지 · 소리 · 3D", "스포츠 · 게임", "전문 서비스"]
assert {it["slug"] for it in SHOP} <= set(PAPERS) and set(PAPERS) == set(FIT), set(PAPERS) ^ {it["slug"] for it in SHOP}
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


def fetch_cover(slug, out_path):
    """커버 사진 (Wikimedia Commons, covers.json) 을 960x300 쯤으로 잘라 저장. 받은 원본은 data/covers/ 에 남긴다"""
    import io, shutil, urllib.request
    from PIL import Image
    covers = json.load(open(os.path.join(HERE, "covers.json")))
    if slug not in covers:                       # 자금세탁: 직접 그린 패턴 그림
        shutil.copy(os.path.join(ROOT, "docs", "datashop", "worldcup", "img", f"{slug}.jpg"), out_path); return None
    c = covers[slug]; cache = os.path.join(HERE, c["local"]) if c.get("local") else os.path.join(HERE, "data", "covers", slug + ".jpg")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        req = urllib.request.Request(c["thumb"], headers={"User-Agent": "ML course datashop"})
        open(cache, "wb").write(urllib.request.urlopen(req, timeout=120).read())
    im = Image.open(cache).convert("RGB"); w, h = 1200, 340
    if c.get("local"):                                 # a diagram: fit it inside (no cropping), white margins
        r = min(w / im.width, h / im.height) * 0.96; im = im.resize((round(im.width * r), round(im.height * r)))
        canvas = Image.new("RGB", (w, h), "white"); canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); canvas.save(out_path, quality=85)
    else:
        r = max(w / im.width, h / im.height); im = im.resize((round(im.width * r), round(im.height * r)))
        l, t = (im.width - w) // 2, (im.height - h) // 2
        im.crop((l, t, l + w, t + h)).save(out_path, quality=78)
    if not c.get("page"):
        return None                                   # drawn by us, no credit line
    return dict(page=c["page"], artist=c["artist"], license=c["license"])


def build_site(items):
    """docs/datashop/: 목록 + 데이터마다 페이지 (커버 · 문제 그림 · 샘플 · 통계 · 그래프 · 결과 · 논문 · 관심 업종) + 준비 중 페이지"""
    import shutil, previews, site_pages, stubs
    out = os.path.join(ROOT, "docs", "datashop")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "shop.css"), "w", encoding="utf-8").write(site_pages.CSS)
    shutil.copy(os.path.join(HERE, "favicon.svg"), os.path.join(out, "favicon.svg"))
    types = {c["id"]: c["type"] for c in json.load(open(os.path.join(HERE, "worldcup", "candidates.json")))}
    page_items = []
    for it in items:
        page_items.append(dict(it, papers=PAPERS[it["slug"]], fit=FIT[it["slug"]], total=len(items), type=types.get(it["slug"], ""),
                               colab=COLAB.format(it["file"]), github=GITHUB.format(it["file"])))
    for k, it in enumerate(page_items):
        d = os.path.join(out, it["slug"]); os.makedirs(d, exist_ok=True)
        it["credit"] = fetch_cover(it["slug"], os.path.join(d, "cover.jpg"))
        shutil.copy(os.path.join(HERE, it["file"] + ".ipynb"), os.path.join(d, it["file"] + ".ipynb"))
        pv = getattr(previews, it["slug"])(os.path.join(d, "thumb.png"))
        res = previews.result_lines(os.path.join(HERE, it["file"] + ".ipynb"))
        prev = page_items[k - 1] if k > 0 else None
        nxt = page_items[k + 1] if k + 1 < len(page_items) else None
        open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(site_pages.render_item(it, prev, nxt, pv, res))
        print("page", it["slug"], len(pv["charts"]), "charts,", len(res), "result lines")
    stub_items = []
    for k, st in enumerate(stubs.STUBS):
        it = dict(st, num=f"{len(items) + k + 1:02d}", type=types.get(st["slug"], ""))
        d = os.path.join(out, it["slug"]); os.makedirs(d, exist_ok=True)
        it["credit"] = fetch_cover(it["slug"], os.path.join(d, "cover.jpg"))
        open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(site_pages.render_stub(it))
        stub_items.append(it); print("stub", it["slug"])
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(site_pages.render_index(page_items, SECTIONS, stub_items))
    print("wrote", out)


if __name__ == "__main__":
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    run = "--no-run" not in sys.argv and "--pages" not in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]          # 이름 일부를 주면 그 노트북만
    if "--pages" not in sys.argv:
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
    if "--nb-only" not in sys.argv:
        build_site(SHOP)
