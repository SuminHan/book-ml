# 2026-10-16 데이터 가게에서 뺀 항목 (카드 이상거래 · 법인카드 · 체스 · 지진)
# ------------------------------------------------------------------
dict(slug="card_fraud", title="카드 이상거래 탐지", sector="금융 · 핀테크",
     keywords=["이상거래탐지 (FDS)", "보이스피싱 · 송금 사기", "설명가능성"],
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
    "card_fraud": ("카드사 · 은행 · 증권, 간편결제 · 핀테크, 금융보안", "FDS (이상거래탐지) 분석가, 리스크 모델러, 자금세탁방지 (AML) 컴플라이언스"),
    "card_fraud": [("Dal Pozzolo et al. (2015) Calibrating Probability with Undersampling for Unbalanced Classification, IEEE SSCI", "https://doi.org/10.1109/SSCI.2015.33")],
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
    "corporate_card": ("지출 관리 · 법인카드 핀테크, 회계 · 감사, 기업 재무팀 · ERP", "핀테크 데이터 사이언티스트, 재무 · 감사 데이터 분석가"),
    "corporate_card": [("Oklahoma Open Data: Purchase Card (PCard) Fiscal Year 2014", "https://data.ok.gov/dataset/purchase-card-pcard-fiscal-year-2014")],
# ------------------------------------------------------------------
dict(slug="chess", title="체스: 누가 이길까?", sector="스포츠 · 게임", keywords=["게임 데이터", "실시간 승률 (중계)", "실력 측정 (레이팅)"],
     teaser="두 사람의 레이팅과 지금까지 둔 수. 중계 화면처럼 '지금 누가 이기고 있나' 를 확률로 낼 수 있을까?",
     data="Lichess 공개 대국 · 2013년 1월", data_src="온라인 체스 Lichess 의 모든 대국 기록 (CC0) 중 2013년 1월 121,332판 (Hugging Face Lichess/standard-chess-games): 결과 · 두 사람 레이팅 · 시간 규칙 · 수 기록",
     size="약 37MB",
     five=[("① 한 행", "대국 하나를 N수 둔 시점 (10 · 20 · 30수)"), ("② 입력 X", "레이팅 차 + 그 시점 판의 기물 점수 차 (폰 1 · 나이트/비숍 3 · 룩 5 · 퀸 9)"), ("③ 출력 y", "백 승 / 무승부 / 흑 승"),
           ("④ 평가지표", "정확도 · 로그 손실"), ("⑤ 기준선", "레이팅만 보기")],
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
("md", """## 문제 하나 골라 보기: N수 둔 판 → 결과 (중계의 '실시간 승률')

레이팅만 보면 첫 수를 두기 전에 이미 답이 정해진다. 실제 중계 화면의 승률 막대는 **지금 판** 을 본다. 그래서 한 행을 '대국 하나를 N수 둔 시점' 으로 바꾼다.
- 판을 다시 두어 보려면 `python-chess` 가 필요하다 (아래 첫 줄).
- 판에서 뽑는 값은 가장 단순한 것 하나: **기물 점수 차** (백 − 흑, 폰 1 · 나이트/비숍 3 · 룩 5 · 퀸 9).
- N수까지 둔 값만 쓰니 누수가 아니다. 단, N수보다 일찍 끝난 대국은 그 시점이 없어서 빠진다.

**날짜 순서** 로 앞 80% 학습, 뒤 20% 시험. 같은 대국들 위에서 '레이팅만' 과 '레이팅 + 판' 을 비교한다. 빨리 돌리려고 앞 5만 판만 쓴다."""),
("code", """!pip -q install chess"""),
("code", """import re
import chess
from sklearn.linear_model import LogisticRegression

PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def material_after(movetext, plies):
    \"\"\"White-minus-black material after `plies` half-moves, or None if the game ended earlier\"\"\"
    text = re.sub(r"\\{[^}]*\\}", " ", movetext)                      # drop {comments}
    sans = [t.rstrip("?!") for t in text.split() if not re.match(r"^\\d+\\.+$", t) and t not in ("1-0", "0-1", "1/2-1/2", "*")]
    if len(sans) < plies:
        return None
    board = chess.Board()
    for san in sans[:plies]:
        board.push_san(san)
    return sum(PIECE.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1) for p in board.piece_map().values())

def compare_at(games, plies):
    \"\"\"Accuracy and log loss of rating-only vs rating + material, on games that reached `plies`\"\"\"
    mat = games["movetext"].map(lambda m: material_after(m, plies))
    g = games.assign(mat=mat).dropna(subset=["mat", "diff"])
    y = g["Result"].map({"1-0": 0, "1/2-1/2": 1, "0-1": 2}).values
    cut = int(len(g) * 0.8)
    out = {}
    for name, cols in [("rating only", ["diff"]), ("rating + material", ["diff", "mat"])]:
        X = g[cols].values / np.array([100, 1][:len(cols)])
        prob = LogisticRegression(max_iter=2000).fit(X[:cut], y[:cut]).predict_proba(X[cut:])
        out[name] = (np.mean(prob.argmax(1) == y[cut:]), -np.mean(np.log(prob[np.arange(len(prob)), y[cut:]] + 1e-9)))
    return len(g), out

games = df.iloc[:50000]
for moves in [10, 20, 30]:
    n, res = compare_at(games, moves * 2)
    for name, (acc, ll) in res.items():
        print(f"after {moves:2d} moves ({n:,} games)  {name:18s} accuracy {acc:.1%}  log loss {ll:.3f}")"""),
("md", """**관찰**: 10수 뒤에는 판을 봐도 거의 그대로다 (아직 기물을 거의 안 잃었다). 20수, 30수로 갈수록 기물 점수 차가 레이팅보다 더 많은 것을 알려 준다. 반대로 '레이팅만' 의 정확도는 뒤로 갈수록 떨어진다. 오래 버틴 대국일수록 실력이 비슷한 사람들끼리였기 때문이다 (같은 칸에서 비교해야 하는 이유).

`Termination` (어떻게 끝났나) 이나 **둔 수 전체** 를 넣으면? 경기가 끝나야 알 수 있는 값이라 누수다.

## 다른 문제로 바꿔 보기

- **더 좋은 판 정보**: 기물 점수 말고 캐슬링 했나 · 킹 주변 폰 · 둘 수 있는 수의 개수. 아니면 체스 엔진의 평가값 (이 데이터의 `[%eval ...]` 이 붙은 대국).
- **수 예측**: 한 행 = 한 수, 입력 = 지금 판, 출력 = 사람이 둘 다음 수. 레이팅마다 다르게 두는 '사람 같은 AI' (Maia, 아래 논문).
- **부정행위 탐지**: 사람의 레이팅에 비해 너무 엔진 같은 수만 두는 계정 찾기."""),
]),
    "chess": ('게임 회사, e스포츠 · 매칭 시스템, 교육용 AI', '게임 데이터 사이언티스트, 추천 · 매칭 엔지니어'),
    "chess": [('McIlroy-Young et al. (2020) Aligning Superhuman AI with Human Behavior: Chess as a Model System (Maia), KDD', 'https://arxiv.org/abs/2006.01855')],
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
    "earthquakes": ("재난 · 안전 (기상 · 지질 기관), 보험 · 재보험, 건설 · 인프라", "시공간 데이터 사이언티스트, 재난 위험 모델러, 지구물리 데이터 분석가"),
    "earthquakes": [("U.S. Geological Survey (2017) Advanced National Seismic System (ANSS) Comprehensive Catalog", "https://doi.org/10.5066/F7MS3QZH")],
