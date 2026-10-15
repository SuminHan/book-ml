"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: UCI Wine (scikit-learn 내장) · 항공 승객 (jbrownlee/Datasets) · IBM AML HI-Small (Hugging Face 사본, 로그인 없이, 압축 약 90MB)
· AlphaFold DB (EMBL-EBI API). 슬라이드 숫자 (exp_formulation.py · exp_aml.py) 와 같은 설정."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (problem_formulation) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 문제 정형화와 평가지표 실습: 문제를 직접 정해 보기

모델보다 먼저 **다섯 칸** 을 채운다: ① 한 행은 무엇? ② 입력 X ③ 출력 y ④ 평가지표 ⑤ 기준선.

- **파트 1**: 와인: 같은 데이터로 두 문제 (품종 분류 · 알코올 도수 회귀), 정확도 · MAE · 기준선 직접 만들기
- **파트 2**: 항공 승객: 기준선 (지난달 그대로 · 작년 같은 달) vs kNN 모델
- **파트 3**: IBM AML 자금세탁 거래: 정확도의 함정, 정밀도 · 재현율, 한 행을 거래 → 계좌로
- **파트 4**: AlphaFold: 단백질 서열 X → 3D 좌표 y, 예측 신뢰도 pLDDT

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 코드 칸 아래의 **예상 결과** 와 비교하면 맞았는지 바로 알 수 있습니다."""),
("code", """import io, json, os, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler"""),

# ---------------- 파트 1. 와인
("md", """## 파트 1. 와인 178병: 같은 데이터, 두 문제

먼저 채점 함수 세 개를 만든다. 정답 `y_true` 와 예측 `y_pred` 를 받는다.
- 정확도 = 맞힌 개수 ÷ 전체
- MAE = 오차 절댓값의 평균
- RMSE = 오차 제곱의 평균에 루트"""),
("code", """def accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))      #@ANS return None   # TODO: fraction of exact matches

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))      #@ANS return None   # TODO: mean of |error|

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))      #@ANS return None   # TODO: sqrt of mean of error^2

print("accuracy:", accuracy([0, 1, 2, 2], [0, 1, 1, 2]))          # 0.75
print("MAE :", mae([0, 0, 0, 0], [1, 1, 1, 9]))                  # slide: 3
print("RMSE:", round(rmse([0, 0, 0, 0], [1, 1, 1, 9]), 2))       # slide: 4.58"""),
("code", """wine = load_wine(as_frame=True)
df = wine.data.copy()
df["cultivar"] = wine.target
print(df.shape)
print(df["cultivar"].value_counts().sort_index())
df.head(3)"""),
("md", """### 문제 A: 13성분 → 품종 (분류)

`X` 는 13성분 전부, `y` 는 품종. 학습 70% · 시험 30% (품종 비율을 맞춰서). **기준선** = 학습 데이터에서 가장 많은 품종을 늘 답하기."""),
("code", """X_a = df.drop(columns="cultivar")      #@ANS X_a = None   # TODO: all 13 measurements (drop the answer column)
y_a = df["cultivar"]                   #@ANS y_a = None   # TODO: the cultivar column
Xtr, Xte, ytr, yte = train_test_split(X_a, y_a, test_size=0.3, stratify=y_a, random_state=0)

most_common = ytr.value_counts().idxmax()
base_pred = np.full(len(yte), most_common)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
print(f"baseline (always cultivar {most_common}): accuracy {accuracy(yte, base_pred):.0%}")
print(f"logistic regression           : accuracy {accuracy(yte, model.predict(Xte)):.0%}")"""),
("md", """### 문제 B: 같은 와인, 12성분 → 알코올 도수 (회귀)

이번엔 `y` 가 알코올 도수. **입력에서 알코올을 빼야 한다** (답을 입력에 넣으면 안 된다). 품종도 입력에서 뺀다. 같은 행으로 학습 · 시험을 나눈다. **기준선** = 학습 데이터의 평균 도수를 늘 답하기."""),
("code", """X_b = df.drop(columns=["alcohol", "cultivar"])      #@ANS X_b = None   # TODO: drop the answer (alcohol) and cultivar
y_b = df["alcohol"]                                  #@ANS y_b = None   # TODO: alcohol column
Xtr_b, Xte_b = X_b.loc[Xtr.index], X_b.loc[Xte.index]
ytr_b, yte_b = y_b.loc[Xtr.index], y_b.loc[Xte.index]

base_pred = np.full(len(yte_b), ytr_b.mean())
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr_b, ytr_b)
print(f"baseline (always {ytr_b.mean():.2f}): MAE {mae(yte_b, base_pred):.2f}  RMSE {rmse(yte_b, base_pred):.2f}")
print(f"linear regression      : MAE {mae(yte_b, reg.predict(Xte_b)):.2f}  RMSE {rmse(yte_b, reg.predict(Xte_b)):.2f}")"""),
("md", """**생각해 보기**: 문제 B 의 입력에 `alcohol` 을 그대로 두면 MAE 가 얼마가 될까? 한 번 해 보고, 그 점수가 왜 의미 없는지 말해 보자."""),

# ---------------- 파트 2. 항공 승객
("md", """## 파트 2. 항공 승객: 기준선이 모델을 이길 수 있다

한 달 국제선 승객 수 (천 명), 1949~1960. 앞 10년 (120달) 으로 학습, 뒤 2년 (24달) 을 예측한다. **시간 순서대로** 나눈다."""),
("code", """url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
air = pd.read_csv(url)
series = air["Passengers"].values.astype(float)
N_TRAIN, W = 120, 12
print(len(series), "months;", air["Month"].iloc[0], "~", air["Month"].iloc[-1])

X = np.stack([series[i:i + W] for i in range(len(series) - W)])     # past 12 months
y = series[W:]                                                      # next month
is_train = np.arange(W, len(series)) < N_TRAIN
y_test = y[~is_train]
print(X.shape, "test months:", len(y_test))"""),
("md", """세 가지 기준선을 만든다. `X` 의 각 행은 지난 12달 (`X[:, -1]` = 지난달, `X[:, 0]` = 12달 전 = 작년 같은 달)."""),
("code", """base_mean = np.full(len(y_test), series[:N_TRAIN].mean())   # always the training-period average
base_last = X[~is_train, -1]      #@ANS base_last = None     # TODO: last month's value
base_season = X[~is_train, 0]     #@ANS base_season = None   # TODO: same month last year

mu, sd = series[:N_TRAIN].mean(), series[:N_TRAIN].std()
knn = KNeighborsRegressor(3).fit((X[is_train] - mu) / sd, (y[is_train] - mu) / sd)
pred_knn = knn.predict((X[~is_train] - mu) / sd) * sd + mu

results = {"training average": base_mean, "last month": base_last, "same month last year": base_season, "kNN model": pred_knn}
for name, pred in results.items():
    print(f"{name:22s} MAE {mae(y_test, pred):5.1f}")
plt.figure(figsize=(7, 3)); plt.bar(results.keys(), [mae(y_test, p) for p in results.values()], color=["gray", "navy", "navy", "red"])
plt.ylabel("test MAE (1000s)"); plt.xticks(fontsize=8); plt.show()"""),
("md", """**관찰**: kNN 모델 (74) 이 '지난달 그대로' (44) 보다 못하다. 기준선을 재 보지 않았다면 몰랐을 일이다. (다음 차시 정규화 실습에서 창마다 표준화하면 이 모델이 10 까지 내려간다.)"""),

# ---------------- 파트 3. AML
("md", """## 파트 3. IBM AML: 자금세탁 거래 찾기

IBM 이 가상의 사람 · 회사 · 은행을 시뮬레이션해 만든 합성 거래 데이터 (Altman et al., NeurIPS 2023). 원본은 Kaggle 에 있고, 여기서는 로그인 없이 받을 수 있는 Hugging Face 사본에서 받는다.

압축 파일 약 90MB, 풀면 약 480MB. 처음엔 몇 분 걸린다."""),
("code", """os.makedirs("data", exist_ok=True)
URL = "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip"
ZIP = "data/HI-Small_Trans.csv.zip"
if not os.path.exists(ZIP):
    urllib.request.urlretrieve(URL, ZIP)
print(round(os.path.getsize(ZIP) / 1e6), "MB")"""),
("code", """with zipfile.ZipFile(ZIP) as z:
    df = pd.read_csv(z.open("HI-Small_Trans.csv"))
print(df.shape)
print(df.dtypes)
df.head(3)"""),
("md", """### ① 한 행 = 거래 한 건

출력 `y` 는 `Is Laundering` (0/1). 세탁 거래는 몇 건, 몇 % 일까? 그리고 **모든 거래에 '정상' (0)** 이라고 답하면 정확도는?"""),
("code", """y = df["Is Laundering"].values
print("transactions:", len(y), "  laundering:", y.sum(), f"({y.mean():.3%})")

all_normal = np.zeros(len(y), dtype=int)      #@ANS all_normal = None   # TODO: predict 0 for every transaction
print(f"'always normal' accuracy: {accuracy(y, all_normal):.3%}")"""),
("md", """정확도가 매우 높지만 세탁을 하나도 못 잡는다. 결제 형식별로 세탁 비율을 보고, **가장 비율이 높은 형식이면 '세탁'** 이라고 답하는 규칙 하나를 만들어 보자."""),
("code", """by_format = df.groupby("Payment Format")["Is Laundering"].agg(["mean", "sum", "size"]).sort_values("mean", ascending=False)
print(by_format)
fmt = by_format.index[0]
rule = (df["Payment Format"] == fmt).values.astype(int)      #@ANS rule = None   # TODO: 1 if the payment format is fmt, else 0
print(f"rule '{fmt} -> laundering' accuracy: {accuracy(y, rule):.3%}")"""),
("md", """### 정밀도와 재현율: 세는 법을 바꾼다

- **재현율** = 잡았다 ÷ 실제 세탁 전부 (실제 세탁 중 몇 개를 잡았나)
- **정밀도** = 잡았다 ÷ '세탁' 이라고 한 것 전부 (찍은 것 중 진짜는 몇 개)

(2주차 정밀도와 재현율에서 자세히 다룬다.)"""),
("code", """def precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))     # caught
    fp = np.sum((y_pred == 1) & (y_true == 0))     # false alarm
    fn = np.sum((y_pred == 0) & (y_true == 1))     # missed
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0      #@ANS precision = None   # TODO: caught / everything we called laundering
    recall = tp / (tp + fn)                                 #@ANS recall = None      # TODO: caught / all real laundering
    return tp, fp, fn, precision, recall

for name, pred in [("always normal", all_normal), (f"rule: {fmt}", rule)]:
    tp, fp, fn, p, r = precision_recall(y, pred)
    print(f"{name:16s} caught {tp:5d}  false alarms {fp:7d}  missed {fn:5d}  precision {p:.1%}  recall {r:.1%}")"""),
("md", """### ② 한 행 = 계좌 하나

은행이 실제로 하는 일은 '이 계좌를 조사할까?' 다. 보낸 쪽 계좌 (`From Bank` + `Account`) 하나를 한 행으로, **세탁 거래를 한 번이라도 보냈으면 1** 로 정해 보자."""),
("code", """sender = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
account_y = df.groupby(sender)["Is Laundering"].max()      #@ANS account_y = None   # TODO: per sender account, 1 if any of its transactions is laundering (groupby + max)
print("accounts:", len(account_y), "  laundering accounts:", account_y.sum(), f"({account_y.mean():.2%})")
print(f"transaction-level rate {y.mean():.3%}  ->  account-level rate {account_y.mean():.2%}")"""),
("md", """**관찰**: 같은 파일인데 한 행을 무엇으로 정하느냐에 따라 세탁 비율이 달라진다. 이 데이터를 쓴 논문들마다 '세탁 비율' 이 다르게 적힌 이유다.

**생각해 보기**: 시험 데이터는 어떻게 나눠야 할까? `Timestamp` 열을 보고, 무작위로 섞으면 왜 안 되는지 말해 보자."""),

# ---------------- 파트 4. AlphaFold
("md", """## 파트 4. AlphaFold: 서열 → 3D 구조

AlphaFold DB 의 API 에 UniProt 번호를 주면 예측 구조 파일 주소가 온다. 헤모글로빈 알파 (P69905) 와 알파-시누클레인 (P37840) 을 받자."""),
("code", """def alphafold_pdb(uniprot_id):
    \"\"\"Ask the AlphaFold DB API for the predicted structure and return the PDB file as text\"\"\"
    headers = {"User-Agent": "Mozilla/5.0 (ML course lab)"}
    meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    meta = json.loads(urllib.request.urlopen(urllib.request.Request(meta_url, headers=headers), timeout=60).read())[0]
    pdb = urllib.request.urlopen(urllib.request.Request(meta["pdbUrl"], headers=headers), timeout=60).read().decode()
    return meta, pdb

proteins = {}
for uid in ["P69905", "P37840"]:
    meta, pdb = alphafold_pdb(uid)
    proteins[uid] = pdb
    print(uid, meta["uniprotDescription"], "|", meta["pdbUrl"])
print("\\n".join(proteins["P69905"].split("\\n")[:3]))
print([l for l in proteins["P69905"].split("\\n") if l.startswith("ATOM")][:2])"""),
("md", """PDB 파일은 원자 하나가 한 줄 (`ATOM ...`). 칸의 위치가 정해져 있다:
- 18~20번째 글자: 아미노산 이름 (예: `MET`)
- 13~16번째: 원자 이름. 아미노산의 중심 원자는 `CA`
- 31~38 / 39~46 / 47~54번째: x, y, z 좌표 (Å)
- 61~66번째: 예측 신뢰도 pLDDT (AlphaFold 는 이 칸에 넣는다)

파이썬 문자열은 0부터 세니까 `line[17:20]`, `line[12:16]`, `line[30:38]` … 처럼 자른다. `CA` 원자 줄만 골라 서열 · 좌표 · pLDDT 를 꺼내자."""),
("code", """AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I",
          LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")

def read_ca(pdb_text):
    \"\"\"Return (sequence string, CA coordinates (n, 3), pLDDT (n,)) from a PDB file\"\"\"
    seq, xyz, plddt = [], [], []
    for line in pdb_text.split("\\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":      #@ANS if None:   # TODO: an ATOM line whose atom name (line[12:16]) is CA
            seq.append(AA[line[17:20]])
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddt.append(float(line[60:66]))
    return "".join(seq), np.array(xyz), np.array(plddt)

seq, xyz, plddt = read_ca(proteins["P69905"])
print("X (sequence):", len(seq), "letters:", seq[:40], "...")
print("y (CA coordinates):", xyz.shape)
print(f"pLDDT mean {plddt.mean():.1f}, share above 70: {(plddt > 70).mean():.0%}")"""),
("code", """fig = plt.figure(figsize=(11, 4))
for k, uid in enumerate(["P69905", "P37840"]):
    seq, xyz, plddt = read_ca(proteins[uid])
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.plot(*xyz.T, color="lightgray", lw=1)
    ax.scatter(*xyz.T, c=plddt, cmap="RdYlBu", vmin=30, vmax=100, s=12)
    ax.set_title(f"{uid}: {len(seq)} aa, mean pLDDT {plddt.mean():.0f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax = fig.add_subplot(1, 3, 3)
for uid in ["P69905", "P37840"]:
    ax.plot(read_ca(proteins[uid])[2], label=uid)
ax.axhline(70, color="gray", ls=":"); ax.set_xlabel("residue"); ax.set_ylabel("pLDDT"); ax.legend()
plt.tight_layout(); plt.show()"""),
("md", """**관찰**: 헤모글로빈 알파는 거의 전부 pLDDT 90 이상. 알파-시누클레인은 뒤쪽 (약 100번 이후) 이 70 아래로 떨어진다. 이 부분은 실제로도 정해진 모양 없이 흐느적거린다. 출력에 '얼마나 자신 있는지' 까지 넣은 것도 문제 정의의 일부다.

**확인 질문**
1. 파트 1 문제 B 에서 입력에 `alcohol` 을 남기면 MAE 는? 그 점수가 의미 없는 이유는?
2. 파트 2 에서 kNN 이 '지난달 그대로' 에 진 이유는 무엇일까? (힌트: 학습 기간 최댓값과 시험 기간 값을 비교해 보자)
3. 파트 3 에서 정확도 · 정밀도 · 재현율 중 은행이 가장 중요하게 볼 것은? 놓친 세탁 한 건과 헛짚은 정상 고객 한 명의 손해를 비교해 말해 보자.
4. 파트 3 에서 '전부 세탁' 이라고 답하면 재현율과 정밀도는 각각 얼마가 될까?
5. 파트 4 의 문제를 다섯 칸 (한 행 · 입력 · 출력 · 평가지표 · 기준선) 으로 써 보자. 시험 데이터는 어떻게 골라야 공정할까?

데이터 출처: UCI Wine (Aeberhard & Forina, 1991) · Box & Jenkins (1976) 항공 승객 · IBM Transactions for Anti Money Laundering (Altman et al., NeurIPS 2023, Kaggle) · AlphaFold Protein Structure Database (EMBL-EBI, CC BY 4.0; Jumper et al., Nature 2021)."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    sys.path.insert(0, os.path.join(HERE, "../../../../tools"))
    from expected import add_expected                         # 학생용 코드 칸 아래에 '예상 결과' (정답을 실제 실행한 출력)
    print(add_expected(f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb"))
