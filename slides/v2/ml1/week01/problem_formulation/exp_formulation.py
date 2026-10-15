"""문제 정형화 실험 (AML 제외) → assets/formulation.json. 실습 노트북과 같은 데이터 · 같은 설정.
  와인 (UCI Wine 178병): 문제 A = 13성분 → 품종 (분류, 정확도), 문제 B = 알코올을 뺀 12성분 → 알코올 도수 (회귀, MAE)
     학습 70% · 시험 30% (random_state=0, 품종 층화), 표준화 + 로지스틱회귀 / 선형회귀, 기준선 = 가장 많은 품종 / 학습 평균
  항공 승객 (1949~1960): 앞 10년 학습 · 뒤 2년 시험, 다음 달 예측. 기준선 = 지난달 그대로 · 작년 같은 달, 모델 = kNN (k = 3) 12달 창
  AlphaFold DB: 헤모글로빈 알파 (P69905) · 알파-시누클레인 (P37840) 예측 구조, 잔기별 pLDDT (PDB 의 B-factor 칸)"""
import json, os
import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "lab", "data")
J = {}

# ---- 와인: 같은 데이터, 두 문제
w = load_wine(as_frame=True); X, y = w.data, w.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
major = ytr.value_counts().idxmax()
feat = X.drop(columns="alcohol")
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(feat.loc[Xtr.index], Xtr["alcohol"])
J["wine"] = dict(n=len(X), n_class=int(y.nunique()), counts=y.value_counts().sort_index().tolist(),
                 acc_model=float(clf.score(Xte, yte)), acc_base=float((yte == major).mean()),
                 alc_lo=float(X.alcohol.min()), alc_hi=float(X.alcohol.max()),
                 mae_model=float(np.abs(reg.predict(feat.loc[Xte.index]) - Xte["alcohol"]).mean()),
                 mae_base=float(np.abs(Xtr["alcohol"].mean() - Xte["alcohol"]).mean()))

# ---- 항공 승객: 기준선 vs 모델
p = pd.read_csv(os.path.join(D, "air.csv")).Passengers.values.astype(float)
W, NTR = 12, 120
Xw = np.stack([p[i:i + W] for i in range(len(p) - W)]); yw = p[W:]
tr = np.arange(W, len(p)) < NTR
mu, sd = p[:NTR].mean(), p[:NTR].std()
knn = KNeighborsRegressor(3).fit((Xw[tr] - mu) / sd, (yw[tr] - mu) / sd)
pred_knn = knn.predict((Xw[~tr] - mu) / sd) * sd + mu
mae = lambda a: float(np.abs(a - yw[~tr]).mean())
J["air"] = dict(mae_last=mae(Xw[~tr, -1]), mae_season=mae(Xw[~tr, 0]), mae_knn=mae(pred_knn), mae_mean=mae(np.full((~tr).sum(), p[:NTR].mean())))


# ---- AlphaFold: 서열 → 좌표 + 잔기별 신뢰도
def read_pdb(path):
    seq, xyz, plddt = [], [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            seq.append(line[17:20]); xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])]); plddt.append(float(line[60:66]))
    return seq, np.array(xyz), np.array(plddt)


AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
          MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
J["af"] = {}
for uid, name in (("P69905", "hemoglobin alpha"), ("P37840", "alpha-synuclein")):
    seq, xyz, pl = read_pdb(os.path.join(D, f"{uid}.pdb"))
    J["af"][uid] = dict(name=name, seq="".join(AA[s] for s in seq), xyz=xyz.tolist(), plddt=pl.tolist(),
                        mean_plddt=float(pl.mean()), frac_high=float((pl > 70).mean()))

# ---- 회귀 지표 손 계산 예: 오차 [1, 1, 1, 9]
e = np.array([1, 1, 1, 9]); J["toy"] = dict(mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())))

json.dump(J, open(os.path.join(HERE, "assets", "formulation.json"), "w"))
print("wine", {k: v for k, v in J["wine"].items()})
print("air", J["air"]); print({u: (a["name"], len(a["seq"]), round(a["mean_plddt"], 1), round(a["frac_high"], 2)) for u, a in J["af"].items()})
print(J["af"]["P69905"]["seq"]); print("toy", J["toy"])
