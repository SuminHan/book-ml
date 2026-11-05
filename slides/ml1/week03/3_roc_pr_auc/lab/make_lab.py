"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 슬라이드의 점수 100개 (data.py 와 같은 배열) · Ozone Level Detection 8hr (UCI, OpenML 에서 내려받기)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (pr_auc) 이 파일명 앞에 붙는다

CELLS = [
("md", """# ROC-AUC 와 PR-AUC 실습

같은 AI 인데 ROC-AUC 0.98, PR-AUC 0.85. 두 점수를 직접 구현하고, 데이터가 쏠릴수록 어떻게 달라지는지 본다.

- **파트 1**: ROC-AUC (짝 세기) 와 AP (줄 세우기) 직접 구현
- **파트 2**: 건강 학생을 10배 · 100배로 늘리면
- **파트 3**: 실제 데이터 (오존 경보, 양성 6%): 두 모델 비교

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score"""),
("md", """## 슬라이드의 예시 데이터

학교 100명 중 독감 10명, AI 의 '독감일 확률' 점수 (지난 시간과 같은 점수)."""),
("code", """# Same example as the slides: AI scores for 10 sick and 90 healthy students
sick = np.array([0.95, 0.91, 0.86, 0.82, 0.77, 0.71, 0.63, 0.56, 0.44, 0.31])
healthy = np.array([0.81, 0.66, 0.58, 0.53, 0.33, 0.32, 0.27, 0.25, 0.25, 0.25, 0.24, 0.21, 0.21, 0.2, 0.2, 0.2, 0.19, 0.19,
                    0.19, 0.17, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
                    0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09,
                    0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05,
                    0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01])
print(len(sick), "sick,", len(healthy), "healthy")"""),
("md", """## 파트 1. 두 점수를 직접 구현

- **ROC-AUC**: 독감-건강 짝 $10 \\times 90 = 900$ 개 중 독감 점수가 더 높은 짝의 비율 (같으면 0.5 개)
- **AP (PR-AUC)**: 점수 순으로 줄 세워, 독감 학생을 찾을 때마다 그때까지의 정밀도를 적고 평균"""),
("code", """def roc_auc_pairs(pos, neg):
    wins = 0.0
    for s in pos:
        wins += np.sum(s > neg) + 0.5 * np.sum(s == neg)  #@ANS wins += None  # TODO: count pairs this positive wins
    return wins / (len(pos) * len(neg))

def average_precision(pos, neg):
    scores = np.r_[pos, neg]
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-scores, kind="stable")           # high score first
    found, precisions = 0, []
    for k, i in enumerate(order, start=1):
        if labels[i] == 1:
            found += 1
            precisions.append(found / k)                  #@ANS precisions.append(None)  # TODO: precision so far
    return np.mean(precisions)

auc, ap = roc_auc_pairs(sick, healthy), average_precision(sick, healthy)
y = np.r_[np.ones(10), np.zeros(90)]; s = np.r_[sick, healthy]
print(f"ROC-AUC: mine {auc:.3f}  sklearn {roc_auc_score(y, s):.3f}")
print(f"AP     : mine {ap:.3f}  sklearn {average_precision_score(y, s):.3f}")"""),
("md", """## 파트 2. 건강한 학생을 늘리면

같은 점수의 건강 학생을 10배, 100배로 늘린다 (AI 의 실력은 그대로)."""),
("code", """for k in [1, 10, 100]:
    neg = np.tile(healthy, k)
    y_k = np.r_[np.ones(10), np.zeros(len(neg))]
    s_k = np.r_[sick, neg]
    print(f"healthy x{k:3d} ({len(neg):5d}): ROC-AUC {roc_auc_score(y_k, s_k):.3f}   AP {average_precision_score(y_k, s_k):.3f}   positive rate {10 / len(y_k):.3f}")"""),
("md", """**관찰**: ROC-AUC 는 그대로인데 AP 는 떨어진다. AP 옆의 양성 비율(찍기 AI 의 바닥) 도 같이 보자.

## 파트 3. 실제 데이터: 오존 경보일 예측

휴스턴 지역 7년치 날씨 측정값 72개로 그날이 오존 경보일(8시간 기준)인지 예측. 경보일은 약 6%. OpenML 에서 내려받는다."""),
("code", """from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

data = fetch_openml(name="ozone-level-8hr", version=1, as_frame=True)
X = data.data.to_numpy(float)
y = (data.target.astype(str) == "2").astype(int)          # 2 = ozone day
print(X.shape, "positive rate", round(y.mean(), 3))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)"""),
("code", """models = {
    "logistic": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)),
    "forest":   make_pipeline(SimpleImputer(), RandomForestClassifier(300, random_state=0)),
}
for name, model in models.items():
    s_te = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    roc = roc_auc_score(y_te, s_te)
    ap = average_precision_score(y_te, s_te)          #@ANS ap = None  # TODO: PR-AUC (average precision)
    print(f"{name:8s}: ROC-AUC {roc:.3f}   PR-AUC {ap:.3f}   (baseline {y_te.mean():.3f})")"""),
("md", """**확인 질문**
1. 파트 1 의 짝 세기가 ROC 곡선 아래 넓이와 같은 이유를 슬라이드의 계단 그림으로 설명해 보자.
2. 파트 2 에서 AP 는 떨어지는데 AP ÷ (양성 비율) 은 어떻게 변하나?
3. 파트 3 에서 두 모델의 ROC-AUC 차이와 PR-AUC 차이 중 어느 쪽이 더 큰가? 오존 경보 서비스를 만든다면 어느 점수로 모델을 고르겠는가?
4. (도전) `sklearn.metrics.precision_recall_curve` 로 파트 3 의 PR 곡선을 두 모델 모두 그리고, 바닥선(양성 비율) 을 함께 그려 보자.

데이터 출처: Ozone Level Detection, UCI Machine Learning Repository (Zhang, Fan, Li 2008), OpenML 사본."""),
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
