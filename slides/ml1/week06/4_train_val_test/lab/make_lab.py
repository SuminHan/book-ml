"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 동전 실험 · 환자 · 시간 (data.py 와 같은 생성 규칙) · Diabetes (scikit-learn 내장). notebooks/ml1/chapter06_3 을 바탕으로 줄임."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (train_val_test) 이 파일명 앞에 붙는다

CELLS = [
("md", """# train / val / test 와 데이터 누수 실습

시험 데이터의 정보가 학습으로 새면 성능이 부풀려진다. 세 가지 누수를 직접 만들고 고친다.

- **파트 1**: 동전 실험: 전체로 특징 고르기 vs `Pipeline`
- **파트 2**: 환자 누수 (`GroupKFold`) · 시간 누수 (`TimeSeriesSplit`)
- **파트 3**: 당뇨병 3분할: val 로 고르고 test 는 한 번

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor"""),
("md", """## 파트 1. 동전 실험

100명 × 의미 없는 난수 5,000개, 정답 = 동전 (앞 50 · 뒤 50). 정답과 가장 관련 있어 보이는 숫자 20개를 골라 로지스틱 회귀."""),
("code", """rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))
y = np.array([0, 1] * 50); rng.shuffle(y)
folds = StratifiedKFold(5, shuffle=True, random_state=0)

# WRONG: pick the 20 features using all 100 people, then cross-validate
picked = SelectKBest(f_classif, k=20).fit(X, y).get_support()
leak = cross_val_score(LogisticRegression(max_iter=2000), X[:, picked], y, cv=folds).mean()

# RIGHT: put the feature picking inside the pipeline -> it is refit on each training fold only
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000))   #@ANS pipe = None  # TODO: SelectKBest(k=20) then LogisticRegression
honest = cross_val_score(pipe, X, y, cv=folds).mean()
print(f"leaky : {leak:.2f}")
print(f"honest: {honest:.2f}   (coin flip = 0.50)")"""),
("md", """## 파트 2. 환자 누수와 시간 누수

### 환자 100명 × 사진 5장
한 환자의 사진 5장은 서로 아주 비슷하다. 사진 단위로 섞어 나누면 같은 환자가 양쪽에 들어간다."""),
("code", """r = np.random.default_rng(51)
P, T = 100, 5
value = r.normal(0, 2.0, P)                               # each patient's true value
center = r.normal(0, 2.0, (P, 2))                         # each patient's 'look'
Xp = (center[:, None, :] + r.normal(0, 0.1, (P, T, 2))).reshape(P * T, 2)
yp = np.repeat(value, T)
patient = np.repeat(np.arange(P), T)                      # which patient each photo belongs to

knn = KNeighborsRegressor(1)
random_cv = -cross_val_score(knn, Xp, yp, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error").mean()
group_cv = -cross_val_score(knn, Xp, yp, cv=GroupKFold(5), groups=patient, scoring="neg_root_mean_squared_error").mean()   #@ANS group_cv = None  # TODO: GroupKFold with groups=patient
print(f"photos shuffled : RMSE {random_cv:.2f}")
print(f"patients kept together: RMSE {group_cv:.2f}")"""),
("code", """# 365 days, the value rises by 1 every day
t = np.arange(365.0)
yt = 100 + t + np.random.default_rng(71).normal(0, 0.5, 365)
Xt = t.reshape(-1, 1)
knn5 = KNeighborsRegressor(5)
shuffled = -cross_val_score(knn5, Xt, yt, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error").mean()
in_order = -cross_val_score(knn5, Xt, yt, cv=TimeSeriesSplit(5), scoring="neg_root_mean_squared_error").mean()   #@ANS in_order = None  # TODO: TimeSeriesSplit(5)
print(f"days shuffled : RMSE {shuffled:.2f}")
print(f"past -> future: RMSE {in_order:.2f}")"""),
("md", """## 파트 3. 당뇨병: 3분할 파이프라인

1. 먼저 test 를 떼어 둔다 (20%). 2. 남은 것을 train / val 로. 3. val 로 λ 를 고른다. 4. test 는 한 번."""),
("code", """from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
Xd, yd = load_diabetes(return_X_y=True)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xd, yd, test_size=0.4, random_state=0)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
print("sizes:", len(X_tr), len(X_va), len(X_te))

def rmse(model, X, y):
    return np.sqrt(np.mean((model.predict(X) - y) ** 2))

val_scores = {}
for lam in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    val_scores[lam] = rmse(Ridge(lam).fit(X_tr, y_tr), X_va, y_va)   #@ANS val_scores[lam] = None  # TODO: fit on train, score on val
print({k: round(float(v), 2) for k, v in val_scores.items()})
best = min(val_scores, key=val_scores.get)
print(f"chosen lambda = {best}   ->  test RMSE (look once) = {rmse(Ridge(best).fit(X_tr, y_tr), X_te, y_te):.2f}")"""),
("md", """**확인 질문**
1. 파트 1 에서 `k=20` 을 `k=200` 으로 바꾸면 leaky 점수는 어떻게 되나? honest 는?
2. 파트 2 의 환자 예에서, 실제 병원에 이 모델을 쓴다면 어느 RMSE 를 믿어야 하나? 이유는?
3. 파트 2 의 시간 예에서 `TimeSeriesSplit` 은 조각을 어떻게 나누나? `list(TimeSeriesSplit(5).split(Xt))` 로 확인해 보자.
4. 파트 3 에서 test 점수를 본 뒤 λ 를 다시 고르고 싶어졌다. 어떻게 해야 정직한가?

데이터 출처: Efron et al. (2004) Diabetes, scikit-learn. 동전 · 환자 · 시간 데이터는 설명용으로 만든 것."""),
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
