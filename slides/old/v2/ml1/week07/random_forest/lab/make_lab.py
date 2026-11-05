"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Iris · Titanic (seaborn-data 사본) · Pima Indians Diabetes (jbrownlee/Datasets 사본), 처음 실행 때 내려받기."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (random_forest) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 랜덤 포레스트 실습

서로 다른 나무 여럿의 다수결. 숲을 직접 만들고, OOB 와 특징 중요도를 본다.

- **파트 1**: 부트스트랩 + 나무 + 다수결로 숲 직접 만들기 (붓꽃)
- **파트 2**: OOB 점수 직접 계산, 나무 수에 따른 정확도 (당뇨병)
- **파트 3**: 타이타닉: 한 그루 vs 숲, 특징 중요도

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold"""),
("code", """def download(url, name):
    os.makedirs("data", exist_ok=True)
    path = f"data/{name}"
    if not os.path.exists(path):
        import urllib.request; urllib.request.urlretrieve(url, path)
    return path

SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
iris = pd.read_csv(download(SEABORN + "iris.csv", "iris.csv"))
X = iris[["petal_length", "petal_width"]].to_numpy(); y = iris["species"].to_numpy()"""),
("md", """## 파트 1. 숲 직접 만들기

1. 부트스트랩: 150송이에서 복원 추출로 150송이  2. 나무 한 그루 (질문마다 특징 1개만 무작위로: `max_features=1`)  3. 100번 반복  4. 다수결"""),
("code", """def bootstrap_indices(n, rng):
    return rng.integers(0, n, size=n)                     #@ANS return None  # TODO: n random indices WITH replacement

def grow_forest(X, y, n_trees, seed=0):
    rng = np.random.default_rng(seed)
    trees, in_bag = [], []
    for k in range(n_trees):
        idx = bootstrap_indices(len(y), rng)
        tree = DecisionTreeClassifier(max_features=1, random_state=k).fit(X[idx], y[idx])
        trees.append(tree); in_bag.append(np.unique(idx))
    return trees, in_bag

def forest_votes(trees, x):
    votes = [t.predict(x.reshape(1, -1))[0] for t in trees]
    labels, counts = np.unique(votes, return_counts=True)
    return {str(l): int(c) for l, c in zip(labels, counts)}

trees, in_bag = grow_forest(X, y, 100)
print("distinct flowers in the first bootstrap sample:", len(in_bag[0]), "/ 150")
for flower in [[5.0, 1.7], [4.9, 1.6], [1.5, 0.2]]:
    print(flower, forest_votes(trees, np.array(flower)))"""),
("md", """## 파트 2. OOB 점수와 나무 수

당뇨병 768명. 데이터마다, 그 데이터를 **학습에 안 쓴** 나무들만 모아 다수결 → 맞았나?"""),
("code", """cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree", "age", "outcome"]
pima = pd.read_csv(download("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", "pima.csv"), header=None, names=cols)
Xp, yp = pima[cols[:-1]].to_numpy(float), pima["outcome"].to_numpy()

trees_p, in_bag_p = grow_forest(Xp, yp, 300)
correct, counted = 0, 0
for i in range(len(yp)):
    voters = [t for t, bag in zip(trees_p, in_bag_p) if i not in bag]   #@ANS voters = None  # TODO: trees that did NOT see row i
    if voters:
        votes = [t.predict(Xp[i:i + 1])[0] for t in voters]
        correct += int(np.bincount(votes).argmax() == yp[i]); counted += 1
print(f"my OOB accuracy: {correct / counted:.3f}  (rows scored: {counted})")
print("sklearn oob_score_:", round(RandomForestClassifier(300, oob_score=True, random_state=0).fit(Xp, yp).oob_score_, 3))"""),
("code", """cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("one full tree:", round(cross_val_score(DecisionTreeClassifier(random_state=0), Xp, yp, cv=cv).mean(), 3))
ns = [1, 3, 10, 30, 100, 300]
accs = [cross_val_score(RandomForestClassifier(n, random_state=0), Xp, yp, cv=cv).mean() for n in ns]
plt.semilogx(ns, accs, "o-"); plt.xlabel("number of trees"); plt.ylabel("5-fold CV accuracy"); plt.show()
print([round(float(a), 3) for a in accs])"""),
("md", """## 파트 3. 타이타닉: 한 그루 vs 숲"""),
("code", """titanic = pd.read_csv(download(SEABORN + "titanic.csv", "titanic.csv"))
df = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].copy()
df["sex"] = (df["sex"] == "female").astype(int)
df["age"] = df["age"].fillna(df["age"].median())
Xt, yt = df.drop(columns="survived").to_numpy(), df["survived"].to_numpy()
names = list(df.columns[1:])
tree_acc = cross_val_score(DecisionTreeClassifier(random_state=0), Xt, yt, cv=cv).mean()
forest = RandomForestClassifier(300, random_state=0)
forest_acc = cross_val_score(forest, Xt, yt, cv=cv).mean()   #@ANS forest_acc = None  # TODO: 5-fold CV accuracy of the forest
print(f"one tree {tree_acc:.3f}   forest {forest_acc:.3f}")
imp = forest.fit(Xt, yt).feature_importances_
for n, v in sorted(zip(names, imp), key=lambda t: -t[1]):
    print(f"{n:6s} {v:.3f}")"""),
("md", """**확인 질문**
1. 파트 1 에서 `max_features=1` 을 지우면 (모든 특징을 후보로) 표가 어떻게 달라지나? 나무들이 더 닮을까?
2. 파트 2 의 직접 계산한 OOB 와 scikit-learn 의 값이 조금 다른 이유는? (힌트: 나무를 만드는 방법)
3. 파트 3 에서 가장 중요한 특징은? 결정 트리 실습의 첫 질문과 같은가?
4. (도전) 파트 3 에서 `fare` 대신 아무 뜻 없는 난수 열을 넣으면 그 열의 중요도는 0 일까? 왜 아닐까?

데이터 출처: Fisher (1936) Iris · Titanic (seaborn-data) · Pima Indians Diabetes (NIDDK, UCI)."""),
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
