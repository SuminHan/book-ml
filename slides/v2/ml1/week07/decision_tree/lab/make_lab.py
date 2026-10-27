"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Iris · Titanic (seaborn-data 사본, 처음 실행 때 내려받기)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (decision_tree) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 결정 트리 실습

예 · 아니오 질문으로 나누기. 지니 불순도로 가장 좋은 질문을 직접 찾고, 실제 데이터에서 깊이를 고른다.

- **파트 1**: 지니 불순도 · 가장 좋은 질문 찾기 직접 구현 → 붓꽃 0.333, 0.11
- **파트 2**: scikit-learn 트리 그리기, 새 꽃의 경로
- **파트 3**: 타이타닉 891명: 최대 깊이를 교차 검증으로

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt"""),
("code", """def download(name):
    os.makedirs("data", exist_ok=True)
    path = f"data/{name}.csv"
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{name}.csv", path)
    return pd.read_csv(path)

iris = download("iris")
X = iris[["petal_length", "petal_width"]].to_numpy()
y = iris["species"].to_numpy()
print(iris["species"].value_counts().to_dict())"""),
("md", """## 파트 1. 지니 불순도와 가장 좋은 질문

$$\\text{Gini} = 1 - \\sum_k p_k^2, \\qquad \\text{after a split} = \\frac{n_L}{n}\\text{Gini}_L + \\frac{n_R}{n}\\text{Gini}_R$$"""),
("code", """def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)                             #@ANS return None  # TODO: 1 - sum of squared proportions

def split_gini(X, y, feature, threshold):
    left = X[:, feature] < threshold
    n, nl = len(y), left.sum()
    return nl / n * gini(y[left]) + (n - nl) / n * gini(y[~left])   #@ANS return None  # TODO: weighted average of both sides

print("root:", round(gini(y), 3))
print("petal length < 2.45:", round(split_gini(X, y, 0, 2.45), 3))
print("petal width  < 1.75:", round(split_gini(X, y, 1, 1.75), 3))"""),
("code", """def best_split(X, y):
    best = (None, None, np.inf)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for t in (values[:-1] + values[1:]) / 2:           # midpoints between sorted values
            g = split_gini(X, y, f, t)
            if g < best[2]:
                best = (f, t, g)                          #@ANS best = None  # TODO: remember this feature, threshold and score
    return best

f, t, g = best_split(X, y)
print(f"best first question: feature {f} < {t:.2f}  (gini {g:.3f})")
rest = X[:, 0] >= 2.45                                    # the 100 non-setosa flowers
f2, t2, g2 = best_split(X[rest], y[rest])
print(f"best second question: feature {f2} < {t2:.2f}  (gini {g2:.3f})")"""),
("md", """## 파트 2. scikit-learn 의 트리"""),
("code", """from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
print(export_text(tree, feature_names=["petal_length", "petal_width"]))
plt.figure(figsize=(9, 5)); plot_tree(tree, feature_names=["petal_length", "petal_width"], class_names=list(tree.classes_), filled=True); plt.show()
new_flower = np.array([[5.1, 1.8]])
print("new flower (5.1, 1.8) ->", tree.predict(new_flower)[0], "  path nodes:", tree.decision_path(new_flower).indices.tolist())"""),
("md", """## 파트 3. 타이타닉: 누가 살아남았나

승객 891명의 등급, 성별, 나이, 요금 등으로 생존을 예측한다. 성별은 0/1 로 바꾸고, 나이가 빈 칸은 학습 데이터의 중앙값으로 채운다 (간단히 하려고 전체 중앙값을 쓰지만, 엄밀히는 지난 주의 누수 규칙대로 Pipeline 안에서 해야 한다)."""),
("code", """from sklearn.model_selection import cross_val_score, StratifiedKFold
titanic = download("titanic")
df = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]].copy()
df["sex"] = (df["sex"] == "female").astype(int)
df["age"] = df["age"].fillna(df["age"].median())
Xt, yt = df.drop(columns="survived").to_numpy(), df["survived"].to_numpy()
names = list(df.columns[1:])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
depths = [1, 2, 3, 4, 5, 6, 8, 10, None]
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    train_acc = m.fit(Xt, yt).score(Xt, yt)
    cv_acc = cross_val_score(m, Xt, yt, cv=cv).mean()    #@ANS cv_acc = None  # TODO: 5-fold CV accuracy
    print(f"max_depth={str(d):4s}: train {train_acc:.3f}   CV {cv_acc:.3f}   leaves {m.get_n_leaves()}")"""),
("code", """best_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xt, yt)
print(export_text(best_tree, feature_names=names))"""),
("md", """**확인 질문**
1. 파트 1 에서 꽃잎 길이 2.45 와 꽃잎 너비 0.8 이 같은 점수다. `best_split` 은 어느 쪽을 고르나? 왜?
2. 파트 3 에서 교차 검증이 가장 높은 깊이는? 끝까지 키운 트리와 학습 · 교차 검증 정확도를 비교하자.
3. 파트 3 의 깊이 3 트리에서 첫 질문은? 타이타닉 이야기와 맞는가?
4. (도전) `min_samples_leaf` 를 1, 5, 20, 50 으로 바꿔 보자. 깊이 제한과 어떤 점이 비슷한가?

데이터 출처: Fisher (1936) Iris · Titanic passenger data (seaborn-data 사본)."""),
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
