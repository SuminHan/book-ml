"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: iris (scikit-learn 내장, 150송이). 노트북 첫 셀이 일부러 더러운 값을 섞는다 (seed 고정)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (dirty_data) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 더티 데이터 실습: 지저분한 iris 청소하기

붓꽃(iris) 150송이 데이터에 **일부러** 더러운 값을 섞어 두었다. 슬라이드의 세 가지 모양(결측치 · 중복 · 이상치)을 찾아서 고치고, 고치기 전과 후에 모델 정확도가 얼마나 다른지 본다.

- **파트 0**: 더러운 데이터 받기, 첫눈에 훑어보기
- **파트 1**: 결측치: 빈칸 · 0 · -999 찾기
- **파트 2**: 중복과 라벨 표기: `'Setosa'`, `' setosa'` 는 같은 꽃?
- **파트 3**: 이상치: 51cm 꽃받침은 오타일까?
- **파트 4**: 청소 전 vs 후: 같은 모델의 정확도 비교

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# ---- Make the data dirty on purpose (just run this cell, don't edit) ----
iris = load_iris(as_frame=True)
clean = iris.frame.rename(columns={"sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width",
                                   "petal length (cm)": "petal_length", "petal width (cm)": "petal_width"})
clean["species"] = iris.target_names[iris.target]
clean = clean.drop(columns="target")

def make_dirty(clean, seed=2026):
    \"\"\"Mix missing values, duplicates, outliers and inconsistent labels into clean iris\"\"\"
    rng = np.random.default_rng(seed)
    d = clean.copy()
    d.loc[rng.choice(150, 5, replace=False), "petal_length"] = np.nan           # real blanks
    d.loc[rng.choice(150, 4, replace=False), "sepal_width"] = -999              # 'missing' written as -999
    d.loc[rng.choice(np.where(clean.species != "setosa")[0], 5, replace=False), "petal_width"] = 0   # 'missing' written as 0
    d.loc[rng.choice(150, 3, replace=False), "sepal_length"] *= 10             # typo: written in mm instead of cm
    styles = [str.capitalize, lambda s: " " + s, lambda s: s + " ", str.upper]
    for i, f in zip(rng.choice(150, 4, replace=False), styles):              # same label, different spelling
        d.loc[i, "species"] = f(d.loc[i, "species"])
    dups = d.loc[rng.choice(150, 8, replace=False)].copy()                    # the same record twice
    dups.iloc[:3, dups.columns.get_loc("species")] = [str.upper(v) for v in dups.species.iloc[:3]]  # 3 of them with different spelling only
    return pd.concat([d, dups]).sample(frac=1, random_state=0).reset_index(drop=True)

df = make_dirty(clean)
print(df.shape); df.head()"""),
("md", """## 파트 0. 첫눈에 훑어보기

`describe()` 로 각 열의 최솟값 · 최댓값 · 평균을 본다. 이상한 점이 몇 개 보이나?"""),
("code", """df.describe().round(2)"""),
("md", """## 파트 1. 결측치: 빈칸 · 0 · -999

빈칸(`NaN`)은 `isna()` 가 찾아 주지만, `0` 과 `-999` 는 숫자인 척한다. 꽃받침·꽃잎 길이가 0cm 이하일 수 있을까?"""),
("code", """num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
print("Blank (NaN) count:\\n", df[num_cols].isna().sum(), "\\n")
print("Values <= 0:\\n", (df[num_cols] <= 0).sum())"""),
("code", """# Turn 'missing' values pretending to be numbers (<= 0) into real blanks (NaN)
df[num_cols] = df[num_cols].mask(df[num_cols] <= 0)        #@ANS df[num_cols] = None   # TODO: <= 0 -> NaN (hint: .mask)
print(df[num_cols].isna().sum())"""),
("md", """## 파트 2. 중복과 라벨 표기

`value_counts()` 로 꽃 종류를 세어 보자. 종류가 3개여야 하는데 몇 개인가?"""),
("code", """print(df["species"].value_counts(), "\\n")
print("Exact duplicate rows:", df.duplicated().sum())"""),
("code", """# One spelling: strip spaces + lowercase
df["species"] = df["species"].str.strip().str.lower()      #@ANS df["species"] = None    # TODO: strip + lowercase
print(df["species"].value_counts(), "\\n")
print("Duplicate rows after fixing labels:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("After dropping duplicates:", df.shape)"""),
("md", """**관찰**: 표기를 맞추기 전에는 컴퓨터가 찾은 중복 수가 실제보다 적다. 띄어쓰기나 대소문자 하나만 달라도 다른 줄로 본다. 무엇을 같은 것으로 볼지는 사람이 정한다."""),
("md", """## 파트 3. 이상치: 51cm 꽃받침?

붓꽃의 꽃받침 길이는 보통 4~8cm 다. 상자 그림으로 튀는 값을 찾아보자."""),
("code", """df[num_cols].plot(kind="box", figsize=(8, 3.5)); plt.ylabel("cm"); plt.show()
print(df.loc[df.sepal_length > 10, ["sepal_length", "species"]])"""),
("code", """# Domain knowledge: a sepal longer than 10 cm is a typo in mm -> divide by 10 to get cm
too_big = df.sepal_length > 10
df.loc[too_big, "sepal_length"] = df.loc[too_big, "sepal_length"] / 10      #@ANS df.loc[too_big, "sepal_length"] = None   # TODO: mm → cm
print(df.sepal_length.describe().round(2))"""),
("md", """**생각해 보기**: 여기서는 '10 으로 나누기' 로 고쳤다. 만약 꽃받침이 9.5cm 라면? 오타인지 진짜 큰 꽃인지는 데이터만 봐서는 모른다. 지울지, 고칠지, 남길지는 그 분야를 아는 사람이 정한다."""),
("md", """## 파트 4. 청소 전 vs 후: 모델은 얼마나 달라질까?

남은 빈칸은 그 꽃 종류의 평균으로 채우고 (꽃 종류별 평균), 같은 모델(로지스틱회귀, 5-겹 교차검증) 로 정확도를 비교한다. 비교용 '청소 안 한' 데이터는 첫 셀의 `make_dirty(clean)` 결과에서 빈칸만 0 으로 채운 것이다."""),
("code", """from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Cleaned data: fill the remaining blanks with the mean of each species
filled = df.copy()
filled[num_cols] = filled.groupby("species")[num_cols].transform(lambda c: c.fillna(c.mean()))   #@ANS filled[num_cols] = None   # TODO: fill with the per-species mean

# Uncleaned data: same make_dirty(clean) as the first cell, only blanks filled with 0
raw = make_dirty(clean).fillna(0)

import warnings; warnings.filterwarnings("ignore")      # hide warnings caused by 1-member 'species'
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
for name, data in [("not cleaned", raw), ("cleaned", filled), ("original clean iris", clean)]:
    acc = cross_val_score(model, data[num_cols], data["species"], cv=5)
    print(f"{name:20s}: {data.species.nunique()} species, accuracy {acc.mean():.1%}")"""),
("md", """**확인 질문**
1. 파트 1 에서 `-999` 를 그대로 두고 꽃받침 너비의 평균을 내면 얼마가 되나? 실제 평균(약 3cm)과 비교해 보자.
2. 파트 2 에서 표기를 맞추기 전과 후에 `duplicated()` 가 찾은 수가 다른 이유는?
3. 파트 3 에서 꽃받침 9.5cm 는 오타일까 진짜일까? 무엇을 더 알아야 판단할 수 있을까?
4. 파트 4 에서 '청소 안 함' 의 꽃 종류가 3개가 아닌 이유는? 모델은 무엇을 배우게 될까?

데이터 출처: Fisher (1936), Iris (scikit-learn 내장)."""),
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
