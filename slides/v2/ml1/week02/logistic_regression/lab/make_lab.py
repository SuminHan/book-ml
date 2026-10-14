"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 공부 시간 → 합격 (슬라이드와 같은 예시 40명, data.py 와 같은 생성식) · Titanic (seaborn-data 에서 내려받기)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (logistic_regression) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 로지스틱회귀 실습 (25분)

$\\hat y = \\sigma(w_0 + w_1 x)$ = 1일 확률. 슬라이드의 숫자를 재현하고, 실제 타이타닉 승객 데이터로 해석과 임계값을 다룬다.

- **파트 1** (8분): 선형회귀 경사하강법에서 한 줄 바꿔 로지스틱회귀 → 7시간 92% 재현
- **파트 2** (9분): 타이타닉: 성별 · 객실 등급 · 나이로 생존 예측, 오즈 배수 읽기
- **파트 3** (8분): 임계값을 바꿔 정밀도 · 재현율 보기, PR 곡선과 PR-AUC

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))                         #@ANS return None             # TODO: σ(z) = 1 / (1 + e^{-z})

# Same example as the slides: study hours of 40 students -> pass (1) / fail (0)
r = np.random.default_rng(11); x = np.round(r.uniform(0.5, 10, 40), 1)
y = (r.uniform(size=40) < sigmoid(-6 + 1.2 * x)).astype(int)
print("passed", y.sum(), "/", len(y))"""),
("md", """## 파트 1. 선형회귀 코드에서 한 줄만 바꾸기

선형회귀의 기울기: $\\frac{1}{m}\\sum(\\hat y - y)\\,x$ (단 $\\hat y = w_0 + w_1 x$).
로지스틱회귀도 **같은 식**이고 $\\hat y = \\sigma(w_0 + w_1 x)$ 만 다르다."""),
("code", """def gradient_descent(alpha=0.1, steps=20000):
    w0, w1 = 0.0, 0.0
    for _ in range(steps):
        y_hat = sigmoid(w0 + w1 * x)                     #@ANS y_hat = None            # TODO: the only line that differs from linear regression
        err = y_hat - y
        w0, w1 = w0 - alpha * err.mean(), w1 - alpha * (err * x).mean()
    return w0, w1

w0, w1 = gradient_descent()
print(f"w0 = {w0:.2f}, w1 = {w1:.2f}")
print(f"decision boundary x = {-w0 / w1:.2f} h,  7 h -> pass probability {sigmoid(w0 + 7 * w1):.0%}")
m_ = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)       # large C = almost no regularization
print(f"sklearn: w0 = {m_.intercept_[0]:.2f}, w1 = {m_.coef_[0, 0]:.2f}")

xs = np.linspace(0, 10.5, 200)
plt.scatter(x, y, c=y, cmap="coolwarm_r"); plt.plot(xs, sigmoid(w0 + w1 * xs))
plt.axvline(-w0 / w1, ls="--", color="gray"); plt.xlabel("study hours"); plt.ylabel("pass probability"); plt.show()"""),
("md", """## 파트 2. 타이타닉: 누가 살아남았나

1912년 타이타닉 승객 891명. `survived` (1 = 생존) 를 성별 · 객실 등급 · 나이로 예측한다. 처음 실행하면 CSV 를 내려받는다."""),
("code", """url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/titanic.csv"):
    import urllib.request; urllib.request.urlretrieve(url, "data/titanic.csv")
df = pd.read_csv("data/titanic.csv")
df = df.dropna(subset=["age"])                             # drop passengers with unknown age (see the dirty data lesson!)
df["female"] = (df["sex"] == "female").astype(int)
df["first_class"] = (df["pclass"] == 1).astype(int)
df["third_class"] = (df["pclass"] == 3).astype(int)
feats = ["female", "first_class", "third_class", "age"]
print(len(df), "passengers, survival rate", f"{df.survived.mean():.0%}")
df[feats + ["survived"]].head()"""),
("code", """model = LogisticRegression(max_iter=1000).fit(df[feats], df["survived"])
for name, w in zip(feats, model.coef_[0]):
    odds = np.exp(w)                                     #@ANS odds = None             # TODO: odds ratio e^w
    print(f"{name:12s} w = {w:+.3f}   odds x{odds:.2f}")
print("\\nReading: the female odds ratio = how many times higher a woman's survival odds are than a man's, at the same class and age")"""),
("code", """# Ask the model: 20-year-old woman in 1st class vs 20-year-old man in 3rd class
ask = pd.DataFrame([[1, 1, 0, 20], [0, 0, 1, 20]], columns=feats)
for desc, p in zip(["woman, 20, 1st class", "man, 20, 3rd class"], model.predict_proba(ask)[:, 1]):
    print(f"{desc}: survival probability {p:.0%}")"""),
("md", """## 파트 3. 임계값과 PR 곡선

`predict` 는 확률 0.5 를 기준으로 자른다. 기준을 바꾸면 정밀도와 재현율이 어떻게 바뀔까?"""),
("code", """from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score
p = model.predict_proba(df[feats])[:, 1]
for th in [0.3, 0.5, 0.7]:
    pred = (p >= th).astype(int)                         #@ANS pred = None             # TODO: 1 if probability >= th
    print(f"threshold {th}: precision {precision_score(df.survived, pred):.2f}  recall {recall_score(df.survived, pred):.2f}")

prec, rec, _ = precision_recall_curve(df.survived, p)
plt.plot(rec, prec); plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"PR curve  (PR-AUC = {average_precision_score(df.survived, p):.2f}; random guessing = survival rate {df.survived.mean():.2f})")
plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 선형회귀와 다른 줄은 정확히 어디인가? 기울기 식은 왜 같은 모양일까? (슬라이드의 약분)
2. 파트 2 에서 `third_class` 의 오즈 배수가 1 보다 작다는 것은 무슨 뜻인가?
3. 파트 2 에서 나이를 모르는 승객 177명을 뺐다. 이게 결과를 어떻게 바꿀 수 있을까?
4. 구명보트에 태울 사람을 고르는 상황이라면 임계값을 올릴까 내릴까? 이유는?

데이터 출처: Titanic passenger data (Encyclopedia Titanica 기반, seaborn-data 사본)."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    print(lab.verify(f"{HERE}/{TOPIC}_solution.ipynb"))
