"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 공부시간 → 점수 (슬라이드와 같은 예시 30명, data.py 와 같은 생성식) · Auto MPG (UCI, seaborn-data 에서 내려받기)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (linear_regression) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 선형회귀 실습

직선 $h = w_0 + w_1 x$, 비용 $J$ (오차 제곱의 평균), 경사하강법. 슬라이드의 숫자를 직접 재현하고 실제 자동차 데이터에 써 본다.

- **파트 1**: 경사하강법 직접 구현 → 7시간 75.4점 재현 · 학습률 바꿔 발산 보기
- **파트 2**: 자동차 398대: 마력 → 연비, 직선 vs $x^2$ vs $\\log x$
- **파트 3**: 차수를 올리며 학습 / 시험 오차 비교 (과적합)

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Same example as the slides: study hours and scores of 30 students
r = np.random.default_rng(7); x = r.uniform(0.5, 10, 30)
y = 40 + 5 * x + r.normal(0, 5, 30)
plt.scatter(x, y); plt.xlabel("study hours"); plt.ylabel("score"); plt.show()"""),
("md", """## 파트 1. 경사하강법 직접 구현

$$J = \\frac{1}{2m}\\sum (h - y)^2, \\qquad \\frac{\\partial J}{\\partial w_0} = \\frac{1}{m}\\sum (h - y), \\qquad \\frac{\\partial J}{\\partial w_1} = \\frac{1}{m}\\sum (h - y)\\,x$$"""),
("code", """def cost(w0, w1, x, y):
    return np.mean((w0 + w1 * x - y) ** 2) / 2

def gradient(w0, w1, x, y):
    err = w0 + w1 * x - y                                # error h - y
    g0 = err.mean()                                      #@ANS g0 = None               # TODO: ∂J/∂w0
    g1 = (err * x).mean()                                #@ANS g1 = None               # TODO: ∂J/∂w1
    return g0, g1

def gradient_descent(x, y, alpha, steps, w0=0.0, w1=0.0):
    history = []
    for _ in range(steps):
        g0, g1 = gradient(w0, w1, x, y)
        w0, w1 = w0 - alpha * g0, w1 - alpha * g1        #@ANS w0, w1 = None, None     # TODO: one step
        history.append(cost(w0, w1, x, y))
    return w0, w1, history

w0, w1, hist = gradient_descent(x, y, alpha=0.02, steps=5000)
print(f"w0 = {w0:.2f}, w1 = {w1:.2f}  ->  7 hours gives {w0 + 7 * w1:.1f} points")
m_ = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"sklearn: w0 = {m_.intercept_:.2f}, w1 = {m_.coef_[0]:.2f}")
plt.plot(hist); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("cost J"); plt.show()"""),
("code", """# Try larger learning rates: 0.05 and 0.06
for alpha in [0.05, 0.06]:
    _, _, h = gradient_descent(x, y, alpha, steps=30)
    print(f"alpha={alpha}: first J={h[0]:.1f}  after 30 steps J={h[-1]:.3g}")"""),
("md", """**관찰**: 0.06 은 발산한다. 슬라이드의 오른쪽 그림처럼 바닥을 건너뛰며 매번 더 높이 튄다.

$R^2$ 와 RMSE 도 확인하자 (슬라이드: RMSE 4.40, $R^2$ 0.915)."""),
("code", """pred = w0 + w1 * x
rmse = np.sqrt(np.mean((pred - y) ** 2))                 #@ANS rmse = None             # TODO: RMSE (no 1/2)
print(f"RMSE = {rmse:.2f},  R² = {r2_score(y, pred):.3f}")"""),
("md", """## 파트 2. 실제 자동차 데이터: 마력 → 연비

Auto MPG: 1970~82년 자동차 398대. 마력(horsepower) 이 클수록 연비(mpg, 갤런당 마일) 는 떨어지는데, 직선일까? 처음 실행하면 CSV 를 내려받는다."""),
("code", """import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/mpg.csv"):
    import urllib.request; urllib.request.urlretrieve(url, "data/mpg.csv")
df = pd.read_csv("data/mpg.csv").dropna(subset=["horsepower"])
hp, mpg = df["horsepower"].to_numpy(float), df["mpg"].to_numpy(float)
print(len(df), "cars");  print(df[["name", "horsepower", "mpg"]].head())
plt.scatter(hp, mpg, s=12); plt.xlabel("horsepower"); plt.ylabel("mpg"); plt.show()"""),
("code", """features = {
    "x":        hp.reshape(-1, 1),
    "x, x²":    np.c_[hp, hp ** 2],
    "log x":    np.log(hp).reshape(-1, 1),             #@ANS "log x":    None,         # TODO: one column, log-transformed
}
hs = np.linspace(hp.min(), hp.max(), 200)
plt.scatter(hp, mpg, s=10, color="lightgray")
for name, F in features.items():
    model = LinearRegression().fit(F, mpg)
    print(f"{name:8s}: R² = {r2_score(mpg, model.predict(F)):.3f}")
    Fs = {"x": hs.reshape(-1, 1), "x, x²": np.c_[hs, hs ** 2], "log x": np.log(hs).reshape(-1, 1)}[name]
    plt.plot(hs, model.predict(Fs), lw=2, label=name)
plt.legend(); plt.xlabel("horsepower"); plt.ylabel("mpg"); plt.show()"""),
("md", """## 파트 3. 차수를 계속 올리면? (과적합)

자동차 **25대만** 학습에 쓰고, 나머지 367대로 시험한다. 데이터가 적을 때 차수를 올리면 어떻게 될까?"""),
("code", """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

h_tr, h_te, m_tr, m_te = train_test_split(hp.reshape(-1, 1), mpg, train_size=25, random_state=0)
degrees = [1, 2, 3, 4, 6, 8, 10]
tr_err, te_err = [], []
for k in degrees:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(k), LinearRegression()).fit(h_tr, m_tr)
    tr_err.append(np.sqrt(mean_squared_error(m_tr, model.predict(h_tr))))
    te_err.append(np.sqrt(mean_squared_error(m_te, model.predict(h_te))))   #@ANS te_err.append(None)  # TODO: test RMSE
    print(f"degree {k:2d}: train RMSE {tr_err[-1]:6.2f}   test RMSE {te_err[-1]:8.2f}")
plt.plot(degrees, tr_err, "o-", label="train"); plt.plot(degrees, te_err, "s-", label="test")
plt.yscale("log"); plt.xlabel("degree"); plt.ylabel("RMSE (mpg)"); plt.legend(); plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 학습률 0.06 이 발산하는 이유를 슬라이드의 그림으로 설명해 보자.
2. 파트 2 에서 가장 좋은 특징은? 그 모델도 '선형'회귀인 이유는?
3. 파트 3 에서 알맞은 차수는 몇이라고 생각하나? 학습 오차만 보고 고르면 몇 차를 고르게 될까? `train_size` 를 200 으로 늘리면 어떻게 달라질까?
4. (도전) `df` 의 `weight`(무게) 를 특징으로 추가하면 $R^2$ 이 얼마나 오를까?

데이터 출처: Auto MPG, UCI Machine Learning Repository (Quinlan, 1993), seaborn-data 사본."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    print(lab.verify(f"{HERE}/{TOPIC}_solution.ipynb"))
