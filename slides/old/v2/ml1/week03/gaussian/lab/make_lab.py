"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 골턴 보드 시뮬레이션 · Palmer Penguins (seaborn-data 사본에서 내려받기)."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (gaussian) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 가우시안 분포 실습

작은 효과의 합이 종 모양을 만든다. 골턴 보드를 시뮬레이션하고, 실제 펭귄 데이터에 종을 맞추고, 종 두 개로 분류해 본다.

- **파트 1**: 골턴 보드 시뮬레이션
- **파트 2**: 펭귄 날개 길이에 가우시안 맞추기 ($\\hat\\mu$, $\\hat\\sigma$, 68% 규칙)
- **파트 3**: 가우시안 + 베이즈 정리로 펭귄 종 맞히기

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
from math import comb"""),
("md", """## 파트 1. 골턴 보드

공은 못 10줄을 지나며 매번 오른쪽(1) 또는 왼쪽(0) 으로 반반 튄다. 공이 떨어지는 칸 = 오른쪽으로 튄 횟수.
슬라이드: 공 1000개면 가운데 칸에 약 246개."""),
("code", """def galton(n_balls, n_rows, seed=0):
    rng = np.random.default_rng(seed)
    bounces = rng.integers(0, 2, size=(n_balls, n_rows))  # 1 = bounced right
    return bounces.sum(axis=1)                            #@ANS return None  # TODO: bin of each ball = number of right bounces

bins = galton(1000, 10)
counts = np.bincount(bins, minlength=11)
expected = [round(1000 * comb(10, k) / 2 ** 10) for k in range(11)]
print("simulated:", counts.tolist())
print("expected :", expected)
plt.bar(range(11), counts); plt.xlabel("bin (right bounces)"); plt.ylabel("balls"); plt.show()"""),
("code", """# More rows -> smoother bell
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, rows in zip(axs, [3, 10, 50]):
    b = galton(20000, rows)
    ax.hist(b, bins=np.arange(rows + 2) - 0.5); ax.set_title(f"{rows} rows")
plt.show()"""),
("md", """## 파트 2. 실제 데이터에 종 맞추기

남극 펭귄 3종(Adelie, Chinstrap, Gentoo) 333마리의 날개 길이(mm). 처음 실행하면 CSV 를 내려받는다.

$$\\hat\\mu = \\frac{1}{m}\\sum x^{(i)}, \\qquad \\hat\\sigma^2 = \\frac{1}{m}\\sum \\big(x^{(i)} - \\hat\\mu\\big)^2$$"""),
("code", """import pandas as pd
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/penguins.csv"):
    import urllib.request; urllib.request.urlretrieve(url, "data/penguins.csv")
df = pd.read_csv("data/penguins.csv").dropna()
print(df["species"].value_counts())
df.head()"""),
("code", """def fit_gaussian(x):
    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2))               #@ANS sigma = None  # TODO: MLE standard deviation (divide by m)
    return mu, sigma

def gaussian_pdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

xs = np.linspace(165, 235, 300)
params = {}
for sp in ["Adelie", "Chinstrap", "Gentoo"]:
    x = df.loc[df["species"] == sp, "flipper_length_mm"].to_numpy(float)
    mu, sigma = fit_gaussian(x)
    params[sp] = (mu, sigma)
    inside = np.mean(np.abs(x - mu) <= sigma)             #@ANS inside = None  # TODO: fraction within mu ± sigma
    print(f"{sp:9s}: n={len(x):3d}  mu={mu:6.1f}  sigma={sigma:4.1f}  within 1 sigma: {inside:.0%}")
    plt.hist(x, bins=20, density=True, alpha=0.35)
    plt.plot(xs, gaussian_pdf(xs, mu, sigma), lw=2, label=sp)
plt.xlabel("flipper length (mm)"); plt.legend(); plt.show()"""),
("md", """**관찰**: 1σ 안에 든 비율이 68% 에 가까운가?

## 파트 3. 종 두 개로 분류하기

Adelie 와 Chinstrap 은 날개 길이가 겹친다. 날개 길이 하나만 보고 둘을 구별해 보자.

$$P(\\text{Adelie} \\mid x) = \\frac{p(x \\mid \\text{Adelie})\\,P(\\text{Adelie})}{p(x \\mid \\text{Adelie})\\,P(\\text{Adelie}) + p(x \\mid \\text{Chinstrap})\\,P(\\text{Chinstrap})}$$"""),
("code", """two = df[df["species"].isin(["Adelie", "Chinstrap"])]
x = two["flipper_length_mm"].to_numpy(float)
is_adelie = (two["species"] == "Adelie").to_numpy()

def posterior_adelie(x, params, prior_adelie):
    a = gaussian_pdf(x, *params["Adelie"]) * prior_adelie
    c = gaussian_pdf(x, *params["Chinstrap"]) * (1 - prior_adelie)   #@ANS c = None  # TODO: Chinstrap likelihood x prior
    return a / (a + c)

prior = is_adelie.mean()
for name, p in [("equal prior 0.5", 0.5), (f"data prior {prior:.2f}", prior)]:
    pred = posterior_adelie(x, params, p) >= 0.5
    print(f"{name:16s}: accuracy {np.mean(pred == is_adelie):.3f}   P(Adelie | 195 mm) = {posterior_adelie(195.0, params, p):.2f}")"""),
("code", """# Compare with scikit-learn's Gaussian naive Bayes (same idea, one feature)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB().fit(x.reshape(-1, 1), is_adelie)
print("GaussianNB accuracy:", round(nb.score(x.reshape(-1, 1), is_adelie), 3))
print("GaussianNB means:", nb.theta_.ravel().round(1), " stds:", np.sqrt(nb.var_).ravel().round(1))"""),
("md", """**확인 질문**
1. 파트 1 에서 시뮬레이션 결과가 기대값과 조금 다른 이유는? 공을 10만 개로 늘리면?
2. 파트 2 에서 세 종 중 종 모양에 가장 잘 맞는 것은? 68% 규칙에서 가장 벗어난 종은?
3. 파트 3 에서 사전확률을 반반으로 둘 때와 데이터 비율로 둘 때 정확도가 왜 다른가? 195 mm 펭귄의 판정은 어떻게 바뀌나?
4. (도전) 날개 길이 대신 `bill_length_mm` 를 쓰면 Adelie / Chinstrap 구별이 얼마나 좋아지나?

데이터 출처: Gorman, Williams, Fraser (2014) PLoS ONE 9(3) · Palmer Station LTER, seaborn-data 사본."""),
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
