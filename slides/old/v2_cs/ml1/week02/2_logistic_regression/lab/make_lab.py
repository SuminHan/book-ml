"""실습 노트북 원본: 이 파일을 고치고 실행하면 logistic_regression.ipynb(학생용) / logistic_regression_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 슬라이드와 같은 lichess 2013년 1월 체스 117,003판 (../data/chess_2013_01.csv.gz, prep_chess.py).
노트북은 lab/data/ 에 없으면 GitHub 저장소의 ../data/ 파일을 내려받는다 (lab/data/ 는 .gitignore)."""
import os, re, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = re.sub(r"^\d+_", "", os.path.basename(os.path.dirname(HERE)))   # 2_logistic_regression → logistic_regression
REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
DATA_URL = f"https://raw.githubusercontent.com/SuminHan/book-ml/main/{os.path.dirname(REL)}/data/chess_2013_01.csv.gz"

CELLS = [
("md", """# 로지스틱회귀 실습: 레이팅 차이 → 승률

$\\hat y = \\sigma(w_0 + w_1 x)$ = 백이 이길 확률. 슬라이드와 같은 lichess 체스 117,003판으로 슬라이드의 숫자를 재현하고, 해석과 임계값을 다룬다.

- **파트 1**: 선형회귀 경사하강법에서 한 줄 바꿔 로지스틱회귀 → "백이 200점 높으면 74%" 재현, Elo 공식과 비교
- **파트 2**: 백 레이팅 · 흑 레이팅을 따로 넣고 오즈 배수 읽기
- **파트 3**: 임계값을 바꿔 정밀도 · 재현율 보기, PR 곡선과 PR-AUC

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression"""),
("md", """## 데이터: lichess 체스 대국

2013년 1월 lichess 에서 둔 레이팅 대국 중 승패가 난 판 (무승부 제외), 레이팅 차이 ±800 안.
열은 `white` (백 레이팅), `black` (흑 레이팅), `white_win` (1 = 백 승, 0 = 흑 승).
백 · 흑은 레이팅과 상관없이 정해지므로 레이팅 차이 $x$ = 백 − 흑 은 음수도 많다. 처음 실행하면 CSV 를 내려받는다."""),
("code", f"""url = "{DATA_URL}"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/chess_2013_01.csv.gz"):
    urllib.request.urlretrieve(url, "data/chess_2013_01.csv.gz")
df = pd.read_csv("data/chess_2013_01.csv.gz")
df["diff"] = df["white"] - df["black"]                    # rating difference x = white - black
print(len(df), "games, white wins", f"{{df.white_win.mean():.1%}}")
print("white rated higher", f"{{(df['diff'] > 0).mean():.1%}}", "| black rated higher", f"{{(df['diff'] < 0).mean():.1%}}")
df.head()"""),
("md", """0 과 1 점만 찍으면 모양이 안 보인다. 100점 구간마다 **백 승률**을 세어 보자 (슬라이드의 S 자 그림)."""),
("code", """bins = np.arange(-800, 801, 100)
rate = df.groupby(pd.cut(df["diff"], bins, right=False), observed=True)["white_win"].mean()
mid = bins[:-1] + 50
print(pd.DataFrame({"bin center": mid, "white win rate": rate.values.round(3)}).to_string(index=False))
plt.plot(mid, rate.values, "o-"); plt.axhline(0.5, ls="--", color="gray")
plt.xlabel("rating difference (white - black)"); plt.ylabel("white win rate"); plt.show()"""),
("md", """## 파트 1. 시그모이드를 만들고, 선형회귀 코드에서 한 줄만 바꾸기

선형회귀의 기울기: $\\frac{1}{m}\\sum(\\hat y - y)\\,x$ (단 $\\hat y = w_0 + w_1 x$).
로지스틱회귀도 **같은 식**이고 $\\hat y = \\sigma(w_0 + w_1 x)$ 만 다르다."""),
("code", """def sigmoid(z):
    return 1 / (1 + np.exp(-z))                         #@ANS return None             # TODO: σ(z) = 1 / (1 + e^{-z})

for z in [-4, -1, 0, 1, 4]:
    print(f"sigmoid({z:2d}) = {sigmoid(z):.3f}")
zs = np.linspace(-8, 8, 200)
plt.plot(zs, sigmoid(zs)); plt.axhline(0.5, ls="--", color="gray"); plt.xlabel("z"); plt.ylabel("sigmoid(z)"); plt.show()"""),
("md", """**확인**: $z = 0$ 이면 정확히 0.5, $z$ 가 크면 1, 작으면 0 에 가까워진다. 어떤 실수든 0~1 사이 확률로 바꾼다.

이제 이 함수로 경사하강법을 돌린다. 레이팅 차이는 수백 점이라 그대로 넣으면 학습률을 맞추기 어렵다.
**100 으로 나눠** (단위: 100점) 넣고, 다 배운 뒤 $w_1$ 을 100 으로 나눠 1점당 값으로 되돌린다."""),
("code", """def gradient_descent(x, y, alpha=1.0, steps=3000):
    w0, w1 = 0.0, 0.0
    for _ in range(steps):
        y_hat = sigmoid(w0 + w1 * x)                     #@ANS y_hat = None            # TODO: the only line that differs from linear regression
        err = y_hat - y
        w0, w1 = w0 - alpha * err.mean(), w1 - alpha * (err * x).mean()
    return w0, w1

x = df["diff"].values / 100                               # unit: 100 rating points
y = df["white_win"].values
w0, w1_100 = gradient_descent(x, y)
w1 = w1_100 / 100                                         # back to per 1 rating point
print(f"w0 = {w0:.2f}, w1 = {w1:.4f} per point")
print(f"equal ratings (x = 0): white wins {sigmoid(w0):.0%}")
print(f"50% point x = {-w0 / w1:.0f}")
print(f"white 200 points higher: white wins {sigmoid(w0 + 200 * w1):.0%}")
m_ = LogisticRegression(C=1e6).fit(df[["diff"]], y)       # large C = almost no regularization
print(f"sklearn: w0 = {m_.intercept_[0]:.2f}, w1 = {m_.coef_[0, 0]:.4f}")"""),
("md", """체스의 **Elo 공식**은 $\\frac{1}{1 + 10^{-x/400}}$ 이다. 이것도 시그모이드다: $10^{-x/400} = e^{-x \\ln 10 / 400}$ 이므로 $\\sigma(0.00576\\,x)$.
실제 데이터로 배운 곡선과 겹쳐 보자."""),
("code", """def elo(x):
    return 1 / (1 + 10 ** (-x / 400))                    #@ANS return None             # TODO: Elo expected score 1 / (1 + 10^(-x/400))

print(f"Elo slope ln(10)/400 = {np.log(10) / 400:.5f}  vs  learned w1 = {w1:.5f}")
print(f"white 200 points higher: Elo {elo(200):.0%}, logistic regression {sigmoid(w0 + 200 * w1):.0%}")
xs = np.linspace(-800, 800, 300)
plt.plot(mid, rate.values, "o", color="gray", label="actual (100-point bins)")
plt.plot(xs, sigmoid(w0 + w1 * xs), lw=3, label="logistic regression")
plt.plot(xs, elo(xs), "--", label="Elo formula")
plt.xlabel("rating difference (white - black)"); plt.ylabel("P(white wins)"); plt.legend(); plt.show()"""),
("md", """**확인**: 배운 기울기가 Elo 보다 조금 작다. 2013년 lichess 초기 레이팅은 아직 덜 맞춰져 있어서, 레이팅 차이가 실력 차이를 Elo 가 가정한 만큼 다 보여 주지는 못한다.

## 파트 2. 백 레이팅 · 흑 레이팅을 따로 넣기

차이 하나 대신 두 특징 $\\hat y = \\sigma(w_0 + w_1\\,\\text{white} + w_2\\,\\text{black})$.
계수 $w$ 하나는 "그 특징이 1 늘면 오즈가 $e^{w}$ 배". 여기서는 레이팅을 100 으로 나눠 넣어 **100점당 오즈 배수**로 읽는다."""),
("code", """feats = ["white", "black"]
X = df[feats] / 100                                       # unit: 100 rating points
model = LogisticRegression(max_iter=1000).fit(X, df["white_win"])
for name, w in zip(feats, model.coef_[0]):
    odds = np.exp(w)                                     #@ANS odds = None             # TODO: odds ratio e^w
    print(f"{name:6s} w = {w:+.3f} per 100 points   odds x{odds:.2f}")
w_white, w_black = model.coef_[0]
print(f"\\nw_black / w_white = {w_black / w_white:.3f}   (close to -1: only the difference matters)")
print(f"boundary (50%): white = black {-model.intercept_[0] / w_white * 100:+.0f}   (white moves first: a small edge)")"""),
("code", """# Ask the model: same games from both sides
ask = pd.DataFrame([[1500, 1500], [1700, 1500], [1500, 1700], [1900, 1500]], columns=feats) / 100
for (wh, bl), p in zip(ask.values * 100, model.predict_proba(ask)[:, 1]):
    print(f"white {wh:.0f} vs black {bl:.0f}: P(white wins) {p:.0%}")
print(f"\\n400 points -> odds x{np.exp(4 * w_white):.1f} (data)  vs  x10 (Elo)")"""),
("md", """## 파트 3. 임계값과 PR 곡선

`predict` 는 확률 0.5 를 기준으로 자른다. 기준을 바꾸면 정밀도와 재현율이 어떻게 바뀔까?
(양성 = 백 승. 정밀도 = 백 승이라 한 것 중 진짜 백 승, 재현율 = 진짜 백 승 중 찾은 비율)"""),
("code", """from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score
p = model.predict_proba(X)[:, 1]
for th in [0.3, 0.5, 0.7]:
    pred = (p >= th).astype(int)                         #@ANS pred = None             # TODO: 1 if probability >= th
    print(f"threshold {th}: precision {precision_score(df.white_win, pred):.2f}  recall {recall_score(df.white_win, pred):.2f}  "
          f"({pred.mean():.0%} of games called a white win)")

prec, rec, _ = precision_recall_curve(df.white_win, p)
plt.plot(rec, prec); plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"PR curve  (PR-AUC = {average_precision_score(df.white_win, p):.2f}; random guessing = white win rate {df.white_win.mean():.2f})")
plt.show()"""),
("md", """**확인 질문**
1. 파트 1 에서 선형회귀와 다른 줄은 정확히 어디인가? 기울기 식은 왜 같은 모양일까? (슬라이드의 약분)
2. 파트 1 에서 레이팅 차이를 100 으로 나누지 않고 그대로 넣으면 `alpha=1.0` 에서 무슨 일이 생길까? 직접 바꿔 보자.
3. 파트 2 에서 `black` 의 오즈 배수가 1 보다 작다는 것은 무슨 뜻인가? 두 계수의 크기가 거의 같다는 것은?
4. 레이팅이 같아도 백 승률이 50% 를 조금 넘는다. 왜일까? 결정 경계에서 어떻게 보이나?
5. 임계값 0.9 로 "백이 확실히 이긴다" 고만 말하면 정밀도와 재현율은 어떻게 될까? PR-AUC 가 0.5 근처인 무작위 추측보다 얼마나 나은가?

데이터 출처: lichess.org 공개 데이터베이스 (database.lichess.org, CC0), 2013년 1월 레이팅 대국."""),
]

if __name__ == "__main__":
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    sys.path.insert(0, os.path.expanduser("~/book-ml/slides/v2/tools"))
    from expected import add_expected                         # 학생용 코드 칸 아래에 '예상 결과' (정답을 실제 실행한 출력)
    print(add_expected(f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb"))
