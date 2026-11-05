"""이 덱에만 있는 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림 (logit_sigmoid.png) 은 원본 덱 slides/v2/ml1/week02/logistic_regression/figs/ 를 그대로 쓴다.
데이터: data/chess_2013_01.csv.gz (prep_chess.py) — lichess 2013년 1월, 승패가 난 117,003판, 레이팅 차이 ±800 안.
예측: 레이팅 차이 x = 백 − 흑 일 때 백이 이길 (white_win = 1) 확률. 슬라이드 숫자는 맨 아래 print 로 확인.
옛 판: _이전/ (호텔 예약: 구간별 비율이 S 자가 아님) · _이전/flight/ (항공편 지연)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#888888"
O = "figs/"
os.makedirs(O, exist_ok=True)
BINS = list(range(-800, 801, 100))                       # 100점 구간
XL = (-800, 800)


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_logistic(X, y, alpha=1.0, steps=3000):
    """경사하강법 (교차 엔트로피), X 에 1 열 포함 (레이팅은 100 으로 나눠 넣고 계수를 되돌린다)."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= alpha * X.T @ (sig(X @ w) - y) / len(y)
    return w


def bin_rates(x, y):
    mid, rate, cnt = [], [], []
    for a, b in zip(BINS[:-1], BINS[1:]):
        k = (x >= a) & (x < b); mid.append((a + b) / 2); rate.append(y[k].mean()); cnt.append(k.sum())
    return np.array(mid), np.array(rate), np.array(cnt)


def points(ax, x, y, n_show, seed):
    rng = np.random.default_rng(seed); i = rng.choice(len(x), n_show, replace=False)
    ax.scatter(x[i], y[i] + rng.uniform(-0.05, 0.05, n_show), s=24, c=np.where(y[i] == 1, BLUE, RED), alpha=0.3, edgecolors="none")


def yaxis(ax):
    ax.set_xlabel("레이팅 차이 (백 − 흑)"); ax.set_yticks([0, 0.5, 1], ["흑 승 0", "0.5", "백 승 1"]); ax.set_xlim(*XL)


def line_fig(x, y, k):
    """질문: 0/1 점에 직선을 그으면 0 아래 · 1 위로 나간다."""
    xs = np.linspace(*XL, 300); lo, hi = -k[1] / k[0], (1 - k[1]) / k[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(1, 1.6, color=RED, alpha=0.08); ax.axhspan(-0.6, 0, color=RED, alpha=0.08)
    points(ax, x, y, 400, 0)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3)
    ax.text(hi - 20, 1.32, f"+{hi:.0f}점 넘으면\n확률 > 1 ?", color=RED, fontsize=15, fontweight="bold", va="center", ha="right")
    ax.text(lo + 20, -0.32, f"{lo:.0f}점 아래면\n확률 < 0 ?", color=RED, fontsize=15, fontweight="bold", va="center")
    yaxis(ax); ax.set_ylim(-0.55, 1.55)
    fig.tight_layout(); fig.savefig(O + "chess_line.png"); plt.close(fig)
    return lo, hi


def rate_fig(x, y, k):
    """점을 100점 구간마다 세면: 비율은 0~1 안에서 S 자."""
    mid, rate, cnt = bin_rates(x, y); xs = np.linspace(*XL, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(1, 1.3, color=RED, alpha=0.08); ax.axhspan(-0.3, 0, color=RED, alpha=0.08)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=2.5, ls="--", label="직선 (앞 쪽)")
    ax.plot(mid, rate, "o", color=BLUE, ms=9, label="100점 구간마다 실제 백 승률")
    for m_, r_ in zip(mid, rate):
        if m_ in (-750, -550, -250, 50, 250, 550, 750):
            ax.annotate(f"{r_:.0%}", (m_, r_), (0, 12 if m_ < 0 else -24), textcoords="offset points", ha="center", fontsize=13, color=BLUE)
    ax.set_xlabel("레이팅 차이 (백 − 흑)"); ax.set_ylabel("백 승률"); ax.set_xlim(*XL); ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"]); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "chess_rate.png"); plt.close(fig)
    return {int(m_): (round(r_, 3), int(c)) for m_, r_, c in zip(mid, rate, cnt)}


def fit_fig(x, y, w):
    mid, rate, _ = bin_rates(x, y); xs = np.linspace(*XL, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    points(ax, x, y, 300, 1)
    ax.plot(mid, rate, "o", color=GRAY, ms=6, label="실제 백 승률 (100점 구간)")
    ax.plot(xs, sig(w[0] + w[1] * xs), color=BLUE, lw=3, label="로지스틱회귀")
    ax.plot(xs, 1 / (1 + 10 ** (-xs / 400)), color=GOLD, lw=2, ls="--", label="Elo 공식")
    p200 = sig(w[0] + 200 * w[1])
    ax.annotate(f"200점 높으면 → {p200:.0%}", (200, p200), (330, 0.42), fontsize=15, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
    yaxis(ax); ax.set_ylim(-0.08, 1.08); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "chess_fit.png"); plt.close(fig)


def fig_2d(wh, bl, y, w, n_show=500, seed=2):
    """(흑 레이팅, 백 레이팅) 평면의 표본 점과 경계 직선 w0 + w1 white + w2 black = 0."""
    rng = np.random.default_rng(seed); i = rng.choice(len(wh), n_show, replace=False)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(bl[i], wh[i], s=22, c=np.where(y[i] == 1, BLUE, RED), alpha=0.45, edgecolors="none")
    bs = np.linspace(900, 2400, 50); ax.plot(bs, -(w[0] + w[2] * bs) / w[1], color=GOLD, lw=3)
    ax.text(1000, 2200, "백 승 쪽", color=BLUE, fontsize=15, fontweight="bold"); ax.text(1950, 1050, "흑 승 쪽", color=RED, fontsize=15, fontweight="bold")
    ax.set_xlabel("흑 레이팅"); ax.set_ylabel("백 레이팅"); ax.set_xlim(900, 2400); ax.set_ylim(900, 2400); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(O + "chess_2d.png"); plt.close(fig)


def threshold_fig(x, y, w):
    p = sig(w[0] + w[1] * x); ts = np.linspace(0.1, 0.9, 81); P, R = [], []
    for t in ts:
        pred = p >= t; tp = (pred & (y == 1)).sum(); P.append(tp / max(pred.sum(), 1)); R.append(tp / (y == 1).sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ts, P, color=BLUE, lw=3, label="정밀도: 백 승이라 한 것 중 진짜 백 승"); ax.plot(ts, R, color=RED, lw=3, label="재현율: 진짜 백 승 중 찾은 비율")
    ax.axvline(0.5, color=GRAY, ls=":"); ax.set_xlabel("임계값 (이 확률 이상이면 백 승으로 판정)"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=12, loc="lower left")
    fig.tight_layout(); fig.savefig(O + "chess_threshold.png"); plt.close(fig)
    i = np.argmin(abs(ts - 0.5)); return P[i], R[i]


d = pd.read_csv("data/chess_2013_01.csv.gz")
wh, bl, y = d.white.values.astype(float), d.black.values.astype(float), d.white_win.values.astype(float)
x = wh - bl
k = np.polyfit(x, y, 1)
w1 = fit_logistic(np.c_[np.ones_like(x), x / 100], y); w1 = np.array([w1[0], w1[1] / 100])
w2 = fit_logistic(np.c_[np.ones_like(x), (wh - 1600) / 100, (bl - 1600) / 100], y, steps=6000)
w2 = np.array([w2[0] - 16 * (w2[1] + w2[2]), w2[1] / 100, w2[2] / 100])          # 1600 을 뺀 만큼 절편으로 되돌린다
lo, hi = line_fig(x, y, k); rates = rate_fig(x, y, k); fit_fig(x, y, w1); fig_2d(wh, bl, y, w2); pr = threshold_fig(x, y, w1)
print(len(d), "games, white wins", int(y.sum()), round(y.mean(), 3))
print("line", k.round(5), "< 0 below", round(lo), "> 1 above", round(hi))
print("rates", rates)
print("1D w", w1.round(5), "p(0)", round(sig(w1[0]), 3), "50% at", round(-w1[0] / w1[1], 1), "p(+200)", round(sig(w1[0] + 200 * w1[1]), 3),
      "odds/100", round(np.exp(100 * w1[1]), 2), "odds/400", round(np.exp(400 * w1[1]), 2), "Elo slope", round(np.log(10) / 400, 5), "Elo p(+200)", round(1 / (1 + 10 ** -0.5), 3))
print("2D w", w2.round(5), "boundary white - black =", round(-w2[0] / w2[1], 1), "(w2/w1", round(w2[2] / w2[1], 3), ")")
print("threshold 0.5: precision", round(pr[0], 3), "recall", round(pr[1], 3))
