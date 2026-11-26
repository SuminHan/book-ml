"""이 덱에만 있는 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림 (logit_sigmoid.png) 은 원본 덱 slides/v2/ml1/week02/logistic_regression/figs/ 를 그대로 쓴다.
데이터: data/flights_2024_01.csv.gz (prep_flights.py) — 미국 국내선 2024년 1월, 출발 지연 -30~90분 497,164편.
예측: 출발이 x 분 늦으면 도착이 15분 이상 늦을 (late = 1) 확률. 슬라이드 숫자는 맨 아래 print 로 확인.
옛 호텔 예약 판은 _이전/ (구간별 취소 비율이 S 자가 아니라 바꿨다)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#888888"
O = "figs/"
os.makedirs(O, exist_ok=True)
BINS = list(range(-30, 91, 5))                           # 5분 구간


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_logistic(X, y, alpha=1.0, steps=3000):
    """경사하강법 (교차 엔트로피), X 에 1 열 포함 (특징은 10 으로 나눠 넣고 계수를 되돌린다)."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= alpha * X.T @ (sig(X @ w) - y) / len(y)
    return w


def bin_rates(x, y):
    mid, rate = [], []
    for a, b in zip(BINS[:-1], BINS[1:]):
        k = (x >= a) & (x < b); mid.append((a + b - 1) / 2); rate.append(y[k].mean())
    return np.array(mid), np.array(rate)


def points(ax, x, y, n_show, seed):
    rng = np.random.default_rng(seed); i = rng.choice(len(x), n_show, replace=False)
    jx, jy = rng.uniform(-0.5, 0.5, n_show), rng.uniform(-0.05, 0.05, n_show)
    ax.scatter(x[i] + jx, y[i] + jy, s=26, c=np.where(y[i] == 1, RED, BLUE), alpha=0.3, edgecolors="none")


def line_fig(x, y, k):
    """질문: 0/1 점에 직선을 그으면 0 아래 · 1 위로 나간다."""
    xs = np.linspace(-30, 90, 300); lo, hi = -k[1] / k[0], (1 - k[1]) / k[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(1, 1.6, color=RED, alpha=0.08); ax.axhspan(-0.6, 0, color=RED, alpha=0.08)
    points(ax, x, y, 400, 0)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3)
    ax.text(hi + 2, 1.3, f"{hi:.0f}분 넘으면\n확률 > 1 ?", color=RED, fontsize=15, fontweight="bold", va="center")
    ax.text(-29, -0.35, f"{-lo:.0f}분 넘게 일찍 출발하면 확률 < 0 ?", color=RED, fontsize=15, fontweight="bold", va="center")
    ax.set_xlabel("출발이 몇 분 늦었나 (음수 = 일찍)"); ax.set_yticks([0, 0.5, 1], ["정시 0", "0.5", "지연 1"])
    ax.set_xlim(-30, 90); ax.set_ylim(-0.55, 1.55)
    fig.tight_layout(); fig.savefig(O + "flight_line.png"); plt.close(fig)
    return lo, hi


def rate_fig(x, y, k):
    """점을 5분 구간마다 세면: 비율은 0~1 안에서 S 자."""
    mid, rate = bin_rates(x, y); xs = np.linspace(-30, 90, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(1, 1.4, color=RED, alpha=0.08); ax.axhspan(-0.3, 0, color=RED, alpha=0.08)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=2.5, ls="--", label="직선 (앞 쪽)")
    ax.plot(mid, rate, "o", color=RED, ms=8, label="5분 구간마다 실제 지연 비율")
    for m_, r_ in zip(mid, rate):
        if m_ in (-8, 2, 17, 22, 32, 57):
            ax.annotate(f"{r_:.0%}", (m_, r_), (0, 12 if m_ < 30 else -24), textcoords="offset points", ha="center", fontsize=13, color=RED)
    ax.set_xlabel("출발이 몇 분 늦었나 (음수 = 일찍)"); ax.set_ylabel("도착 지연 비율"); ax.set_xlim(-30, 90); ax.set_ylim(-0.3, 1.4)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"]); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "flight_rate.png"); plt.close(fig)
    return dict(zip(mid, rate.round(3)))


def fit_fig(x, y, w):
    mid, rate = bin_rates(x, y); xs = np.linspace(-30, 90, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    points(ax, x, y, 300, 1)
    ax.plot(mid, rate, "o", color=GRAY, ms=6, label="실제 지연 비율 (5분 구간)")
    ax.plot(xs, sig(w[0] + w[1] * xs), color=BLUE, lw=3, label="로지스틱회귀")
    b = -w[0] / w[1]; ax.axvline(b, color=GRAY, ls=":"); ax.text(b + 1, 0.12, f"50%: {b:.1f}분", color=GRAY, fontsize=13)
    p30 = sig(w[0] + 30 * w[1])
    ax.annotate(f"30분 늦게 출발 → {p30:.0%}", (30, p30), (42, 0.45), fontsize=15, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("출발이 몇 분 늦었나 (음수 = 일찍)"); ax.set_yticks([0, 0.5, 1], ["정시 0", "0.5", "지연 1"]); ax.set_xlim(-30, 90); ax.set_ylim(-0.08, 1.08)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "flight_fit.png"); plt.close(fig)


def fig_2d(x, t, y, w, n_show=500, seed=2):
    """(출발 지연, 활주로 대기) 평면의 표본 점과 경계 직선."""
    rng = np.random.default_rng(seed); i = rng.choice(len(x), n_show, replace=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(x[i] + rng.uniform(-0.5, 0.5, n_show), t[i] + rng.uniform(-0.5, 0.5, n_show), s=24, c=np.where(y[i] == 1, RED, BLUE), alpha=0.45, edgecolors="none")
    ts = np.linspace(0, 60, 50); ax.plot(-(w[0] + w[2] * ts) / w[1], ts, color=GOLD, lw=3)
    ax.text(55, 8, "지연 쪽", color=RED, fontsize=15, fontweight="bold"); ax.text(-27, 52, "정시 쪽", color=BLUE, fontsize=15, fontweight="bold")
    ax.set_xlabel("출발이 몇 분 늦었나"); ax.set_ylabel("활주로까지 대기 (분)"); ax.set_xlim(-30, 90); ax.set_ylim(0, 60)
    fig.tight_layout(); fig.savefig(O + "flight_2d.png"); plt.close(fig)


def threshold_fig(x, y, w):
    p = sig(w[0] + w[1] * x); ts = np.linspace(0.05, 0.95, 91); P, R = [], []
    for t in ts:
        pred = p >= t; tp = (pred & (y == 1)).sum(); P.append(tp / max(pred.sum(), 1)); R.append(tp / (y == 1).sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ts, P, color=BLUE, lw=3, label="정밀도: 지연이라 한 것 중 진짜 지연"); ax.plot(ts, R, color=RED, lw=3, label="재현율: 진짜 지연 중 찾은 비율")
    ax.axvline(0.5, color=GRAY, ls=":"); ax.set_xlabel("임계값 (이 확률 이상이면 지연으로 판정)"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=12, loc="lower left")
    fig.tight_layout(); fig.savefig(O + "flight_threshold.png"); plt.close(fig)
    i = np.argmin(abs(ts - 0.5)); return P[i], R[i]


d = pd.read_csv("data/flights_2024_01.csv.gz")
x, t, y = d.dep_delay.values.astype(float), d.taxi_out.values.astype(float), d.late.values.astype(float)
k = np.polyfit(x, y, 1)
w1 = fit_logistic(np.c_[np.ones_like(x), x / 10], y); w1 = np.array([w1[0], w1[1] / 10])
w2 = fit_logistic(np.c_[np.ones_like(x), x / 10, t / 10], y, steps=6000); w2 = np.array([w2[0], w2[1] / 10, w2[2] / 10])
lo, hi = line_fig(x, y, k); rates = rate_fig(x, y, k); fit_fig(x, y, w1); fig_2d(x, t, y, w2); pr = threshold_fig(x, y, w1)
print(len(d), "flights, late", int(y.sum()), round(y.mean(), 3))
print("line", k.round(4), "< 0 before", round(lo, 1), "> 1 after", round(hi, 1))
print("rates", rates)
print("1D w", w1.round(3), "50% at", round(-w1[0] / w1[1], 1), "p(0)", round(sig(w1[0]), 3), "p(30)", round(sig(w1[0] + 30 * w1[1]), 3),
      "odds/min", round(np.exp(w1[1]), 3), "odds/10min", round(np.exp(10 * w1[1]), 2))
print("2D w", w2.round(3), "boundary taxi 10", round(-(w2[0] + 10 * w2[2]) / w2[1], 1), "taxi 30", round(-(w2[0] + 30 * w2[2]) / w2[1], 1), "ratio", round(w2[2] / w2[1], 2))
print("threshold 0.5: precision", round(pr[0], 3), "recall", round(pr[1], 3))
