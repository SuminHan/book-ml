"""이 덱에만 있는 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림은 원본 덱 slides/v2/ml1/week02/logistic_regression/figs/ 를 그대로 쓴다.
데이터: 호텔 예약 119,390건 (Antonio, Almeida & Nunes 2019) = ~/book-ml/notebooks/datashop/data/hotels.csv — 쇼츠 lessons/ml1/logistic_regression 과 같은 데이터."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#888888"
O = "figs/"
os.makedirs(O, exist_ok=True)


def hotel_line(t, y, n_show=300, seed=0):
    """며칠 전 예약 t → 취소 y (0/1) 에 직선을 그으면: 표본 점 + 전체 데이터로 맞춘 직선."""
    k = np.polyfit(t, y, 1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(t), n_show, replace=False)
    jit = rng.uniform(-0.07, 0.07, n_show)
    xs = np.linspace(0, 740, 300)
    over = (1 - k[1]) / k[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(t[i], y[i] + jit, s=38, c=np.where(y[i] == 1, RED, BLUE), alpha=0.35, edgecolors="none", zorder=3)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3)
    ax.axhspan(1, 1.6, color=RED, alpha=0.08)
    ax.axvline(over, color=GRAY, ls=":")
    ax.text(over + 10, 1.3, f"{over:.0f}일 넘으면\n확률 > 1 ?", color=RED, fontsize=15, fontweight="bold", va="center")
    same_day = y[t == 0].mean()
    ax.annotate(f"당일 예약 {k[1]:.0%} ? (실제 {same_day:.0%})", (0, k[1]), (60, 0.62), fontsize=15, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("며칠 전에 예약했나 (일)"); ax.set_yticks([0, 0.5, 1], ["유지 0", "0.5", "취소 1"])
    ax.set_xlim(0, 740); ax.set_ylim(-0.12, 1.55)
    fig.tight_layout(); fig.savefig(O + "hotel_line.png"); plt.close(fig)
    return k, over, same_day


d = pd.read_csv(os.path.expanduser("~/book-ml/notebooks/datashop/data/hotels.csv"))
k, over, same_day = hotel_line(d["lead_time"].values, d["is_canceled"].values)
print("line", k.round(5), "hits 1 at", round(over), "days, same-day actual", round(same_day, 3))


# ---- 호텔 취소 로지스틱회귀 (슬라이드 숫자: data 아래 print 로 확인) ----
def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_logistic(X, y, alpha=0.5, steps=4000):
    """경사하강법 (교차 엔트로피), X 에 1 열 포함. 특징은 미리 비슷한 크기로 (개월 / 12, 요청 수 / 2) 맞춰 넣고 계수를 되돌린다."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= alpha * X.T @ (sig(X @ w) - y) / len(y)
    return w


def hotel_fit(m, y, w, n_show=300, seed=0):
    """개월 m → 취소 확률 곡선 + 1개월 구간별 실제 취소 비율 + 표본 점."""
    rng = np.random.default_rng(seed); i = rng.choice(len(m), n_show, replace=False); jit = rng.uniform(-0.05, 0.05, n_show)
    xs = np.linspace(0, 24, 300)
    edges = np.arange(0, 25); mid = edges[:-1] + 0.5
    cnt = np.array([((m >= a) & (m < a + 1)).sum() for a in edges[:-1]])
    rate = np.array([y[(m >= a) & (m < a + 1)].mean() if c >= 200 else np.nan for a, c in zip(edges[:-1], cnt)])   # 200건 미만 구간은 빼기
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(m[i], y[i] + jit, s=30, c=np.where(y[i] == 1, RED, BLUE), alpha=0.3, edgecolors="none")
    ax.plot(mid, rate, "o", color=GRAY, ms=6, label="실제 취소 비율 (1개월 구간, 200건 이상)")
    ax.plot(xs, sig(w[0] + w[1] * xs), color=BLUE, lw=3, label="로지스틱회귀")
    b = -w[0] / w[1]; ax.axvline(b, color=GRAY, ls=":"); ax.text(b + 0.3, 0.08, f"50%: {b:.1f}개월", color=GRAY, fontsize=13)
    ax.annotate(f"12개월 → {sig(w[0] + 12 * w[1]):.0%}", (12, sig(w[0] + 12 * w[1])), (14.5, 0.45), fontsize=15, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("몇 개월 전에 예약했나"); ax.set_yticks([0, 0.5, 1], ["유지 0", "0.5", "취소 1"]); ax.set_xlim(0, 24); ax.set_ylim(-0.08, 1.08)
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "hotel_fit.png"); plt.close(fig)


def hotel_2d(m, r, y, w, n_show=400, seed=1):
    """(개월, 특별 요청 수) 평면의 표본 점과 경계 직선 w0 + w1 m + w2 r = 0."""
    rng = np.random.default_rng(seed); i = rng.choice(len(m), n_show, replace=False)
    jr = r[i] + rng.uniform(-0.25, 0.25, n_show)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(m[i], jr, s=26, c=np.where(y[i] == 1, RED, BLUE), alpha=0.45, edgecolors="none")
    rs = np.linspace(-0.4, 3.4, 50); ax.plot(-(w[0] + w[2] * rs) / w[1], rs, color=GOLD, lw=3)
    ax.text(16.5, 0.35, "취소 쪽", color=RED, fontsize=15, fontweight="bold"); ax.text(1.0, 2.9, "유지 쪽", color=BLUE, fontsize=15, fontweight="bold")
    ax.set_xlabel("몇 개월 전에 예약했나"); ax.set_ylabel("특별 요청 수"); ax.set_xlim(0, 24); ax.set_ylim(-0.5, 3.5); ax.set_yticks([0, 1, 2, 3])
    fig.tight_layout(); fig.savefig(O + "hotel_2d.png"); plt.close(fig)


def hotel_threshold(m, y, w):
    """임계값에 따른 정밀도 · 재현율 (취소 = 양성)."""
    p = sig(w[0] + w[1] * m); ts = np.linspace(0.2, 0.85, 66); P, R = [], []
    for t in ts:
        pred = p >= t; tp = (pred & (y == 1)).sum()
        P.append(tp / max(pred.sum(), 1)); R.append(tp / (y == 1).sum())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ts, P, color=BLUE, lw=3, label="정밀도: 취소라 한 것 중 진짜 취소"); ax.plot(ts, R, color=RED, lw=3, label="재현율: 진짜 취소 중 찾은 비율")
    ax.axvline(0.5, color=GRAY, ls=":"); ax.set_xlabel("임계값 (이 확률 이상이면 취소로 판정)"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=12)
    fig.tight_layout(); fig.savefig(O + "hotel_threshold.png"); plt.close(fig)
    i = np.argmin(abs(ts - 0.5)); return P[i], R[i]


m, r = d["lead_time"].values / 30, d["total_of_special_requests"].values
y = d["is_canceled"].values.astype(float)
w1 = fit_logistic(np.c_[np.ones_like(m), m / 12], y); w1 = np.array([w1[0], w1[1] / 12])
keep = r <= 3                                            # 요청 4 · 5개는 380건뿐이라 그림 · 학습에서 뺀다
w2 = fit_logistic(np.c_[np.ones(keep.sum()), m[keep] / 12, r[keep] / 2], y[keep]); w2 = np.array([w2[0], w2[1] / 12, w2[2] / 2])
hotel_fit(m, y, w1); hotel_2d(m[keep], r[keep], y[keep], w2); pr = hotel_threshold(m, y, w1)
print("1D w", w1.round(3), "50% at", round(-w1[0] / w1[1], 2), "months; p(0)", round(sig(w1[0]), 3), "p(12)", round(sig(w1[0] + 12 * w1[1]), 3),
      "actual 12m", round(y[(m >= 11.5) & (m < 12.5)].mean(), 3), "odds/month", round(np.exp(w1[1]), 3))
print("2D w", w2.round(3), "boundary r=0", round(-w2[0] / w2[1], 1), "r=2", round(-(w2[0] + 2 * w2[2]) / w2[1], 1), "odds/request", round(np.exp(w2[2]), 2))
print("threshold 0.5: precision", round(pr[0], 3), "recall", round(pr[1], 3), "canceled", int(y.sum()), "/", len(y))


def hotel_rate(t, y):
    """0/1 점 대신 '구간마다 몇 % 가 취소됐나' 를 세면: 비율이 0~1 안에서 오르다 눕는다. 직선과 비교."""
    edges = [0, 1, 8, 31, 61, 91, 121, 181, 241, 301, 366, 451, 541, 740]       # 일
    labels = ["당일", "1주", "1달", "2달", "3달", "4달", "6달", "8달", "10달", "1년", "15달", "18달", "그 이상"]
    rate, cnt, mid = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (t >= a) & (t < b); rate.append(y[k].mean()); cnt.append(k.sum()); mid.append((a + b - 1) / 2)
    k = np.polyfit(t, y, 1); xs = np.linspace(0, 740, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(1, 1.3, color=RED, alpha=0.08)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=2.5, ls="--", label="직선 (앞 쪽)")
    ax.plot(mid, rate, "o-", color=RED, lw=2.5, ms=8, label="구간마다 실제 취소 비율")
    for x, r_, lab in zip(mid, rate, labels):
        if lab in ("당일", "1달", "3달", "6달", "1년", "18달"):
            ax.annotate(f"{lab}\n{r_:.0%}", (x, r_), (0, -34 if lab != "당일" else 14), textcoords="offset points", ha="center", fontsize=12, color=RED)
    ax.set_xlabel("며칠 전에 예약했나 (일)"); ax.set_ylabel("취소된 비율"); ax.set_xlim(-10, 740); ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"]); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "hotel_rate.png"); plt.close(fig)
    return dict(zip(labels, [(round(r_, 3), int(c)) for r_, c in zip(rate, cnt)]))


print("rates", hotel_rate(d["lead_time"].values, d["is_canceled"].values))
