"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (lessons/ml1/linear_regression 과 같은 데이터)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"
x, y = study()
xs = np.linspace(0, 10.5, 100)


def base(ax, x, y):
    ax.scatter(x, y, s=55, color=BLUE, zorder=3)
    ax.set_xlabel("공부 시간 (시간)"); ax.set_ylabel("점수"); ax.set_xlim(0, 10.5); ax.set_ylim(30, 105)


# 1. 질문
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, x, y)
ax.axvline(7, color=RED, ls="--", lw=2); ax.text(7.15, 35, "7시간이면?", color=RED, fontsize=17, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "lr_hook.png"); plt.close(fig)

# 2. 직선은 무수히 많다
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, x, y)
for (a, b), c in zip(((60, 2), (30, 7), (45, 4), (W0, W1)), (GRAY, GRAY, GRAY, RED)):
    ax.plot(xs, a + b * xs, color=c, lw=2.5 if c == RED else 1.8, alpha=1 if c == RED else 0.7)
ax.text(0.3, 98, "어느 직선이 가장 좋을까?", fontsize=16, color=RED)
fig.tight_layout(); fig.savefig(O + "lr_lines.png"); plt.close(fig)

# 3. 오차를 제곱해서 평균
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, x, y)
a, b = 45, 4
ax.plot(xs, a + b * xs, color=GOLD, lw=2.5)
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, a + b * xi], color=RED, lw=1.5)
ax.text(0.3, 98, "빨간 선 = 오차 $h(x) - y$", fontsize=16, color=RED)
fig.tight_layout(); fig.savefig(O + "lr_residuals.png"); plt.close(fig)

# 4. J 는 그릇 모양 (등고선) + 경사하강 경로
def J(w0, w1, x, y): return np.mean((w0 + w1 * x[:, None, None] - y[:, None, None]) ** 2, axis=0) / 2
W0g, W1g = np.meshgrid(np.linspace(0, 80, 200), np.linspace(-2, 12, 200))
Z = J(W0g, W1g, x, y)
def gd(x, y, alpha, steps, w=(5.0, 0.0)):
    w0, w1 = w; path = [(w0, w1)]
    for _ in range(steps):
        r = w0 + w1 * x - y; w0, w1 = w0 - alpha * r.mean(), w1 - alpha * (r * x).mean(); path.append((w0, w1))
    return np.array(path)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.contour(W0g, W1g, Z, levels=np.geomspace(Z.min() * 1.05, Z.max(), 18), cmap="Blues_r", linewidths=1.2)
p = gd(x, y, 0.02, 4000)
ax.plot(p[:, 0], p[:, 1], color=RED, lw=2); ax.scatter(*p[0], color=RED, s=60); ax.scatter(W0, W1, marker="*", s=320, color=GOLD, zorder=4)
ax.text(W0 + 3, W1 - 1.6, "바닥", fontsize=15, color=GOLD, fontweight="bold")
ax.set_xlabel("$w_0$ (절편)"); ax.set_ylabel("$w_1$ (기울기)")
fig.tight_layout(); fig.savefig(O + "lr_bowl.png"); plt.close(fig)

# 5. 학습률: 적당 vs 너무 큼 (w1 만 움직이는 1차원 단면)
def J1(w1, x, y): return np.mean((W0 + w1 * x - y) ** 2) / 2
def path1(x, y, alpha, n=6, w1=0.0):
    out = [w1]
    for _ in range(n):
        w1 = w1 - alpha * np.mean((W0 + w1 * x - y) * x); out.append(w1)
    return np.array(out)
grid = np.linspace(-14, 24, 300)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, alpha, title, c in ((axs[0], 0.01, "학습률 적당", BLUE), (axs[1], 0.06, "학습률 너무 큼 → 발산", RED)):
    ax.plot(grid, [J1(g, x, y) for g in grid], color=GRAY, lw=2)
    pth = path1(x, y, alpha, 6 if alpha < 0.03 else 4)
    ax.plot(pth, [J1(g, x, y) for g in pth], "o-", color=c, lw=2)
    ax.set_title(title, fontsize=16, color=c); ax.set_xlabel("$w_1$"); ax.set_yticks([])
axs[0].set_ylabel("비용 $J$")
fig.tight_layout(); fig.savefig(O + "lr_alpha.png"); plt.close(fig)

# 6. 결과: 7시간 → 75.4점
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, x, y)
ax.plot(xs, W0 + W1 * xs, color=RED, lw=3)
ax.scatter([7], [PRED7], s=200, color=GOLD, zorder=4, edgecolor="black")
ax.annotate(f"7시간 → {PRED7}점", (7, PRED7), (3.2, 92), fontsize=17, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
fig.tight_layout(); fig.savefig(O + "lr_fit.png"); plt.close(fig)

# 7. 곡선: 제동거리 직선 vs x²
v, d = braking(); vs = np.linspace(15, 125, 100)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, k, c in ((axs[0], 1, GOLD), (axs[1], 2, RED)):
    cf = np.polyfit(v, d, k)
    ax.scatter(v, d, s=40, color=BLUE); ax.plot(vs, np.polyval(cf, vs), color=c, lw=3)
    ax.set_title(("직선" if k == 1 else "$x^2$ 특징 추가") + f"   RMSE {BRAKE_RMSE[k]} m", fontsize=16, color=c)
    ax.set_xlabel("속도 (km/h)")
axs[0].set_ylabel("제동거리 (m)")
fig.tight_layout(); fig.savefig(O + "lr_brake.png"); plt.close(fig)

# 8. 로그 변환: 광고비
xa, ya = adspend()
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, f, lab, key in ((axs[0], xa, "광고비 $x$", "x"), (axs[1], np.log(xa), "$\\log x$", "log")):
    cf = np.polyfit(f, ya, 1); fs = np.linspace(f.min(), f.max(), 100)
    ax.scatter(f, ya, s=40, color=BLUE); ax.plot(fs, np.polyval(cf, fs), color=RED if key == "log" else GOLD, lw=3)
    ax.set_title(f"$x$ 축 = {lab}   $R^2$ = {AD_R2[key]:.2f}", fontsize=16); ax.set_xlabel(lab)
axs[0].set_ylabel("방문자 (명)")
fig.tight_layout(); fig.savefig(O + "lr_log.png"); plt.close(fig)

# 9. 과적합: 차수별 학습/시험 RMSE
fig, ax = plt.subplots(figsize=(8.5, 4.6))
tr = [OVERFIT[k][0] for k in DEGREES]; te = [OVERFIT[k][1] for k in DEGREES]
ax.plot(DEGREES, tr, "o-", color=BLUE, lw=2.5, label="학습 RMSE"); ax.plot(DEGREES, te, "s-", color=RED, lw=2.5, label="시험 RMSE")
ax.set_yscale("log"); ax.set_xlabel("다항식 차수"); ax.set_ylabel("RMSE (m, 로그 눈금)"); ax.legend(frameon=False)
ax.annotate("2차: 시험 4.5 m (최소)", (2, 4.5), (5.2, 4.2), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("15차: 시험 1,429 m", (15, 1428.9), (9.3, 600), fontsize=14, color=RED)
ax.set_xticks(DEGREES)
fig.tight_layout(); fig.savefig(O + "lr_degree.png"); plt.close(fig)
print("ok")
