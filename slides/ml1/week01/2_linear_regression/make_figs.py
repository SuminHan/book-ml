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


# 0. 학습의 뼈대: 데이터 -> 모델 -> 손실 -> 최적화, 최적화가 모델의 파라미터를 고치는 고리 (Goodfellow et al. 2016, 5.10)
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(10.5, 3.9)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.9)
boxes = [("① 데이터", "(공부 시간, 점수)\n30쌍", BLUE), ("② 모델", "직선\n$w_0 + w_1 x$", GOLD),
         ("③ 손실함수", "오차 제곱 평균\n$J(w_0, w_1)$", RED), ("④ 최적화", "경사하강법\n$w$ 를 고친다", GREEN)]
bw, gap = 2.25, 0.5
for i, (title, sub, c) in enumerate(boxes):
    x0 = i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x0, 1.2), bw, 2.0, boxstyle="round,pad=0.03", fc=c, ec="none"))
    ax.text(x0 + bw / 2, 2.75, title, ha="center", va="center", color="white", fontsize=18, fontweight="bold")
    ax.text(x0 + bw / 2, 1.85, sub, ha="center", va="center", color="white", fontsize=14, linespacing=1.4)
    if i < 3:
        ax.annotate("", (x0 + bw + gap - 0.05, 2.2), (x0 + bw + 0.05, 2.2), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
x_mod, x_opt = 1 * (bw + gap) + bw / 2, 3 * (bw + gap) + bw / 2
ax.plot([x_opt, x_opt, x_mod], [1.15, 0.8, 0.8], color=GRAY, lw=2)
ax.annotate("", (x_mod, 1.15), (x_mod, 0.8), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
ax.text((x_mod + x_opt) / 2, 0.4, "손실이 줄어들 때까지 반복", ha="center", fontsize=14, color=GRAY)
fig.savefig(O + "lr_recipe.png", bbox_inches="tight"); plt.close(fig)

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
def gd(x, y, lr, steps, w=(5.0, 0.0)):
    w0, w1 = w; path = [(w0, w1)]
    for _ in range(steps):
        r = w0 + w1 * x - y; w0, w1 = w0 - lr * r.mean(), w1 - lr * (r * x).mean(); path.append((w0, w1))
    return np.array(path)
# 가로 · 세로 눈금을 같게 (aspect=1): 그래야 걸음 방향이 등고선에 수직 (가장 가파른 방향) 으로 보인다
Wa, Wb = np.meshgrid(np.linspace(0, 45, 400), np.linspace(-1, 12, 200))
Za = J(Wa, Wb, x, y)
fig, ax = plt.subplots(figsize=(12, 4.0))
ax.contour(Wa, Wb, Za, levels=np.geomspace(Za.min() * 1.02, Za.max(), 22), cmap="Blues_r", linewidths=1.0)
p = gd(x, y, 0.02, 4000)
ax.plot(p[:, 0], p[:, 1], color=RED, lw=1.5, alpha=0.6)
for k in range(3):   # first steps: arrows
    ax.annotate("", p[k + 1], p[k], arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=16))
for k in [0, 1, 2, 3, 100, 1000]:
    ax.scatter(*p[k], color=RED, s=40, zorder=4)
    dx, dy = {0: (0.6, -0.6), 1: (-1.6, 0.0), 2: (-1.6, 0.3), 3: (0.4, 0.6), 100: (0, 0.6), 1000: (-1.5, 0.7)}[k]
    ax.text(p[k][0] + dx, p[k][1] + dy, f"{k}", fontsize=14, color=RED, fontweight="bold")
ax.scatter(W0, W1, marker="*", s=320, color=GOLD, zorder=5)
ax.text(W0 + 1.0, W1 - 1.3, "바닥", fontsize=15, color=GOLD, fontweight="bold")
ax.set_aspect("equal"); ax.set_xlim(0, 45); ax.set_ylim(-1, 12)
ax.set_xlabel("$w_0$ (절편)"); ax.set_ylabel("$w_1$ (기울기)")
fig.tight_layout(); fig.savefig(O + "lr_bowl.png", bbox_inches="tight"); plt.close(fig)

# 4a. 왜 w1 쪽만 가파른가: w0 를 2 올리면 직선이 2 만큼 위로, w1 을 2 올리면 x 배만큼 (x = 10 이면 20)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.0))
for ax, (d0, d1), ttl in ((axs[0], (2, 0), "$w_0$ 를 2 올리면: 모든 학생 +2점"), (axs[1], (0, 2), "$w_1$ 을 2 올리면: 학생마다 $+2x$ 점 (10시간이면 +20)")):
    base(ax, x, y)
    ax.plot(xs, W0 + W1 * xs, color=GRAY, lw=2.5)
    ax.plot(xs, (W0 + d0) + (W1 + d1) * xs, color=RED, lw=2.5)
    for xi in (2, 6, 10):
        ax.annotate("", (xi, (W0 + d0) + (W1 + d1) * xi), (xi, W0 + W1 * xi), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.set_title(ttl, fontsize=16, color=RED); ax.set_ylim(30, 125)
fig.tight_layout(); fig.savefig(O + "lr_nudge.png"); plt.close(fig)

# 4b. 같은 그릇, 학습률 네 가지 (같은 출발점 (5, 0), 처음 300걸음; 눈금은 가로 · 세로 같게)
def steps_to_bottom(x, y, lr, w=(5.0, 0.0), tol=1.01, cap=100000):
    """steps until J is within 1% of the minimum (None if it blows up)."""
    best = J(np.array(W0), np.array(W1), x, y).item(); a, b = w
    for k in range(cap):
        r = a + b * x - y
        if np.mean(r ** 2) / 2 <= best * tol: return k
        if abs(a) > 1e6: return None
        a, b = a - lr * r.mean(), b - lr * (r * x).mean()
fig, axs = plt.subplots(2, 2, figsize=(13, 4.9))
for ax, lr in zip(axs.flat, (0.005, 0.02, 0.05, 0.055)):
    ax.contour(Wa, Wb, Za, levels=np.geomspace(Za.min() * 1.02, Za.max(), 14), cmap="Blues_r", linewidths=0.8)
    p = gd(x, y, lr, 300 if lr < 0.054 else 14)
    ok = np.isfinite(p).all(axis=1) & (np.abs(p[:, 0]) < 1e4)
    p = p[ok]
    ax.plot(p[:, 0], p[:, 1], color=RED, lw=1.3)
    ax.scatter(p[:12, 0], p[:12, 1], color=RED, s=14, zorder=4)
    ax.scatter(W0, W1, marker="*", s=180, color=GOLD, zorder=5)
    n = steps_to_bottom(x, y, lr)
    ax.set_title(f"η = {lr}:  " + (f"바닥까지 {n:,}걸음" if n else "발산 (바닥을 건너뛰며 튕겨 나간다)"),
                 fontsize=14, color=RED if n is None else BLUE)
    ax.set_aspect("equal"); ax.set_xlim(0, 45); ax.set_ylim(-1, 12); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("$w_0$", fontsize=12, labelpad=1); ax.set_ylabel("$w_1$", fontsize=12, labelpad=1)
fig.tight_layout(); fig.savefig(O + "lr_rates.png", bbox_inches="tight"); plt.close(fig)

# 4c. 입력 표준화: 원래 x / 평균 빼기 / 표준화 — 그릇 모양과 경사하강 경로 (출발점 (5, 0), 각자 가장 빠른 학습률 근처)
def bowl_panel(ax, x, y, lr, xlim, ylim, title):
    Wa_, Wb_ = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 200))
    Z_ = J(Wa_, Wb_, x, y)
    ax.contour(Wa_, Wb_, Z_, levels=np.geomspace(Z_.min() * 1.02, Z_.max(), 14), cmap="Blues_r", linewidths=0.8)
    p = gd(x, y, lr, 2000)
    A = np.c_[np.ones_like(x), x]; wb = np.linalg.lstsq(A, y, rcond=None)[0]
    n = steps_to_bottom_w(x, y, lr, wb)
    ax.plot(p[:, 0], p[:, 1], color=RED, lw=1.3); ax.scatter(p[:12, 0], p[:12, 1], color=RED, s=16, zorder=4)
    ax.scatter(*wb, marker="*", s=200, color=GOLD, zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("$w_0$", fontsize=12, labelpad=1); ax.set_ylabel("$w_1$", fontsize=12, labelpad=1)
    ax.set_title(f"{title}\nη = {lr}: 바닥까지 {n:,}걸음", fontsize=19, color=BLUE)
def steps_to_bottom_w(x, y, lr, wb, w=(5.0, 0.0), tol=1.01):
    best = np.mean((wb[0] + wb[1] * x - y) ** 2) / 2; a, b = w
    for k in range(100000):
        r = a + b * x - y
        if np.mean(r ** 2) / 2 <= best * tol: return k
        a, b = a - lr * r.mean(), b - lr * (r * x).mean()
xc = x - x.mean(); xz = xc / x.std()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[45 / 13, 75 / 25, 75 / 25]))
bowl_panel(axs[0], x, y, 0.05, (0, 45), (-1, 12), "원래 x (공부 시간)")
bowl_panel(axs[1], xc, y, 0.2, (0, 75), (-6, 19), "평균 빼기: $x - \\mu$")
bowl_panel(axs[2], xz, y, 1.0, (0, 75), (-6, 19), "z-score: $(x - \\mu)\\,/\\,\\sigma$")
fig.tight_layout(); fig.savefig(O + "lr_scaling.png", bbox_inches="tight"); plt.close(fig)

# 5. 학습률: 적당 vs 너무 큼 (w1 만 움직이는 1차원 단면)
def J1(w1, x, y): return np.mean((W0 + w1 * x - y) ** 2) / 2
def path1(x, y, lr, n=6, w1=0.0):
    out = [w1]
    for _ in range(n):
        w1 = w1 - lr * np.mean((W0 + w1 * x - y) * x); out.append(w1)
    return np.array(out)
grid = np.linspace(-14, 24, 300)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, lr, title, c in ((axs[0], 0.01, "학습률 적당", BLUE), (axs[1], 0.06, "학습률 너무 큼 → 발산", RED)):
    ax.plot(grid, [J1(g, x, y) for g in grid], color=GRAY, lw=2)
    pth = path1(x, y, lr, 6 if lr < 0.03 else 4)
    ax.plot(pth, [J1(g, x, y) for g in pth], "o-", color=c, lw=2)
    ax.set_title(title, fontsize=16, color=c); ax.set_xlabel("$w_1$"); ax.set_yticks([])
axs[0].set_ylabel("비용 $J$")
fig.tight_layout(); fig.savefig(O + "lr_alpha.png"); plt.close(fig)

# 6. 결과: 7시간 → 75.3점
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
