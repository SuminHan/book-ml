"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"
tt = np.linspace(0, 24, 400)
M = {k: fit(k) for k in ("none", "ridge", "lasso")}
COL = {"none": RED, "ridge": BLUE, "lasso": GREEN}


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def base(ax):
    ax.scatter(HOURS, TEMPS, s=70, color="black", zorder=4, label="잰 기온 12개")
    ax.plot(tt, truth(tt), ls="--", color=GRAY, lw=2, label="실제 (예시의 참 곡선)")
    ax.set_xlabel("시각 (시)"); ax.set_ylabel("기온 (°C)"); ax.set_xlim(0, 24); ax.set_ylim(5, 25); ax.set_xticks([0, 6, 12, 18, 24])


def curve(ax, k, label):
    ax.plot(tt, M[k].predict((tt / 24)[:, None]), color=COL[k], lw=3, label=label)


# 1. 질문
fig, ax = plt.subplots(figsize=(8.5, 4.5)); base(ax); curve(ax, "none", f"9차 곡선 (가장 큰 가중치 {MAXW['none']:,})")
ax.legend(frameon=False, loc="upper right", fontsize=12)
save(fig, "rg_hook.png")

# 2. 가중치 크기
fig, ax = plt.subplots(figsize=(10, 4.2))
idx = np.arange(1, DEG + 1)
for j, (k, name) in enumerate((("none", "정규화 없음"), ("ridge", "L2 릿지"), ("lasso", "L1 라쏘"))):
    w = np.abs(M[k][-1].coef_)
    ax.bar(idx + (j - 1) * 0.27, np.maximum(w, 1e-2), width=0.27, color=COL[k], label=name)
ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels([f"$x^{i}$" for i in idx]); ax.set_ylabel("|가중치| (로그 눈금)")
ax.legend(frameon=False, ncol=3, loc="upper center"); ax.set_ylim(1e-2, 3e5)
save(fig, "rg_weights.png")

# 3. 릿지
fig, ax = plt.subplots(figsize=(8.5, 4.5)); base(ax); curve(ax, "none", f"정규화 없음 (오차 {ERR['none']}도)"); curve(ax, "ridge", f"L2 릿지 (오차 {ERR['ridge']}도)")
ax.legend(frameon=False, loc="upper right", fontsize=12)
save(fig, "rg_ridge.png")

# 4. 라쏘: 계수 + 곡선
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
w = M["lasso"][-1].coef_
axs[0].bar(idx, w, color=[GREEN if abs(v) > 1e-6 else "white" for v in w], edgecolor=GREEN, lw=1.5)
for i, v in zip(idx, w):
    if abs(v) < 1e-6: axs[0].text(i, 0.5, "0", ha="center", fontsize=15, color=RED, fontweight="bold")
axs[0].axhline(0, color="black", lw=1); axs[0].set_xticks(idx); axs[0].set_xticklabels([f"$x^{i}$" for i in idx]); axs[0].set_title(f"라쏘 가중치: 9개 중 {LASSO_ZEROS}개가 0", fontsize=15)
base(axs[1]); curve(axs[1], "lasso", f"L1 라쏘 (오차 {ERR['lasso']}도)"); axs[1].legend(frameon=False, fontsize=11, loc="upper right")
save(fig, "rg_lasso.png")

# 5. 기하: 원 vs 마름모
w1, w2 = np.meshgrid(np.linspace(-1.6, 3.2, 300), np.linspace(-1.6, 3.2, 300))
c = np.array([2.2, 1.1]); A = np.array([[1.0, 0.6], [0.6, 1.2]])
J = A[0, 0] * (w1 - c[0]) ** 2 + 2 * A[0, 1] * (w1 - c[0]) * (w2 - c[1]) + A[1, 1] * (w2 - c[1]) ** 2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, kind in zip(axs, ("L2", "L1")):
    ax.contour(w1, w2, J, levels=[0.1, 0.4, 0.9, 1.6, 2.5, 3.6], colors=GRAY, linewidths=1)
    ax.scatter(*c, marker="*", s=250, color="black", zorder=4)
    if kind == "L2":
        th = np.linspace(0, 2 * np.pi, 200); ax.fill(np.cos(th), np.sin(th), color=BLUE, alpha=0.25)
        # 원 위에서 J 가 가장 작은 점
        pts = np.c_[np.cos(th), np.sin(th)]; d = pts - c; v = np.einsum("ij,jk,ik->i", d, A, d); p = pts[v.argmin()]
        ax.scatter(*p, s=120, color=BLUE, zorder=5); ax.set_title("L2: 원 $w_1^2 + w_2^2 \\leq t$", fontsize=15)
    else:
        ax.fill([1, 0, -1, 0], [0, 1, 0, -1], color=GREEN, alpha=0.3)
        ax.scatter(1, 0, s=120, color=GREEN, zorder=5); ax.annotate("꼭짓점: $w_2 = 0$", (1, 0), (1.3, -1.2), fontsize=14, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
        ax.set_title("L1: 마름모 $|w_1| + |w_2| \\leq t$", fontsize=15)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8); ax.set_aspect("equal"); ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
save(fig, "rg_geo.png")

# 6. 계수 경로: 진짜 특징 x1 하나 + 잡음 특징 x2 하나 (notebooks/ml1/chapter06_1 §3 과 같은 실험, 30개 점)
from sklearn.linear_model import Ridge as _R, Lasso as _L
rng = np.random.default_rng(0)
x1 = rng.uniform(-1, 1, 30); x2 = rng.normal(0, 1, 30); yy = 3 * x1 + rng.normal(0, 1, 30)
Xp = np.c_[x1, x2]; lams = np.logspace(-2, np.log10(2.5), 60)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, Mdl, t in ((axs[0], _R, "L2 릿지: 줄어들지만 0 은 아님"), (axs[1], _L, "L1 라쏘: 정확히 0 에 도착")):
    W = np.array([Mdl(alpha=l * (30 if Mdl is _R else 1)).fit(Xp, yy).coef_ for l in lams])
    ax.plot(lams, W[:, 0], color=BLUE, lw=3, label="진짜 특징 $x_1$"); ax.plot(lams, W[:, 1], color=RED, lw=3, label="잡음 특징 $x_2$")
    ax.axhline(0, color="black", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("λ (로그 눈금)"); ax.set_title(t, fontsize=15)
axs[0].set_ylabel("가중치"); axs[0].legend(frameon=False)
save(fig, "rg_path.png")

# 7. λ 에 따른 오차
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.plot(LAMS, SWEEP, "o-", color=BLUE, lw=2.5); ax.set_xscale("log"); ax.set_xlabel("λ (로그 눈금)"); ax.set_ylabel("실제 기온과 오차 (도)")
ax.annotate("너무 약함: 출렁", (1e-6, SWEEP[0]), (3e-6, 2.6), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("너무 셈: 납작", (100, SWEEP[-1]), (2, 3.8), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("바닥 0.99", (1e-3, 0.99), (2e-3, 0.6), fontsize=14, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
ax.set_ylim(0.4, 4.2)
save(fig, "rg_sweep.png")
print("ok")
