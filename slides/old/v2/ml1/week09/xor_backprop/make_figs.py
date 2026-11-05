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


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def pts(ax, t):
    for (a, b), v in zip(X, t):
        ax.scatter(a, b, s=320, color=RED if v else BLUE, marker="o" if v else "s", zorder=3, edgecolor="black")
        ax.text(a + 0.09, b + 0.09, str(int(v)), fontsize=16, fontweight="bold")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_aspect("equal"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


# 1. AND / OR / XOR
xs = np.linspace(-0.4, 1.4, 2)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (name, t, line) in zip(axs, (("AND", [0, 0, 0, 1], 1.5), ("OR", [0, 1, 1, 1], 0.5), ("XOR", [0, 1, 1, 0], None))):
    pts(ax, t); ax.set_title(name, fontsize=18, fontweight="bold")
    if line is not None: ax.plot(xs, line - xs, color=GREEN, lw=3)
    else: ax.text(0.5, 0.5, "?", fontsize=40, color=RED, ha="center", va="center", fontweight="bold")
save(fig, "xo_gates.png")

# 2. 직선 두 개
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
pts(axs[0], T); axs[0].plot(xs, 0.5 - xs, color=GREEN, lw=3, label="$h_1$: OR ($x_1 + x_2 > 0.5$)"); axs[0].plot(xs, 1.5 - xs, color=GOLD, lw=3, label="$h_2$: NAND ($x_1 + x_2 < 1.5$)")
axs[0].fill_between(xs, 0.5 - xs, 1.5 - xs, color=RED, alpha=0.1); axs[0].legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
axs[0].set_title("숨은 층: 직선 두 개", fontsize=15)
H = (X @ HAND["W1"] + HAND["b1"] > 0).astype(float)
for (a, b), v, (x1, x2) in zip(H, T, X):
    axs[1].scatter(a, b, s=320, color=RED if v else BLUE, marker="o" if v else "s", edgecolor="black", zorder=3)
axs[1].text(0.62, 1.15, "입력 (0,1)·(1,0)", fontsize=12); axs[1].text(0.95, -0.25, "입력 (1,1)", fontsize=12); axs[1].text(-0.35, 1.15, "입력 (0,0)", fontsize=12)
axs[1].plot(xs, 1.5 - xs, color=BLUE, lw=3); axs[1].set_xlim(-0.4, 1.4); axs[1].set_ylim(-0.4, 1.4); axs[1].set_aspect("equal")
axs[1].set_xlabel("$h_1$"); axs[1].set_ylabel("$h_2$"); axs[1].set_title("($h_1$, $h_2$) 공간에서는 직선 하나", fontsize=15)
save(fig, "xo_twolines.png")

# 3. 학습 과정 경계
out, losses = train(X, T)
gx, gy = np.meshgrid(np.linspace(-0.4, 1.4, 200), np.linspace(-0.4, 1.4, 200))
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
for ax, e in zip(axs, (0, 150, 400, 4000)):
    Z = predict(out[e], np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[0, 0.5, 1], colors=[BLUE, RED], alpha=0.15); ax.contour(gx, gy, Z, levels=[0.5], colors="black")
    pts(ax, T); ax.set_title(f"{e}번: 정확도 {out[e][4]:.0%}", fontsize=14); ax.set_xlabel(""); ax.set_ylabel("")
save(fig, "xo_train.png")

# 4. 손실 곡선
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(losses, color=BLUE, lw=2.5); ax.set_yscale("log"); ax.set_xlabel("학습 횟수"); ax.set_ylabel("손실 L (로그 눈금)")
save(fig, "xo_loss.png")
print("ok")
