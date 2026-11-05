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
gx, gy = np.meshgrid(np.linspace(-4.5, 1.5, 300), np.linspace(-1.8, 1.8, 200)); Z = (gx ** 2 + 50 * gy ** 2) / 2


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def valley(ax, path, c, title):
    ax.contour(gx, gy, Z, levels=np.geomspace(0.05, 80, 12), colors=GRAY, linewidths=0.8)
    p = path[:STEPS[title[1]] + 1] if isinstance(title, tuple) else path
    ax.plot(p[:, 0], p[:, 1], "o-", color=c, lw=1.6, ms=3); ax.scatter(0, 0, marker="*", s=250, color=GOLD, zorder=4, edgecolor="black")
    ax.set_xlim(-4.5, 1.5); ax.set_ylim(-1.8, 1.8); ax.set_xticks([]); ax.set_yticks([])


for name, path, c, key, t in (("gd", GD, RED, "gd", "경사하강법"), ("mom", MOM, BLUE, "mom", "모멘텀"), ("adam", ADAM, GREEN, "adam", "Adam")):
    fig, ax = plt.subplots(figsize=(7, 4.2)); valley(ax, path[:STEPS[key] + 1], c, t)
    ax.set_title(f"{t}: {STEPS[key]}걸음 (학습률 {LR[key]})", fontsize=16); save(fig, f"ma_{name}.png")

fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, (path, c, key, t) in zip(axs, ((GD, RED, "gd", "경사하강법"), (MOM, BLUE, "mom", "모멘텀"), (ADAM, GREEN, "adam", "Adam"))):
    valley(ax, path[:STEPS[key] + 1], c, t); ax.set_title(f"{t}: {STEPS[key]}걸음", fontsize=15)
save(fig, "ma_all.png")

fig, ax = plt.subplots(figsize=(8, 4.2))
for k, c, t in (("sgd", RED, "SGD (lr 0.05)"), ("momentum", BLUE, "모멘텀 (lr 0.05, β 0.9)"), ("adam", GREEN, "Adam (기본값)")):
    ax.semilogy(np.arange(1, 31), DIG[k], color=c, lw=2.5, label=t)
ax.set_xlabel("에폭"); ax.set_ylabel("학습 손실 (로그)"); ax.legend(frameon=False)
save(fig, "ma_digits.png")
print("ok")
