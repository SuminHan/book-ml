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
CM = plt.get_cmap("tab10")


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def digit_map(ax, Z, y=Y, title=None, s=4):
    ax.scatter(*Z.T, c=CM(y), s=s)
    for k in range(10):
        m = np.median(Z[y == k], 0); ax.text(*m, str(k), fontsize=15, weight="bold", ha="center", va="center",
                                             bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=CM(k), lw=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=15)


# 1. PCA vs t-SNE
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
digit_map(axs[0], PCA2, title=f"PCA 2D  (5-NN {ACC['pca']:.0f}%)")
digit_map(axs[1], TS30, title=f"t-SNE  (5-NN {ACC['ts30']:.0f}%)")
save(fig, "ts_maps.png")

# 2. 정규분포 vs t 분포
fig, ax = plt.subplots(figsize=(7, 3.8))
d = np.linspace(0, 5, 300)
ax.plot(d, np.exp(-d ** 2 / 2), color=BLUE, lw=2.5, label="Gaussian (high-dim)")
ax.plot(d, 1 / (1 + d ** 2), color=GOLD, lw=2.5, label="Student t (map)")
for x in (1, 2, 3):
    ax.plot([x, x], [np.exp(-x ** 2 / 2), 1 / (1 + x ** 2)], color=GRAY, ls=":")
ax.set_xlabel("distance"); ax.set_ylabel("similarity"); ax.legend(frameon=False)
save(fig, "ts_kernel.png")

# 3. perplexity
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, Z, p, k in zip(axs, (TS5, TS30, TS100), (5, 30, 100), ("ts5", "ts30", "ts100")):
    digit_map(ax, Z, title=f"perplexity {p}  ({ACC[k]:.1f}%)", s=3)
save(fig, "ts_perp.png")

# 4. 크기가 다른 두 무리
rng = np.random.default_rng(0)
A = rng.normal(0, 1, (100, 10)); B = rng.normal(0, 10, (100, 10)); B[:, 0] += 50
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
axs[0].scatter(*A[:, :2].T, s=12, color=BLUE); axs[0].scatter(*B[:, :2].T, s=12, color=GOLD)
axs[0].set_aspect("equal"); axs[0].set_title(f"원래 (앞 2칸): 퍼짐 {ORIG_RATIO:.1f}배", fontsize=15)
axs[1].scatter(*TWO[TWO_LAB == 0].T, s=12, color=BLUE); axs[1].scatter(*TWO[TWO_LAB == 1].T, s=12, color=GOLD)
axs[1].set_aspect("equal"); axs[1].set_title(f"t-SNE 지도: {MAP_RATIO:.1f}배", fontsize=15)
for ax in axs: ax.set_xticks([]); ax.set_yticks([])
save(fig, "ts_two.png")

# 5. 시드
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, Z, t in zip(axs, (TS30, SEED1, SEED2), ("init PCA", "random seed 1", "random seed 2")):
    digit_map(ax, Z, title=t, s=3)
save(fig, "ts_seeds.png")
# 6. 숫자 한 장 = 64개 숫자
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.imshow(IMG, cmap="gray_r")
for r in range(8):
    for c in range(8):
        ax.text(c, r, f"{int(IMG[r, c])}", ha="center", va="center", fontsize=11, color="white" if IMG[r, c] > 8 else "k")
ax.set_xticks([]); ax.set_yticks([])
save(fig, "ts_digit.png")
print("figs ok")
