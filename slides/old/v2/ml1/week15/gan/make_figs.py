"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 고리: 단계별
fig, axs = plt.subplots(1, 5, figsize=(13, 3.0))
for ax, s, g in zip(axs, RING_SNAP, Z["ring_gen"]):
    ax.scatter(*Z["ring_real"].T, s=6, color=GRAY, alpha=0.5); ax.scatter(*g.T, s=6, color=RED)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{s}걸음\nD: 진짜 {RING_D[s][0]:.2f} · 가짜 {RING_D[s][1]:.2f}", fontsize=12)
save(fig, "gan_ring.png")

# 2. 봉우리 8개: 시드 0 vs 시드 3 (2,000 · 3,500 · 5,000걸음)
fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.4))
for r, seed in enumerate((0, 3)):
    for c, step in enumerate((2000, 3500, 5000)):
        ax = axs[r, c]; k = EIGHT_STEPS.index(step); g = Z["eight_snaps"][seed, k]
        ax.scatter(*Z["eight_real"].T, s=5, color=GRAY, alpha=0.4); ax.scatter(*g.T, s=6, color=RED if seed == 3 else BLUE)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"시드 {seed}, {step}걸음: {EIGHT[seed][k]}개", fontsize=13)
save(fig, "gan_eight.png")

# 3. 손글씨 숫자: 단계별 16장
fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))
for ax, s, g in zip(axs, Z["digit_snap"], Z["digit_gen"]):
    big = np.ones((4 * 9 - 1, 4 * 9 - 1))
    for k in range(16):
        r, c = divmod(k, 4); big[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = 1 - g[k].reshape(8, 8)
    kinds = sum(v > 0 for v in DIG[int(s)][0])
    ax.imshow(big, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(f"{int(s):,}걸음 ({kinds}종류)" if s else "0걸음 (학습 전)", fontsize=14)
save(fig, "gan_digits.png")

# 4. 위조범 vs 경찰 그림
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")
def box(x, y, t, c, w=2.2, h=0.9, fc="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=c, lw=2)); ax.text(x, y, t, ha="center", va="center", color=c, fontsize=14)
box(1.2, 2.2, "잡음 z", GRAY); box(3.9, 2.2, "생성자 G\n(위조범)", RED, fc="#fbeaea"); box(3.9, 0.7, "진짜 데이터", BLUE, fc="#e8e9f6")
box(6.8, 1.45, "판별자 D\n(경찰)", GOLD, fc="#f3f0e8"); box(9.0, 1.45, "진짜일 확률", GOLD, w=1.9)
for (a, b), (c, d) in [((2.3, 2.2), (2.8, 2.2)), ((5.0, 2.2), (5.7, 1.6)), ((5.0, 0.7), (5.7, 1.3)), ((7.9, 1.45), (8.05, 1.45))]:
    ax.annotate("", (c, d), (a, b), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.8))
ax.annotate("", (3.9, 2.68), (6.8, 1.95), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5, ls="--", connectionstyle="arc3,rad=0.35"))
ax.text(5.6, 3.25, "들킨 만큼 고친다 (기울기)", color=RED, fontsize=12, ha="center")
save(fig, "gan_scheme.png")
print("figs ok")
