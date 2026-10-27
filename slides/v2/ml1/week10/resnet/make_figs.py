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


# 1. 논문 오답률
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(2); w = 0.35
for k, (kind, c, t) in enumerate((("plain", RED, "일반 신경망"), ("res", GREEN, "ResNet"))):
    v = [ERR[(kind, 18)], ERR[(kind, 34)]]; b = ax.bar(x + (k - 0.5) * w, v, w, color=c, label=t)
    for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val + 0.1, f"{val:.1f}", ha="center", fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["18층", "34층"]); ax.set_ylim(24, 29.5); ax.set_ylabel("ImageNet 오답률 (%)"); ax.legend(frameon=False, loc="upper left")
save(fig, "rn_paper.png")

# 2. 잔차 블록 그림
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, res, t in ((axs[0], False, "일반 블록"), (axs[1], True, "잔차 블록")):
    ax.axis("off"); ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_title(t, fontsize=16)
    for y, txt in ((4.6, "층 (가중치)"), (3.1, "ReLU"), (1.6, "층 (가중치)")):
        ax.add_patch(FancyBboxPatch((1.5, y - 0.35), 2.4, 0.7, boxstyle="round,pad=0.05", fc="#eef", ec=BLUE)); ax.text(2.7, y, txt, ha="center", va="center", fontsize=14)
    ax.annotate("", (2.7, 4.95), (2.7, 5.8), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(2.9, 5.6, "x", fontsize=16)
    for a, b in ((4.25, 3.45), (2.75, 1.95)): ax.annotate("", (2.7, b), (2.7, a), arrowprops=dict(arrowstyle="->", lw=2))
    if res:
        ax.plot([2.7, 5.0, 5.0], [5.4, 5.4, 0.6], color=GREEN, lw=3); ax.annotate("", (3.05, 0.6), (5.0, 0.6), arrowprops=dict(arrowstyle="->", lw=3, color=GREEN))
        ax.annotate("", (2.7, 0.85), (2.7, 1.25), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(5.1, 3.1, "지름길\n(그대로)", color=GREEN, fontsize=14); ax.text(2.7, 0.6, "+", ha="center", va="center", fontsize=26, color=GREEN, fontweight="bold")
        ax.text(2.7, 0.0, "F(x) + x", ha="center", fontsize=16)
    else:
        ax.annotate("", (2.7, 0.5), (2.7, 1.25), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(2.7, 0.0, "F(x)", ha="center", fontsize=16)
save(fig, "rn_block.png")

# 3. 우리 실험: 깊이별 학습 정확도
fig, ax = plt.subplots(figsize=(8, 4.2))
L = [LAYERS[b] for b in BLOCKS]
ax.plot(L, [TRAIN_END[(b, False)] for b in BLOCKS], "o-", color=RED, lw=3, label="일반 신경망")
ax.plot(L, [TRAIN_END[(b, True)] for b in BLOCKS], "s-", color=GREEN, lw=3, label="잔차 연결")
for b in BLOCKS:
    ax.text(LAYERS[b], TRAIN_END[(b, False)] - 8, f"{TRAIN_END[(b, False)]}%", ha="center", color=RED, fontsize=13)
    ax.text(LAYERS[b], TRAIN_END[(b, True)] + 3, f"{TRAIN_END[(b, True)]}%", ha="center", color=GREEN, fontsize=13)
ax.set_xticks(L); ax.set_xlabel("층 수"); ax.set_ylabel("학습 정확도 (%, 5 에폭)"); ax.set_ylim(0, 108); ax.legend(frameon=False, loc="center left")
save(fig, "rn_ours.png")
print("ok")
