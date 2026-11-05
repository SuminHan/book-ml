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


def box(ax, x, y, text, color, w=1.4, h=0.6, fc="white", fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)


# 1. 창 (앞뒤 2칸)
words = ["귀여운", "작은", "고양이가", "창가에서", "잔다"]
fig, ax = plt.subplots(figsize=(10, 1.9)); ax.set_xlim(-0.9, 8.1); ax.set_ylim(-0.7, 0.9); ax.axis("off")
for k, w in enumerate(words):
    c = RED if k == 2 else BLUE
    box(ax, k * 1.8, 0, w, c, fc="#fbeaea" if k == 2 else "#e8e9f6")
ax.add_patch(FancyBboxPatch((-0.85, -0.5), 8.0, 1.0, boxstyle="round,pad=0.05", fc="none", ec=GOLD, lw=2, ls="--"))
ax.text(3.6, 0.72, "창: 앞뒤 2칸", ha="center", color="#7a6a40", fontsize=14)
save(fig, "w2v_window.png")

# 2. Skip-gram vs CBOW
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, sg in zip(axs, (True, False)):
    ax.set_xlim(-1.2, 5.6); ax.set_ylim(-0.4, 3.0); ax.axis("off")
    box(ax, 2.2, 1.3, "고양이가", RED, fc="#fbeaea")
    for k, w in enumerate(["귀여운", "작은", "창가에서", "잔다"]):
        x = [-0.4, 1.2, 3.2, 4.8][k]; y = 2.55 if sg else 0.05
        box(ax, x, y, w, BLUE, w=1.35, fc="#e8e9f6", fs=13)
        if sg: ax.annotate("", (x, y - 0.32), (2.2, 1.62), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
        else: ax.annotate("", (2.2, 0.98), (x, y + 0.32), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
    ax.set_title("Skip-gram: 가운데 → 주변" if sg else "CBOW: 주변 → 가운데", fontsize=16, color=RED if sg else BLUE)
save(fig, "w2v_arch.png")

# 3. 나라 → 수도 (PCA 2D)
from sklearn.decomposition import PCA
ws = [w for p in PAIRS for w in p]; M = np.array([EMB[IX[w]] for w in ws]); P2 = PCA(2).fit_transform(M)
fig, ax = plt.subplots(figsize=(7.8, 5.2))
for k, (a, b) in enumerate(PAIRS):
    pa, pb = P2[2 * k], P2[2 * k + 1]
    ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
    ax.scatter(*pa, color=BLUE, s=40); ax.scatter(*pb, color=RED, s=40)
    ax.annotate(a, pa, (7, 2), textcoords="offset points", color=BLUE, fontsize=13); ax.annotate(b, pb, (7, -4), textcoords="offset points", color=RED, fontsize=13)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
save(fig, "w2v_capitals.png")
print("figs ok")
