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


def draw_graph(ax, colors, ring=(), label_ids=(0, 33), size=260):
    for a, b in EDGES:
        ax.plot(*POS[[a, b]].T, color="#cccccc", lw=1, zorder=1)
    ax.scatter(*POS.T, s=size, c=colors, edgecolors="white", linewidths=1.2, zorder=2)
    for i in ring:
        ax.scatter(*POS[i], s=size * 2.6, facecolors="none", edgecolors=RED, linewidths=2.5, zorder=3)
    for i in label_ids:
        ax.text(*(POS[i] + [0, 0.085]), {0: "관장 (0)", 33: "회장 (33)"}.get(i, str(i)), ha="center", fontsize=13, weight="bold")
    ax.set_aspect("equal"); ax.axis("off")


CLS = {0: BLUE, 1: GOLD}

# 1. 질문: 이름표 2명
fig, ax = plt.subplots(figsize=(6.4, 5.4))
draw_graph(ax, [CLS[Y[i]] if i in (0, 33) else "#bbbbbb" for i in range(N)])
save(fig, "gc_question.png")

# 2. 정답 (실제로 갈라진 편)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
draw_graph(axs[0], [CLS[Y[i]] for i in range(N)]); axs[0].set_title("실제로 갈라진 편", fontsize=16)
draw_graph(axs[1], [CLS[PRED[i]] for i in range(N)], ring=WRONG, label_ids=(0, 33, 8)); axs[1].set_title("GCN 예측 (이름표 2명으로)", fontsize=16)
save(fig, "gc_result.png")

# 3. 손 계산 그래프 (4명)
fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
P = np.array([[0, 0], [-0.9, -0.8], [0.9, -0.8], [1.9, 0]])
for ax, vals, t in zip(axs, (HAND_X, HAND_H1, HAND_H2), ("처음", "1층 뒤", "2층 뒤")):
    for a, b in HAND_EDGES: ax.plot(*P[[a, b]].T, color=GRAY, lw=2, zorder=1)
    ax.scatter(*P.T, s=900, c=vals, cmap="coolwarm_r", vmin=-1, vmax=1, edgecolors="k", zorder=2)
    for k, (p, v) in enumerate(zip(P, vals)):
        ax.text(p[0], p[1], HAND_NAMES[k], ha="center", va="center", fontsize=14, weight="bold")
        ax.text(p[0], p[1] - 0.42, f"{v:+.2f}", ha="center", va="top", fontsize=14)
    ax.set_title(t, fontsize=15); ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.7, 0.5); ax.axis("off")
save(fig, "gc_hand.png")

# 4. 층 수
fig, ax = plt.subplots(figsize=(7.5, 4.2))
xs = np.arange(len(LAYERS))
bars = ax.bar(xs, [LAYER_MEAN[k] for k in LAYERS], color=[GRAY if k in (0, 8) else (BLUE if k == 2 else "#8c8fc8") for k in LAYERS])
for x, k in zip(xs, LAYERS):
    ax.text(x, LAYER_MEAN[k] + 0.6, f"{LAYER_MEAN[k]:.1f}", ha="center", fontsize=13)
ax.axhline(32, ls=":", color=GRAY); ax.set_ylim(0, 35)
ax.set_xticks(xs); ax.set_xticklabels([("no graph" if k == 0 else str(k)) for k in LAYERS])
ax.set_xlabel("GCN layers"); ax.set_ylabel("correct (of 32, mean of 10 seeds)")
save(fig, "gc_layers.png")
print("figs ok")
