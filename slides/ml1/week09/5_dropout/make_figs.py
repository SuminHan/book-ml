"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#bbbbbb", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 신경망 그림: 다 켬 vs 무작위로 끔 (두 번)
def net(ax, off, title):
    layers = [4, 6, 6, 3]; pos = {}
    for l, n in enumerate(layers):
        for i in range(n): pos[(l, i)] = (l, (n - 1) / 2 - i)
    for l in range(3):
        for i in range(layers[l]):
            for j in range(layers[l + 1]):
                if (l, i) in off or (l + 1, j) in off: continue
                ax.plot(*zip(pos[(l, i)], pos[(l + 1, j)]), color="#cccccc", lw=0.8, zorder=1)
    for k, (x, y) in pos.items():
        o = k in off
        ax.scatter(x, y, s=260, color="white" if o else (BLUE if k[0] in (1, 2) else GOLD), edgecolor=GRAY if o else "black", zorder=3)
        if o: ax.text(x, y, "×", ha="center", va="center", color=RED, fontsize=16, fontweight="bold", zorder=4)
    ax.set_title(title, fontsize=14); ax.axis("off"); ax.set_xlim(-0.4, 3.4); ax.set_ylim(-3, 3)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
net(axs[0], set(), "시험 때: 모두 켠다")
net(axs[1], {(1, 0), (1, 3), (1, 4), (2, 1), (2, 2), (2, 5)}, "학습 1번째 배치")
net(axs[2], {(1, 1), (1, 2), (1, 5), (2, 0), (2, 3), (2, 4)}, "학습 2번째 배치")
save(fig, "do_net.png")

# 2. 학습 곡선: p=0 vs p=0.5
fig, axs = plt.subplots(1, 2, figsize=(12, 4.0))
for ax, p, c in ((axs[0], "0.0", RED), (axs[1], "0.5", GREEN)):
    h = np.array(J[p]) * 100; ep = np.arange(1, len(h) + 1)
    ax.plot(ep, h[:, 0], color=BLUE, lw=2.5, label="학습 2,000장"); ax.plot(ep, h[:, 1], color=c, lw=2.5, label="시험 1만 장")
    ax.set_ylim(85, 100.5); ax.set_xlabel("에폭"); ax.set_title(f"드롭아웃 p = {p}: 시험 오답 {ERRORS[p]}개", fontsize=14); ax.legend(frameon=False, loc="lower right")
axs[0].set_ylabel("정확도 (%)")
save(fig, "do_curves.png")

# 3. p 별 오답
fig, ax = plt.subplots(figsize=(7.5, 4.0))
ps = list(ERRORS); v = [ERRORS[p] for p in ps]
b = ax.bar([f"p = {p}" for p in ps], v, color=[RED] + [GREEN] * 3)
for r, x in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x + 8, str(x), ha="center", fontsize=15, fontweight="bold")
ax.set_ylabel("시험 1만 장 중 오답"); ax.set_ylim(500, 830)
save(fig, "do_p.png")
print("ok")
