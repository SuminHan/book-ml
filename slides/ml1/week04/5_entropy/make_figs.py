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


# 1. 놀라움 곡선
p = np.linspace(0.005, 1, 400)
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(p, -np.log2(p), color=BLUE, lw=3)
for q, c, t in ((0.9, BLUE, "브라질 승 90%: 0.15"), (0.1, RED, "한국 승 10%: 3.32"), (0.01, RED, "1%: 6.64")):
    ax.scatter([q], [surprise(q)], s=100, color=c, zorder=4); ax.annotate(t, (q, surprise(q)), (q + 0.08, surprise(q) + 0.6), fontsize=14, color=c)
ax.set_xlabel("확률 p"); ax.set_ylabel("놀라움 $-\\log_2 p$ (비트)"); ax.set_ylim(0, 8)
save(fig, "en_surprise.png")

# 2. 두 결과의 엔트로피 곡선
fig, ax = plt.subplots(figsize=(8, 4.3))
pp = np.linspace(0.001, 0.999, 400); ax.plot(pp, [entropy([a, 1 - a]) for a in pp], color=BLUE, lw=3)
for q, t, c in ((0.1, "한국 vs 브라질 0.47", RED), (0.5, "한일전 1.0 (최대)", GREEN), (0.01, "조기축구 vs 브라질 0.08", GRAY)):
    ax.scatter([q], [entropy([q, 1 - q])], s=100, color=c, zorder=4); ax.annotate(t, (q, entropy([q, 1 - q])), (q + 0.06, entropy([q, 1 - q]) + 0.06), fontsize=13, color=c)
ax.set_xlabel("한쪽이 이길 확률 p"); ax.set_ylabel("엔트로피 H (비트)"); ax.set_ylim(0, 1.15)
save(fig, "en_binary.png")

# 3. 지니 vs 엔트로피 (두 종)
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(pp, [entropy([a, 1 - a]) for a in pp], color=BLUE, lw=3, label="엔트로피 (비트)")
ax.plot(pp, 1 - pp ** 2 - (1 - pp) ** 2, color=RED, lw=3, label="지니")
ax.plot(pp, 2 * (1 - pp ** 2 - (1 - pp) ** 2), color=RED, lw=2, ls="--", label="지니 × 2")
ax.set_xlabel("한 종의 비율 p"); ax.legend(frameon=False); ax.set_ylim(0, 1.1)
save(fig, "en_gini.png")
print("ok")
