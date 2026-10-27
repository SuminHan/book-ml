"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def box(ax, x, y, w, h, text, c, fs=13):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc="white", ec=c, lw=2.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=c)


def op(ax, x, y, sym):
    ax.add_patch(Circle((x, y), 0.3, fc="white", ec="black", lw=2, zorder=3)); ax.text(x, y, sym, ha="center", va="center", fontsize=18, zorder=4)


def arrow(ax, a, b, c, lw=2):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", lw=lw, color=c))


# 1. GRU 셀: 기억은 h 하나, 갱신 문 z 가 지킬 비율 · 리셋 문 r 이 후보를 만들 때 볼 비율
fig, ax = plt.subplots(figsize=(11, 4.8)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2)
arrow(ax, (0.3, 4.3), (10.6, 4.3), GOLD, lw=4)
ax.text(0.3, 4.6, "$h_{t-1}$", fontsize=16, color="#7a6a3f"); ax.text(10.2, 4.6, "$h_t$", fontsize=16, color="#7a6a3f")
ax.text(5.5, 4.75, "기억은 $h$ 하나 (셀 상태 없음)", ha="center", fontsize=14, color="#7a6a3f")
op(ax, 3.4, 4.3, "×"); op(ax, 7.4, 4.3, "+")
box(ax, 3.4, 2.3, 2.3, 0.8, "갱신 문 $z$", RED); ax.text(3.4, 1.55, "얼마나 지킬까", ha="center", fontsize=12, color=GRAY)
arrow(ax, (3.4, 2.75), (3.4, 3.95), RED)
box(ax, 5.6, 1.0, 2.1, 0.8, "리셋 문 $r$", GREEN); ax.text(5.6, 0.3, "후보를 만들 때 옛 기억을 얼마나 볼까", ha="center", fontsize=12, color=GRAY)
box(ax, 7.4, 2.3, 2.3, 0.8, "후보 $\\tilde h$", BLUE)
arrow(ax, (6.3, 1.4), (6.9, 1.9), GREEN)
op(ax, 7.4, 3.3, "×"); arrow(ax, (7.4, 2.75), (7.4, 3.0), BLUE); arrow(ax, (7.4, 3.6), (7.4, 4.0), BLUE)
ax.text(7.75, 3.3, "$1 - z$", fontsize=14, color=RED, va="center")
ax.text(3.75, 3.55, "$z$", fontsize=14, color=RED)
ax.text(0.2, 0.3, "입력: 지금 $x_t$ 와 $h_{t-1}$ → 문은 0~1 (시그모이드)", fontsize=12, color=GRAY)
save(fig, "gru_cell.png")

# 2. 학습 곡선 (조각 128)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
for kind, c, t in (("rnn", RED, "RNN"), ("lstm", GREEN, "LSTM"), ("gru", BLUE, "GRU")):
    L = np.array(CURVE[kind]); s = np.convolve(L, np.ones(50) / 50, mode="valid"); ax.plot(np.arange(49, len(L)), s, color=c, lw=2.5, label=t)
ax.set_xlabel("학습 걸음 (조각 길이 128)"); ax.set_ylabel("학습 손실"); ax.set_ylim(1.3, 2.6); ax.legend(frameon=False)
save(fig, "gru_curve.png")

# 3. 가중치 수 vs 검증 손실
fig, ax = plt.subplots(figsize=(7.5, 4.0))
names, cols = ("rnn", "gru", "lstm"), (RED, BLUE, GREEN)
b = ax.bar(["RNN\n(1묶음)", "GRU\n(3묶음)", "LSTM\n(4묶음)"], [PARAMS[k] / 1000 for k in names], color=cols, width=0.55)
for r, k in zip(b, names):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 3, f"{PARAMS[k]:,}\n검증 {VAL[k + '_128']:.2f}", ha="center", fontsize=13)
ax.set_ylabel("순환 층 가중치 (천 개)"); ax.set_ylim(0, 175)
save(fig, "gru_params.png")
print("ok")
