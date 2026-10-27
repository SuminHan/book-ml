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


# 1. 0.9 vs 0.999
k = np.arange(101)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(k, 0.9 ** k * 100, color=RED, lw=3, label="걸음마다 × 0.9 (기본 RNN)"); ax.plot(k, 0.999 ** k * 100, color=GREEN, lw=3, label="걸음마다 × 0.999 (LSTM 잊는 문)")
ax.text(100, 0.9 ** 100 * 100 + 3, "0.003%", ha="right", color=RED, fontsize=14); ax.text(100, 0.999 ** 100 * 100 - 8, "90%", ha="right", color=GREEN, fontsize=14)
ax.set_xlabel("걸음"); ax.set_ylabel("남은 기억 (%)"); ax.legend(frameon=False, loc="center right"); ax.set_ylim(0, 105)
save(fig, "ls_decay.png")

# 2. LSTM 셀 그림
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 5)
ax.annotate("", (10.6, 4.2), (0.3, 4.2), arrowprops=dict(arrowstyle="->", lw=4, color=GOLD))
ax.text(0.3, 4.5, "$c_{t-1}$", fontsize=16, color="#7a6a3f"); ax.text(10.2, 4.5, "$c_t$", fontsize=16, color="#7a6a3f")
ax.text(5.4, 4.65, "셀 상태 = 기억의 컨베이어 벨트", ha="center", fontsize=14, color="#7a6a3f")
for x, sym in ((3.0, "×"), (6.0, "+")):
    ax.add_patch(Circle((x, 4.2), 0.32, fc="white", ec="black", lw=2, zorder=3)); ax.text(x, 4.2, sym, ha="center", va="center", fontsize=20, zorder=4)
gates = [(3.0, "잊는 문 $f$", RED, "얼마나 지울까"), (6.0, "적는 문 $i$ · 후보 $\\tilde c$", BLUE, "무엇을 더할까"), (8.8, "꺼내는 문 $o$", GREEN, "얼마나 내보낼까")]
for x, t, c, d in gates:
    ax.add_patch(FancyBboxPatch((x - 1.0, 1.3), 2.0, 0.8, boxstyle="round,pad=0.05", fc="white", ec=c, lw=2.5)); ax.text(x, 1.7, t, ha="center", va="center", fontsize=13, color=c)
    ax.text(x, 0.8, d, ha="center", fontsize=12, color=GRAY)
    if x < 8: ax.annotate("", (x, 3.85), (x, 2.15), arrowprops=dict(arrowstyle="->", lw=2, color=c))
ax.annotate("", (8.8, 2.15), (8.8, 3.85), arrowprops=dict(arrowstyle="->", lw=2, color=GOLD)); ax.annotate("", (10.6, 1.7), (9.85, 1.7), arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN)); ax.text(10.65, 1.7, "$h_t$", fontsize=16, va="center")
ax.text(0.2, 0.3, "입력: 지금 단어 $x_t$ 와 이전 출력 $h_{t-1}$ → 세 문 모두 0~1 (시그모이드)", fontsize=12, color=GRAY)
save(fig, "ls_cell.png")

# 3. 실험
fig, ax = plt.subplots(figsize=(8.5, 4.0))
for kind, c, t in (("rnn", RED, "RNN"), ("lstm", GREEN, "LSTM")):
    L = np.array(J[f"{kind}_128"]["curve"]); s = np.convolve(L, np.ones(50) / 50, mode="valid"); ax.plot(np.arange(49, len(L)), s, color=c, lw=2.5, label=t)
ax.set_xlabel("학습 걸음 (조각 길이 128)"); ax.set_ylabel("학습 손실"); ax.set_ylim(1.3, 2.6); ax.legend(frameon=False)
save(fig, "ls_curve.png")
print("ok")
