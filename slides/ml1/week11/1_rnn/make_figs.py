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


# 1. 펼친 RNN
fig, ax = plt.subplots(figsize=(12, 3.8)); ax.axis("off"); ax.set_xlim(-0.6, 9.6); ax.set_ylim(-0.3, 3.4)
words = ["dog", "bites", "man"]
for t, w in enumerate(words):
    x = 1 + 3 * t
    ax.add_patch(FancyBboxPatch((x - 0.7, 1.2), 1.4, 0.9, boxstyle="round,pad=0.05", fc="#eef", ec=BLUE, lw=2)); ax.text(x, 1.65, f"$h_{t + 1}$", ha="center", va="center", fontsize=18)
    ax.text(x, 0.1, w, ha="center", fontsize=17, color=GOLD, fontweight="bold"); ax.annotate("", (x, 1.15), (x, 0.45), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(x + 0.12, 0.75, "$W_x$", fontsize=13)
    ax.annotate("", (x, 3.1), (x, 2.15), arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY)); ax.text(x + 0.12, 2.55, "$V$", fontsize=13, color=GRAY)
    if t < 2:
        ax.annotate("", (x + 2.25, 1.65), (x + 0.75, 1.65), arrowprops=dict(arrowstyle="->", lw=2.5, color=RED)); ax.text(x + 1.4, 1.85, "$W_h$", fontsize=14, color=RED)
ax.text(-0.5, 1.65, "$h_0 = 0$", fontsize=14, va="center"); ax.annotate("", (0.25, 1.65), (0.15, 1.65), arrowprops=dict(arrowstyle="->", lw=2.5, color=RED))
save(fig, "rn_unroll.png")

# 2. 두 문장의 메모
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, H, s, c in ((axs[0], H1, S1, BLUE), (axs[1], H2, S2, RED)):
    M = np.array(H); im = ax.imshow(M.T, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(4): ax.text(i, j, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=12)
    ax.set_xticks(range(3)); ax.set_xticklabels(s, fontsize=14); ax.set_yticks(range(4)); ax.set_yticklabels([f"칸 {k + 1}" for k in range(4)], fontsize=12)
    ax.set_title(" ".join(s).capitalize(), fontsize=15, color=c)
save(fig, "rn_memo.png")

# 3. 손실
L = np.array(J["losses"]); k = 25; sm = np.convolve(L, np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(L, color="#cccccc", lw=1); ax.plot(np.arange(k - 1, len(L)), sm, color=BLUE, lw=2.5)
ax.axhline(UNIFORM, color=RED, ls="--"); ax.text(1300, UNIFORM - 0.25, f"아무렇게나 찍기 ln 65 = {UNIFORM}", color=RED, fontsize=13)
ax.set_xlabel("학습 걸음"); ax.set_ylabel("다음 글자 손실 (교차 엔트로피)"); ax.set_ylim(1, 4.5)
save(fig, "rn_loss.png")

# 4. 기억 감쇠
fig, ax = plt.subplots(figsize=(8, 3.6))
k = np.arange(11); ax.bar(k, 0.5 ** k, color=[RED if i == 10 else BLUE for i in k])
ax.text(10, 0.05, "0.001", ha="center", color=RED, fontsize=13); ax.set_xticks(k); ax.set_xlabel("몇 걸음 전의 단어"); ax.set_ylabel("남은 영향 (걸음마다 × 0.5)")
save(fig, "rn_decay.png")
print("ok")
