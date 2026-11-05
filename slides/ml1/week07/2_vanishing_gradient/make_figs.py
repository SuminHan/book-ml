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


# 1. 정답률
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ep = [1, 2, 3]
ax.plot(ep, np.array(ACC_S) * 100, "o-", color=RED, lw=3, label="시그모이드 10층"); ax.plot(ep, np.array(ACC_R) * 100, "s-", color=GREEN, lw=3, label="ReLU 10층")
for e, a, b in zip(ep, ACC_S, ACC_R):
    ax.text(e, a * 100 + 4, f"{a:.0%}", ha="center", color=RED, fontsize=13); ax.text(e, b * 100 - 9, f"{b:.0%}", ha="center", color=GREEN, fontsize=13)
ax.set_xticks(ep); ax.set_xlabel("에폭"); ax.set_ylabel("시험 정답률 (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False, loc="center right")
save(fig, "vg_acc.png")

# 2. 시그모이드와 기울기, ReLU
z = np.linspace(-6, 6, 400); s = 1 / (1 + np.exp(-z))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
axs[0].plot(z, s, color=RED, lw=3, label="σ(z)"); axs[0].plot(z, s * (1 - s), color=GOLD, lw=3, label="σ'(z) (최대 0.25)")
axs[0].axhline(0.25, color=GRAY, ls=":"); axs[0].legend(frameon=False); axs[0].set_title("시그모이드", fontsize=15)
axs[1].plot(z, np.maximum(0, z), color=GREEN, lw=3, label="ReLU(z) = max(0, z)"); axs[1].plot(z, (z > 0).astype(float), color=GOLD, lw=3, label="기울기: 0 또는 1")
axs[1].legend(frameon=False); axs[1].set_title("ReLU", fontsize=15); axs[1].set_ylim(-0.3, 3)
save(fig, "vg_act.png")

# 3. 층별 기울기
fig, ax = plt.subplots(figsize=(8.5, 4.3))
L = np.arange(1, 11)
ax.semilogy(L, GS, "o-", color=RED, lw=3, label="시그모이드"); ax.semilogy(L, GR, "s-", color=GREEN, lw=3, label="ReLU")
ax.annotate(f"1층 {GS[0]:.1e}", (1, GS[0]), (2, GS[0] / 5), fontsize=13, color=RED); ax.annotate(f"10층 {GS[-1]:.1e}", (10, GS[-1]), (7.2, GS[-1] * 8), fontsize=13, color=RED)
ax.set_xticks(L); ax.set_xlabel("층 (1 = 입구 쪽, 10 = 출구 쪽)"); ax.set_ylabel("가중치 기울기 크기 (로그)"); ax.legend(frameon=False, loc="lower right")
save(fig, "vg_layers.png")

# 4. 초기화
fig, ax = plt.subplots(figsize=(8.5, 4.3))
for k, sc, c in (("표준편차 0.01", 0.01, BLUE), ("표준편차 1", 1.0, RED), ("Xavier $\\sqrt{1/n}$", np.sqrt(1 / N), GOLD), ("He $\\sqrt{2/n}$", np.sqrt(2 / N), GREEN)):
    ax.semilogy(L, init_stds(sc), "o-", color=c, lw=2.5, label=k)
ax.set_xticks(L); ax.set_xlabel("층"); ax.set_ylabel("출력의 표준편차 (로그)"); ax.legend(frameon=False, fontsize=12, loc="center left")
save(fig, "vg_init.png")
print("ok")
