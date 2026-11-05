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


# 1. 학습 곡선
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.plot(STEPS, PLAIN * 100, color=RED, lw=3, label="시그모이드 8층"); ax.plot(STEPS, BN * 100, color=GREEN, lw=3, label="+ BatchNorm")
ax.scatter([250], [BN_250], s=120, color=GREEN, zorder=4, edgecolor="black"); ax.annotate(f"250번: {BN_250}%", (250, BN_250), (420, 80), fontsize=14, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
ax.text(900, 15, f"1,500번 뒤에도 {PLAIN_END}%", color=RED, fontsize=14)
ax.set_xlabel("학습 횟수 (배치)"); ax.set_ylabel("정답률 (%)"); ax.set_ylim(0, 103); ax.legend(frameon=False, loc="center right")
save(fig, "bn_curve.png")

# 2. 손 계산
fig, axs = plt.subplots(3, 1, figsize=(9, 3.6), sharex=False)
for ax, v, t, c in ((axs[0], HAND_X, "원래 값", GRAY), (axs[1], HAND["xhat"], "평균 0 · 표준편차 1 로", BLUE), (axs[2], HAND["y"], f"γ = {GAMMA:g}, β = {BETA:g} 로 다시 조절", GREEN)):
    ax.scatter(v, [0] * 4, s=140, color=c, zorder=3); ax.axhline(0, color="#dddddd", lw=1)
    for x in v: ax.text(x, 0.35, f"{x:g}", ha="center", fontsize=12)
    ax.set_yticks([]); ax.set_xlim(-2.5, 9); ax.set_ylim(-0.6, 0.9); ax.set_title(t, fontsize=13, loc="left"); ax.spines["left"].set_visible(False)
save(fig, "bn_hand.png")

# 3. 8층 시그모이드의 마지막 층 입력 분포 (학습 전, digits 512장)
import torch, torch.nn as nn
from sklearn.datasets import load_digits
X = torch.tensor(load_digits().data[:512] / 16.0, dtype=torch.float32)
def pre_acts(bn, seed=0):
    torch.manual_seed(seed); h = X; outs = []
    for _ in range(8):
        lin = nn.Linear(h.shape[1], 64); z = lin(h)
        if bn: z = (z - z.mean(0)) / torch.sqrt(z.var(0, unbiased=False) + 1e-5)
        outs.append(z.detach().numpy().ravel()); h = torch.sigmoid(z)
    return outs
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for bn, t, c in ((False, "BatchNorm 없음", RED), (True, "BatchNorm", GREEN)):
    spread = [float(np.std(z.reshape(512, 64), axis=0).mean()) for z in pre_acts(bn)]
    ax.semilogy(range(1, 9), spread, "o-", color=c, lw=2.5, label=t)
    ax.text(8.15, spread[-1], f"{spread[-1]:.3f}", color=c, fontsize=13, va="center")
ax.set_xlabel("층"); ax.set_ylabel("그림 512장 사이의 퍼짐\n(뉴런별 표준편차 평균, 로그)"); ax.legend(frameon=False); ax.set_xlim(0.7, 8.9)
save(fig, "bn_dist.png")
print("ok")
