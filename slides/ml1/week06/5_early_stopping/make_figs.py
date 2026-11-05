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
ep = np.arange(1, EPOCHS + 1)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def curves(ax, mark=True):
    ax.plot(ep, ACC_TR * 100, color=BLUE, lw=2.5, label="학습 그림 500장")
    ax.plot(ep, ACC_VA * 100, color=RED, lw=2.5, label="숨긴 그림 500장 (검증)")
    ax.set_xlabel("학습 횟수 (epoch)"); ax.set_ylabel("정확도 (%)"); ax.set_ylim(0, 100); ax.legend(frameon=False, loc="lower right")
    if mark:
        ax.axvline(BEST, color=GOLD, lw=2, ls="--"); ax.scatter([BEST], [ACC_VA[BEST - 1] * 100], s=120, color=GOLD, zorder=4, edgecolor="black")


# 1. 질문
fig, ax = plt.subplots(figsize=(8.5, 4.3)); curves(ax, mark=False)
ax.text(150, 40, "오래 학습할수록 더 좋을까?", fontsize=18, color=RED, fontweight="bold", ha="center")
save(fig, "es_hook.png")

# 2. 학습 곡선 + 31번째
fig, ax = plt.subplots(figsize=(8.5, 4.3)); curves(ax)
ax.annotate(f"{BEST}번째: 검증 {pct(ACC_VA[BEST - 1])}%", (BEST, ACC_VA[BEST - 1] * 100), (60, 60), fontsize=15, color="#7a6a3f", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOLD))
ax.annotate(f"300번째: 학습 {pct(ACC_TR[-1])}% · 검증 {pct(ACC_VA[-1])}%", (300, ACC_VA[-1] * 100), (150, 30), fontsize=14, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "es_curve.png")

# 3. 외운 비율
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(ep, MEM * 100, color=GREEN, lw=2.5)
ax.axvline(BEST, color=GOLD, ls="--", lw=2)
ax.text(BEST + 4, 70, f"{BEST}번째: {pct(MEM[BEST - 1])}장", fontsize=14, color="#7a6a3f"); ax.text(230, 92, f"300번째: {pct(MEM[-1])}장", fontsize=14, color=GREEN)
ax.set_xlabel("학습 횟수 (epoch)"); ax.set_ylabel("틀린 정답 100장 중\n그대로 외운 수"); ax.set_ylim(0, 100)
save(fig, "es_mem.png")

# 4. 맞히던 그림을 틀린다
fig, axs = plt.subplots(1, len(SHOW_VA), figsize=(9, 2.8))
for ax, j in zip(axs, SHOW_VA):
    ax.imshow(IMG[VA[j]].reshape(8, 8), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"정답 {Y[VA[j]]}\n{BEST}번째 {PRED_VA[BEST - 1, j]} → 300번째 {PRED_VA[-1, j]}", fontsize=13)
save(fig, "es_digits.png")

# 5. 참을성
fig, ax = plt.subplots(figsize=(8.5, 4.2)); curves(ax)
for p, c in ((PATIENCE, GREEN),):
    ax.axvspan(BEST, STOP, color=c, alpha=0.15); ax.axvline(STOP, color=c, lw=2)
    ax.text(STOP + 3, 50, f"{PATIENCE}번 더 기다려도 안 나아짐\n→ {STOP}번째에 멈추고 {BEST}번째로 되돌림", fontsize=13, color=GREEN)
ax.set_xlim(0, 120)
save(fig, "es_patience.png")
print("ok")
