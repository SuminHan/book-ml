"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
CAT, DOG = "#e07b39", "#5b8bd9"
O = "figs/"


def chips(ax, cat, dog, title, show_pen=True):
    """칩 100개를 고양이/강아지에 나눠 건 막대. 정답(고양이) 쪽만 남는다."""
    ax.barh([0], [cat], color=CAT, height=0.6)
    ax.barh([0], [dog], left=[cat], color=DOG, height=0.6, alpha=0.35)
    if cat >= 8: ax.text(cat / 2, 0, f"고양이 {cat}", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
    else: ax.text(cat + 1, 0.45, f"고양이 {cat}", ha="left", va="bottom", color=CAT, fontsize=14, fontweight="bold")
    if dog >= 15: ax.text(cat + dog / 2, 0, f"강아지 {dog}", ha="center", va="center", color=DOG, fontsize=15)
    elif dog > 0: ax.text(cat + dog / 2, 0.45, f"강아지 {dog}", ha="center", va="bottom", color=DOG, fontsize=14)
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 0.9); ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", fontweight="bold")
    if show_pen:
        p = pen(cat)
        ax.text(101, 0, "벌점 ∞" if p == float("inf") else f"벌점 {p:.2f}", ha="left", va="center",
                fontsize=17, color=RED if p > 2 else BLUE, fontweight="bold")


# 1. 오늘의 질문
fig, ax = plt.subplots(figsize=(9, 2.4))
chips(ax, *AI_C, "정답은 고양이인데, AI는 '강아지 99%'", show_pen=False)
ax.text(101, 0, "벌점은?", ha="left", va="center", fontsize=19, color=RED, fontweight="bold")
fig.savefig(O + "ce_hook.png", bbox_inches="tight"); plt.close(fig)

# 2. 칩 게임: 세 AI
fig, axs = plt.subplots(3, 1, figsize=(10, 4.6))
for ax, (c, d), name in zip(axs, (AI_A, AI_HALF, AI_C), ("AI A", "AI B (반반)", "AI C")):
    chips(ax, c, d, name, show_pen=False)
    ax.text(101, 0, f"남은 칩 {c}개", ha="left", va="center", fontsize=16, color=CAT, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "ce_chips.png", bbox_inches="tight"); plt.close(fig)

# 3. 벌점 곡선 −log₂(남은 칩/100)
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.linspace(0.6, 100, 400)
ax.plot(x, [pen(v) for v in x], color=BLUE, lw=3)
for v, lab_dx in ((90, -14), (50, 2), (25, 2), (1, 3)):
    ax.scatter([v], [pen(v)], s=90, color=RED, zorder=3)
    ax.text(v + lab_dx, pen(v) + 0.35, f"{v}개 → {pen(v):.2f}", fontsize=14, color=RED)
ax.set_xlabel("정답 쪽에 남은 칩 (100개 중)"); ax.set_ylabel("벌점 (비트)")
ax.set_xlim(0, 102); ax.set_ylim(0, 8)
fig.tight_layout(); fig.savefig(O + "ce_curve.png"); plt.close(fig)

# 4. 파산 vs 반반
fig, axs = plt.subplots(2, 1, figsize=(10, 3.2))
chips(axs[0], *AI_ALLIN, "강아지에 전부 걸었다")
chips(axs[1], *AI_HALF, "모르겠으니 반반")
fig.tight_layout(); fig.savefig(O + "ce_allin.png", bbox_inches="tight"); plt.close(fig)

# 5. CE vs MSE: 손실과 기울기 (정답 y=1, 예측 ŷ = 시그모이드 출력)
q = np.linspace(0.005, 0.995, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.plot(q, -np.log(q), color=BLUE, lw=3, label="교차 엔트로피 $-\\ln \\hat y$")
a1.plot(q, (1 - q) ** 2, color=GOLD, lw=3, label="MSE $(1-\\hat y)^2$")
a1.set_xlabel("정답에 준 확률 $\\hat y$"); a1.set_title("손실", fontsize=16); a1.legend(frameon=False, fontsize=13); a1.set_ylim(0, 5.5)
a2.plot(q, 1 - q, color=BLUE, lw=3, label="교차 엔트로피: $1-\\hat y$")
a2.plot(q, 2 * (1 - q) * q * (1 - q), color=GOLD, lw=3, label="MSE: $2(1-\\hat y)\\,\\hat y(1-\\hat y)$")
a2.annotate("확신하고 틀리면\n기울기가 0에 가깝다\n→ 학습이 멈춘다", xy=(0.03, 0.06), xytext=(0.66, 0.56), fontsize=13, color=GOLD,
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8))
a2.set_xlabel("정답에 준 확률 $\\hat y$"); a2.set_title("기울기 크기 $|\\partial L/\\partial z|$", fontsize=16); a2.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout(); fig.savefig(O + "ce_vs_mse.png"); plt.close(fig)

# 6. 학습 = 정답 쪽에 칩을 더 건다
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(range(len(LEARN)), LEARN, color=CAT, width=0.6)
for i, v in enumerate(LEARN):
    ax.text(i, v + 2, f"벌점 {pen(v):.2f}", ha="center", fontsize=14, color=RED if pen(v) > 2 else BLUE, fontweight="bold")
ax.set_xticks(range(len(LEARN)), [f"{i + 1}" for i in range(len(LEARN))]); ax.set_xlabel("학습 단계")
ax.set_ylabel("고양이에 건 칩"); ax.set_ylim(0, 110)
fig.tight_layout(); fig.savefig(O + "ce_learning.png"); plt.close(fig)

# 7. 다음 단어에 칩 나눠 걸기
fig, ax = plt.subplots(figsize=(9, 3.8))
names = [w for w, _ in WORDS]; vals = [v for _, v in WORDS]
ax.barh(names[::-1], vals[::-1], color=[GRAY, GRAY, BLUE], height=0.6)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 1, i, f"{v}개", va="center", fontsize=16)
ax.set_xlim(0, 100); ax.set_title("“점심 뭐 ___ ?”   정답: 먹지", fontsize=17, loc="left")
ax.text(70, 2, f"벌점 {pen(WORDS[0][1]):.2f}", fontsize=18, color=BLUE, fontweight="bold", va="center")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
fig.tight_layout(); fig.savefig(O + "ce_words.png"); plt.close(fig)
print("ok")
