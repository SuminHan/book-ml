"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#bbbbbb", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 100명 중 독감 10명
fig, ax = plt.subplots(figsize=(6.4, 6.4))
flu = {3, 14, 27, 31, 45, 52, 68, 76, 83, 99}
for i in range(100):
    ax.add_patch(Circle((i % 10, 9 - i // 10), 0.38, color=RED if i in flu else GRAY))
ax.set_xlim(-0.6, 9.6); ax.set_ylim(-0.6, 9.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, "pr_hook.png")

# 2. 두 원: AI 가 고른 12명 · 진짜 독감 10명 · 겹침 8명
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.add_patch(Circle((-0.9, 0), 1.9, color=BLUE, alpha=0.25)); ax.add_patch(Circle((0.9, 0), 1.75, color=RED, alpha=0.25))
ax.text(-2.0, 0, f"{FP}", fontsize=34, ha="center", va="center", color=BLUE, fontweight="bold")
ax.text(0, 0, f"{TP}", fontsize=40, ha="center", va="center", color="black", fontweight="bold")
ax.text(1.9, 0, f"{FN}", fontsize=34, ha="center", va="center", color=RED, fontweight="bold")
ax.text(-1.3, 2.15, f"AI 가 '독감'이라고 고른 {TP + FP}명", fontsize=16, ha="center", color=BLUE, fontweight="bold")
ax.text(1.5, -2.25, f"진짜 독감 {TP + FN}명", fontsize=16, ha="center", color=RED, fontweight="bold")
ax.text(3.5, 1.9, f"둘 다 아닌\n건강 {TN}명", fontsize=14, ha="center", color="#666666")
ax.set_xlim(-3.5, 4.3); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, "pr_venn.png")


def row(ax, items, y, title, col):
    for i, c in enumerate(items):
        ax.add_patch(Circle((i, y), 0.38, color=c))
    ax.text(-0.7, y, title, ha="right", va="center", fontsize=16, color=col, fontweight="bold")


# 3. 정밀도: 고른 12명 줄 세우기
fig, ax = plt.subplots(figsize=(10, 2.0))
row(ax, [RED] * TP + [GRAY] * FP, 0, f"고른 {TP + FP}명", BLUE)
ax.text(12.2, 0, f"진짜 독감 {TP}명 → {TP}/{TP + FP} = {PREC}%", va="center", fontsize=17, color=RED, fontweight="bold")
ax.set_xlim(-3.2, 19.5); ax.set_ylim(-0.7, 0.7); ax.set_aspect("equal"); ax.axis("off")
save(fig, "pr_prec.png")

# 4. 재현율: 독감 10명 불러 모으기
fig, ax = plt.subplots(figsize=(10, 2.0))
row(ax, [RED] * TP + ["white"] * FN, 0, f"독감 {TP + FN}명", RED)
for i in range(TP, TP + FN):
    ax.add_patch(Circle((i, 0), 0.38, fill=False, ls="--", lw=2, color=RED))
ax.text(10.2, 0, f"잡힌 {TP}명 → {TP}/{TP + FN} = {REC}%", va="center", fontsize=17, color=RED, fontweight="bold")
ax.set_xlim(-3.2, 19.5); ax.set_ylim(-0.7, 0.7); ax.set_aspect("equal"); ax.axis("off")
save(fig, "pr_rec.png")

# 5. 점수 분포와 기준선
fig, ax = plt.subplots(figsize=(10, 3.6))
rng = np.random.default_rng(0)                                       # 점이 겹치지 않게 세로로만 흩뿌림 (값은 data.py 그대로)
ax.scatter(HEALTHY, 0.3 + rng.uniform(-0.18, 0.18, len(HEALTHY)), s=40, color=GRAY, edgecolor="#777777", label="건강 90명")
ax.scatter(SICK, 1.0 + rng.uniform(-0.12, 0.12, len(SICK)), s=70, color=RED, label="독감 10명")
ax.axvline(TH, color=BLUE, lw=3, ls="--"); ax.text(TH + 0.01, 1.35, "기준 0.5 이상 = '독감'", color=BLUE, fontsize=15, fontweight="bold")
ax.set_yticks([0.3, 1.0]); ax.set_yticklabels(["건강", "독감"]); ax.set_xlabel("AI 의 '독감일 확률' 점수"); ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.5)
save(fig, "pr_scores.png")

# 6. 기준을 움직이면
ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
P, R = [], []
for t in ths:
    _, p, r, _ = metrics(*confusion(SICK, HEALTHY, t)); P.append(p * 100); R.append(r * 100)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(ths, P, color=BLUE, lw=2.5, label="정밀도"); ax.plot(ths, R, color=RED, lw=2.5, label="재현율")
for t, (pp, rr) in THS.items():
    ax.scatter([t, t], [pp, rr], color=[BLUE, RED], s=60, zorder=3)
ax.set_xlabel("기준 (임계값)"); ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(frameon=False, loc="lower left")
save(fig, "pr_threshold.png")
print("ok")
