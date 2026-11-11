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


def pr_points(sick, healthy):
    pts, found = [], 0
    for k, (_, y) in enumerate(ranking(sick, healthy), 1):
        found += y
        if y: pts.append((found / len(sick), found / k))
    return pts


# 1. 같은 AI, 두 점수
fig, ax = plt.subplots(figsize=(7, 4.4))
b = ax.bar(["ROC-AUC", "PR-AUC"], [ROC_AUC, AP], color=[BLUE, RED], width=0.5)
for r, v in zip(b, [ROC_AUC, AP]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=24, fontweight="bold", color=r.get_facecolor())
ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1]); ax.set_title("같은 AI, 같은 점수 100개", fontsize=17)
save(fig, "pa_hook.png")

# 2. ROC 곡선
x, y = zip(*roc_points(SICK, HEALTHY))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.step(x, y, where="post", color=BLUE, lw=3); ax.fill_between(x, y, step="post", color=BLUE, alpha=0.15)
ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=2); ax.text(0.55, 0.42, "찍기 AI = 0.5", color="#888888", fontsize=14, rotation=38)
ax.scatter([4 / 90], [0.8], s=140, color=GOLD, edgecolor="black", zorder=4)
ax.annotate("기준 0.5\n(4%, 80%)", (4 / 90, 0.8), (0.2, 0.62), fontsize=14, color="#7a6a3f", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOLD))
ax.text(0.45, 0.15, f"넓이 = {ROC_AUC}", fontsize=20, color=BLUE, fontweight="bold")
ax.set_xlabel("거짓 양성 비율 FP / 건강 90명"); ax.set_ylabel("재현율 TP / 독감 10명"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
save(fig, "pa_roc.png")

# 3. PR 곡선
pts = [(0, 1.0)] + pr_points(SICK, HEALTHY)
r_, p_ = zip(*pts)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.step(r_, p_, where="pre", color=RED, lw=3); ax.fill_between(r_, p_, step="pre", color=RED, alpha=0.15)
ax.axhline(0.1, ls="--", color=GRAY, lw=2); ax.text(0.03, 0.13, "찍기 AI = 독감 비율 0.1", color="#888888", fontsize=14)
ax.scatter([0.8], [8 / 12], s=140, color=GOLD, edgecolor="black", zorder=4)
ax.annotate("기준 0.5\n(80%, 67%)", (0.8, 8 / 12), (0.42, 0.42), fontsize=14, color="#7a6a3f", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOLD))
ax.text(0.08, 0.3, f"넓이 = {AP}", fontsize=20, color=RED, fontweight="bold")
ax.set_xlabel("재현율 (놓치지 않았나)"); ax.set_ylabel("정밀도 (AI 말이 맞았나)"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
save(fig, "pa_pr.png")

# 4. 줄 세우기: 독감을 찾을 때마다 정밀도
R = ranking(SICK, HEALTHY)[:16]
_, precs = average_precision(SICK, HEALTHY)
fig, ax = plt.subplots(figsize=(12, 2.6))
j = 0
for k, (s, yv) in enumerate(R, 1):
    ax.add_patch(Circle((k, 0), 0.4, color=RED if yv else GRAY))
    ax.text(k, -0.75, f"{k}", ha="center", fontsize=12, color="#555555")
    if yv:
        ax.text(k, 0.75, f"{j + 1}/{k}", ha="center", fontsize=13, color=RED, fontweight="bold"); j += 1
ax.text(0.2, -0.75, "줄", ha="right", fontsize=12, color="#555555")
ax.set_xlim(-0.3, 16.7); ax.set_ylim(-1.1, 1.2); ax.set_aspect("equal"); ax.axis("off")
save(fig, "pa_rank.png")

# 5. 건강한 학생이 10배가 되면
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (name, vals, c) in zip(axs, (("ROC-AUC", (ROC_AUC, HEALTHY_X10[0]), BLUE), ("PR-AUC", (AP, HEALTHY_X10[1]), RED))):
    b = ax.bar(["건강 90명", "건강 900명"], vals, color=[c, c], alpha=0.9, width=0.55)
    b[1].set_alpha(0.5)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=20, fontweight="bold", color=c)
    ax.set_title(name, fontsize=17, color=c); ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1])
save(fig, "pa_x10.png")
print("ok")
