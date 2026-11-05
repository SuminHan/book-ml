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
BLUE, RED, GOLD, GRAY, ORANGE = "#2E3192", "#d62728", "#A9976B", "#888888", "#f28e2b"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def fruits(ax, sy=1.0, q=True, legend=True, qs=320):
    a = np.array(APPLE); t = np.array(TANG)
    ax.scatter(a[:, 0], a[:, 1] * sy, s=110, color=RED, label="사과", zorder=3)
    ax.scatter(t[:, 0], t[:, 1] * sy, s=110, marker="s", color=ORANGE, label="귤", zorder=3)
    ax.scatter([NOISE[0]], [NOISE[1] * sy], s=110, marker="s", color=ORANGE, edgecolor="black", lw=1.5, zorder=3)
    if q:
        ax.scatter([Q[0]], [Q[1] * sy], s=qs, marker="*", color=GOLD, edgecolor="black", zorder=4, label="처음 보는 과일")
    ax.set_xlabel("크기"); ax.set_ylabel("단맛" if sy == 1 else "단맛 (0~100 점)")
    if legend: ax.legend(frameon=False, loc="lower left", fontsize=13)


# 1. 질문
fig, ax = plt.subplots(figsize=(7.5, 5)); fruits(ax, qs=170)
ax.text(Q[0] + 0.25, Q[1] + 0.35, "?", fontsize=30, color=GOLD, fontweight="bold")
ax.set_xlim(1, 9.5); ax.set_ylim(2, 9.5)
save(fig, "kn_hook.png")

# 2. k=3 이웃
fig, ax = plt.subplots(figsize=(7.5, 5)); fruits(ax)
r = NEAR3[-1][2]
ax.add_patch(Circle(Q, r + 0.05, fill=False, ls="--", lw=2, color=GOLD))
for p, l, d in NEAR3:
    ax.plot([Q[0], p[0]], [Q[1], p[1]], color=GOLD, lw=2)
    ax.text((Q[0] + p[0]) / 2 + 0.08, (Q[1] + p[1]) / 2 - 0.05, f"{d:.2f}", fontsize=13, color="#7a6a3f", fontweight="bold")
ax.set_xlim(4.5, 8); ax.set_ylim(3, 6.3); ax.set_aspect("equal")
save(fig, "kn_k3.png")

# 3. 결정 경계 k = 1, 5, 21
gx, gy = np.meshgrid(np.linspace(1, 9.5, 170), np.linspace(2, 9.5, 150))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, k in zip(axs, (1, 5, 21)):
    Z = np.array([[knn(PTS, (x, y), k)[1] == "사과" for x, y in zip(rx, ry)] for rx, ry in zip(gx, gy)])
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=[ORANGE, RED], alpha=0.15)
    fruits(ax, legend=False); ax.set_title(f"k = {k}", fontsize=17); ax.set_xlim(1, 9.5); ax.set_ylim(2, 9.5)
save(fig, "kn_boundary.png")

# 4. leave-one-out
fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(LOO); v = [LOO[k] for k in ks]
b = ax.bar([str(k) for k in ks], v, color=[RED if x == max(v) else GRAY for x in v])
for r_, x in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x + 0.3, f"{x}/21", ha="center", fontsize=13)
ax.set_xlabel("k"); ax.set_ylabel("맞힌 과일 수 (21개 중)"); ax.set_ylim(0, 23)
save(fig, "kn_loo.png")

# 5. 단위: 단맛을 0~100 점으로
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, sy, t in ((axs[0], 1, "크기 0~10, 단맛 0~10"), (axs[1], 10, "크기 0~10, 단맛 0~100")):
    fruits(ax, sy=sy, legend=False)
    near, vote = knn(PTS, Q, 3, sy=sy)
    for p, l in near:
        ax.plot([Q[0], p[0]], [Q[1] * sy, p[1] * sy], color=GOLD, lw=2.5)
    ax.set_title(f"{t} → k=3 답: {vote}", fontsize=15, color=RED if vote == "사과" else ORANGE)
save(fig, "kn_scale.png")

# 6. 거리 비
ds = sorted(RATIO)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(ds, [RATIO[d] for d in ds], "o-", color=BLUE, lw=2.5); ax.set_xscale("log")
ax.set_xlabel("차원 (로그 눈금)"); ax.set_ylabel("가까운 거리 ÷ 먼 거리"); ax.set_ylim(0, 1)
ax.annotate(f"2차원 {RATIO[2]:.2f}", (2, RATIO[2]), (3, 0.25), fontsize=14, arrowprops=dict(arrowstyle="->"))
ax.annotate(f"1,000차원 {RATIO[1000]:.2f}", (1000, RATIO[1000]), (60, 0.93), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "kn_ratio.png")

# 7. 잡음 특징
fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(NOISE_ACC); v = [NOISE_ACC[k] for k in ks]
b = ax.bar([f"{k:,}" for k in ks], v, color=[BLUE, GRAY, GRAY, RED])
for r_, x in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x + 1.5, f"{x}%", ha="center", fontsize=15, fontweight="bold")
ax.axhline(50, ls="--", color=GRAY); ax.text(3.45, 51.5, "찍기 50%", color=GRAY, fontsize=12, ha="right")
ax.set_xlabel("더한 잡음 특징 수 (원래 특징 2개)"); ax.set_ylabel("kNN 정확도 (%)"); ax.set_ylim(0, 105)
save(fig, "kn_noise.png")
print("ok")
