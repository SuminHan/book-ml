"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 그림 + 창 + 필터 + 계산
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
axs[0].imshow(IMG, cmap="gray_r"); axs[0].add_patch(Rectangle((J0 - 0.5, I0 - 0.5), 3, 3, fill=False, ec=RED, lw=3)); axs[0].set_title("손글씨 7 (28 × 28)", fontsize=14); axs[0].axis("off")
win = IMG[I0:I0 + 3, J0:J0 + 3]
for ax, M, t in ((axs[1], win, "빨간 창의 픽셀"), (axs[2], KH, "가로선 필터 (3 × 3)")):
    ax.imshow(M, cmap="coolwarm" if M is KH else "gray_r", vmin=-1 if M is KH else 0, vmax=1); ax.set_title(t, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{M[i, j]:.1f}" if M is win else f"{int(M[i, j]):+d}", ha="center", va="center", fontsize=15, color="black" if M is KH or M[i, j] < 0.5 else "white")
save(fig, "cn_window.png")

# 2. 특징 지도
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, M, t in ((axs[0], IMG, "입력 28 × 28"), (axs[1], MH, "가로선 필터 → 26 × 26"), (axs[2], MV, "세로선 필터 → 26 × 26")):
    ax.imshow(M, cmap="gray_r" if M is IMG else "magma"); ax.set_title(t, fontsize=14); ax.axis("off")
save(fig, "cn_maps.png")

# 3. 풀링
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
axs[0].imshow(MH, cmap="magma"); axs[0].set_title("26 × 26", fontsize=14); axs[0].axis("off")
axs[1].imshow(PH, cmap="magma"); axs[1].set_title("2 × 2 최대 풀링 → 13 × 13", fontsize=14); axs[1].axis("off")
save(fig, "cn_pool.png")

# 4. 학습된 필터
F = np.load("assets/filters.npy")
fig, axs = plt.subplots(1, 8, figsize=(13, 2.2))
for ax, f in zip(axs, F):
    ax.imshow(f, cmap="coolwarm", vmin=-abs(F).max(), vmax=abs(F).max()); ax.set_xticks([]); ax.set_yticks([])
save(fig, "cn_filters.png")

# 5. 비교
fig, ax = plt.subplots(figsize=(9, 3.0))
names = ["완전연결", "작은 CNN"]
v = [CMP["mlp"][0], CMP["cnn"][0]]
b = ax.barh(names, v, color=[RED, GREEN])
for r, x, a in zip(b, v, (CMP["mlp"][1], CMP["cnn"][1])): ax.text(x + 1500, r.get_y() + r.get_height() / 2, f"{x:,}개 · 정확도 {a}%", va="center", fontsize=14)
ax.set_xlabel("가중치 수 (MNIST, 2 에폭 학습)"); ax.set_xlim(0, 175000); ax.ticklabel_format(axis="x", style="plain")
save(fig, "cn_compare.png")

# 6. 1차원 합성곱 f * g (f = 상자, g = 내려가는 경사, 둘 다 [0, 1])
def f_box(t): return ((t >= 0) & (t <= 1)).astype(float)
def g_ramp(t): return np.where((t >= 0) & (t <= 1), 1 - t, 0.0)
tau = np.linspace(-1.2, 2.6, 1500)
fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(2, 4, width_ratios=[0.8, 1, 1, 1], height_ratios=[1, 1.15], hspace=0.55, wspace=0.25)
for r, (y, c, t) in enumerate(((f_box(tau), BLUE, "f(τ)"), (g_ramp(tau), RED, "g(τ)"))):
    ax = fig.add_subplot(gs[r, 0]); ax.plot(tau, y, color=c, lw=3); ax.set_title(t, fontsize=16, color=c)
    ax.set_ylim(-0.1, 1.25); ax.set_yticks([0, 1]); ax.set_xticks([0, 1]); ax.set_xlim(-0.6, 1.8)
cs = [GOLD, GREEN, "#1f77b4"]
axc = fig.add_subplot(gs[1, 1:]); axc.plot(TS, CONV1D, color="black", lw=3)
for k, (t0, v0) in enumerate(zip(T_SNAP, V_SNAP)):
    ax = fig.add_subplot(gs[0, k + 1]); fy, gy = f_box(tau), g_ramp(t0 - tau)
    ax.fill_between(tau, np.minimum(fy, gy), color=cs[k], alpha=0.8, lw=0)
    ax.plot(tau, fy, color=BLUE, lw=2.5); ax.plot(tau, gy, color=RED, lw=2.5)
    ax.set_title(f"t = {t0}:  area = {v0}", fontsize=15, color=cs[k]); ax.set_ylim(-0.1, 1.25); ax.set_xlim(-1.1, 2.1)
    ax.set_yticks([0, 1]); ax.set_xticks([0, 1]); ax.set_xlabel("τ", fontsize=14)
    axc.plot([t0, t0], [0, v0], color=cs[k], lw=3); axc.plot(t0, v0, "o", color=cs[k], ms=10)
    axc.text(t0 - 0.05 if k == 0 else t0 + 0.04, v0 + 0.03, f"{v0}", color=cs[k], fontsize=15, ha="right" if k == 0 else "left")
axc.set_title("(f * g)(t) = shaded area", fontsize=16); axc.set_xlabel("t", fontsize=14)
axc.set_xlim(-1.1, 2.1); axc.set_ylim(-0.03, 0.65); axc.set_yticks([0, 0.5]); axc.set_xticks([0, 1, 2])
fig.savefig(O + "cn_conv1d.png", bbox_inches="tight"); plt.close(fig)

# 7. 필터 뒤집기: K 와 180° 돌린 K
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, M, t in ((axs[0], KH, "K"), (axs[1], KH_FLIP, "K flipped (180°)")):
    ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1); ax.set_title(t, fontsize=16); ax.axis("off")
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{int(M[i, j]):+d}", ha="center", va="center", fontsize=16)
save(fig, "cn_flip.png")
print("ok")
