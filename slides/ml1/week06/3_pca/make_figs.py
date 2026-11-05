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


def project(X, u, mu=MU):
    return mu + np.outer((X - mu) @ u, u)


# 1. 수학 축 그림자 vs PC1 그림자
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, deg, title, col in [(axs[0], 0, f"수학 점수만: 퍼짐 {frac(0) * 100:.0f}%", GRAY), (axs[1], ANG1, f"대각선 (PC1): 퍼짐 {L1 / (L1 + L2) * 100:.0f}%", BLUE)]:
    u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))]); P = project(SCORES, u)
    t = np.array([-45, 45]); ax.plot(*(MU[:, None] + np.outer(u, t)), color=col, lw=2)
    for a, b in zip(SCORES, P): ax.plot(*np.c_[a, b], color="#dddddd", lw=1, zorder=1)
    ax.scatter(*SCORES.T, s=45, color="k", zorder=2); ax.scatter(*P.T, s=40, color=col, zorder=3)
    ax.set_xlim(25, 108); ax.set_ylim(30, 108); ax.set_aspect("equal"); ax.set_xlabel("math"); ax.set_ylabel("physics"); ax.set_title(title, fontsize=15, color=col)
save(fig, "pc_shadow.png")

# 2. 각도별 남는 퍼짐
fig, ax = plt.subplots(figsize=(7.5, 3.8))
d = np.linspace(0, 180, 361); ax.plot(d, [frac(t) * 100 for t in d], color=BLUE, lw=2.5)
for t, lab, c in [(0, "수학 축", GRAY), (90, "물리 축", GRAY), (ANG1, "PC1", BLUE), (ANG1 + 90, "PC2", RED)]:
    ax.plot(t, frac(t) * 100, "o", color=c, ms=8); ax.annotate(f"{lab} {frac(t) * 100:.0f}%", (t, frac(t) * 100), (6, 6), textcoords="offset points", color=c, fontsize=13)
ax.set_xlabel("direction of the shadow (degrees)"); ax.set_ylabel("variance kept (%)"); ax.set_xticks([0, 45, 90, 135, 180]); ax.set_ylim(0, 105)
save(fig, "pc_angle.png")

# 3. PC1 · PC2 화살표
fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.scatter(*SCORES.T, s=45, color="k")
for u, lam, c, lab in [(PC1, L1, BLUE, "PC1"), (PC2, L2, RED, "PC2")]:
    v = u * 2 * np.sqrt(lam); ax.annotate("", MU + v, MU, arrowprops=dict(arrowstyle="->", color=c, lw=3)); ax.text(*(MU + v * 1.12), lab, color=c, fontsize=16, weight="bold", ha="center")
ax.set_xlim(25, 108); ax.set_ylim(30, 108); ax.set_aspect("equal"); ax.set_xlabel("math"); ax.set_ylabel("physics")
save(fig, "pc_axes.png")

# 4. 손글씨 숫자 복원
pca = PCA().fit(DIG); x = DIG[0]; z = pca.transform(x[None])[0]
ks = [2, 5, 10, DIG_K90, 64]
fig, axs = plt.subplots(1, len(ks) + 1, figsize=(11, 2.6))
axs[0].imshow(x.reshape(8, 8), cmap="gray_r"); axs[0].set_title("original", fontsize=13)
for ax, k in zip(axs[1:], ks):
    r = pca.mean_ + z[:k] @ pca.components_[:k]; ax.imshow(r.reshape(8, 8), cmap="gray_r")
    ax.set_title(f"{k} numbers\n{DIG_CUM[k - 1] * 100:.0f}%", fontsize=13)
for ax in axs: ax.axis("off")
save(fig, "pc_digit.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(np.arange(1, 65), DIG_CUM * 100, color=BLUE, lw=2.5); ax.axhline(90, ls=":", color=GRAY)
ax.plot(DIG_K90, DIG_CUM[DIG_K90 - 1] * 100, "o", color=RED); ax.annotate(f"{DIG_K90} components → 90%", (DIG_K90, 90), (10, -30), textcoords="offset points", color=RED, fontsize=13)
ax.plot(2, DIG_CUM[1] * 100, "o", color=GOLD); ax.annotate(f"2 → {DIG_CUM[1] * 100:.1f}%", (2, DIG_CUM[1] * 100), (10, -5), textcoords="offset points", color="#7a6a40", fontsize=13)
ax.set_xlabel("number of components"); ax.set_ylabel("variance kept (%)"); ax.set_ylim(0, 102)
save(fig, "pc_digit_curve.png")

# 5. 붓꽃 2차원
iris = load_iris(); Z = PCA(2).fit_transform(iris.data)
fig, ax = plt.subplots(figsize=(6, 4.2))
for k, (c, nm) in enumerate(zip((BLUE, GOLD, RED), iris.target_names)):
    ax.scatter(*Z[iris.target == k].T, s=25, color=c, label=nm)
ax.set_xlabel(f"PC1 ({IRIS[0]}%)"); ax.set_ylabel(f"PC2 ({IRIS[1]}%)"); ax.legend(frameon=False, fontsize=12)
save(fig, "pc_iris.png")
print("figs ok")
