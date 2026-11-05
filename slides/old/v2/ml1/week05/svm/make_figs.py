"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#999999"
O = "figs/"
X = np.array(RED_PTS + BLUE_PTS, float); y = np.r_[np.ones(10), -np.ones(10)]
xs = np.linspace(0, 10, 2)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def pts(ax, X=X, y=y):
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, zorder=3); ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, marker="s", zorder=3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def street(ax, m, X=X, label=True):
    w, b = m.coef_[0], m.intercept_[0]
    line = lambda c: (c - b - w[0] * xs) / w[1]
    ax.fill_between(xs, line(-1), line(1), color=GOLD, alpha=0.18)
    ax.plot(xs, line(0), color="black", lw=2.5); ax.plot(xs, line(1), "--", color=RED, lw=1.5); ax.plot(xs, line(-1), "--", color=BLUE, lw=1.5)
    sv = X[m.support_]; ax.scatter(sv[:, 0], sv[:, 1], s=320, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)


M = SVC(kernel="linear", C=1e6).fit(X, y)

# 1. 질문: 가르는 선 세 개
fig, ax = plt.subplots(figsize=(5.6, 5.6)); pts(ax)
for (a, c), col in (((1.3, -2.2), GRAY), ((0.55, 1.6), GRAY), ((1.0, 0.25), "black")):
    ax.plot(xs, a * xs + c, color=col, lw=2.2)
ax.text(8.6, 9.3, "?", fontsize=34, color=RED, fontweight="bold")
save(fig, "sv_hook.png")

# 2. 점에 바짝 붙은 선
fig, ax = plt.subplots(figsize=(5.6, 5.6)); pts(ax)
ax.plot(xs, 1.3 * xs - 2.2, color=GRAY, lw=2.5)
newp = (4.2, 3.5)
ax.scatter(*newp, s=160, marker="s", color="white", edgecolor=BLUE, lw=2.5, zorder=4)
ax.annotate("새 파랑 점: 조금만 비껴도\n선의 빨강 쪽으로", newp, (5.2, 6.4), fontsize=13, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
save(fig, "sv_risky.png")

# 3. 가장 넓은 길
fig, ax = plt.subplots(figsize=(5.8, 5.8)); pts(ax); street(ax, M)
ax.annotate("", (2.0 + 1.59, 4.5 - 1.59), (2.0, 4.5), arrowprops=dict(arrowstyle="<->", lw=2))
ax.text(3.2, 3.0, f"마진 {MARGIN}", fontsize=15, fontweight="bold", rotation=-45)
save(fig, "sv_street.png")

# 4. 멀리 있는 점 옮기기
X2 = X.copy(); X2[RED_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
M2 = SVC(kernel="linear", C=1e6).fit(X2, y)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pts(axs[0]); street(axs[0], M); axs[0].set_title("처음", fontsize=16)
pts(axs[1], X2); street(axs[1], M2, X2); axs[1].set_title("멀리 있는 빨강 점을 옮겨도", fontsize=16)
axs[1].annotate("", FAR_MOVE[1], FAR_MOVE[0], arrowprops=dict(arrowstyle="->", lw=2, color=RED))
save(fig, "sv_move.png")

# 5. 소프트 마진: C
Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [-1]]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, C in zip(axs, (100, 0.05)):
    m = SVC(kernel="linear", C=C).fit(Xo, yo); pts(ax, Xo, yo); street(ax, m, Xo)
    ax.set_title(f"C = {C}: 마진 {2 / np.linalg.norm(m.coef_[0]):.2f}, 서포트 벡터 {len(m.support_)}개", fontsize=14)
save(fig, "sv_soft.png")

# 6. 원 모양 데이터와 들어 올리기
Xc, yc = make_circles(120, factor=0.4, noise=0.06, random_state=1)
fig = plt.figure(figsize=(12, 4.6))
ax = fig.add_subplot(1, 3, 1)
ax.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], color=RED, s=25); ax.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], color=BLUE, s=25, marker="s")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("직선으로는 못 가른다", fontsize=15)
ax3 = fig.add_subplot(1, 3, 2, projection="3d")
z = (Xc ** 2).sum(1)
ax3.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], z[yc == 1], color=RED, s=18); ax3.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], z[yc == 0], color=BLUE, s=18, marker="s")
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 2), np.linspace(-1.2, 1.2, 2)); ax3.plot_surface(gx, gy, np.full_like(gx, 0.5), alpha=0.25, color=GOLD)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.set_zlabel("높이 $x_1^2 + x_2^2$", fontsize=12); ax3.set_title("가운데서 멀수록 높이", fontsize=15)
ax = fig.add_subplot(1, 3, 3)
ax.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], color=RED, s=25); ax.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], color=BLUE, s=25, marker="s")
t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.sqrt(0.5) * np.cos(t), np.sqrt(0.5) * np.sin(t), color="black", lw=2.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("내려다보면 경계는 원", fontsize=15)
save(fig, "sv_lift.png")

# 7. RBF 의 γ
Xm, ym = make_moons(150, noise=0.25, random_state=0)
gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.3, 1.8, 250))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, g in zip(axs, (0.1, 2, 100)):
    m = SVC(kernel="rbf", gamma=g, C=1).fit(Xm, ym)
    Z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z > 0, alpha=0.15, colors=[BLUE, RED]); ax.contour(gx, gy, Z, levels=[0], colors="black", linewidths=2)
    ax.scatter(Xm[ym == 1, 0], Xm[ym == 1, 1], color=RED, s=15); ax.scatter(Xm[ym == 0, 0], Xm[ym == 0, 1], color=BLUE, s=15, marker="s")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ = {g}", fontsize=16)
save(fig, "sv_gamma.png")
print("ok")
