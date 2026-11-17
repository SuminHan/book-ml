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
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.tick_params(labelsize=11, length=3); ax.set_xlabel("$x_1$", fontsize=13, labelpad=1); ax.set_ylabel("$x_2$", fontsize=13, rotation=0, labelpad=8)


def origin_axes(ax):
    """원점 (0, 0) 을 지나는 x1 · x2 축 (화살표 끝 + 이름)"""
    ax.spines["left"].set_position("zero"); ax.spines["bottom"].set_position("zero")
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, ms=7)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms=7)
    ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1]); ax.tick_params(labelsize=12)
    ax.text(1.0, 0.0, "  $x_1$", transform=ax.get_yaxis_transform(), va="center", fontsize=16)
    ax.text(0.0, 0.97, "  $x_2$", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=16)


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
_a = np.array([2.0, 4.5]); _b = _a + (2 / np.linalg.norm(M.coef_[0])) * np.array([1, -1]) / np.sqrt(2)   # 빨강 가장자리 위 서포트 벡터에서 파랑 가장자리까지
assert abs(M.decision_function([_b])[0] + 1) < 1e-2
ax.annotate("", _b, _a, arrowprops=dict(arrowstyle="<->", lw=2, shrinkA=0, shrinkB=0))
ax.text(_b[0] + 0.15, _b[1] - 0.15, f"길 폭 2m = {MARGIN}", fontsize=15, fontweight="bold", ha="left", va="top")
save(fig, "sv_street.png")

# 4. 멀리 있는 점 옮기기
X2 = X.copy(); X2[RED_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
M2 = SVC(kernel="linear", C=1e6).fit(X2, y)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pts(axs[0]); street(axs[0], M); axs[0].set_title("처음", fontsize=16)
pts(axs[1], X2); street(axs[1], M2, X2); axs[1].set_title("멀리 있는 빨강 점을 옮겨도", fontsize=16)
axs[1].annotate("", FAR_MOVE[1], FAR_MOVE[0], arrowprops=dict(arrowstyle="->", lw=2, color=RED, shrinkA=9, shrinkB=9))
axs[1].add_patch(plt.Circle(FAR_MOVE[0], 0.32, fill=False, ls="--", lw=2, ec=RED, zorder=4)); axs[1].text(FAR_MOVE[0][0] + 0.45, FAR_MOVE[0][1] - 0.15, "원래 자리", fontsize=12, color=RED)
save(fig, "sv_move.png")

# 5. 소프트 마진: C
Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [-1]]
fig, axs = plt.subplots(1, 3, figsize=(14, 5.6))
for ax, C in zip(axs, (100, 1, 0.05)):
    m = SVC(kernel="linear", C=C).fit(Xo, yo); pts(ax, Xo, yo); street(ax, m, Xo)
    bud, mg, nsv = BUDGET[C]
    ax.set_title(f"예산 B = {bud:.2f}  (sklearn C = {C})\nm = {mg:.2f}, 서포트 벡터 {nsv}개", fontsize=14)
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


# 가장자리 ±1 은 약속: 같은 경계 x1 + x2 = 3 을 (w, b) = ((1,1), -3) 과 2배 ((2,2), -6) 로 쓸 때
SP = np.array([(2, 2), (3, 2.4), (2.3, 3.3), (3.5, 3.4)], float)        # +1 (red)
SN = np.array([(1, 1), (0.2, 0.3), (1.0, -0.1), (-0.2, 1.1)], float)    # -1 (blue)
LAB = {(1, 1): (-0.55, 0.3), (2, 2): (0.2, -0.35), (0.2, 0.3): (-0.15, -0.42), (3.5, 3.4): (-0.05, -0.45)}
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, (k, ttl) in zip(axs, ((1, "$\\mathbf{w} = (1, 1),\\ b = -3$\n가장 가까운 점에서 값이 딱 ±1"),
                              (2, "2배: $\\mathbf{w} = (2, 2),\\ b = -6$ (같은 경계)\n값도 2배, ±1 선이 점에서 떨어진다"))):
    g = np.linspace(-0.6, 4.6, 2)
    ax.fill_between(g, -1, 3 - g, color=BLUE, alpha=0.06); ax.fill_between(g, 3 - g, 6, color=RED, alpha=0.06)
    ax.plot(g, 3 - g, color="black", lw=2.5)
    for c in (-1, 1):
        ax.plot(g, (3 + c / k) - g, color=GRAY, lw=1.8, ls="--")
    ax.scatter(SP[:, 0], SP[:, 1], s=90, color=RED, zorder=3); ax.scatter(SN[:, 0], SN[:, 1], s=90, color=BLUE, marker="s", zorder=3)
    for (px, py), (dx, dy) in LAB.items():
        v = k * (px + py - 3)
        ax.text(px + dx, py + dy, f"{v:+g}", fontsize=16, fontweight="bold", color=RED if v > 0 else BLUE)
    for c, lab, col in ((3, "값 0", "black"), (3 + 1 / k, "값 +1", GRAY), (3 - 1 / k, "값 −1", GRAY)):
        ax.text(c / 2 - 1.3, c / 2 + 1.3 + 0.06, lab, fontsize=12, color=col, rotation=-45, ha="center", va="bottom")
    ax.set_xlim(-0.5, 4.2); ax.set_ylim(-0.5, 4.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(ttl, fontsize=15)
axs[0].annotate("", (1.7, 2.3), (1.2, 1.8), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=18))
axs[0].text(1.05, 2.3, "$\\mathbf{w}$", fontsize=19, color=GOLD, fontweight="bold")
axs[0].annotate("", (3.1, 0.9), (2.1, -0.1), arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
axs[0].text(2.95, 1.0, "폭 $= 2/\\Vert\\mathbf{w}\\Vert = \\sqrt{2}$", fontsize=14, bbox=dict(fc="white", ec="none", pad=1.5))
save(fig, "sv_scale.png")


# 방향을 돌려 가며 폭 재기 (서포트 벡터가 3개만 남는 과정)
def sweep_street(ax, deg, R=RED_PTS, Bl=BLUE_PTS, width=True):
    wd, cr, cb, tr, tb = sweep(deg, R, Bl)
    t = np.radians(deg + 90); n = np.array([np.cos(t), np.sin(t)]); d = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    s = np.linspace(-20, 20, 2)
    seg = lambda c: c * n[:, None] + d[:, None] * s
    a, b = seg(cr), seg(cb)
    ax.fill(np.r_[a[0], b[0][::-1]], np.r_[a[1], b[1][::-1]], color=GOLD, alpha=0.18, lw=0)
    ax.plot(*seg((cr + cb) / 2), color="black", lw=2.5); ax.plot(*a, "--", color=RED, lw=1.5); ax.plot(*b, "--", color=BLUE, lw=1.5)
    tp = np.r_[tr, tb]; ax.scatter(tp[:, 0], tp[:, 1], s=320, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
    if width:
        ax.set_title(f"기울기 {deg}° · 폭 {wd:.2f}", fontsize=16)
    return wd


fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, dg in zip(axs, (25, 45, 65)):
    pts(ax); sweep_street(ax, dg)
axs[1].title.set_color(RED); axs[1].title.set_fontweight("bold")
save(fig, "sv_sweep3.png")

fig, ax = plt.subplots(figsize=(8, 4.6))
dd = np.linspace(5, 85, 801); ww = [sweep(v)[0] for v in dd]
ax.plot(dd, ww, color="black", lw=2.5)
for dg, wd in SWEEP.items():
    ax.scatter(dg, wd, s=90, color=RED if dg == 45 else GRAY, zorder=3)
    ax.text(dg, wd + 0.18, f"{wd:.2f}", ha="center", fontsize=14, color=RED if dg == 45 else "black", fontweight="bold" if dg == 45 else None)
ax.set_xlabel("선의 기울기 (°)"); ax.set_ylabel("길의 폭"); ax.set_xticks([5, 25, 45, 65, 85]); ax.set_ylim(0, 3.8)
save(fig, "sv_sweep_curve.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pts(axs[0]); sweep_street(axs[0], 45, width=False); axs[0].set_title("점 20개로 풀면", fontsize=16)
S = np.array(SV, float); ys = np.array([1, 1, -1])
pts(axs[1], S, ys); sweep_street(axs[1], 45, SV[:2], SV[2:], width=False); axs[1].set_title("점 3개만 남기고 풀어도", fontsize=16)
save(fig, "sv_only3.png")
print("ok sweep")


# 왜 폭 = 2/||w|| 인가: 같은 예 x1 + x2 = 3, w = (1, 1), b = -3
WE, BE = np.array([1.0, 1.0]), -3.0
NW = np.linalg.norm(WE); U = WE / NW; X0 = np.array([1.5, 1.5])
val = lambda p: WE @ p + BE
g = np.linspace(-0.6, 4.6, 2)


def frame_ax(ax):
    ax.set_xlim(0, 4.0); ax.set_ylim(0, 4.0); ax.set_aspect("equal"); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.tick_params(labelsize=12)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


# (가) w 방향으로 걸으면 값이 일정하게 오른다
fig, ax = plt.subplots(figsize=(6.2, 6.0)); frame_ax(ax)
ax.plot(g, 3 - g, color="black", lw=2.5); ax.text(0.15, 3.05, "값 0 (경계)", fontsize=13, rotation=-45)
for d in (0, 0.5, 1.0, 1.5):
    p = X0 + d * U; v = val(p); assert abs(v - d * NW) < 1e-12
    ax.scatter(*p, s=90, color=GOLD if d else "black", zorder=4)
    ax.text(p[0] + 0.12, p[1] - 0.28, f"d = {d:g}  →  값 {v:.2f}", fontsize=14, color=RED if d else "black", fontweight="bold" if d == 1 else None)
ax.annotate("", X0 + 1.75 * U, X0, arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=18))
ax.text(*(X0 + 1.8 * U + np.array([-0.35, 0.1])), "$\\mathbf{w}$ 방향", fontsize=15, color=GOLD, fontweight="bold")
ax.set_title("$\\mathbf{w} = (1, 1),\\ b = -3$:  1 걸을 때마다 값이 $\\Vert\\mathbf{w}\\Vert = \\sqrt{2} = 1.41$ 오른다", fontsize=13)
save(fig, "sv_walk.png")

# (나) 값 -1, 0, +1 선 사이 간격 = 1/||w||
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, k in zip(axs, (1, 2)):
    frame_ax(ax); nw = k * NW
    for c in (-1, 0, 1):
        ax.plot(g, (3 + c / k) - g, color="black" if c == 0 else GRAY, lw=2.5 if c == 0 else 1.8, ls="-" if c == 0 else "--")
        q = X0 + (c / nw + 0.04) * U - 1.3 * np.array([1, -1]) / np.sqrt(2)
        ax.text(*q, f"값 {c:+d}" if c else "값 0", fontsize=13, rotation=-45, ha="center", va="bottom", rotation_mode="anchor",
                color="black" if c == 0 else "#555555", fontweight="bold" if c == 0 else None)
    a, b = X0 - U / nw, X0 + U / nw
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    ax.text(b[0] + 0.15, b[1] - 0.05, f"폭 $2/\\Vert\\mathbf{{w}}\\Vert$ = {2 / nw:.2f}", fontsize=15, color=RED, fontweight="bold")
    ax.set_title(("$\\mathbf{w} = (1, 1)$,  $\\Vert\\mathbf{w}\\Vert = 1.41$" if k == 1 else "$\\mathbf{w} = (2, 2)$,  $\\Vert\\mathbf{w}\\Vert = 2.83$"), fontsize=15)
save(fig, "sv_width.png")
print("ok width")


# 식 읽기 (벡터를 처음 보는 학생용): w = (1, 1), b = -3 같은 예
def grid_ax(ax, lim=4.3):
    ax.set_xlim(-0.3, lim); ax.set_ylim(-0.3, lim); ax.set_aspect("equal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$", rotation=0, labelpad=12)


# ① 점 = 숫자 두 개 = 화살표, 길이 = 피타고라스
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
ax = axs[0]; grid_ax(ax)
ax.annotate("", (2, 3), (0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5, mutation_scale=18))
ax.scatter(2, 3, s=90, color=RED, zorder=4); ax.text(2.15, 3.1, "$\\mathbf{x} = (2, 3)$", fontsize=17, color=RED)
ax.plot([2, 2], [0, 3], ":", color=GRAY); ax.plot([0, 2], [3, 3], ":", color=GRAY)
ax.set_title("점 하나 = 숫자 두 개 = 화살표", fontsize=16)
ax = axs[1]; grid_ax(ax, 2.3); ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.annotate("", (1, 1), (0, 0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=20))
ax.plot([0, 1], [0, 0], color=GRAY, lw=2); ax.plot([1, 1], [0, 1], color=GRAY, lw=2)
ax.text(0.5, -0.2, "1", ha="center", fontsize=15); ax.text(1.08, 0.5, "1", fontsize=15)
ax.text(0.0, 1.75, "길이 $= \\sqrt{1^2 + 1^2} = 1.41$", fontsize=16, color="#7a6a3f")
ax.text(1.08, 1.12, "$\\mathbf{w} = (1, 1)$", fontsize=17, color=GOLD)
ax.set_title("화살표의 길이 $\\Vert\\mathbf{w}\\Vert$ = 피타고라스", fontsize=16)
save(fig, "sv_vec.png")

# ③ 점수 지도: 격자점마다 x1 + x2 - 3
fig, ax = plt.subplots(figsize=(6.0, 6.0)); grid_ax(ax)
gg = np.linspace(-0.5, 4.5, 2)
ax.fill_between(gg, -1, 3 - gg, color=BLUE, alpha=0.07); ax.fill_between(gg, 3 - gg, 6, color=RED, alpha=0.07)
ax.plot(gg, 3 - gg, color="black", lw=2.5)
for i in range(5):
    for j in range(5):
        v = i + j - 3; col = RED if v > 0 else BLUE if v < 0 else "black"
        ax.text(i, j, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=15, fontweight="bold", color=col,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=col, lw=1.2))
ax.set_title("격자점마다 $x_1 + x_2 - 3$ 계산", fontsize=15)
save(fig, "sv_scoremap.png")

# ④ 선을 따라 걸으면 점수 그대로, w 방향으로 걸으면 오른다
fig, ax = plt.subplots(figsize=(6.0, 6.0)); grid_ax(ax)
ax.plot(gg, 3 - gg, color="black", lw=2.5)
P0 = np.array([1.0, 2.0])
for d, col, lab, off in (((1, -1), GRAY, "선을 따라 $(+1, -1)$\n점수 변화 $1{\\cdot}1 + 1{\\cdot}(-1) = 0$", (0.1, 0.25)),
                         ((1, 1), GOLD, "$\\mathbf{w}$ 방향 $(+1, +1)$\n점수 변화 $1{\\cdot}1 + 1{\\cdot}1 = +2$", (0.15, 0.2))):
    q = P0 + np.array(d)
    ax.annotate("", q, P0, arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=18))
    ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=13, color="black" if col == GRAY else "#7a6a3f")
ax.scatter(*P0, s=90, color="black", zorder=4); ax.text(P0[0] - 0.95, P0[1] - 0.05, "점수 0", fontsize=13)
ax.set_title("$\\mathbf{w}$ 는 경계에 수직인 오르막 방향", fontsize=15)
save(fig, "sv_perp.png")
print("ok read")


# 학습 (‖w‖ = 1): 가장 가까운 점이 민다 → 길이 1 로
def street_m(ax, w, b, m, ring=(), ring_col=GOLD):
    """‖w‖ = 1 인 경계와, 경계에서 거리 m 인 길 가장자리 두 줄"""
    s = np.linspace(-30, 30, 2); u = w / np.linalg.norm(w); d = np.array([-u[1], u[0]]); b = b / np.linalg.norm(w)
    seg = lambda c: (c - b) * u[:, None] + d[:, None] * s
    if m > 0:
        a, c = seg(m), seg(-m)
        ax.fill(np.r_[a[0], c[0][::-1]], np.r_[a[1], c[1][::-1]], color=GOLD, alpha=0.18, lw=0)
        ax.plot(*a, "--", color=RED, lw=1.5); ax.plot(*c, "--", color=BLUE, lw=1.5)
    ax.plot(*seg(0), color="black", lw=2.5)
    if len(ring):
        r = np.array(ring); ax.scatter(r[:, 0], r[:, 1], s=320, facecolors="none", edgecolors=ring_col, lw=2.5, zorder=4)


TR4 = {2: "1걸음 뒤: 첫 선", 3: "2걸음 뒤: 너무 돌았다", 141: "140걸음 뒤", 20000: "20,000걸음"}
wt, bt, pushed, shots = train_maxmin(X, y, snaps=tuple(TR4))
fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, (t, ttl) in zip(axs, TR4.items()):
    w_, b_, act, m_ = shots[t]; pts(ax); street_m(ax, w_, b_, m_, X[act])
    ax.set_title(f"{ttl}  m {m_:.2f}", fontsize=15)
save(fig, "sv_train4.png")

mg = y * (X @ wt + bt); o = np.argsort(mg); Mn = mg.min()
fig, ax = plt.subplots(figsize=(10, 4.2))
cols = [GOLD if mg[i] < Mn + 0.02 else (RED if y[i] > 0 else BLUE) for i in o]
ax.bar(range(20), mg[o], color=cols, alpha=0.9)
ax.axhline(Mn, color="black", ls="--", lw=1.5); ax.text(-0.4, 5.6, f"점선: m = {Mn:.2f} = 길 가장자리", fontsize=14)
ax.text(1, Mn + 0.45, "서포트 벡터", ha="center", fontsize=14, color="#7a6a3f", fontweight="bold")
for k, i in enumerate(o[:3]):
    ax.text(k, mg[i] + 0.12, f"{mg[i]:.2f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([]); ax.set_ylabel("경계까지 거리"); ax.set_xlabel("점 20개 (거리 작은 순)")
save(fig, "sv_slack_bars.png")
print("ok train")


# 힌지 손실: 여유 m → 벌점 max(0, 1 - m)
fig, ax = plt.subplots(figsize=(5.4, 4.3))
m = np.linspace(-1.5, 3, 400)
ax.fill_between([1, 3], 0, 2.7, color=GOLD, alpha=0.12); ax.fill_between([-1.5, 1], 0, 2.7, color=RED, alpha=0.06)
ax.plot(m, np.maximum(0, 1 - m), color="black", lw=3)
ax.axvline(1, color=GRAY, ls="--", lw=1.5)
ax.text(2.0, 1.2, "여유 ≥ 1\n벌점 0\n→ 가만히", ha="center", fontsize=19, color="#7a6a3f", fontweight="bold")
ax.text(-0.45, 0.25, "여유 < 1\n모자란 만큼\n벌점", ha="center", fontsize=18, color=RED, fontweight="bold")
ax.set_xlim(-1.5, 3); ax.set_ylim(0, 2.7); ax.set_xticks([-1, 0, 1, 2, 3])
ax.set_xlabel("여유", fontsize=18); ax.set_ylabel("벌점", fontsize=18)
save(fig, "sv_hinge.png")
print("ok hinge")


# ---- CME 250 Lecture 5 흐름에 맞춘 그림 (초평면 · 약점 · 슬랙 · 커널 비교)
# 초평면: 2차원 = 직선, 3차원 = 평면
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_subplot(1, 2, 1)
g2 = np.linspace(-1.5, 1.5, 2); ax.plot(g2, (-1 - 2 * g2) / 3, color="black", lw=3)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); origin_axes(ax)
ax.set_title("2차원: 초평면 = 직선 (1차원)", fontsize=16)
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
gx, gy = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)); ax3.plot_surface(gx, gy, 0.3 - 0.5 * gx + 0.4 * gy, alpha=0.45, color=GOLD)
ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_zlabel("$x_3$")
ax3.set_title("3차원: 초평면 = 평면 (2차원)", fontsize=16)
save(fig, "sv_hyper.png")

# 반으로 나누기: 1 + 2 x1 + 3 x2
c0, c1, c2 = HYPER
fig, ax = plt.subplots(figsize=(5.8, 5.6))
gg = np.linspace(-1.5, 1.5, 300); G1, G2 = np.meshgrid(gg, gg); F = c0 + c1 * G1 + c2 * G2
ax.contourf(G1, G2, F > 0, levels=[-0.5, 0.5, 1.5], colors=[BLUE, RED], alpha=0.12)
ax.plot(gg, (-c0 - c1 * gg) / c2, color="black", lw=3)
for (px, py) in ((0.8, 0.6), (-0.9, -0.7), (0.5, -0.6)):
    v = c0 + c1 * px + c2 * py
    ax.scatter(px, py, s=70, color="black", zorder=3)
    ax.text(px + 0.07, py + 0.08, f"{v:+.1f}", fontsize=15, fontweight="bold", color=RED if v > 0 else BLUE)
ax.text(0.3, 1.2, "$1 + 2x_1 + 3x_2 > 0$", fontsize=15, color=RED); ax.text(-1.4, -1.35, "$1 + 2x_1 + 3x_2 < 0$", fontsize=15, color=BLUE)
ax.annotate("$1 + 2x_1 + 3x_2 = 0$\n(직선)", (-0.95, 0.3), (-1.45, 0.75), fontsize=14, weight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); origin_axes(ax)
save(fig, "sv_half.png")

# 최대 마진의 약점: 겹치면 불가 · 점 하나에 민감
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pts(axs[0], Xo, yo); axs[0].scatter(*OUTLIER, s=380, facecolors="none", edgecolors=RED, lw=2.5, zorder=4)
axs[0].set_title("겹치면: 가르는 직선이 없다", fontsize=16)
Xn = np.r_[X, [NEWPT]]; yn = np.r_[y, -1]; Mn = SVC(kernel="linear", C=1e6).fit(Xn, yn)
pts(axs[1], Xn, yn); street(axs[1], Mn, Xn)
w0_, b0_ = M.coef_[0], M.intercept_[0]; axs[1].plot(xs, -(b0_ + w0_[0] * xs) / w0_[1], color=GRAY, lw=2, ls=":")
axs[1].scatter(*NEWPT, s=380, facecolors="none", edgecolors=RED, lw=2.5, zorder=5)
axs[1].set_title(f"파란 점 하나 추가: m {M_HARD} → {M_NEW}", fontsize=16)
save(fig, "sv_sensitive.png")

# 슬랙 ε: 예산 안에서 가장자리를 넘는 점
ms = SVC(kernel="linear", C=0.05).fit(Xo, yo); eps = np.maximum(0, 1 - yo * ms.decision_function(Xo))
fig, ax = plt.subplots(figsize=(6.0, 6.0)); pts(ax, Xo, yo); street(ax, ms, Xo)
for i in np.where(eps > 1e-3)[0]:
    col = RED if eps[i] > 1 else "#e8590c"
    tx = {(3.5, 6.0): (5.0, 8.9), (2.5, 5.5): (0.3, 9.4), (3.0, 5.2): (4.4, 4.4), (2.0, 4.5): (0.2, 2.6),
          (6.5, 3.5): (8.0, 6.6), (5.0, 3.0): (2.6, 1.3)}.get(tuple(Xo[i]))
    ax.annotate(f"ε {eps[i]:.2f}", Xo[i], tx, fontsize=13, fontweight="bold", color=col, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1, alpha=0.9), arrowprops=dict(arrowstyle="-", color=col, lw=1.5))
ax.set_title(f"ε > 0: 가장자리를 넘음 · ε > 1: 틀림\nΣε = {eps.sum():.2f} (예산 B)", fontsize=14)
save(fig, "sv_slack.png")

# 커널 비교: 3차 다항 vs RBF (같은 데이터)
Xk, yk = make_circles(160, factor=0.45, noise=0.12, random_state=3)
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (ttl, mk) in zip(axs, (("선형 커널", SVC(kernel="linear", C=1)), ("3차 다항 커널", SVC(kernel="poly", degree=3, coef0=1, C=1)), ("RBF 커널", SVC(kernel="rbf", gamma=1, C=1)))):
    mk.fit(Xk, yk); Z = mk.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z > 0, alpha=0.15, colors=[BLUE, RED]); ax.contour(gx, gy, Z, levels=[0], colors="black", linewidths=2.5)
    ax.scatter(Xk[yk == 1, 0], Xk[yk == 1, 1], color=RED, s=16); ax.scatter(Xk[yk == 0, 0], Xk[yk == 0, 1], color=BLUE, s=16, marker="s")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(f"{ttl}: 학습 정확도 {mk.score(Xk, yk):.0%}", fontsize=15)
save(fig, "sv_kernels.png")
print("ok cme")


# 값과 거리: 같은 직선을 두 식으로 — 길이 1 로 나누면 값 = 거리
fig, ax = plt.subplots(figsize=(6.2, 5.8)); frame_ax(ax)
ax.plot(g, 3 - g, color="black", lw=2.5)
P1 = X0 + 1.0 * U
ax.annotate("", P1, X0, arrowprops=dict(arrowstyle="<->", color=RED, lw=2.5, shrinkA=0, shrinkB=0))
ax.text(*(X0 + 0.5 * U + np.array([-0.62, 0.12])), "거리 1", fontsize=16, color=RED, fontweight="bold")
ax.scatter(*P1, s=110, color=RED, zorder=4); ax.scatter(*X0, s=70, color="black", zorder=4)
v1, v2 = val(P1), (WE / NW) @ P1 + BE / NW
assert abs(v1 - NW) < 1e-9 and abs(v2 - 1) < 1e-9
ax.text(P1[0] - 0.2, P1[1] + 0.95, f"$x_1 + x_2 - 3$  →  {v1:.2f}", fontsize=15, color="#555555",
        bbox=dict(fc="white", ec=GRAY, pad=3))
ax.text(P1[0] - 0.2, P1[1] + 0.45, f"$0.71x_1 + 0.71x_2 - 2.12$  →  {v2:.2f}", fontsize=15, color=RED, fontweight="bold",
        bbox=dict(fc="white", ec=RED, pad=3))
ax.text(0.1, 3.1, "같은 직선", fontsize=14, rotation=-45)
ax.set_xlim(0, 4.6)
save(fig, "sv_dist.png")
print("ok dist")


# 참고: 점곱 = ‖β‖‖x‖cos θ → ‖β‖ = 1 이면 β 방향 그림자, 경계의 그림자 −β0 를 빼면 거리
Bu, B0u = WE / NW, BE / NW                                  # (0.71, 0.71), −2.12
PX = np.array([1.0, 3.6])
sh, shb = Bu @ PX, -B0u                                     # 그림자 3.89, 경계 2.12
dist = sh - shb; assert abs(dist - (Bu @ PX + B0u)) < 1e-12 and abs(dist - 1.13) < 0.01 and abs(sh - 3.25) < 0.01
th = np.degrees(np.arccos(sh / np.linalg.norm(PX)))
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.set_xlim(-0.9, 4.6); ax.set_ylim(-1.3, 4.3); ax.set_aspect("equal"); ax.grid(alpha=0.25)
t = np.linspace(-0.9, 4.3, 2); ax.plot(t, t, color=GOLD, lw=1.5, ls="--")             # β 방향 축
tt = np.linspace(-0.9, 4.3, 2); ax.plot(tt, 3 - tt, color="black", lw=2.5)                  # 경계
F = sh * Bu; Fb = shb * Bu
ax.annotate("", PX, (0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5, mutation_scale=16))
ax.annotate("", Bu, (0, 0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3.5, mutation_scale=18))
# 치수선: β 축에서 오른쪽 아래로 띄워 그린다 (원점 · 경계 · x 의 그림자 끝에 눈금)
nrm = np.array([1, -1]) / np.sqrt(2)
def dim(a, b, off, col, lab, lw=2.5):
    pa, pb = a * Bu + off * nrm, b * Bu + off * nrm
    ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="<->", color=col, lw=lw, shrinkA=0, shrinkB=0))
    for q, base in ((pa, a * Bu), (pb, b * Bu)):
        ax.plot(*zip(base, q + 0.08 * nrm), color=col, lw=1, ls=":")
    mid = (pa + pb) / 2 + 0.22 * nrm
    ax.text(*mid, lab, fontsize=13, color=col, rotation=45, ha="center", va="center", fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
dim(0, sh, 0.95, BLUE, f"x 의 w 방향 위치 $\\Vert\\mathbf{{x}}\\Vert\\cos\\theta$ = {sh:.2f}")
dim(0, shb, 0.45, "#555555", f"경계의 위치 {shb:.2f} = $-b$")
ax.annotate("", F, Fb, arrowprops=dict(arrowstyle="<->", color="#2b8a3e", lw=3, shrinkA=0, shrinkB=0))
ax.plot(*zip(PX, F), color=GRAY, lw=1.5, ls=":")
ax.scatter(*PX, s=80, color=RED, zorder=4); ax.scatter(*Fb, s=50, color="black", zorder=4); ax.scatter(0, 0, s=50, color="black", zorder=5); ax.text(-0.35, -0.3, "(0, 0)", fontsize=13)
ax.text(PX[0] + 0.08, PX[1] + 0.1, "$\\mathbf{x} = (1, 3.6)$", fontsize=15, color=RED)
ax.text(-0.2, 0.55, "$\\mathbf{w}$", fontsize=16, color="#7a6a3f", fontweight="bold")
ax.text(0.4, 0.6, "θ", fontsize=16)
ax.text(0.95, 2.3, f"거리 {dist:.2f}", fontsize=15, color="#2b8a3e", fontweight="bold")
origin_axes(ax); ax.set_xticks([1, 2, 3, 4]); ax.set_yticks([1, 2, 3, 4])
save(fig, "sv_cos.png")
print("ok cos", round(th, 1))


# 학습: 걸음마다 m (가장 가까운 점까지 거리) 이 어떻게 바뀌나
T = np.unique(np.logspace(0, np.log10(MM_T), 300).astype(int)); T = T[T >= 2]
_, _, _, sha = train_maxmin(X, y, snaps=tuple(T))
mm = np.array([sha[t][3] for t in T])
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(T, mm, color="black", lw=2); ax.set_xscale("log")
ax.axhline(1.59, color=GOLD, ls="--", lw=1.5); ax.text(2.2, 1.66, "SVM 답 1.59", fontsize=12, color="#7a6a3f")
ax.annotate("2걸음 뒤: 너무 돌아서 (7.8, 5) 가 가장 가까움", (3, sha[3][3]), (6, 0.25), fontsize=12, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
ax.scatter(3, sha[3][3], s=70, color=BLUE, zorder=4)
ax.set_xticks([2, 10, 100, 1000, 10000, 20000]); ax.set_xticklabels(["2", "10", "100", "1,000", "10,000", "20,000"])
ax.set_xlabel("걸음 (로그 눈금)", fontsize=13); ax.set_ylabel("m (가장 가까운 점까지 거리)", fontsize=13)
ax.set_ylim(0, 1.9); ax.tick_params(labelsize=12)
save(fig, "sv_trainpath.png")


def train_anim(X, y, path="figs/sv_train_anim.mp4", fps=20):
    """경사하강 2만 걸음을 로그 간격 프레임으로: 길 · 미는 점 · w, b · 폭"""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    F = np.unique(np.r_[np.arange(1, 30), np.logspace(np.log10(30), np.log10(TRAIN_T), 230).astype(int)])
    F = np.r_[F, [TRAIN_T] * 40]
    _, _, _, sh = train_svm(X, y, snaps=tuple(F))
    seen = np.zeros(len(X), bool)
    fig, ax = plt.subplots(figsize=(7.2, 8)); fig.subplots_adjust(top=0.80, bottom=0.08, left=0.1, right=0.97)
    def draw(i):
        t = F[i]; w_, b_, act = sh[t]; ax.clear(); pts(ax, X, y)
        if t > 1:
            seen[act] = True
        if np.linalg.norm(w_) > 0:
            street_wb(ax, w_, b_, X[act])
            wtxt = f"폭 {2 / np.linalg.norm(w_):.2f}"
        else:
            ax.scatter(X[:, 0], X[:, 1], s=320, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4); wtxt = "선 없음 (w = 0)"
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        fig.texts.clear()
        fig.text(0.5, 0.95, f"{t:,}걸음", ha="center", fontsize=24, weight="bold")
        fig.text(0.5, 0.895, f"w = ({w_[0]:.2f}, {w_[1]:.2f}),  b = {b_:.2f},  {wtxt}", ha="center", fontsize=16)
        fig.text(0.5, 0.845, f"미는 점 (금색 동그라미) {act.sum()}개", ha="center", fontsize=15, color="#7a6a3f")
    an = FuncAnimation(fig, draw, frames=len(F))
    ff = open(os.path.expanduser("~/Documents/manim_shorts/_env/ffpath")).read().strip()
    plt.rcParams["animation.ffmpeg_path"] = ff
    an.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400)); draw(len(F) - 60); fig.savefig("figs/sv_train_anim_thumb.png"); plt.close(fig)


if "anim" in sys.argv:
    train_anim(X, y)


# 학습 1걸음 따라가기: 평균 두 개 → w 방향 → 원점 지나는 수직선
def step1_fig(X, y, path="sv_step1.png"):
    mr, mb = X[y > 0].mean(0), X[y < 0].mean(0); w1 = TRAIN_LR * (y[:, None] * X).sum(0)
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax, X, y)
    ax.scatter(*mr, s=420, marker="*", color=RED, edgecolors="black", lw=1.2, zorder=5)
    ax.scatter(*mb, s=420, marker="*", color=BLUE, edgecolors="black", lw=1.2, zorder=5)
    ax.annotate(f"빨강 평균\n({mr[0]:.2f}, {mr[1]:.2f})", mr, (0.3, 2.6), fontsize=15, color=RED, weight="bold", arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.annotate(f"파랑 평균\n({mb[0]:.2f}, {mb[1]:.2f})", mb, (7.2, 6.4), fontsize=15, color=BLUE, weight="bold", arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    ax.annotate("", mr, mb, arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=22))
    ax.text(4.9, 5.6, "w 방향", fontsize=16, color="#7a6a3f", weight="bold")
    yy = -w1[0] * xs / w1[1]; ax.plot(xs, yy, color="black", lw=2.5)
    ax.text(1.0, 0.25, "← 경계: 원점 지남", fontsize=14)
    ax.scatter(0, 0, s=60, color="black", zorder=5, clip_on=False)
    ax.set_title("1걸음 뒤 첫 선", fontsize=16)
    save(fig, path)


step1_fig(X, y)


def train_anim_slow(X, y, path="figs/sv_train_slow.mp4", fps=20):
    """0걸음부터 천천히: 점 20개가 하나씩 더해지는 1걸음 → 길이 1 로 → 가장 가까운 점이 민다"""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    D = np.unique(np.r_[np.arange(2, 40), np.logspace(np.log10(40), np.log10(MM_T), 200).astype(int), list(MM_FIRST.values())])
    _, _, _, sh = train_maxmin(X, y, snaps=tuple(D))
    ev = [("zero", None, 60)] + [("add", k, 9) for k in range(len(X))] + [("sum", None, 60), ("line", None, 70)]
    for t in D:
        ev.append(("step", t, 45 if t in MM_FIRST.values() else (25 if t <= 6 else 1)))
    ev.append(("step", MM_T, 60))
    frames = [(k, v) for k, v, n in ev for _ in range(n)]
    fig = plt.figure(figsize=(7.2, 9.6)); ax = fig.add_axes([0.1, 0.27, 0.86, 0.6])
    raw = MM_LR * (y[:, None] * X).sum(0); w1 = raw / np.linalg.norm(raw)

    def head(a, b_, c, col="black"):
        fig.texts.clear()
        fig.text(0.5, 0.95, a, ha="center", fontsize=22, weight="bold")
        fig.text(0.5, 0.905, b_, ha="center", fontsize=15)
        fig.text(0.5, 0.06, c, ha="center", fontsize=15, color=col, linespacing=1.6)

    def draw(i):
        kind, v = frames[i]; ax.clear(); pts(ax, X, y); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ring = lambda P, col=GOLD: ax.scatter(P[:, 0], P[:, 1], s=330, facecolors="none", edgecolors=col, lw=2.5, zorder=4)
        if kind == "zero":
            ring(X)
            head("0걸음", "w = (0, 0),  b = 0  →  선이 아직 없다",
                 "어느 점이든 값 = 0  →  모두 똑같이 가장 가깝다\n→  20개 모두 민다")
        elif kind == "add":
            s = (y[: v + 1, None] * X[: v + 1]).sum(0); p = X[v]
            ring(X[: v + 1], GRAY); ring(p[None], RED if y[v] > 0 else BLUE)
            head("1걸음 계산 중", f"점 {v + 1} / 20:  {'+' if y[v] > 0 else '−'} ({p[0]:g}, {p[1]:g})",
                 f"빨강은 좌표를 더하고, 파랑은 뺀다\n합 = ({s[0]:.1f}, {s[1]:.1f})", RED if y[v] > 0 else BLUE)
        elif kind == "sum":
            ring(X, GRAY)
            head("1걸음 계산 끝", "w = 0.05 × (−44.1, 39.2) = (−2.21, 1.96)",
                 f"길이 2.95 > 1  →  2.95 로 나눠 길이 1 로\nw = ({w1[0]:.2f}, {w1[1]:.2f}),  b = 0")
        elif kind == "line":
            ax.plot(xs, -w1[0] * xs / w1[1], color="black", lw=2.5)
            ax.annotate("", 2.5 * w1 + np.array([5, 5]), (5, 5), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3, mutation_scale=20))
            ax.text(4.2, 6.3, "w 방향\n(빨강 쪽)", fontsize=13, color="#7a6a3f")
            head("1걸음 결과: 첫 선", f"w = ({w1[0]:.2f}, {w1[1]:.2f}),  b = 0",
                 "경계는 w 에 수직, 원점 (0, 0) 을 지난다")
        else:
            w_, b_, act, m_ = sh[v]; street_m(ax, w_, b_, m_, X[act])
            k = np.where(act)[0][0]; p = X[k]; col = RED if y[k] > 0 else BLUE
            msg = f"가장 가까운 점 ({p[0]:g}, {p[1]:g}) 이 민다  →  길이 1 로"
            if v in MM_FIRST.values():
                msg = f"{v:,}걸음: ({p[0]:g}, {p[1]:g}) 이 처음으로 가장 가까워짐!"
            if v == MM_T:
                msg = "서포트 벡터 3개가 번갈아 민다\nm = 1.59 (SVM 답)"; col = "#7a6a3f"
            head(f"{min(v, MM_T):,}걸음", f"w = ({w_[0]:.2f}, {w_[1]:.2f}),  b = {b_:.2f},  m = {m_:.2f}", msg, col)
    an = FuncAnimation(fig, draw, frames=len(frames))
    plt.rcParams["animation.ffmpeg_path"] = open(os.path.expanduser("~/Documents/manim_shorts/_env/ffpath")).read().strip()
    an.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400)); draw(0); fig.savefig("figs/sv_train_slow_thumb.png"); plt.close(fig)


if "slow" in sys.argv:
    train_anim_slow(X, y)


# 가장 쉬운 방법: 두 평균의 수직이등분선 → SVM 이 가장 가까운 점 기준으로 다듬는다
def line_dist(ax, w, b, X, y, col="black", ls="-", lw=2.5):
    """경계 w·x + b = 0 을 그리고, 양쪽에서 가장 가까운 점까지 수선과 거리 글자"""
    ax.plot(xs, (-b - w[0] * xs) / w[1], color=col, ls=ls, lw=lw)
    u = w / np.linalg.norm(w); d = y * (X @ w + b) / np.linalg.norm(w); out = {}
    for s, c in ((1, RED), (-1, BLUE)):
        i = np.where(y == s)[0][np.argmin(d[y == s])]; p = X[i]; f = p - s * d[i] * u
        ax.plot([p[0], f[0]], [p[1], f[1]], color=c, lw=2.5)
        ax.scatter(*p, s=330, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
        out[s] = (p, d[i])
    return out


def centroid_figs(X, y):
    mr, mb = X[y > 0].mean(0), X[y < 0].mean(0); wc = mr - mb; bc = -wc @ (mr + mb) / 2
    # ① 평균 선 하나만
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax, X, y)
    for m_, c in ((mr, RED), (mb, BLUE)):
        ax.scatter(*m_, s=480, marker="*", color=c, edgecolors="black", lw=1.2, zorder=5)
    ax.plot(*np.c_[mr, mb], color=GOLD, lw=2.5, ls="--")
    ax.scatter(*((mr + mb) / 2), s=60, color="black", zorder=5)
    ax.plot(xs, (-bc - wc[0] * xs) / wc[1], color="black", lw=2.8)
    ax.annotate("빨강 평균", mr, (0.4, 9.2), fontsize=15, color=RED, weight="bold", arrowprops=dict(arrowstyle="-", color=RED))
    ax.annotate("파랑 평균", mb, (7.6, 0.4), fontsize=15, color=BLUE, weight="bold", arrowprops=dict(arrowstyle="-", color=BLUE))
    ax.text(4.8, 4.0, "한가운데", fontsize=14)
    save(fig, "sv_centroid.png")
    # ② 평균 선 vs SVM
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.2))
    for ax, (w_, b_, ttl) in zip(axs, ((wc, bc, "평균의 한가운데"), (np.array(W_B[0]), W_B[1], "SVM"))):
        pts(ax, X, y); o = line_dist(ax, w_, b_, X, y)
        for s, (p, dd) in o.items():
            q = (p + p - s * dd * (np.array(w_) / np.linalg.norm(w_))) / 2; ax.text(q[0] + (0.0 if s < 0 else 0.3), q[1] + (-1.3 if s < 0 else 0.6), f"{dd:.2f}", fontsize=20, weight="bold", color=RED if s > 0 else BLUE)
        ax.set_title(ttl, fontsize=18)
    axs[1].plot(xs, (-bc - wc[0] * xs) / wc[1], color=GRAY, ls="--", lw=1.5)
    save(fig, "sv_centroid_vs.png")


W_B = ((-1 / np.sqrt(2), 1 / np.sqrt(2)), -1 / (4 * np.sqrt(2)))   # SVM 답 (‖w‖ = 1): w = (−0.71, 0.71), b = −0.18
centroid_figs(X, y)


def train_anim_bad(X, y, w0, b0, name, path, T=6000, fps=20, ylim=10):
    """엉뚱한 선에서 출발 (‖w‖ = 1): 가장 가까운 점 (틀린 점이면 값이 음수) 이 밀어 선을 옮긴다"""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    n0 = np.linalg.norm(w0); w0, b0 = np.array(w0, float) / n0, b0 / n0
    S = list(range(1, 6)) + list(np.unique(np.r_[np.logspace(np.log10(6), np.log10(T), 150).astype(int), T]))
    _, _, _, hist = train_maxmin(X, y, T=T, w0=w0, b0=b0, snaps=tuple(S))
    frames = [(s, k) for s, n in [(s, 30 if s <= 5 else 1) for s in S] + [(T, 60)] for k in range(n)]
    fig = plt.figure(figsize=(7.2, 9.6)); ax = fig.add_axes([0.11, 0.25, 0.85, 0.6])
    eq = lambda w_, b_: f"x2 = {-w_[0] / w_[1]:.2f}·x1 {-b_ / w_[1]:+.2f}"
    bsv = np.array([-1, 1]) / np.sqrt(2), -1 / (4 * np.sqrt(2))

    def draw(i):
        s, k = frames[i]; w_, b_, act, m_ = hist[s]; ax.clear(); pts(ax, X, y); ax.set_ylim(0, ylim); ax.set_yticks(range(0, ylim + 1, 2))
        if 1 < s <= 5 and k < 30:
            pw, pb, _, _ = hist[s - 1]; ax.plot(xs, (-pb - pw[0] * xs) / pw[1], color=GRAY, ls="--", lw=1.5)
        last = i >= len(frames) - 60
        if last:
            ax.plot(xs, (-bsv[1] - bsv[0][0] * xs) / bsv[0][1], color=GOLD, ls="--", lw=2)
        ax.plot(xs, (-b_ - w_[0] * xs) / w_[1], color="black", lw=2.8)
        j = np.where(act)[0][0]; p = X[j]; col = RED if y[j] > 0 else BLUE
        ax.scatter(*p, s=340, facecolors="none", edgecolors=col, lw=2.5, zorder=4)
        fig.texts.clear()
        fig.text(0.5, 0.955, f"{name}  ·  {s:,}걸음", ha="center", fontsize=21, weight="bold")
        fig.text(0.5, 0.905, f"선: {eq(w_, b_)}   m = {m_:.2f}", ha="center", fontsize=17)
        if last:
            msg = "금색 점선 = SVM 답  x2 = 1.00·x1 +0.25\n같은 자리에 도착"; col = "#7a6a3f"
        else:
            side = "파랑" if y[j] > 0 else "빨강"
            msg = (f"값이 가장 작은 점 ({p[0]:g}, {p[1]:g}){' (틀림, 값 < 0)' if m_ < 0 else ''}\n"
                   f"→ 선을 {side} 쪽으로 밀어낸다")
        fig.text(0.5, 0.115, "동그라미 = 이번 걸음에 가장 가까운 점 (값이 가장 작은 점)", ha="center", fontsize=13, color="#555")
        fig.text(0.5, 0.035, msg, ha="center", fontsize=15, color=col, linespacing=1.5)
    an = FuncAnimation(fig, draw, frames=len(frames))
    plt.rcParams["animation.ffmpeg_path"] = open(os.path.expanduser("~/Documents/manim_shorts/_env/ffpath")).read().strip()
    an.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400)); draw(0); fig.savefig(path.replace(".mp4", "_thumb.png")); plt.close(fig)


if "bad" in sys.argv:
    train_anim_bad(X, y, (-1, 1), -10, "위로 밀린 선에서 출발", "figs/sv_bad_up.mp4", ylim=12)
    train_anim_bad(X, y, (-3, 1), 0, "가파른 선에서 출발", "figs/sv_bad_steep.mp4")


def b_rate_fig(X, y, path="sv_b_rate.png", T=20000):
    """b 학습률을 1배 · 5배로 했을 때 걸음마다 b (로그 가로축). 1배는 b 가 거의 안 움직인다."""
    ts = np.unique(np.logspace(0, np.log10(T), 300).astype(int))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for sb, col, lab in ((1, GRAY, r"$\eta_2 = \eta_1$"), (MM_SB, BLUE, rf"$\eta_2 = {MM_SB}\,\eta_1$")):
        *_, shots = train_maxmin(X, y, T=T, sb=sb, snaps=set(ts))
        ax.plot(ts, [shots[t][1] for t in ts], color=col, lw=2.8, label=lab)
        ax.annotate(f"2만 걸음: {shots[ts[-1]][1]:.2f}", (ts[-1], shots[ts[-1]][1]), xytext=(-4, 12), textcoords="offset points", ha="right", color=col, fontsize=13)
    ax.axhline(B_ANS, color=RED, ls="--", lw=1.8); ax.text(1.2, B_ANS - 0.012, f"SVM 답 b = {B_ANS:.2f}", color=RED, fontsize=13, va="top")
    ax.set_xscale("log"); ax.set_xlabel("걸음 (로그 눈금)"); ax.set_ylabel("b", rotation=0, labelpad=10)
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlim(1, T * 1.3); ax.legend(frameon=False, fontsize=13, loc="upper right")
    save(fig, path)


if "brate" in sys.argv:
    B_ANS = -0.18
    b_rate_fig(X, y); sys.exit()


def train_anim_click(X, y, path="figs/sv_train_click.mp4", fps=20, slow=(1, 2, 3, 4, 5, 6)):
    """한 걸음 = ① 값 재기 → ② 가장 작은 점 클릭 → ③ 민다 (선 이동) → ④ ‖w‖ 로 나눔 (선은 그대로). 앞 몇 걸음은 천천히, 뒤는 빨리."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.patches import Polygon
    fast = np.unique(np.r_[np.arange(slow[-1] + 1, 60), np.logspace(np.log10(60), np.log10(MM_T), 120).astype(int), list(MM_FIRST.values())])
    _, _, _, sh = train_maxmin(X, y, snaps=tuple(slow) + tuple(fast))
    P = {"val": 28, "move": 14, "click": 10, "push": 26, "norm": 30}
    frames = []
    for t in slow:
        for ph, n in P.items():
            frames += [(t, ph, j / max(n - 1, 1)) for j in range(n)]
    for t in fast:
        frames += [(t, "fast", 0)] * (30 if t in MM_FIRST.values() and t > slow[-1] else 2)
    frames += [(MM_T, "fast", 0)] * 70
    fig = plt.figure(figsize=(7.2, 9.6)); ax = fig.add_axes([0.1, 0.27, 0.86, 0.6])

    def head(a, b_, c, col="black"):
        fig.texts.clear()
        fig.text(0.5, 0.95, a, ha="center", fontsize=22, weight="bold")
        fig.text(0.5, 0.905, b_, ha="center", fontsize=15)
        fig.text(0.5, 0.05, c, ha="center", fontsize=15, color=col, linespacing=1.6)

    def line(w_, b_, **kw):
        if np.linalg.norm(w_) > 1e-9:
            if abs(w_[1]) > 1e-6:
                ax.plot(xs, -(w_[0] * xs + b_) / w_[1], **kw)
            else:
                ax.axvline(-b_ / w_[0], **kw)

    def cursor(p):
        c = np.array([[0, 0], [0, -0.75], [0.2, -0.57], [0.36, -0.9], [0.48, -0.84], [0.33, -0.52], [0.58, -0.52]])
        ax.add_patch(Polygon(c + p, closed=True, fc="white", ec="black", lw=1.6, zorder=6))

    def draw(i):
        t, ph, a = frames[i]; ax.clear(); pts(ax, X, y); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        w_, b_, act, m_ = sh[t]; ks = np.where(act)[0]
        eta = MM_LR / np.sqrt(t); w2 = w_ + eta * (y[ks, None] * X[ks]).sum(0); b2 = b_ + MM_SB * eta * y[ks].sum()
        n2 = np.linalg.norm(w2); w3, b3 = (w2 / n2, b2 / n2) if n2 > 1 else (w2, b2)
        v = y * (X @ w_ + b_)
        ring = lambda Q, col=GOLD, s=330, lw=2.5: ax.scatter(Q[:, 0], Q[:, 1], s=s, facecolors="none", edgecolors=col, lw=lw, zorder=4)
        nm = "20개 모두" if len(ks) > 1 else f"({X[ks[0], 0]:g}, {X[ks[0], 1]:g})"
        kcol = GOLD if len(ks) > 1 else (RED if y[ks[0]] > 0 else BLUE)
        if ph == "fast":
            street_m(ax, w_, b_, m_, X[act])
            msg = f"가장 가까운 점 {nm} 이 민다"
            if t in MM_FIRST.values() and t > slow[-1]:
                msg = f"({X[ks[0], 0]:g}, {X[ks[0], 1]:g}) 이 처음으로 가장 가까워짐"
            if t == MM_T:
                msg = "서포트 벡터 3개가 번갈아 민다\nm = 1.59 (SVM 답)"; kcol = "#7a6a3f"
            head(f"{t - 1:,}걸음 뒤", f"w = ({w_[0]:.2f}, {w_[1]:.2f}),  b = {b_:.2f},  m = {m_:.2f}", msg, kcol if t != MM_T else kcol)
            return
        if ph in ("val", "move", "click"):
            line(w_, b_, color="black", lw=2.5)
            for j in range(len(X)):
                lo = act[j]
                ax.text(X[j, 0] + 0.22, X[j, 1] + 0.2, f"{v[j]:.2f}", fontsize=12 if lo else 9, weight="bold" if lo else None,
                        color="#7a6a3f" if lo else GRAY, zorder=5)
            if ph == "val":
                head(f"{t}걸음째  ① 값 재기", f"w = ({w_[0]:.2f}, {w_[1]:.2f}),  b = {b_:.2f}",
                     r"점마다 값 $y_i(\mathbf{w}\cdot\mathbf{x}_i + b)$ = 경계까지 거리" + "\n" + ("선이 없어서 모두 0" if t == 1 else f"가장 작은 값 = m = {m_:.2f}"))
            else:
                tgt = X[ks].mean(0) if len(ks) > 1 else X[ks[0]]
                start = np.array([8.6, 1.0])
                p = start + (tgt - start) * (min(a, 1) if ph == "move" else 1)
                if ph == "click":
                    ring(X[ks], kcol, s=330 + 900 * a, lw=3 * (1 - a) + 0.5)
                    ring(X[ks], kcol)
                if len(ks) == 1 or ph == "move":
                    cursor(p + np.array([0.05, -0.05]))
                head(f"{t}걸음째  ② 가장 작은 점 클릭", f"k = argmin 값  →  {nm}",
                     f"값 {m_:.2f}  =  m" if len(ks) == 1 else "값이 모두 0 으로 같다  →  20개 모두", kcol)
        elif ph == "push":
            s = a * a * (3 - 2 * a)
            wi, bi = w_ + s * (w2 - w_), b_ + s * (b2 - b_)
            line(w_, b_, color=GRAY, lw=1.5, ls=":")
            line(wi, bi, color="black", lw=2.5)
            ring(X[ks], kcol)
            sg = "+" if (len(ks) == 1 and y[ks[0]] > 0) else ("−" if len(ks) == 1 else "")
            body = (rf"$\mathbf{{w}} \leftarrow \mathbf{{w}} {sg} \eta_1\,({X[ks[0], 0]:g}, {X[ks[0], 1]:g})$,   $b \leftarrow b {sg} \eta_2$" if len(ks) == 1
                    else r"$\mathbf{w} \leftarrow \mathbf{w} + \eta_1\,$(빨강 합 − 파랑 합)")
            head(rf"{t}걸음째  ③ 민다  ($\eta_1$ = {eta:.3f}, $\eta_2$ = {MM_SB * eta:.3f})", body,
                 f"w = ({w2[0]:.2f}, {w2[1]:.2f}),  b = {b2:.2f}\n선이 그 점에서 멀어진다", kcol)
        else:
            line(w3, b3, color="black", lw=2.5)
            v3 = y * (X @ w3 + b3); m3 = v3.min()
            street_m(ax, w3, b3, m3) if a > 0.5 else None
            if n2 > 1:
                head(f"{t}걸음째  ④ 길이 1 로", rf"$\|\mathbf{{w}}\|$ = {n2:.2f}  →  w, b 를 {n2:.2f} 로 나눈다",
                     f"w = ({w3[0]:.2f}, {w3[1]:.2f}),  b = {b3:.2f}\n선은 그대로, 숫자만 바뀐다  →  새 m = {m3:.2f}")
            else:
                head(f"{t}걸음째  ④ 길이 1 로", rf"$\|\mathbf{{w}}\|$ = {n2:.2f} ≤ 1  →  그대로", f"새 m = {m3:.2f}")
    an = FuncAnimation(fig, draw, frames=len(frames))
    plt.rcParams["animation.ffmpeg_path"] = open(os.path.expanduser("~/Documents/manim_shorts/_env/ffpath")).read().strip()
    an.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    for nm_, k in (("thumb", 28 + 14 + 9), ("p1", 10), ("p2", 28 + 14 + 9), ("p3", 28 + 14 + 10 + 25), ("p4", 28 + 14 + 10 + 26 + 29)):
        draw(k + 108); fig.savefig(f"figs/sv_train_click_{nm_}.png")
    plt.close(fig)
    return len(frames) / fps


if "click" in sys.argv:
    print("click", train_anim_click(X, y)); sys.exit()


def fold_fig(X, y, w, b, path="sv_fold.png"):
    """거울로 접기: y_i 를 곱하는 것 = 파란 점을 경계(거울)에 비춰 빨강 쪽으로 넘기기. 접은 뒤 가장 가까운 거리 = m"""
    w = np.asarray(w, float); w, b = w / np.linalg.norm(w), b / np.linalg.norm(w)
    v = y * (X @ w + b); m = v.min(); sv = v < m + 0.02
    Xf = X.copy(); bl = y < 0; Xf[bl] = X[bl] - 2 * (X[bl] @ w + b)[:, None] * w
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, ttl in zip(axs, ("접기 전", "접은 뒤: 파랑을 경계(거울)에 비춰 넘김")):
        ax.set_title(ttl, fontsize=16)
    pts(axs[0], X, y); street_m(axs[0], w, b, m, X[sv])
    ax = axs[1]; pts(ax, X[~bl], y[~bl])
    ax.scatter(Xf[bl, 0], Xf[bl, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=2, zorder=3)
    for p, q in zip(X[bl], Xf[bl]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, alpha=0.15, lw=1)
    ax.scatter(X[bl, 0], X[bl, 1], s=40, color=BLUE, marker="s", alpha=0.15)
    s = np.linspace(-30, 30, 2); d = np.array([-w[1], w[0]])
    seg = lambda c: (c - b) * w[:, None] + d[:, None] * s
    ax.plot(*seg(0), color="black", lw=2.5); ax.plot(*seg(m), "--", color=GOLD, lw=2)
    Q = np.where(sv)[0]; ax.scatter(Xf[Q, 0], Xf[Q, 1], s=330, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
    ax.text(9.7, 0.35, f"접은 뒤 가장 가까운 거리\nm = {m:.2f}", fontsize=13, color="#7a6a3f", ha="right", bbox=dict(fc="white", ec="none", alpha=0.9))
    save(fig, path)


def push_fig(X, y, t=2, path="sv_push.png"):
    """한 번 밀기 = 평행이동 (b) + 회전 (w). 실제 학습 t 걸음째 숫자."""
    _, _, _, sh = train_maxmin(X, y, snaps=(t,), T=t)
    w, b, act, m = sh[t]; k = np.where(act)[0][0]; xk, yk = X[k], y[k]
    e1 = MM_LR / np.sqrt(t); e2 = MM_SB * e1
    wb, bb = w, b + e2 * yk                       # b 만
    wf, bf = w + e1 * yk * xk, b + e2 * yk        # w 까지
    dist = lambda w_, b_: yk * (w_ @ xk + b_) / np.linalg.norm(w_)
    fig, ax = plt.subplots(figsize=(6.6, 6.0)); pts(ax, X, y)
    def L(w_, b_, **kw):
        ax.plot(xs, -(w_[0] * xs + b_) / w_[1], **kw)
    L(w, b, color=GRAY, lw=2, ls=":", label=f"전: 거리 {dist(w, b):.2f}")
    L(wb, bb, color=GOLD, lw=2.5, ls="--", label=rf"$b$ 만 ($+\eta_2 y_k$): 평행이동, 거리 {dist(wb, bb):.2f}")
    L(wf, bf, color="black", lw=2.8, label=rf"$\mathbf{{w}}$ 까지 ($+\eta_1 y_k \mathbf{{x}}_k$): 회전 추가, 거리 {dist(wf, bf):.2f}")
    ax.scatter(*xk, s=380, facecolors="none", edgecolors=RED if yk > 0 else BLUE, lw=3, zorder=5)
    ax.text(xk[0] - 0.3, xk[1] + 0.45, f"k = ({xk[0]:g}, {xk[1]:g})", fontsize=13, ha="right", color=RED if yk > 0 else BLUE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=12.5)
    save(fig, path)
    return dist(w, b), dist(wb, bb), dist(wf, bf)


if "fold" in sys.argv:
    fold_fig(X, y, *W_B)
    print(push_fig(X, y)); sys.exit()
