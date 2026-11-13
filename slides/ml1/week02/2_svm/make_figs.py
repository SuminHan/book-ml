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
_a = np.array([2.0, 4.5]); _b = _a + (2 / np.linalg.norm(M.coef_[0])) * np.array([1, -1]) / np.sqrt(2)   # 빨강 가장자리 위 서포트 벡터에서 파랑 가장자리까지
assert abs(M.decision_function([_b])[0] + 1) < 1e-2
ax.annotate("", _b, _a, arrowprops=dict(arrowstyle="<->", lw=2, shrinkA=0, shrinkB=0))
ax.text(_b[0] + 0.15, _b[1] - 0.15, f"마진 {MARGIN}", fontsize=15, fontweight="bold", ha="left", va="top")
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
    ax.set_xlim(-0.2, 4.0); ax.set_ylim(-0.2, 4.0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
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


# 학습: 여유 < 1 인 점만 민다 (경사하강)
def street_wb(ax, w, b, ring=()):
    s = np.linspace(-30, 30, 2); nrm = np.linalg.norm(w); u = w / nrm; d = np.array([-u[1], u[0]])
    seg = lambda c: ((c - b) / nrm) * u[:, None] + d[:, None] * s
    a, c = seg(1), seg(-1)
    ax.fill(np.r_[a[0], c[0][::-1]], np.r_[a[1], c[1][::-1]], color=GOLD, alpha=0.18, lw=0)
    ax.plot(*seg(0), color="black", lw=2.5); ax.plot(*a, "--", color=RED, lw=1.5); ax.plot(*c, "--", color=BLUE, lw=1.5)
    if len(ring):
        r = np.array(ring); ax.scatter(r[:, 0], r[:, 1], s=320, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)


wt, bt, pushed, shots = train_svm(X, y, snaps=SNAPS)
fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, t in zip(axs, SNAPS):
    w_, b_, act = shots[t]; pts(ax)
    if t == SNAPS[0]:
        street_wb(ax, w_, b_, X); ttl = "1걸음: 20개 모두 밀어 첫 선"
    elif t == SNAPS[-1]:
        street_wb(ax, w_, b_, SV); ttl = f"{t:,}걸음: 폭 {2 / np.linalg.norm(w_):.2f}"
    else:
        new = [k for k, v in TOUCH.items() if v <= t]
        street_wb(ax, w_, b_, new); ttl = f"{t}걸음: 폭 {2 / np.linalg.norm(w_):.2f}, 미는 점 {len(new)}개"
    ax.set_title(ttl, fontsize=15)
save(fig, "sv_train4.png")

mg = y * (X @ wt + bt); o = np.argsort(mg)
fig, ax = plt.subplots(figsize=(10, 4.2))
cols = [GOLD if mg[i] < 1.02 else (RED if y[i] > 0 else BLUE) for i in o]
ax.bar(range(20), mg[o], color=cols, alpha=0.9)
ax.axhline(1, color="black", ls="--", lw=1.5); ax.text(-0.4, 2.8, "점선: 여유 1 = 길 가장자리", fontsize=14)
ax.text(1, 1.32, "서포트 벡터", ha="center", fontsize=14, color="#7a6a3f", fontweight="bold")
for k, i in enumerate(o[:3]):
    ax.text(k, mg[i] + 0.08, f"{mg[i]:.2f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([]); ax.set_ylabel("여유  $y_i(\\mathbf{w}\\cdot\\mathbf{x}_i + b)$"); ax.set_xlabel("점 20개 (여유 작은 순)")
save(fig, "sv_slack_bars.png")
print("ok train")
