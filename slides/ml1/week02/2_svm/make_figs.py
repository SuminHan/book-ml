"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py.
덱 전체가 z-score 좌표 하나만 쓴다 (sv_zscore.png 의 왼쪽만 '안 했을 때' 원래 좌표). 옛 0~10 좌표판은 _이전/make_figs_v4_원좌표.py."""
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
BLUE, RED, GOLD, GRAY, GOLDT = "#2E3192", "#d62728", "#A9976B", "#999999", "#7a6a3f"
O = "figs/"
X = np.array(RED_PTS + BLUE_PTS, float); y = np.r_[np.ones(10), -np.ones(10)]
L = 2.2                                                    # 그림 범위 −L ~ L
xs = np.linspace(-L - 1, L + 1, 2)
WA, BA, MA = np.array(W_ANS), B_ANS, M_ANS


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def pts(ax, X=X, y=y, lim=L):
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, zorder=3); ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, marker="s", zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks(range(-2, 3)); ax.set_yticks(range(-2, 3))
    ax.tick_params(labelsize=11, length=3); ax.set_xlabel("$x_1$", fontsize=13, labelpad=1); ax.set_ylabel("$x_2$", fontsize=13, rotation=0, labelpad=8)
    ax.axhline(0, color=GRAY, lw=0.9, zorder=0); ax.axvline(0, color=GRAY, lw=0.9, zorder=0)          # x1 = 0, x2 = 0 축


def street_m(ax, w, b, m, ring=(), ring_col=GOLD, s_len=10):
    """‖w‖ = 1 로 나눈 경계와, 경계에서 거리 m 인 길 가장자리 두 줄"""
    s = np.linspace(-s_len, s_len, 2); n = np.linalg.norm(w); u = w / n; d = np.array([-u[1], u[0]]); b = b / n
    seg = lambda c: (c - b) * u[:, None] + d[:, None] * s
    if m > 0:
        a, c = seg(m), seg(-m)
        ax.fill(np.r_[a[0], c[0][::-1]], np.r_[a[1], c[1][::-1]], color=GOLD, alpha=0.18, lw=0)
        ax.plot(*a, "--", color=RED, lw=1.5); ax.plot(*c, "--", color=BLUE, lw=1.5)
    ax.plot(*seg(0), color="black", lw=2.5)
    if len(ring):
        r = np.array(ring); ax.scatter(r[:, 0], r[:, 1], s=320, facecolors="none", edgecolors=ring_col, lw=2.5, zorder=4)


def line(ax, w, b, **kw):
    s = np.linspace(-10, 10, 2); n = np.linalg.norm(w); u = w / n; d = np.array([-u[1], u[0]])
    ax.plot(*((-b / n) * u[:, None] + d[:, None] * s), **kw)


def svm_street(ax, m, X=X):
    w, b, mm = unit(m); street_m(ax, w, b, mm, X[m.support_])


def origin_axes(ax):
    """원점 (0, 0) 을 지나는 x1 · x2 축 (화살표 끝 + 이름)"""
    ax.spines["left"].set_position("zero"); ax.spines["bottom"].set_position("zero")
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, ms=7)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms=7)
    ax.set_xticks([-1, 1]); ax.set_yticks([-1, 1]); ax.tick_params(labelsize=12)
    ax.text(1.0, 0.0, "  $x_1$", transform=ax.get_yaxis_transform(), va="center", fontsize=16)
    ax.text(0.0, 0.97, "  $x_2$", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=16)


arw = lambda ax, p, q, col, lw=3: ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=22, shrinkA=0, shrinkB=0), zorder=6)
M = SVC(kernel="linear", C=1e6).fit(X, y)


def fig_raw_z():
    """z-score: 원래 점 (0~10) → (x − 평균) / 표준편차"""
    A = np.array(RAW_RED + RAW_BLUE, float)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    ax = axs[0]; ax.scatter(A[y > 0, 0], A[y > 0, 1], s=80, color=RED, zorder=3); ax.scatter(A[y < 0, 0], A[y < 0, 1], s=80, color=BLUE, marker="s", zorder=3)
    ax.scatter(*RAW_MU, s=420, marker="+", color="black", lw=3, zorder=5); ax.text(RAW_MU[0] - 2.2, RAW_MU[1] - 1.0, f"평균 ({RAW_MU[0]}, {RAW_MU[1]})", fontsize=13)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("원래 $x_1$"); ax.set_ylabel("원래 $x_2$"); ax.set_title(f"원래 값 (표준편차 {RAW_SD[0]}, {RAW_SD[1]})", fontsize=15)
    ax = axs[1]; pts(ax); ax.scatter(0, 0, s=420, marker="+", color="black", lw=3, zorder=5); ax.text(0.12, 0.12, "평균 (0, 0)", fontsize=13)
    ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_title("z-score 후 (평균 0, 표준편차 1)", fontsize=15)
    save(fig, "sv_raw_z.png")


def fig_hook():
    fig, ax = plt.subplots(figsize=(5.6, 5.6)); pts(ax)
    for (a, c), col in (((1.6, -0.1), GRAY), ((0.55, 0.3), GRAY), ((1.0, 0.05), "black")):
        assert (np.sign(X[:, 1] - (a * X[:, 0] + c)) == y).all(), (a, c)          # 세 선 모두 완벽하게 가른다
        ax.plot(xs, a * xs + c, color=col, lw=2.2)
    ax.text(0.45, 1.55, "?", fontsize=34, color=RED, fontweight="bold")
    save(fig, "sv_hook.png")


def line_dist(ax, w, b, X, y, col="black", ls="-", lw=2.5):
    """경계 w·x + b = 0 을 그리고, 양쪽에서 가장 가까운 점까지 수선"""
    line(ax, w, b, color=col, ls=ls, lw=lw)
    u = w / np.linalg.norm(w); d = y * (X @ w + b) / np.linalg.norm(w); out = {}
    for s, c in ((1, RED), (-1, BLUE)):
        i = np.where(y == s)[0][np.argmin(d[y == s])]; p = X[i]; f = p - s * d[i] * u
        ax.plot([p[0], f[0]], [p[1], f[1]], color=c, lw=2.5)
        ax.scatter(*p, s=330, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
        out[s] = (p, d[i])
    return out


def fig_centroid():
    mr, mb = X[y > 0].mean(0), X[y < 0].mean(0); wc = mr - mb; bc = -wc @ (mr + mb) / 2
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax)
    for m_, c in ((mr, RED), (mb, BLUE)):
        ax.scatter(*m_, s=480, marker="*", color=c, edgecolors="black", lw=1.2, zorder=5)
    ax.plot(*np.c_[mr, mb], color=GOLD, lw=2.5, ls="--")
    ax.scatter(*((mr + mb) / 2), s=60, color="black", zorder=5)
    line(ax, wc, bc, color="black", lw=2.8)
    ax.annotate("빨강 평균", mr, (-2.0, 2.0), fontsize=15, color=RED, weight="bold", arrowprops=dict(arrowstyle="-", color=RED))
    ax.annotate("파랑 평균", mb, (0.9, -2.05), fontsize=15, color=BLUE, weight="bold", arrowprops=dict(arrowstyle="-", color=BLUE))
    ax.text(0.15, -0.25, "한가운데", fontsize=14)
    save(fig, "sv_centroid.png")
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.2))
    for ax, (w_, b_, ttl) in zip(axs, ((wc, bc, "평균의 한가운데"), (WA, BA, "SVM"))):
        pts(ax); o = line_dist(ax, w_, b_, X, y)
        for s, (p, dd) in o.items():
            ax.text(p[0] + (-0.85 if s < 0 else -0.95), p[1] + (-0.55 if s < 0 else 0.25), f"{dd:.2f}", fontsize=20, weight="bold", color=RED if s > 0 else BLUE)
        ax.set_title(ttl, fontsize=18)
    line(axs[1], wc, bc, color=GRAY, ls="--", lw=1.5)
    save(fig, "sv_centroid_vs.png")


def fig_hyper():
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


def fig_street():
    fig, ax = plt.subplots(figsize=(5.8, 5.8)); pts(ax); svm_street(ax, M)
    a = np.array(SV[1]); b = a + 2 * MA * WA
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", lw=2, shrinkA=0, shrinkB=0))
    ax.text(-2.05, -0.75, f"길 폭 2m = {2 * MA:.2f}", fontsize=15, fontweight="bold", ha="left", va="top")
    save(fig, "sv_street.png")
    X2 = X.copy(); X2[10 + BLUE_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
    M2 = SVC(kernel="linear", C=1e6).fit(X2, y)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    pts(axs[0]); svm_street(axs[0], M); axs[0].set_title("처음", fontsize=16)
    pts(axs[1], X2); svm_street(axs[1], M2, X2); axs[1].set_title("멀리 있는 파랑 점을 옮겨도", fontsize=16)
    axs[1].annotate("", FAR_MOVE[1], FAR_MOVE[0], arrowprops=dict(arrowstyle="->", lw=2, color=BLUE, shrinkA=9, shrinkB=9))
    axs[1].add_patch(plt.Circle(FAR_MOVE[0], 0.13, fill=False, ls="--", lw=2, ec=BLUE, zorder=4)); axs[1].text(FAR_MOVE[0][0] - 1.05, FAR_MOVE[0][1] - 0.1, "원래 자리", fontsize=12, color=BLUE)
    save(fig, "sv_move.png")


def fig_fold():
    v = y * (X @ WA + BA); sv = v < MA + 0.01
    Xf = X.copy(); bl = y < 0; Xf[bl] = X[bl] - 2 * (X[bl] @ WA + BA)[:, None] * WA
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    axs[0].set_title("접기 전", fontsize=16); axs[1].set_title("접은 뒤: 파랑을 경계(거울)에 비춰 넘김", fontsize=16)
    pts(axs[0]); street_m(axs[0], WA, BA, MA, X[sv])
    ax = axs[1]; pts(ax, X[~bl], y[~bl])
    ax.scatter(Xf[bl, 0], Xf[bl, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=2, zorder=3)
    for p, q in zip(X[bl], Xf[bl]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, alpha=0.15, lw=1)
    ax.scatter(X[bl, 0], X[bl, 1], s=40, color=BLUE, marker="s", alpha=0.15)
    s = np.linspace(-10, 10, 2); d = np.array([-WA[1], WA[0]]); seg = lambda c: (c - BA) * WA[:, None] + d[:, None] * s
    ax.plot(*seg(0), color="black", lw=2.5); ax.plot(*seg(MA), "--", color=GOLD, lw=2)
    Q = np.where(sv)[0]; ax.scatter(Xf[Q, 0], Xf[Q, 1], s=330, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
    ax.text(2.1, -2.05, f"접은 뒤 가장 가까운 거리\nm = {MA:.2f}", fontsize=13, color=GOLDT, ha="right", bbox=dict(fc="white", ec="none", alpha=0.9))
    save(fig, "sv_fold.png")


COS_W, COS_B, COS_X = np.array([1, 1]) / np.sqrt(2), -1 / np.sqrt(2), np.array([0.4, 1.6])   # 경계 x1 + x2 = 1, 점 (0.4, 1.6)


def fig_cos():
    """점곱 = ‖w‖‖x‖cos θ → ‖w‖ = 1 이면 w 방향 위치, 경계의 위치 −b 를 빼면 거리"""
    Bu, PX = COS_W, COS_X
    sh, shb = Bu @ PX, -COS_B; dist = sh - shb
    assert abs(sh - 1.41) < 0.01 and abs(shb - 0.71) < 0.01 and abs(dist - 0.71) < 0.01
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.set_xlim(-0.6, 2.2); ax.set_ylim(-0.7, 2.1); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    t = np.linspace(-0.6, 2.1, 2); ax.plot(t, t, color=GOLD, lw=1.5, ls="--")
    ax.plot(t, 1 - t, color="black", lw=2.5)
    F = sh * Bu; Fb = shb * Bu
    arw(ax, (0, 0), PX, RED, 2.5); arw(ax, (0, 0), 0.5 * Bu, GOLD, 3.5)
    nrm = np.array([1, -1]) / np.sqrt(2)

    def dim(a, b, off, col, lab):
        pa, pb = a * Bu + off * nrm, b * Bu + off * nrm
        ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="<->", color=col, lw=2.5, shrinkA=0, shrinkB=0))
        for q, base in ((pa, a * Bu), (pb, b * Bu)):
            ax.plot(*zip(base, q + 0.04 * nrm), color=col, lw=1, ls=":")
        mid = (pa + pb) / 2 + 0.11 * nrm
        ax.text(*mid, lab, fontsize=13, color=col, rotation=45, ha="center", va="center", fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    dim(0, sh, 0.5, BLUE, f"x 의 w 방향 위치 $\\Vert\\mathbf{{x}}\\Vert\\cos\\theta$ = {sh:.2f}")
    dim(0, shb, 0.24, "#555555", f"경계의 위치 {shb:.2f} = $-b$")
    ax.annotate("", F, Fb, arrowprops=dict(arrowstyle="<->", color="#2b8a3e", lw=3, shrinkA=0, shrinkB=0))
    ax.plot(*zip(PX, F), color=GRAY, lw=1.5, ls=":")
    ax.scatter(*PX, s=80, color=RED, zorder=4); ax.scatter(*Fb, s=50, color="black", zorder=4); ax.scatter(0, 0, s=50, color="black", zorder=5); ax.text(-0.3, -0.18, "(0, 0)", fontsize=13)
    ax.text(PX[0] + 0.05, PX[1] + 0.05, f"$\\mathbf{{x}} = ({PX[0]:g}, {PX[1]:g})$", fontsize=15, color=RED)
    ax.text(-0.12, 0.3, "$\\mathbf{w}$", fontsize=16, color=GOLDT, fontweight="bold")
    ax.text(0.17, 0.33, "θ", fontsize=16)
    ax.text(0.15, 1.05, f"거리 {dist:.2f}", fontsize=15, color="#2b8a3e", fontweight="bold")
    ax.text(1.45, -0.6, "경계 $x_1 + x_2 = 1$", fontsize=13)
    origin_axes(ax); ax.set_xticks([1, 2]); ax.set_yticks([1, 2])
    save(fig, "sv_cos.png")


def fig_push(t=2, S=1.0, KW=10, KB=10):
    """한 번 밀기 (식 순서대로): ① w' = w + η y_k x_k → 법선이 돈다, ② b' = b + η y_k → 그 선이 수직으로 평행이동. 화살표는 KW · KB 배"""
    _, _, _, sh = train_maxmin(X, y, snaps=(t,), T=t)
    w, b, act, m = sh[t]; k = np.where(act)[0][0]; xk, yk = X[k], y[k]; col = RED if yk > 0 else BLUE
    eta = MM_LR / np.sqrt(t); wf = w + eta * yk * xk; bb = b + eta * yk
    dist = lambda w_, b_: yk * (w_ @ xk + b_) / np.linalg.norm(w_)
    u = w / np.linalg.norm(w)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.0))
    ax = axs[0]; pts(ax); ax.set_title(r"① $\mathbf{w}' = \mathbf{w} + \eta\, y_k \mathbf{x}_k$ : 법선이 돈다", fontsize=15)
    line(ax, w, b, color=GRAY, lw=2, ls=":"); line(ax, wf, b, color="black", lw=2.6)
    ax.plot([0, xk[0]], [0, xk[1]], color=col, ls="--", lw=1.4); ax.text(xk[0] / 2 + 0.12, xk[1] / 2 - 0.05, r"$\mathbf{x}_k$", color=col, fontsize=15)
    o = np.array([0.9, 1.0]); o = o - (w @ o + b) * u
    tip = o + S * u; add = S * KW * eta * yk * xk; tip2 = tip + add
    arw(ax, o, tip, GRAY); ax.text(*(tip + np.array([-0.3, -0.05])), r"$\mathbf{w}$", fontsize=15, color=GRAY)
    arw(ax, tip, tip2, col, 2.5); ax.text(*(tip + add / 2 + np.array([0.06, 0.08])), r"$-\eta\,\mathbf{x}_k$" if yk < 0 else r"$+\eta\,\mathbf{x}_k$", fontsize=14, color=col)
    arw(ax, o, tip2, "black"); ax.text(*(tip2 + np.array([0.05, 0.08])), r"$\mathbf{w}'$", fontsize=15)
    ax.scatter(*xk, s=380, facecolors="none", edgecolors=col, lw=3, zorder=5)
    ax.text(-2.1, -2.05, f"화살표 {KW}배 · 거리 {dist(w, b):.3f} → {dist(wf, b):.3f}", fontsize=12.5)
    ax = axs[1]; pts(ax); ax.set_title(r"② $b' = b + \eta\, y_k$ : 선에 수직으로", fontsize=15)
    uf = wf / np.linalg.norm(wf)
    line(ax, wf, b, color=GRAY, lw=2, ls=":"); line(ax, wf, bb, color="black", lw=2.6)
    foot = xk - (wf @ xk + b) / np.linalg.norm(wf) * uf
    for s_ in (-1.2, 0, 1.2):
        p = foot + s_ * np.array([-uf[1], uf[0]]); arw(ax, p, p - KB * (bb - b) / np.linalg.norm(wf) * uf, GOLD)
    ax.scatter(*xk, s=380, facecolors="none", edgecolors=col, lw=3, zorder=5)
    ax.text(-2.1, -2.05, f"화살표 {KB}배 · 거리 {dist(wf, b):.3f} → {dist(wf, bb):.3f}", fontsize=12.5, color=GOLDT)
    save(fig, "sv_push_dir.png")
    return xk, yk, dist(w, b), dist(wf, b), dist(wf, bb)


def fig_zscore(T=MM_T):
    """z-score 안 하면: 원래 점의 x2 에 +SHIFT 하고 그대로 학습 → 실패 (왼쪽, 원래 좌표). z-score 한 덱의 점으로 학습 → 답 (오른쪽)."""
    A = np.array(RAW_RED + RAW_BLUE, float) + np.array([0, SHIFT])
    w1, b1, _, _ = train_maxmin(A, y, T=T); v = y * (A @ w1 + b1); bad = v < 0
    wz, bz, _, _ = train_maxmin(X, y, T=T); mz = (y * (X @ wz + bz)).min()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
    ax = axs[0]
    ax.scatter(A[y > 0, 0], A[y > 0, 1], s=80, color=RED, zorder=3); ax.scatter(A[y < 0, 0], A[y < 0, 1], s=80, color=BLUE, marker="s", zorder=3)
    street_m(ax, w1, b1, 0, s_len=400); ax.scatter(A[bad, 0], A[bad, 1], s=330, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=4)
    ax.set_xlim(-1, 10); ax.set_ylim(SHIFT, SHIFT + 10); ax.set_aspect("equal")
    ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(SHIFT, SHIFT + 11, 2))
    ax.set_xlabel("원래 $x_1$"); ax.set_ylabel("원래 $x_2$ + 100")
    ax.set_title(f"z-score 안 하고 학습: 2만 걸음 뒤\n틀린 점 {bad.sum()}개", fontsize=15)
    ax = axs[1]; pts(ax); street_m(ax, wz, bz, mz)
    ax.set_title(f"z-score 하고 학습: 2만 걸음 뒤\nm = {mz:.2f} (SVM 답)", fontsize=15)
    save(fig, "sv_zscore.png")
    return int(bad.sum()), round(mz, 2)


def fig_step1():
    """1걸음 뒤 첫 선: w = 0.05 × Σ y_i x_i 를 길이 1 로, b = 0 이라 원점을 지난다"""
    w1 = np.array(STEP1_W)
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax)
    arw(ax, (0, 0), 1.3 * w1, GOLD, 3.5)
    ax.text(*(1.3 * w1 + np.array([-0.35, 0.12])), f"$\\mathbf{{w}}$ = ({w1[0]:.2f}, {w1[1]:.2f})", fontsize=15, color=GOLDT, weight="bold")
    line(ax, w1, 0.0, color="black", lw=2.5)
    ax.scatter(0, 0, s=60, color="black", zorder=5); ax.text(0.1, -0.25, "(0, 0)", fontsize=13)
    ax.set_title("1걸음 뒤 첫 선 ($b = 0$, 원점 지남)", fontsize=16)
    save(fig, "sv_step1.png")


TR8 = {2: "1걸음 뒤", 3: "2걸음 뒤", 4: "3걸음 뒤", 5: "4걸음 뒤", 6: "5걸음 뒤", 7: "6걸음 뒤", 10: "9걸음 뒤", 20000: "2만 걸음 뒤"}


def fig_train():
    wt, bt, pushed, shots = train_maxmin(X, y, snaps=tuple(TR8))
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.6))
    for ax, (t, ttl) in zip(axs.ravel(), TR8.items()):
        w_, b_, act, m_ = shots[t]; pts(ax); street_m(ax, w_, b_, m_, X[act])
        ax.set_title(f"{ttl}  m {m_:.2f}", fontsize=15)
    save(fig, "sv_train4.png")
    mg = y * (X @ wt + bt); o = np.argsort(mg); Mn = mg.min()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [GOLD if mg[i] < Mn + 0.01 else (RED if y[i] > 0 else BLUE) for i in o]
    ax.bar(range(20), mg[o], color=cols, alpha=0.9)
    ax.axhline(Mn, color="black", ls="--", lw=1.5); ax.text(-0.4, 2.35, f"점선: m = {Mn:.2f} = 길 가장자리", fontsize=14)
    ax.text(1, Mn + 0.2, "서포트 벡터", ha="center", fontsize=14, color=GOLDT, fontweight="bold")
    for k, i in enumerate(o[:3]):
        ax.text(k, mg[i] + 0.05, f"{mg[i]:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_ylabel("경계까지 거리"); ax.set_xlabel("점 20개 (거리 작은 순)")
    save(fig, "sv_slack_bars.png")
    T = np.unique(np.logspace(0, np.log10(MM_T), 300).astype(int)); T = T[T >= 2]
    _, _, _, sha = train_maxmin(X, y, snaps=tuple(T))
    mm = np.array([sha[t][3] for t in T])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(T - 1, mm, color="black", lw=2); ax.set_xscale("log")
    ax.axhline(MA, color=GOLD, ls="--", lw=1.5); ax.text(1.1, MA + 0.006, f"SVM 답 {MA:.2f}", fontsize=12, color=GOLDT)
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlabel("걸음 뒤 (로그 눈금)", fontsize=13); ax.set_ylabel("m (가장 가까운 점까지 거리)", fontsize=13)
    ax.set_ylim(0.55, 0.70); ax.tick_params(labelsize=12)
    save(fig, "sv_trainpath.png")
    return {t: (np.round(shots[t][0], 2), round(shots[t][1], 2), round(shots[t][3], 2), X[shots[t][2]].tolist()) for t in TR8}, [round(mg[i], 2) for i in o[:4]]


def fig_soft():
    Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [1]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    pts(axs[0], Xo, yo); axs[0].scatter(*OUTLIER, s=380, facecolors="none", edgecolors=BLUE, lw=2.5, zorder=4)
    axs[0].set_title("겹치면: 가르는 직선이 없다", fontsize=16)
    Xn = np.r_[X, [NEWPT]]; yn = np.r_[y, 1]; Mn = SVC(kernel="linear", C=1e6).fit(Xn, yn)
    pts(axs[1], Xn, yn); svm_street(axs[1], Mn, Xn); line(axs[1], WA, BA, color=GRAY, lw=2, ls=":")
    axs[1].scatter(*NEWPT, s=380, facecolors="none", edgecolors=BLUE, lw=2.5, zorder=5)
    axs[1].set_title(f"빨간 점 하나 추가: m {MA:.2f} → {M_NEW}", fontsize=16)
    save(fig, "sv_sensitive.png")
    Cs = sorted(BUDGET, reverse=True)
    ms = SVC(kernel="linear", C=Cs[-1]).fit(Xo, yo); eps = np.maximum(0, 1 - yo * ms.decision_function(Xo))
    fig, ax = plt.subplots(figsize=(6.0, 6.0)); pts(ax, Xo, yo); svm_street(ax, ms, Xo)
    lab = []
    for i in np.where(eps > 1e-3)[0]:
        c = RED if eps[i] > 1 else "#e8590c"; p = Xo[i]
        off = np.array([-0.95, 0.35]) if yo[i] > 0 else np.array([0.5, -0.2])
        if eps[i] > 1:
            off = np.array([-0.95, -0.55])
        ax.annotate(f"ε {eps[i]:.2f}", p, p + off, fontsize=12, fontweight="bold", color=c, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.9), arrowprops=dict(arrowstyle="-", color=c, lw=1.2))
        lab.append((tuple(p), round(eps[i], 2)))
    ax.set_title(f"ε > 0: 가장자리를 넘음 · ε > 1: 틀림\nΣε = {eps.sum():.2f} (예산 B)", fontsize=14)
    save(fig, "sv_slack.png")
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.6))
    for ax, C in zip(axs, Cs):
        m = SVC(kernel="linear", C=C).fit(Xo, yo); pts(ax, Xo, yo); svm_street(ax, m, Xo)
        bud, mg, nsv = BUDGET[C]
        ax.set_title(f"예산 B = {bud:.2f}  (sklearn C = {C})\nm = {mg:.2f}, 서포트 벡터 {nsv}개", fontsize=14)
    save(fig, "sv_soft.png")
    return lab


def fig_kernel():
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
    Xm, ym = make_moons(150, noise=0.25, random_state=0)
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.3, 1.8, 250))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, g in zip(axs, (0.1, 2, 100)):
        m = SVC(kernel="rbf", gamma=g, C=1).fit(Xm, ym)
        Zg = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, Zg > 0, alpha=0.15, colors=[BLUE, RED]); ax.contour(gx, gy, Zg, levels=[0], colors="black", linewidths=2)
        ax.scatter(Xm[ym == 1, 0], Xm[ym == 1, 1], color=RED, s=15); ax.scatter(Xm[ym == 0, 0], Xm[ym == 0, 1], color=BLUE, s=15, marker="s")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ = {g}", fontsize=16)
    save(fig, "sv_gamma.png")
    Xk, yk = make_circles(160, factor=0.45, noise=0.12, random_state=3)
    gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (ttl, mk) in zip(axs, (("선형 커널", SVC(kernel="linear", C=1)), ("3차 다항 커널", SVC(kernel="poly", degree=3, coef0=1, C=1)), ("RBF 커널", SVC(kernel="rbf", gamma=1, C=1)))):
        mk.fit(Xk, yk); Zk = mk.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, Zk > 0, alpha=0.15, colors=[BLUE, RED]); ax.contour(gx, gy, Zk, levels=[0], colors="black", linewidths=2.5)
        ax.scatter(Xk[yk == 1, 0], Xk[yk == 1, 1], color=RED, s=16); ax.scatter(Xk[yk == 0, 0], Xk[yk == 0, 1], color=BLUE, s=16, marker="s")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(f"{ttl}: 학습 정확도 {mk.score(Xk, yk):.0%}", fontsize=15)
    save(fig, "sv_kernels.png")


FIGS = {"rawz": fig_raw_z, "hook": fig_hook, "centroid": fig_centroid, "hyper": fig_hyper, "street": fig_street, "fold": fig_fold,
        "cos": fig_cos, "push": fig_push, "zscore": fig_zscore, "step1": fig_step1, "train": fig_train, "soft": fig_soft, "kernel": fig_kernel}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
