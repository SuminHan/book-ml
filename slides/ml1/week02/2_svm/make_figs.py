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
    ax.scatter(*RAW_MU, s=420, marker="+", color="black", lw=3, zorder=5); ax.text(RAW_MU[0] - 4.9, RAW_MU[1] + 1.0, f"평균 ({RAW_MU[0]}, {RAW_MU[1]})", fontsize=13)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("원래 $x_1$"); ax.set_ylabel("원래 $x_2$"); ax.set_title(f"원래 값 (표준편차 {RAW_SD[0]}, {RAW_SD[1]})", fontsize=15)
    ax = axs[1]; pts(ax); ax.scatter(0, 0, s=420, marker="+", color="black", lw=3, zorder=5); ax.text(-1.95, 0.15, "평균 (0, 0)", fontsize=13)
    ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_title("z-score 후 (평균 0, 표준편차 1)", fontsize=15)
    save(fig, "sv_raw_z.png")


def fig_hook_d():
    """분리 초평면 쪽: 세 선 + 검은 선 기준 D(x) > 0 (A, 옅은 빨강) / D(x) < 0 (B, 옅은 파랑) 영역"""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    g = np.linspace(-L, L, 400); G1, G2 = np.meshgrid(g, g); Dg = G2 - (-1.0 * G1 - 0.05)
    ax.contourf(G1, G2, Dg > 0, levels=[-0.5, 0.5, 1.5], colors=[BLUE, RED], alpha=0.10, zorder=0)
    pts(ax)
    for (a, c), col in (((-0.625, -0.06), GRAY), ((-1.82, 0.55), GRAY), ((-1.0, -0.05), "black")):
        ax.plot(xs, a * xs + c, color=col, lw=2.2)
    ax.text(0.15, 1.95, "$D(\\mathbf{x}) > 0$: A", fontsize=16, color=RED, fontweight="bold", va="top")
    ax.text(-2.1, -1.85, "$D(\\mathbf{x}) < 0$: B", fontsize=16, color=BLUE, fontweight="bold", va="center")
    save(fig, "sv_hook_d.png")


def fig_hook():
    fig, ax = plt.subplots(figsize=(5.6, 5.6)); pts(ax)
    for (a, c), col in (((-0.625, -0.06), GRAY), ((-1.82, 0.55), GRAY), ((-1.0, -0.05), "black")):
        assert (np.sign(X[:, 1] - (a * X[:, 0] + c)) == y).all(), (a, c)          # 세 선 모두 완벽하게 가른다
        ax.plot(xs, a * xs + c, color=col, lw=2.2)
    ax.text(-1.2, 1.4, "?", fontsize=34, color=RED, fontweight="bold")
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


def fig_flip():
    """파랑 부호 뒤집기: −x = 원점 기준 180° 회전 (점대칭). 왼쪽 원래, 오른쪽 뒤집은 뒤"""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
    pts(axs[0]); axs[0].set_title("원래", fontsize=16)
    ax = axs[1]; pts(ax, X[y > 0], y[y > 0]); F = -X[y < 0]
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=40, color=BLUE, marker="s", alpha=0.18)
    ax.scatter(F[:, 0], F[:, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=2, zorder=3)
    for p in X[y < 0]:
        ax.plot([p[0], -p[0]], [p[1], -p[1]], color=BLUE, lw=0.8, alpha=0.2)
    ex = np.array([-1.0, -0.6])
    ax.plot([ex[0], -ex[0]], [ex[1], -ex[1]], color=BLUE, lw=1.6, ls="--", zorder=2)
    ax.annotate("", -ex, ex, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, connectionstyle="arc3,rad=-0.9", shrinkA=8, shrinkB=8), zorder=4)
    ax.text(-1.55, 0.8, "원점 기준\n180° 회전", fontsize=14, color=BLUE, weight="bold", ha="center")
    ax.text(ex[0] - 0.05, ex[1] - 0.3, "$\\mathbf{x}$", fontsize=15, color=BLUE, ha="center")
    ax.text(-ex[0] + 0.05, -ex[1] - 0.3, "$-\\mathbf{x}$", fontsize=15, color=BLUE, ha="center")
    ax.set_title("파랑 $\\times (-1)$: 20개가 모두 한쪽에", fontsize=16)
    save(fig, "sv_flip.png")


def fig_people():
    """원 논문 저자 네 명 (사진 출처는 덱 끝 '이미지 출처' 쪽): 1992 Boser · Guyon · Vapnik, 1995 Cortes · Vapnik"""
    import matplotlib.image as mpimg
    P = [("boser", "Bernhard Boser"), ("guyon", "Isabelle Guyon"), ("vapnik", "Vladimir Vapnik"), ("cortes", "Corinna Cortes")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (f, nm) in zip(axs, P):
        ax.imshow(mpimg.imread(f"figs/people/{f}.jpg")); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#999999")
        ax.set_xlabel(nm, fontsize=26, labelpad=8)
    save(fig, "sv_people.png")


def fig_flip_mean():
    """한 줄 세 장: ① 원래 ② 파랑 × (−1) = 원점 기준 180° 회전 (20개가 한쪽에) ③ 20개 y_k x_k 의 평균 = w, 그에 수직이고 원점을 지나는 선 (b = 0)"""
    a = (y[:, None] * X).mean(0); F = -X[y < 0]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.9))
    pts(axs[0]); axs[0].set_title("① 원래 점 20개", fontsize=17)
    ax = axs[1]; pts(ax, X[y > 0], y[y > 0])
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=40, color=BLUE, marker="s", alpha=0.18)
    ax.scatter(F[:, 0], F[:, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=2, zorder=3)
    for p in X[y < 0]:
        ax.plot([p[0], -p[0]], [p[1], -p[1]], color=BLUE, lw=0.8, alpha=0.2)
    ex = np.array([-1.0, -0.6])
    ax.plot([ex[0], -ex[0]], [ex[1], -ex[1]], color=BLUE, lw=1.6, ls="--", zorder=2)
    ax.annotate("", -ex, ex, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, connectionstyle="arc3,rad=-0.9", shrinkA=8, shrinkB=8), zorder=4)
    ax.text(-1.55, 0.8, "원점 기준\n180° 회전", fontsize=14, color=BLUE, weight="bold", ha="center")
    ax.text(ex[0] - 0.05, ex[1] - 0.3, "$\\mathbf{x}$", fontsize=15, color=BLUE, ha="center")
    ax.text(-ex[0] + 0.05, -ex[1] - 0.3, "$-\\mathbf{x}$", fontsize=15, color=BLUE, ha="center")
    ax.set_title("② 파랑 $\\times (-1)$: 20개가 모두 한쪽에", fontsize=17)
    ax = axs[2]; pts(ax, X[y > 0], y[y > 0])
    ax.scatter(F[:, 0], F[:, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=1.8, zorder=3)
    arw(ax, (0, 0), a, GOLD, 4); line(ax, a, 0.0, color="black", lw=2.8)
    ax.text(-2.05, -1.25, f"평균 $\\mathbf{{w}}$ = ({MEAN_W[0]:.2f}, {MEAN_W[1]:.2f})", fontsize=14, color=GOLDT, weight="bold", ha="left", bbox=dict(fc="white", ec="none", alpha=0.9))
    ax.annotate("", a * 0.45, (-0.9, -1.05), arrowprops=dict(arrowstyle="-", color=GOLDT, lw=1))
    ax.set_title("③ 20개의 평균 = $\\mathbf{w}$, 수직인 선 ($b = 0$)", fontsize=17)
    save(fig, "sv_flip_mean.png")


def fig_centroid():
    """가장 쉬운 방법: 빨강은 그대로, 파랑은 부호를 뒤집어 20개 y_i x_i 의 평균 = w, b = 0 → 원점을 지나고 w 에 수직인 선 (= 학습 1걸음 뒤 첫 선)"""
    a = (y[:, None] * X).mean(0)
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax)
    F = -X[y < 0]
    ax.scatter(F[:, 0], F[:, 1], s=80, facecolors="none", edgecolors=BLUE, marker="s", lw=1.8, zorder=3)
    for p in X[y < 0]:
        ax.annotate("", -p, p, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, alpha=0.25, shrinkA=6, shrinkB=6))
    arw(ax, (0, 0), a, GOLD, 4)
    ax.text(0.15, -1.55, f"평균 $\\mathbf{{w}}$ = ({MEAN_W[0]:.2f}, {MEAN_W[1]:.2f})", fontsize=14, color=GOLDT, weight="bold", ha="left", bbox=dict(fc="white", ec="none", alpha=0.9))
    ax.annotate("", a * 0.6, (0.9, -1.35), arrowprops=dict(arrowstyle="-", color=GOLDT, lw=1))
    line(ax, a, 0.0, color="black", lw=2.8)
    ax.text(0.9, -2.05, "□ = 파랑 $\\times (-1)$", fontsize=13, color=BLUE)
    save(fig, "sv_centroid.png")
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.2))
    for ax, (w_, b_, ttl) in zip(axs, ((a, 0.0, "모든 벡터의 평균"), (WA, BA, "SVM"))):
        pts(ax); o = line_dist(ax, w_, b_, X, y)
        for s, (p, dd) in o.items():
            ax.text(p[0] + (-0.8 if s < 0 else 0.25), p[1] + (0.75 if s < 0 else -0.55), f"{dd:.2f}", fontsize=20, weight="bold", color=RED if s > 0 else BLUE)
        ax.set_title(ttl, fontsize=18)
    line(axs[1], a, 0.0, color=GRAY, ls="--", lw=1.5)
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
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", lw=2, color="#666666", shrinkA=0, shrinkB=0))
    ax.text(0.25, -1.75, f"길 폭 2M = {2 * MA:.2f}", fontsize=15, fontweight="bold", color="#666666", ha="left", va="top")
    save(fig, "sv_street.png")
    X2 = X.copy(); X2[10 + BLUE_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
    M2 = SVC(kernel="linear", C=1e6).fit(X2, y)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    pts(axs[0]); svm_street(axs[0], M); axs[0].set_title("처음", fontsize=16)
    pts(axs[1], X2); svm_street(axs[1], M2, X2); axs[1].set_title("멀리 있는 파랑 점을 옮겨도", fontsize=16)
    axs[1].annotate("", FAR_MOVE[1], FAR_MOVE[0], arrowprops=dict(arrowstyle="->", lw=2, color=BLUE, shrinkA=9, shrinkB=9))
    axs[1].add_patch(plt.Circle(FAR_MOVE[0], 0.13, fill=False, ls="--", lw=2, ec=BLUE, zorder=4)); axs[1].text(FAR_MOVE[0][0] + 0.2, FAR_MOVE[0][1] - 0.15, "원래 자리", fontsize=12, color=BLUE)
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
    ax.text(2.1, -2.05, f"접은 뒤 가장 가까운 거리\nM = {MA:.2f}", fontsize=13, color=GOLDT, ha="right", bbox=dict(fc="white", ec="none", alpha=0.9))
    save(fig, "sv_fold.png")


COS_W, COS_B, COS_X = np.array([1, 1]) / np.sqrt(2), 0.5 / np.sqrt(2), np.array([0.4, 1.6])   # 경계 x1 + x2 = −0.5 (b = +0.35, 값 = w·x + b 를 바로 더하기), 점 (0.4, 1.6)


def fig_cos():
    """점곱 = ‖w‖‖x‖cos θ → ‖w‖ = 1 이면 w 방향 위치, 경계의 위치 −b 를 빼면 거리"""
    Bu, PX = COS_W, COS_X
    sh, shb = Bu @ PX, -COS_B; dist = sh - shb
    assert abs(sh - 1.41) < 0.01 and abs(COS_B - 0.35) < 0.01 and abs(dist - 1.77) < 0.01 and abs(sh + COS_B - dist) < 1e-9
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.set_xlim(-1.2, 2.2); ax.set_ylim(-1.0, 2.1); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    t = np.linspace(-1.2, 2.1, 2); ax.plot(t, t, color=GOLD, lw=1.5, ls="--")
    ax.plot(t, -0.5 - t, color="black", lw=2.5)
    F = sh * Bu; Fb = shb * Bu
    arw(ax, (0, 0), PX, RED, 2.5); arw(ax, (0, 0), 0.5 * Bu, GOLD, 3.5)
    nrm = np.array([1, -1]) / np.sqrt(2)

    def dim(a, b, off, col, lab):
        pa, pb = a * Bu + off * nrm, b * Bu + off * nrm
        ax.annotate("", pb, pa, arrowprops=dict(arrowstyle="<->", color=col, lw=2.5, shrinkA=0, shrinkB=0))
        for q, base in ((pa, a * Bu), (pb, b * Bu)):
            ax.plot(*zip(base, q + 0.04 * nrm), color=col, lw=1, ls=":")
        mid = (pa + pb) / 2 + 0.11 * nrm
        if lab: ax.text(*mid, lab, fontsize=13, color=col, rotation=45, ha="center", va="center", fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    dim(0, sh, 0.5, BLUE, f"x 의 w 방향 위치 $\\Vert\\mathbf{{x}}\\Vert\\cos\\theta$ = {sh:.2f}")
    dim(0, shb, 0.24, "#555555", "")
    mid = shb / 2 * Bu + 0.24 * nrm
    ax.annotate(f"원점에서 경계까지\nb = {COS_B:.2f}", mid, (0.55, -0.75), fontsize=13, color="#555555", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=1), bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    ax.annotate("", F, Fb, arrowprops=dict(arrowstyle="<->", color="#2b8a3e", lw=3, shrinkA=0, shrinkB=0))
    ax.plot(*zip(PX, F), color=GRAY, lw=1.5, ls=":")
    ax.scatter(*PX, s=80, color=RED, zorder=4); ax.scatter(*Fb, s=50, color="black", zorder=4); ax.scatter(0, 0, s=50, color="black", zorder=5)
    ax.text(PX[0] + 0.05, PX[1] + 0.05, f"$\\mathbf{{x}} = ({PX[0]:g}, {PX[1]:g})$", fontsize=15, color=RED)
    ax.text(-0.12, 0.3, "$\\mathbf{w}$", fontsize=16, color=GOLDT, fontweight="bold")
    ax.text(0.17, 0.33, "θ", fontsize=16)
    foot = PX - dist * Bu                                                          # 점 x 에서 경계선에 내린 수선의 발
    ax.annotate("", foot, PX, arrowprops=dict(arrowstyle="<->", color="#2b8a3e", lw=3, shrinkA=4, shrinkB=0), zorder=5)
    ax.scatter(*foot, s=30, color="#2b8a3e", zorder=5)
    ax.text(-0.6, 1.75, f"점 x 에서 경계선까지\n거리 {dist:.2f}", fontsize=14, color="#2b8a3e", fontweight="bold", ha="center")
    ax.text(-1.15, -0.85, "경계 $x_1 + x_2 = -0.5$", fontsize=13)
    origin_axes(ax); ax.set_xticks([1, 2]); ax.set_yticks([1, 2])
    save(fig, "sv_cos.png")


def fig_push(t=2, S=1.1, KW=10, KB=10):
    """한 번 밀기 (식 순서대로): ① w' = w + η y_k x_k → 법선이 돈다, ② b' = b + η y_k → 그 선이 수직으로 평행이동. 화살표는 KW · KB 배"""
    _, _, _, sh = train_maxmin(X, y, snaps=(t,), T=t)
    w, b, act, m = sh[t]; k = np.where(act)[0][0]; xk, yk = X[k], y[k]; col = RED if yk > 0 else BLUE
    eta = MM_LR / np.sqrt(t); wf = w + eta * yk * xk; bb = b + eta * yk
    wf = wf / max(1.0, np.linalg.norm(wf))                                # 3단계: w 만 길이 1 로 (b 는 그대로)
    dist = lambda w_, b_: yk * (w_ @ xk + b_) / np.linalg.norm(w_)
    u = w / np.linalg.norm(w)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.0))
    ax = axs[0]; pts(ax); ax.set_title(r"① $\mathbf{w}' = \mathbf{w} + \eta\, y_k \mathbf{x}_k$ : 법선이 돈다", fontsize=15)
    line(ax, w, b, color=GRAY, lw=2, ls=":"); line(ax, wf, b, color="black", lw=2.6)
    ax.plot([0, xk[0]], [0, xk[1]], color=col, ls="--", lw=1.4); ax.text(xk[0] / 2 - 0.1, xk[1] / 2 + 0.12, r"$\mathbf{x}_k$", color=col, fontsize=15)
    ft = xk - (w @ xk + b) * u                                            # 점 k 에서 원래 선으로 내린 수선의 발
    ax.plot(*zip(xk, ft), color=col, lw=2, ls=":")
    o = np.array([1.0, -1.0]); o = o - (w @ o + b) * u
    tip = o + S * u; add = S * KW * eta * yk * xk; tip2 = tip + add; n = np.array([-u[1], u[0]])
    arw(ax, o, tip, GRAY); ax.text(*((o + tip) / 2 + 0.22 * n), r"$\mathbf{w}$", fontsize=16, color=GRAY, ha="center", va="center")
    arw(ax, tip, tip2, col, 2.5); ax.text(*((tip + tip2) / 2 + np.array([0.0, 0.22])), r"$-\eta\,\mathbf{x}_k$" if yk < 0 else r"$+\eta\,\mathbf{x}_k$", fontsize=14, color=col, ha="center", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
    arw(ax, o, tip2, "black"); ax.text(*((o + tip2) / 2 - 0.22 * n), r"$\mathbf{w}'$", fontsize=16, ha="center", va="center")
    ax.scatter(*xk, s=380, facecolors="none", edgecolors=col, lw=3, zorder=5)
    ax.text(2.1, 2.05, f"화살표 {KW}배 · 거리 {dist(w, b):.3f} → {dist(wf, b):.3f}", fontsize=12.5, ha="right")
    ax = axs[1]; pts(ax); ax.set_title(r"② $b' = b + \eta\, y_k$ : 선에 수직으로", fontsize=15)
    uf = wf / np.linalg.norm(wf)
    line(ax, wf, b, color=GRAY, lw=2, ls=":"); line(ax, wf, bb, color="black", lw=2.6)
    foot = xk - (wf @ xk + b) / np.linalg.norm(wf) * uf
    for s_ in (-1.2, 0, 1.2):
        p = foot + s_ * np.array([-uf[1], uf[0]]); arw(ax, p, p - KB * (bb - b) / np.linalg.norm(wf) * uf, GOLD)
    ax.scatter(*xk, s=380, facecolors="none", edgecolors=col, lw=3, zorder=5)
    ax.text(2.1, 2.05, f"화살표 {KB}배 · 거리 {dist(wf, b):.3f} → {dist(wf, bb):.3f}", fontsize=12.5, color=GOLDT, ha="right")
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
    ax.set_title(f"z-score 하고 학습: 2만 걸음 뒤\nM = {mz:.2f} (SVM 답)", fontsize=15)
    save(fig, "sv_zscore.png")
    return int(bad.sum()), round(mz, 2)


def fig_step1():
    """1걸음 뒤 첫 선: w = 0.05 × Σ y_i x_i 를 길이 1 로, b = 0 이라 원점을 지난다"""
    w1 = np.array(STEP1_W)
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); pts(ax)
    arw(ax, (0, 0), 1.3 * w1, GOLD, 3.5)
    ax.text(*(1.3 * w1 + np.array([-0.35, 0.12])), f"$\\mathbf{{w}}$ = ({w1[0]:.2f}, {w1[1]:.2f})", fontsize=15, color=GOLDT, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85))
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
        ax.set_title(f"{ttl}  M {m_:.2f}", fontsize=15)
    save(fig, "sv_train4.png")
    mg = y * (X @ wt + bt); o = np.argsort(mg); Mn = mg.min()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [GOLD if mg[i] < Mn + 0.01 else (RED if y[i] > 0 else BLUE) for i in o]
    ax.bar(range(20), mg[o], color=cols, alpha=0.9)
    ax.axhline(Mn, color="black", ls="--", lw=1.5); ax.text(-0.4, 2.35, f"점선: M = {Mn:.2f} = 길 가장자리", fontsize=14)
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
    ax.set_xlabel("걸음 뒤 (로그 눈금)", fontsize=13); ax.set_ylabel("M (가장 가까운 점까지 거리)", fontsize=13)
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
    axs[1].set_title(f"빨간 점 하나 추가: M {MA:.2f} → {M_NEW}", fontsize=16)
    save(fig, "sv_sensitive.png")
    sol = {C: train_soft(Xo, yo, C, T=200000 if C == 100 else MM_T)[:3] for C in SOFT_C}   # 직접 짠 소프트 마진 학습 (data.train_soft, 논문 C 꼴)
    C = 0.5; w, b, m = sol[C]; v = yo * (Xo @ w + b); xi = np.maximum(0, 1 - v)          # ξ_k = 모자란 값 (가장 가까운 값 = 1 기준)
    fig, ax = plt.subplots(figsize=(6.0, 6.0)); pts(ax, Xo, yo); street_m(ax, w, b, m, Xo[v <= 1 + 2e-3])
    lab = []
    for i in np.where(xi > 1e-3)[0]:
        c = RED if xi[i] > 1 else "#e8590c"; p = Xo[i]
        POS = {(0.7, 0.2): (1.15, -0.2), (-0.6, -0.4): (-0.95, 0.4), (0.1, -1.1): (0.75, -1.55), (-0.2, -0.6): (0.45, -0.5)}   # 글자 자리 (다른 점과 안 겹치게)
        ax.annotate(f"ξ {xi[i]:.2f}", p, POS.get(tuple(np.round(p, 1)), tuple(p + np.array([0.5, 0.4]))), fontsize=12, fontweight="bold", color=c, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.9), arrowprops=dict(arrowstyle="-", color=c, lw=1.2))
        lab.append((tuple(p), round(xi[i], 2)))
    ax.set_title(f"C = 0.5: M = {m:.2f}\nξ > 0: 길 안 · ξ > 1: 틀림", fontsize=14)
    save(fig, "sv_slack.png")
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.6))
    for ax, C in zip(axs, (100, 3, 0.5)):
        w, b, m = sol[C]; v = yo * (Xo @ w + b); pts(ax, Xo, yo); street_m(ax, w, b, m, Xo[v <= 1 + 2e-3])
        mg, nsv, _ = SOFT_C[C]
        ax.set_title(f"C = {C:g}\nM = {mg:.2f}, 서포트 벡터 {nsv}개", fontsize=14)
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
    for ax, (ttl, mk) in zip(axs, (("선형 커널", SVC(kernel="linear", C=1)), ("다항 커널 (q = 3)", SVC(kernel="poly", degree=3, gamma=1, coef0=1, C=1)), ("RBF 커널 (γ = 1)", SVC(kernel="rbf", gamma=1, C=1)))):
        mk.fit(Xk, yk); Zk = mk.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, Zk > 0, alpha=0.15, colors=[BLUE, RED]); ax.contour(gx, gy, Zk, levels=[0], colors="black", linewidths=2.5)
        ax.scatter(Xk[yk == 1, 0], Xk[yk == 1, 1], color=RED, s=16); ax.scatter(Xk[yk == 0, 0], Xk[yk == 0, 1], color=BLUE, s=16, marker="s")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(f"{ttl}\n학습 정확도 {mk.score(Xk, yk):.0%}", fontsize=15)
    save(fig, "sv_kernels.png")


def fig_lock():
    """왜 2~3개로 정해지나: ① 한쪽만 닿음 → 평행이동하면 넓어짐 ② 양쪽 1개씩 → 돌리면 넓어짐 ③ 3개 닿음 → 더 못 넓힘"""
    def best_b(u):
        r, q = (X[y > 0] @ u).min(), (X[y < 0] @ u).max(); return (r - q) / 2, -(r + q) / 2
    th0 = np.arctan2(WA[1], WA[0])
    u2 = np.array([np.cos(th0 - np.radians(14)), np.sin(th0 - np.radians(14))]); m2, b2 = best_b(u2)
    m1 = MA - 0.15; b1 = BA - 0.15                                    # 최적 방향, 파랑 쪽으로 0.15 치우친 선
    cases = [(WA, b1, "① 한쪽만 닿음", "평행이동하면 넓어진다"), (u2, b2, "② 양쪽 1개씩 닿음", "돌리면 넓어진다"), (WA, BA, "③ 3개 닿음", "더 못 넓힌다 = 답")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6)); out = []
    for ax, (w_, b_, ttl, sub) in zip(axs, cases):
        v = y * (X @ w_ + b_); m_ = v.min(); touch = v < m_ + 0.005
        pts(ax); street_m(ax, w_, b_, m_, X[touch])
        ax.set_title(f"{ttl}: M = {m_:.2f}\n→ {sub}", fontsize=15, color="black" if ttl[0] != "③" else GOLDT)
        out.append((round(m_, 2), int(touch.sum())))
    arw(axs[0], np.array([-0.2, -0.6]) + 0.0, np.array([-0.2, -0.6]) - 0.35 * WA, GOLD, 3)
    from matplotlib.patches import FancyArrowPatch
    axs[1].add_patch(FancyArrowPatch((1.2, -1.6), (1.75, -0.9), connectionstyle="arc3,rad=0.4", arrowstyle="-|>", mutation_scale=20, color=GOLD, lw=3))
    save(fig, "sv_lock.png")
    return out


def fig_two_three():
    """2개 vs 3개: ① 가장 가까운 쌍의 수직이등분선 길이 깨끗하면 2개 (작은 예) ② 우리 데이터: 그 길에 점이 걸린다 ③ 같은 색 2개가 기울기, 반대 색 1개가 폭"""
    TR, TB = np.array([(0.5, 0.5), (1.5, 1.5), (0.5, 1.6), (1.6, 0.6)]), np.array([(-0.5, -0.5), (-1.5, -1.2), (-0.8, -1.6), (-1.6, -0.3)])
    Xt = np.r_[TR, TB]; yt = np.r_[np.ones(4), -np.ones(4)]
    mt = SVC(kernel="linear", C=1e6).fit(Xt, yt); wt, bt, mmt = unit(mt); assert len(mt.support_) == 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6)); out = {"toy": (np.round(wt, 2), round(bt, 2), round(mmt, 2))}
    ax = axs[0]; pts(ax, Xt, yt); street_m(ax, wt, bt, mmt, Xt[mt.support_])
    p, q = Xt[mt.support_]; ax.plot(*zip(p, q), color=GOLD, lw=2, ls="--")
    ax.set_title(f"① 2개: 가장 가까운 쌍의 수직이등분선\n길 안이 비었다 → M = {mmt:.2f}", fontsize=14)
    R, B = X[y > 0], X[y < 0]; D = np.linalg.norm(R[:, None] - B[None], axis=2); i, j = np.unravel_index(D.argmin(), D.shape)
    r, s = R[i], B[j]; u = (r - s) / np.linalg.norm(r - s); bb = -u @ (r + s) / 2; mh = np.linalg.norm(r - s) / 2
    ax = axs[1]; pts(ax); street_m(ax, u, bb, mh, np.array([r, s])); ax.plot(*zip(r, s), color=GOLD, lw=2, ls="--")
    v = y * (X @ u + bb); bad = (v < mh - 1e-6)
    ax.scatter(X[bad, 0], X[bad, 1], s=420, facecolors="none", edgecolors=RED, lw=3, zorder=6)
    ax.set_title(f"② 우리 데이터: 같은 방법이면 M = {mh:.2f}\n그런데 길 안에 점이 걸린다", fontsize=14)
    out["pair"] = (r.tolist(), s.tolist(), round(mh, 2), X[bad].tolist())
    ax = axs[2]; pts(ax); street_m(ax, WA, BA, MA, np.array(SV))
    b1, b2 = np.array(SV[1]), np.array(SV[2]); d = (b2 - b1) / np.linalg.norm(b2 - b1)
    ax.plot(*zip(b1 - 1.2 * d, b2 + 1.2 * d), color=BLUE, lw=3.5, alpha=0.5)
    rr = np.array(SV[0]); foot = rr - 2 * MA * WA
    ax.annotate("", foot, rr, arrowprops=dict(arrowstyle="<->", color="#666666", lw=2.2, shrinkA=0, shrinkB=0))
    ax.text(*(rr - MA * WA + np.array([0.25, -0.35])), f"2M = {2 * MA:.2f}", fontsize=13, color="#666666", weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_title("③ 3개: 파랑 2개가 기울기, 빨강 1개가 폭\n" + f"M = {MA:.2f}", fontsize=14, color=GOLDT)
    save(fig, "sv_two_three.png")
    return out


TR9 = {2: "1걸음 뒤", 3: "2걸음 뒤", 4: "3걸음 뒤", 5: "4걸음 뒤", 6: "5걸음 뒤", 11: "10걸음 뒤", 101: "100걸음 뒤", 1001: "1,000걸음 뒤", 20000: "2만 걸음 뒤"}


def fig_train9():
    """걸음별 9장 (3장씩 3쪽): 길 · 미는 점 · 경계선 식"""
    _, _, _, sh = train_maxmin(X, y, snaps=tuple(TR9)); out = []
    items = list(TR9.items())
    for part in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6))
        for ax, (t, ttl) in zip(axs, items[3 * part:3 * part + 3]):
            w_, b_, act, m_ = sh[t]; pts(ax); street_m(ax, w_, b_, m_, X[act])
            sg = "+" if b_ >= 0 else "−"
            ax.set_title(f"{ttl}   M = {m_:.2f}\n${w_[0]:.2f}\\,x_1 + {w_[1]:.2f}\\,x_2 {sg} {abs(b_):.2f} = 0$", fontsize=18)
            out.append((t, round(w_[0], 2), round(w_[1], 2), round(b_, 2), round(m_, 2), X[act].tolist()))
        save(fig, f"sv_train9_{part + 1}.png")
    return out


TS9 = {2: "1걸음 뒤", 6: "5걸음 뒤", 11: "10걸음 뒤", 21: "20걸음 뒤", 51: "50걸음 뒤", 101: "100걸음 뒤", 201: "200걸음 뒤", 1001: "1,000걸음 뒤", 20000: "2만 걸음 뒤"}


def fig_soft9(C=3):
    """소프트 마진 걸음별 9장 (3장씩 3쪽): 길 (M = 1/‖w‖) · 미는 점 V (값 y_k D(x_k) < 1 인 점 모두) · 경계선 식. 이상치 포함 21개, C = 3"""
    Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [1]]
    _, _, _, sh = train_soft(Xo, yo, C, snaps=tuple(TS9)); out = []
    items = list(TS9.items())
    for part in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6))
        for ax, (t, ttl) in zip(axs, items[3 * part:3 * part + 3]):
            w_, b_, V, m_ = sh[t]; pts(ax, Xo, yo)
            if m_ > 1e-6:
                street_m(ax, w_, b_, m_, Xo[V], ring_col=RED)
            else:
                line(ax, w_, b_, color="black", lw=2.6); ax.scatter(*Xo[V].T, s=380, facecolors="none", edgecolors=RED, lw=2.5, zorder=5)
            sg = "+" if b_ >= 0 else "−"
            ax.set_title(f"{ttl}   M = {m_:.2f}, 미는 점 {int(V.sum())}개\n${w_[0]:.2f}\\,x_1 + {w_[1]:.2f}\\,x_2 {sg} {abs(b_):.2f} = 0$", fontsize=17)
            out.append((t, round(w_[0], 2), round(w_[1], 2), round(b_, 2), round(m_, 2), int(V.sum())))
        save(fig, f"sv_soft9_{part + 1}.png")
    return out


def fig_rbf():
    """RBF 커널 = 두 점의 가까움 점수: K = exp(−γ d²), 같은 점이면 1, 멀면 0. γ 0.5 · 2 · 10"""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    d = np.linspace(0, 3, 300)
    for g, c in ((0.5, GOLD), (2, "black"), (10, RED)):
        ax.plot(d, np.exp(-g * d ** 2), color=c, lw=3, label=f"γ = {g}")
    ax.set_xlabel("두 점 사이 거리 $\\Vert\\mathbf{x} - \\mathbf{x}'\\Vert$"); ax.set_ylabel("$K(\\mathbf{x}, \\mathbf{x}')$"); ax.legend(frameon=False, fontsize=14)
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 3)
    save(fig, "sv_rbf.png")


def fig_rbf_degree(G=(0.25, 1, 2.5, 5), D=22):
    """γ 에 따라 RBF 의 무게가 몰리는 차수: (x·x′ = 1 일 때) d 차 항 (2γ)^d / d! 를 합 1 로 나눈 막대. 가장 큰 무게 ≈ 2γ 차 (포아송 분포 모양)"""
    from math import factorial
    fig, axs = plt.subplots(1, len(G), figsize=(15, 4.4), sharey=True); out = {}
    d = np.arange(D)
    for ax, g in zip(axs, G):
        wts = np.array([(2 * g) ** k / factorial(k) for k in d]); wts = wts / wts.sum()
        ax.bar(d, wts, color=GOLD, alpha=0.9); top = int(np.argmax(wts))
        ax.bar([top], [wts[top]], color=RED)
        ax.set_title(f"γ = {g}: 무게 중심 ≈ {2 * g:g}차", fontsize=15)
        ax.set_xlabel("다항 차수 q"); ax.set_xticks(range(0, D, 4))
        out[g] = (top, round(float((d * wts).sum()), 2))
    axs[0].set_ylabel("q 차 항의 몫")
    save(fig, "sv_rbf_degree.png")
    return out


def fig_hills(G=1.0):
    """반달 데이터 (z-score): 원점 언덕 1개를 셋째 특징으로 (x1, x2, e^{-γ‖x‖²}) + 선형 SVM  vs  RBF SVM (서포트 벡터마다 언덕). 배경 = f(x), 검은 선 = 경계"""
    Xd, yd = make_moons(150, noise=0.15, random_state=0); Xd = (Xd - Xd.mean(0)) / Xd.std(0)
    hill = lambda A: np.c_[A, np.exp(-G * (A ** 2).sum(1))]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6)); out = {}
    L = 2.9; gx, gy = np.meshgrid(np.linspace(-L, L, 260), np.linspace(-L, L, 260)); P = np.c_[gx.ravel(), gy.ravel()]
    for r, ax in enumerate(axs):
        if r == 0:
            m = SVC(kernel="linear", C=1).fit(hill(Xd), yd); Z = m.decision_function(hill(P)); acc = m.score(hill(Xd), yd)
            ax.scatter([0], [0], marker="*", s=480, color=GOLD, edgecolor="black", zorder=6)
            ttl = f"특징 3개: $x_1, x_2$, 원점 언덕 높이\n정확도 {acc:.0%}"
        else:
            m = SVC(kernel="rbf", gamma=G, C=1).fit(Xd, yd); Z = m.decision_function(P); acc = m.score(Xd, yd)
            sv = Xd[m.support_]; sg = np.sign(m.dual_coef_[0])
            ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none", edgecolors=np.where(sg > 0, RED, BLUE), lw=1.8, zorder=5)
            ttl = f"RBF: 서포트 벡터 {len(sv)}개 자리마다 언덕\n정확도 {acc:.0%}"
        Z = Z.reshape(gx.shape); lim = np.abs(Z).max()
        ax.contourf(gx, gy, Z, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r", alpha=0.45)
        ax.contour(gx, gy, Z, levels=[0], colors="black", linewidths=2.5)
        ax.scatter(Xd[yd == 1, 0], Xd[yd == 1, 1], color=RED, s=14, zorder=4); ax.scatter(Xd[yd == 0, 0], Xd[yd == 0, 1], color=BLUE, s=14, marker="s", zorder=4)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(ttl, fontsize=15)
        for s_ in ax.spines.values(): s_.set_visible(True)
        out[r] = round(acc, 3)
    save(fig, "sv_hills.png")
    return out


def fig_hill_count(G=1.0):
    """RBF 피처 수: 반달 150점 → 언덕 후보 150개 (γ 하나, 폭 같음) → 학습 뒤 α ≠ 0 인 서포트 벡터 언덕만 남음"""
    Xd, yd = make_moons(150, noise=0.15, random_state=0); Xd = (Xd - Xd.mean(0)) / Xd.std(0)
    m = SVC(kernel="rbf", gamma=G, C=1).fit(Xd, yd); sv = m.support_; al = m.dual_coef_[0]
    L = 2.9; gx, gy = np.meshgrid(np.linspace(-L, L, 240), np.linspace(-L, L, 240)); P = np.c_[gx.ravel(), gy.ravel()]
    hills = lambda C_, W: (np.exp(-G * ((P[:, None, :] - C_[None]) ** 2).sum(2)) * W).sum(1).reshape(gx.shape)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
    Z0 = hills(Xd, np.where(yd == 1, 1.0, -1.0)) / 8                       # 후보 언덕: 높이를 모두 같게 (부호만)
    Z1 = m.decision_function(P).reshape(gx.shape)
    for ax, Z, ttl in ((axs[0], Z0, f"① 언덕 후보 {len(Xd)}개 = 학습 점마다 하나\n(폭은 γ = {G:g} 하나로 모두 같다)"),
                       (axs[1], Z1, f"③ 학습 뒤: α ≠ 0 인 언덕 {len(sv)}개만 남음\n(나머지 {len(Xd) - len(sv)}개는 α = 0)")):
        lim = np.abs(Z).max(); ax.contourf(gx, gy, Z, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r", alpha=0.45)
        ax.scatter(Xd[yd == 1, 0], Xd[yd == 1, 1], color=RED, s=14, zorder=4); ax.scatter(Xd[yd == 0, 0], Xd[yd == 0, 1], color=BLUE, s=14, marker="s", zorder=4)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(ttl, fontsize=14)
        for s_ in ax.spines.values(): s_.set_visible(True)
    axs[0].scatter(Xd[:, 0], Xd[:, 1], s=110, facecolors="none", edgecolors=np.where(yd == 1, RED, BLUE), lw=1.0, alpha=0.7, zorder=5)
    axs[1].scatter(Xd[sv, 0], Xd[sv, 1], s=130, facecolors="none", edgecolors=np.where(al > 0, RED, BLUE), lw=1.8, zorder=5)
    axs[1].contour(gx, gy, Z1, levels=[0], colors="black", linewidths=2.5)
    save(fig, "sv_hill_count.png")
    return len(Xd), len(sv)


FIGS = {"hookd": fig_hook_d, "people": fig_people, "flipmean": fig_flip_mean, "rawz": fig_raw_z, "hook": fig_hook, "flip": fig_flip, "centroid": fig_centroid, "hyper": fig_hyper, "street": fig_street, "lock": fig_lock, "twothree": fig_two_three, "fold": fig_fold,
        "cos": fig_cos, "push": fig_push, "zscore": fig_zscore, "step1": fig_step1, "train": fig_train, "train9": fig_train9, "soft": fig_soft, "soft9": fig_soft9}
# 커널 그림 (fig_kernel · fig_rbf · fig_rbf_degree · fig_hills · fig_hill_count) 은 ../3_svm_kernel/make_figs.py 가 불러서 그 폴더 figs/ 에 만든다

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
