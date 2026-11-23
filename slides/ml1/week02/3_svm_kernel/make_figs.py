"""커널 SVM 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다).
앞 차시 덱의 그림 함수(../2_svm/make_figs.py: fig_kernel · fig_rbf · fig_rbf_degree · fig_hills · fig_hill_count)를 불러 이 폴더 figs/ 에 만들고,
이 덱에서 새로 그리는 그림(kn_*)은 여기 있다. 숫자는 data.py (이 폴더) 와 ../2_svm/data.py."""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2_svm"))
import make_figs as mf                                     # ../2_svm/make_figs.py (그 폴더 data.py 를 불러온다)
os.chdir(HERE)
mf.O = os.path.join(HERE, "figs") + "/"
_s = importlib.util.spec_from_file_location("kdata", os.path.join(HERE, "data.py")); kd = importlib.util.module_from_spec(_s); _s.loader.exec_module(kd)
import numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
BLUE, RED, GOLD, GRAY = mf.BLUE, mf.RED, mf.GOLD, mf.GRAY


def fig_recap():
    """지난 시간 한 쪽: ① 경계와 값 = 거리 ② 최대 마진 ③ 소프트 마진 (λ = 5)"""
    X, y, WA, BA = mf.X, mf.y, mf.WA, mf.BA
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    ax = axs[0]; mf.pts(ax); mf.street_m(ax, WA, BA, 0)
    p = np.array([-0.2, -1.4]); d = -(WA @ p + BA); q = p + d * WA                  # ‖w‖ = 1 이라 값 = 거리, q = 경계 위 발
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="<->", lw=2, color="black", shrinkA=6, shrinkB=0), zorder=6)
    ax.text(-0.45, -1.62, f"값 = 거리 = {d:.2f}", fontsize=14, fontweight="bold", va="top")
    ax.text(-1.05, 2.05, "$\\mathbf{w}\\cdot\\mathbf{x} + b = 0$", fontsize=14, va="top")
    ax.set_title("① 경계: $\\Vert\\mathbf{w}\\Vert = 1$ 이면 값 = 거리", fontsize=15)
    ax = axs[1]; mf.pts(ax); mf.svm_street(ax, mf.M)
    ax.set_title(f"② 최대 마진: m = {mf.MA:.2f}\n경계는 서포트 벡터 3개가 정한다", fontsize=15)
    Xo = np.r_[X, [mf.OUTLIER]]; yo = np.r_[y, [1]]
    w, b, m = mf.train_soft(Xo, yo, 5)[:3]; v = yo * (Xo @ w + b)
    ax = axs[2]; mf.pts(ax, Xo, yo); mf.street_m(ax, w, b, m, Xo[v <= m + 2e-3])
    ax.set_title(f"③ 소프트 마진 λ = 5: m = {m:.2f}\n길 안에 점을 둬도 된다", fontsize=15)
    mf.save(fig, "kn_recap.png")
    return d


def fig_svarrow():
    """w = 파랑 서포트 벡터 → 빨강 서포트 벡터 화살표 두 개의 합 × 0.37"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    r, b1, b2 = (np.array(s) for s in mf.SV); ar, a1, a2 = (mf.ALPHA[tuple(s)] for s in mf.SV)
    assert abs(ar + a1 + a2) < 0.01 and abs(a1 - a2) < 0.01           # 빨강 무게 = 파랑 무게, 파랑 둘 같다
    c = -a1; d1, d2 = r - b1, r - b2; w = c * (d1 + d2)
    assert np.allclose(w, ar * r + a1 * b1 + a2 * b2) and np.allclose(w, WA, atol=0.02)
    GREEN = "#2b8a3e"
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    ax = axs[0]; mf.pts(ax, np.zeros((0, 2)), np.zeros(0)); mf.street_m(ax, WA, BA, MA, np.array(mf.SV))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, alpha=0.25, zorder=3); ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, alpha=0.25, marker="s", zorder=3)
    ax.scatter(*r, s=90, color=RED, zorder=5); ax.scatter(*np.c_[b1, b2], s=90, color=BLUE, marker="s", zorder=5)
    mf.arw(ax, b1, r, "#e8590c"); mf.arw(ax, b2, r, GREEN)
    BB = dict(fc="white", ec="none", pad=2, alpha=0.9)
    ax.text(0.35, -1.55, f"$r - b_1$ = ({d1[0]:.1f}, {d1[1]:.1f})", color="#e8590c", fontsize=14, fontweight="bold", bbox=BB, zorder=7)
    ax.text(0.35, -1.95, f"$r - b_2$ = ({d2[0]:.1f}, {d2[1]:.1f})", color=GREEN, fontsize=14, fontweight="bold", bbox=BB, zorder=7)
    for p, t, o in ((r, "$r$", (0.12, 0.1)), (b1, "$b_1$", (-0.3, 0.12)), (b2, "$b_2$", (-0.32, -0.2))):
        ax.text(p[0] + o[0], p[1] + o[1], t, fontsize=15, fontweight="bold")
    ax.set_title("① 파랑 서포트 벡터 → 빨강 서포트 벡터", fontsize=15)
    ax = axs[1]; O = np.zeros(2)
    ax.set_xlim(-0.9, 2.2); ax.set_ylim(-0.2, 2.2); ax.set_aspect("equal"); ax.tick_params(labelsize=11)
    ax.axhline(0, color=GRAY, lw=0.9, zorder=0); ax.axvline(0, color=GRAY, lw=0.9, zorder=0)
    ax.set_xlabel("$x_1$", fontsize=13); ax.set_ylabel("$x_2$", fontsize=13, rotation=0, labelpad=8)
    mf.arw(ax, O, d1, "#e8590c"); mf.arw(ax, d1, d1 + d2, GREEN); mf.arw(ax, O, w, "black", lw=4)
    ax.plot(*np.c_[O, d1 + d2], ":", color=GRAY, lw=1.5, zorder=0)
    ax.text(*(d1 / 2 + np.array([0.05, -0.22])), "$r - b_1$", color="#e8590c", fontsize=14, fontweight="bold")
    ax.text(*(d1 + d2 / 2 + np.array([0.08, -0.1])), "$r - b_2$", color=GREEN, fontsize=14, fontweight="bold")
    ax.text(*(d1 + d2 + np.array([-0.55, 0.08])), f"합 ({(d1 + d2)[0]:.1f}, {(d1 + d2)[1]:.1f})", fontsize=13)
    ax.text(-0.85, 1.0, f"합 × {c:.2f} = w\n= ({w[0]:.2f}, {w[1]:.2f})", fontsize=14, fontweight="bold")
    ax.set_title(f"② 두 화살표를 더해 {c:.2f} 배 = w", fontsize=15)
    mf.save(fig, "kn_svarrow.png")
    return dict(c=round(c, 2), d1=d1.round(1).tolist(), d2=d2.round(1).tolist(), w=w.round(2).tolist())


def fig_quiz1():
    """퀴즈 Q1: 점 20개, 서포트 벡터 3개 (좌표 · α), 새 점 ★ (답은 안 적는다)"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    SV = [np.array(s) for s in mf.SV]; AL = [mf.ALPHA[tuple(s)] for s in mf.SV]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    mf.pts(ax, np.zeros((0, 2)), np.zeros(0)); mf.street_m(ax, WA, BA, MA, np.array(SV))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, alpha=0.28, zorder=3); ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, alpha=0.28, marker="s", zorder=3)
    off = {0: (0.22, -0.05), 1: (-1.55, 0.3), 2: (0.22, -0.3)}
    for k, (p, a) in enumerate(zip(SV, AL)):
        c = RED if a > 0 else BLUE
        ax.scatter([p[0]], [p[1]], s=90, color=c, marker="o" if a > 0 else "s", zorder=5)
        ax.text(p[0] + off[k][0], p[1] + off[k][1], f"({p[0]:g}, {p[1]:g})\nα = {a:+.2f}".replace("-", "−"), fontsize=13, fontweight="bold", color=c, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(0.3, 2.1, "흐린 점 17개: α = 0", fontsize=13, color="#555555", va="top", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    nx = np.array(kd.NEW_PT)
    ax.scatter([nx[0]], [nx[1]], marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=7)
    ax.text(nx[0] + 0.22, nx[1] - 0.35, f"새 점 ({nx[0]:g}, {nx[1]:g})".replace("-", "−"), fontsize=13, fontweight="bold", color="#2B8A3E", zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_title(f"점 20개: 서포트 벡터 3개, b = {BA:.2f}", fontsize=15)
    mf.save(fig, "kn_quiz1.png")


def fig_quizkid():
    """쉬운 퀴즈 (덱처럼 ‖w‖ = 1): 정수 점 8개, w = (0.6, 0.8), b = −3.9, 마진 2.5, 서포트 벡터 A (4,5) · B (1,1), 새 점 P (6,2) · Q (0,3)"""
    R = np.array([[4, 5], [5, 5], [4, 6], [5, 6]]); B = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    w, b, m = np.array([0.6, 0.8]), -3.9, 2.5
    M = SVC(kernel="linear", C=1e6, tol=1e-10).fit(np.r_[R, B], np.r_[np.ones(4), -np.ones(4)]); n = np.linalg.norm(M.coef_[0])
    assert np.allclose(M.coef_[0] / n, w) and np.isclose(M.intercept_[0] / n, b) and np.allclose(np.abs(M.dual_coef_[0] / n), 0.2)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    t = np.linspace(-2, 8, 2); ln = lambda c: (c - b - w[0] * t) / w[1]           # w·x + b = c 인 직선의 x2
    ax.fill_between(t, ln(-m), ln(m), color=GOLD, alpha=0.18, lw=0)
    ax.plot(t, ln(0), color="black", lw=2.5); ax.plot(t, ln(m), "--", color=RED, lw=1.5); ax.plot(t, ln(-m), "--", color=BLUE, lw=1.5)
    ax.scatter(R[:, 0], R[:, 1], s=110, color=RED, zorder=4); ax.scatter(B[:, 0], B[:, 1], s=110, color=BLUE, marker="s", zorder=4)
    ax.scatter([4, 1], [5, 1], s=420, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=5)
    mf.arw(ax, np.array([1, 1]), np.array([4, 5]), "#e8590c", lw=2.5)
    ax.text(2.2, 5.25, "A (4, 5)", fontsize=14, fontweight="bold", color=RED); ax.text(1.25, 0.45, "B (1, 1)", fontsize=14, fontweight="bold", color=BLUE)
    for p, nm in (((6, 2), "P (6, 2)"), ((0, 3), "Q (0, 3)")):
        ax.scatter(*p, marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=6)
        ax.text(p[0] - 0.45, p[1] + 0.35, nm, fontsize=14, fontweight="bold", color="#2B8A3E")
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.6, 6.6); ax.set_aspect("equal"); ax.set_xticks(range(0, 7)); ax.set_yticks(range(0, 7)); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$", fontsize=14); ax.set_ylabel("$x_2$", fontsize=14, rotation=0, labelpad=10)
    ax.set_title("빨강 4개, 파랑 4개: 경계 $0.6x_1 + 0.8x_2 - 3.9 = 0$", fontsize=14)
    mf.save(fig, "kn_quizkid.png")


def fig_q():
    """오늘의 질문: 원 데이터에 직선 SVM"""
    X, y = kd.Xc, kd.yc
    m = SVC(kernel="linear", C=1).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 300), np.linspace(-1.3, 1.3, 300))
    Z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.contourf(gx, gy, Z > 0, alpha=0.12, colors=[BLUE, RED]); ax.contour(gx, gy, Z, levels=[0], colors="black", linewidths=2.5)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], color=RED, s=28); ax.scatter(X[y < 0, 0], X[y < 0, 1], color=BLUE, s=28, marker="s")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"직선 SVM: 정확도 {kd.QL_ACC:.0%}", fontsize=16)
    mf.save(fig, "kn_q.png")


def fig_lift1d():
    """x → (x, x²): 위에서는 직선 하나, 원래 수직선에서는 구간"""
    R, B, h, c = np.array(kd.L1_RED), np.array(kd.L1_BLUE), kd.L1_H, kd.L1_CUT
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axs[0]
    ax.axhline(0, color=GRAY, lw=1.2, zorder=0)
    ax.scatter(R, 0 * R, color=RED, s=110, zorder=3); ax.scatter(B, 0 * B, color=BLUE, s=110, marker="s", zorder=3)
    ax.text(0, 0.55, "어디를 끊어도\n칼 하나로는 못 가른다", ha="center", fontsize=15)
    ax.set_xlim(-2, 2); ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xlabel("$x$", fontsize=15)
    ax.spines["left"].set_visible(False); ax.set_title("① 수직선 위 (특징 1개: $x$)", fontsize=16)
    ax = axs[1]
    t = np.linspace(-1.9, 1.9, 200); ax.plot(t, t * t, color=GRAY, lw=1, ls="--", zorder=0)
    ax.scatter(R, R * R, color=RED, s=110, zorder=3); ax.scatter(B, B * B, color=BLUE, s=110, marker="s", zorder=3)
    ax.axhline(h, color="black", lw=2.5)
    ax.text(1.95, h + 0.12, f"직선: 높이 = {h}", ha="right", fontsize=14)
    for s in (-c, c):
        ax.plot([s, s], [0, h], color=GOLD, lw=2, ls="--"); ax.scatter([s], [0], color=GOLD, s=60, zorder=4)
    ax.text(0, -0.42, f"원래 자리: $x = \\pm{c:.2f}$ 에서 끊는 구간", ha="center", fontsize=14, color=mf.GOLDT)
    ax.axhline(0, color=GRAY, lw=1, zorder=0)
    ax.set_xlim(-2, 2); ax.set_ylim(-0.6, 3.2); ax.set_xlabel("$x$", fontsize=15); ax.set_ylabel("$x^2$", fontsize=15, rotation=0, labelpad=12)
    ax.set_title("② $x \\to (x,\\ x^2)$ (특징 2개)", fontsize=16)
    mf.save(fig, "kn_lift1d.png")


def fig_dual():
    """지난 시간 점 20개: 서포트 벡터 3개만 α ≠ 0, w = Σ α_i x_i (꼬리 물기)"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    SV = [np.array(s) for s in mf.SV]; AL = [mf.ALPHA[tuple(s)] for s in mf.SV]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    ax = axs[0]; mf.pts(ax, np.zeros((0, 2)), np.zeros(0)); mf.street_m(ax, WA, BA, MA, np.array(SV))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, alpha=0.28, zorder=3)          # α = 0 인 점은 흐리게
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, alpha=0.28, marker="s", zorder=3)
    for s, a in zip(SV, AL):
        ax.scatter([s[0]], [s[1]], s=80, color=RED if a > 0 else BLUE, marker="o" if a > 0 else "s", zorder=5)
    off = {0: (0.2, -0.35), 1: (0.2, 0.2), 2: (0.2, -0.45)}
    for k, (s, a) in enumerate(zip(SV, AL)):
        dx, dy = off[k]
        ax.text(s[0] + dx, s[1] + dy, f"$\\alpha$ = {a:+.2f}".replace("-", "−"), fontsize=14, fontweight="bold", color=RED if a > 0 else BLUE, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(0.25, 2.1, "흐린 점 17개: $\\alpha$ = 0", fontsize=13, color="#555555", va="top", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    nx = np.array(kd.NEW_PT)
    ax.scatter([nx[0]], [nx[1]], marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=7)
    ax.text(nx[0] + 0.2, nx[1] - 0.05, f"새 점 ({nx[0]:g}, {nx[1]:g})\nf = {kd.NEW_F:+.2f} → 빨강".replace("-", "−"), fontsize=13, fontweight="bold",
            color="#2B8A3E", va="top", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_title("① 서포트 벡터 3개만 $\\alpha \\neq 0$", fontsize=16)
    ax = axs[1]
    cur = np.zeros(2); cols = [RED if a > 0 else BLUE for a in AL]
    for k, (s, a, c) in enumerate(zip(SV, AL, cols)):
        v = a * s
        ax.annotate("", cur + v, cur, arrowprops=dict(arrowstyle="-|>", lw=2.5, color=c, mutation_scale=18))
        mid = cur + v / 2
        lab = f"{a:+.2f} · ({s[0]:g}, {s[1]:g})".replace("-", "−")
        ax.text(mid[0] + (0.06 if k != 1 else 0.06), mid[1] + (-0.1 if k == 0 else 0.0), lab, fontsize=13, color=c, ha="left", va="center")
        cur = cur + v
    ax.annotate("", cur, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color="black", mutation_scale=20))
    ax.text(cur[0] / 2 - 0.08, cur[1] / 2 + 0.08, f"합 = $\\mathbf{{w}}$ = ({cur[0]:.2f}, {cur[1]:.2f})", fontsize=14, fontweight="bold", ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    ax.scatter([0], [0], color="black", s=20)
    ax.set_xlim(-0.1, 1.0); ax.set_ylim(-0.1, 1.0); ax.set_aspect("equal"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3)
    ax.set_title("② $\\alpha_i \\mathbf{x}_i$ 를 이어 더하면 $\\mathbf{w}$", fontsize=16)
    mf.save(fig, "kn_dual.png")
    return cur.round(3)


def fig_ktrick():
    """두 길: 3차원으로 옮겨 점곱 (먼 길) vs 점곱하고 제곱 (지름길 = 커널 K), 둘 다 25"""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    def box(cx, cy, w, h, s, ec, fs=17, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02", fc="white", ec=ec, lw=2.2))
        ax.text(cx, cy, s, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.5)
    def arr(a, b, ec, lab, lp):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", lw=2.5, color=ec, mutation_scale=22, shrinkA=0, shrinkB=0))
        ax.text(lp[0], lp[1], lab, ha="center", va="center", fontsize=16, color=ec)
    p, q, pr = kd.K_PX, kd.K_PXP, kd.K_PROD
    f = lambda v: "(" + ", ".join(f"{z:g}" for z in v) + ")"
    box(1.6, 2.5, 2.8, 1.7, f"2차원 점 두 개\n$\\mathbf{{x}}$ = {f(kd.K_X)}\n$\\mathbf{{x}}_i$ = {f(kd.K_XP)}", "black")
    box(6.5, 4.0, 4.0, 1.6, "3차원 점 두 개를 만든다\n$\\varphi(\\mathbf{x})$ = (1, $2\\sqrt{2}$, 4)\n$\\varphi(\\mathbf{x}_i)$ = (9, $3\\sqrt{2}$, 1)", BLUE)
    box(6.5, 1.0, 3.8, 0.9, f"$\\mathbf{{x}}\\cdot\\mathbf{{x}}_i$ = 3 + 2 = {kd.K_DOT:g}", RED, fs=18)
    box(11.4, 2.5, 1.6, 1.2, f"{kd.K_DOT ** 2:g}", "black", fs=30, bold=True)
    arr((3.0, 3.0), (4.5, 3.8), BLUE, "① $\\varphi$ 로 옮기기", (3.3, 3.85))
    arr((8.5, 3.8), (10.5, 2.95), BLUE, "② 점곱  $1{\\cdot}9 + 2\\sqrt{2}{\\cdot}3\\sqrt{2} + 4{\\cdot}1$\n$= 9 + 12 + 4$", (11.0, 4.3))
    arr((3.0, 2.0), (4.5, 1.2), RED, "① 점곱", (3.35, 1.25))
    arr((8.5, 1.2), (10.5, 2.05), RED, f"② 제곱 {kd.K_DOT:g}²", (10.0, 1.15))
    ax.text(0.05, 4.75, "먼 길: 옮기고 점곱", ha="left", va="top", fontsize=17, color=BLUE, fontweight="bold")
    ax.text(12.95, 0.15, "지름길 = 커널 $K(\\mathbf{x}, \\mathbf{x}_i) = (\\mathbf{x}\\cdot\\mathbf{x}_i)^2$", ha="right", fontsize=17, color=RED, fontweight="bold")
    mf.save(fig, "kn_ktrick.png")

def fig_alpha():
    """걸음마다 α: 서포트 벡터 3개는 쌓이고, 나머지 17개는 0 으로"""
    ts = kd.AL_STEPS; H = np.array([kd.AL_HIST[t] for t in ts]); X20 = kd._X20
    sv = [X20.tolist().index(list(s)) for s in mf.SV]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i in range(20):
        if i not in sv:
            ax.plot(ts, H[:, i], color=GRAY, lw=1, alpha=0.6, zorder=1)
    for i, s in zip(sv, mf.SV):
        c = RED if kd._y20[i] > 0 else BLUE
        ax.plot(ts, H[:, i], color=c, lw=3, marker="o" if c == RED else "s", ms=5, zorder=3)
        ax.text(ts[-1] * 1.25, H[-1, i] + (0.05 if s == mf.SV[2] else -0.05 if s == mf.SV[1] else 0), f"({s[0]:g}, {s[1]:g})  α = {H[-1, i]:+.2f}".replace("-", "−"), color=c, fontsize=14, va="center", fontweight="bold")
    ax.text(ts[-1] * 1.25, 0.0, "나머지 17개  α = 0", color="#555555", fontsize=14, va="center")
    ax.plot([0.8, ts[-1]], [0, 0], color="black", lw=0.8); ax.set_ylim(-0.5, 0.85)
    ax.set_xscale("log"); ax.set_xlim(0.8, ts[-1] * 9); ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(["1", "10", "100", "1000", "1만"])
    ax.set_xlabel("걸음", fontsize=15); ax.set_ylabel("$\\alpha_i$", fontsize=16, rotation=0, labelpad=14); ax.tick_params(labelsize=12)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    mf.save(fig, "kn_alpha.png")


FIGS = {"recap": fig_recap, "quizkid": fig_quizkid, "quiz1": fig_quiz1, "svarrow": fig_svarrow, "alpha": fig_alpha, "ktrick": fig_ktrick, "dual": fig_dual, "q": fig_q, "lift1d": fig_lift1d, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree,
        "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
