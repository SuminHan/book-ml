"""SVM 2 (소프트 마진) 덱의 그림 원본. 실행: python make_figs.py [키]. 공용 그림 함수는 ../2_svm/make_figs.py, 숫자는 data.py (이 폴더)."""
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
    """지난 시간 한 쪽: ① 경계와 값 = 거리 ② 최대 마진"""
    X, y, WA, BA = mf.X, mf.y, mf.WA, mf.BA
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.4))
    ax = axs[0]; mf.pts(ax); mf.street_m(ax, WA, BA, 0)
    p = np.array([-0.2, 1.3]); d = WA @ p + BA; q = p - d * WA                     # ‖w‖ = 1 이라 |D| = 거리, q = 경계 위 발. 수선이 다른 점에서 가장 먼 점
    ax.scatter(*p, s=340, facecolors="none", edgecolors="#6b6b6b", lw=2.5, zorder=6)
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="<->", lw=3.2, color="#6b6b6b", shrinkA=6, shrinkB=0), zorder=6)
    ax.text(-2.12, 0.62, f"$|D(\\mathbf{{x}})|$ = 경계까지\n거리 = {abs(d):.2f}", color="#6b6b6b", fontsize=13.5, fontweight="bold", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1), zorder=7)
    ax.text(-1.05, 2.05, "$D(\\mathbf{x}) = \\mathbf{w}\\cdot\\mathbf{x} + b = 0$", fontsize=14, va="top")
    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox          # 제목에서 거리 부분은 굵게
    parts = [TextArea("① 경계: $\\Vert\\mathbf{w}\\Vert = 1$ 이면 ", textprops=dict(fontsize=15)),
             TextArea("$|D(\\mathbf{x})|$ = 경계까지 거리", textprops=dict(fontsize=16, color="#6b6b6b", fontweight="bold"))]
    ax.add_artist(AnchoredOffsetbox(loc="lower center", child=HPacker(children=parts, align="baseline", pad=0, sep=2), frameon=False,
                                    bbox_to_anchor=(0.5, 1.005), bbox_transform=ax.transAxes, borderpad=0.1))
    ax.set_title(" ", fontsize=15)
    ax = axs[1]; mf.pts(ax); mf.svm_street(ax, mf.M)
    ax.set_title(f"② 최대 마진: M = {mf.MA:.2f}\n경계는 서포트 벡터 3개가 정한다", fontsize=15)
    mf.save(fig, "kn_recap.png")
    return d


def fig_svarrow():
    """w* = Σ α_k y_k x_k = 파랑 서포트 벡터 → 빨강 서포트 벡터 화살표 두 개의 합 × 0.55 (논문 식 (16), 식 (13) 기준)"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    r, b1, b2 = (np.array(s) for s in mf.SV); ar, a1, a2 = (mf.ALPHA[tuple(s)] for s in mf.SV)   # α_k ≥ 0, y = +1 · −1 · −1
    assert abs(ar - a1 - a2) < 0.01 and abs(a1 - a2) < 0.01           # Σ α_k y_k = 0, 파랑 둘 같다
    c = a1; d1, d2 = r - b1, r - b2; w = c * (d1 + d2)
    assert np.allclose(w, ar * r - a1 * b1 - a2 * b2) and np.allclose(w, mf.W_STAR, atol=0.02)
    w = np.array(mf.W_STAR)
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
    ax.text(-0.85, 1.0, f"합 × {c:.2f} = $\\mathbf{{w}}^*$\n= ({w[0]:.2f}, {w[1]:.2f})", fontsize=14, fontweight="bold")
    ax.set_title(f"② 두 화살표를 더해 {c:.2f} 배 = $\\mathbf{{w}}^*$", fontsize=15)
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
    """쉬운 퀴즈 (논문 식 (13) 기준: 가장 가까운 값 = 1): 점 8개, w = (0.6, 0.8), b = −2.4, M = 1/‖w‖ = 1, 서포트 벡터 A (2.2, 2.6) · B (1, 1), α = 0.5 · 0.5, 새 점 P (6, 2) · Q (0, 2)"""
    R = np.array([[2.2, 2.6], [4, 3], [3, 4], [5, 2]]); B = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    w, b, m = np.array([0.6, 0.8]), -2.4, 1.0
    M = SVC(kernel="linear", C=1e6, tol=1e-10).fit(np.r_[R, B], np.r_[np.ones(4), -np.ones(4)])
    assert np.allclose(M.coef_[0], w) and np.isclose(M.intercept_[0], b) and np.allclose(np.abs(M.dual_coef_[0]), 0.5) and len(M.support_) == 2
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    t = np.linspace(-2, 8, 2); ln = lambda c: (c - b - w[0] * t) / w[1]           # D(x) = c 인 직선의 x2
    ax.fill_between(t, ln(-m), ln(m), color=GOLD, alpha=0.18, lw=0)
    ax.plot(t, ln(0), color="black", lw=2.5); ax.plot(t, ln(m), "--", color=RED, lw=1.5); ax.plot(t, ln(-m), "--", color=BLUE, lw=1.5)
    ax.scatter(R[:, 0], R[:, 1], s=110, color=RED, zorder=4); ax.scatter(B[:, 0], B[:, 1], s=110, color=BLUE, marker="s", zorder=4)
    ax.scatter([2.2, 1], [2.6, 1], s=420, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=5)
    mf.arw(ax, np.array([1, 1]), np.array([2.2, 2.6]), "#e8590c", lw=2.5)
    ax.text(-0.1, 3.2, "A (2.2, 2.6)", fontsize=14, fontweight="bold", color=RED); ax.text(1.3, 0.5, "B (1, 1)", fontsize=14, fontweight="bold", color=BLUE)
    for p, nm, o in (((6, 2), "P (6, 2)", (-0.45, 0.35)), ((0, 2), "Q (0, 2)", (-0.4, 0.35))):
        ax.scatter(*p, marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=6)
        ax.text(p[0] + o[0], p[1] + o[1], nm, fontsize=14, fontweight="bold", color="#2B8A3E")
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.6, 6.6); ax.set_aspect("equal"); ax.set_xticks(range(0, 7)); ax.set_yticks(range(0, 7)); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$", fontsize=14); ax.set_ylabel("$x_2$", fontsize=14, rotation=0, labelpad=10)
    ax.set_title("빨강 4개, 파랑 4개: 경계 $0.6x_1 + 0.8x_2 - 2.4 = 0$", fontsize=14)
    mf.save(fig, "kn_quizkid.png")


def fig_force():
    """역학 비유: 서포트 벡터가 경계(판)를 미는 힘 = α. 힘 합 0 (안 밀려 감), 돌림힘 합 0 (안 돎)"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    u = WA / np.linalg.norm(WA); dd = np.array([-u[1], u[0]]); bb = BA / np.linalg.norm(WA)
    SV = [np.array(s) for s in mf.SV]; AL = np.array([mf.ALPHA[tuple(s)] * (1 if tuple(s) in mf.RED_PTS else -1) for s in mf.SV])   # α_k y_k
    pos = np.array([p @ dd for p in SV]); tq = AL * pos
    assert abs(AL.sum()) < 1e-9 and abs(tq.sum()) < 0.01                              # 힘의 평형 · 돌림힘의 평형
    k = MA / AL.max()                                                                  # 가장 큰 힘 화살표 = 마진 길이
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    mf.pts(ax, np.zeros((0, 2)), np.zeros(0)); mf.street_m(ax, WA, BA, MA, np.array(SV))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, alpha=0.22, zorder=3); ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, alpha=0.22, marker="s", zorder=3)
    s = np.linspace(-4, 4, 2); ax.plot(*((-bb) * u[:, None] + dd[:, None] * s), color="#555555", lw=7, alpha=0.55, solid_capstyle="butt", zorder=2)   # 판
    for p, a in zip(SV, AL):
        c = RED if a > 0 else BLUE
        ax.scatter(*p, s=90, color=c, marker="o" if a > 0 else "s", zorder=5)
        foot = p - (p @ u + bb) * u                                                    # 경계 위 발
        start = foot + np.sign(a) * abs(a) * k * u
        ax.plot(*np.c_[p, start], ":", color=c, lw=1.5, zorder=4)
        ax.annotate("", foot, start, arrowprops=dict(arrowstyle="-|>", color=c, lw=2 + 4 * abs(a), mutation_scale=24, shrinkA=0, shrinkB=0), zorder=6)
        off = np.array([0.15, 0.12]) if a > 0 else np.array([0.12, -0.32])
        ax.text(*(start + off), f"힘 {abs(a):g}", fontsize=14, fontweight="bold", color=c, zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(0.85, 2.1, "회색 판 = 경계", fontsize=13, color="#444444", va="top")
    ax.set_title(f"빨강 {AL[0]:g} = 파랑 {-AL[1]:g} + {-AL[2]:g}", fontsize=15)
    mf.save(fig, "kn_force.png")
    return dict(pos=pos.round(2).tolist(), torque=tq.round(2).tolist())


def fig_quizkid3():
    """쉬운 퀴즈, 서포트 벡터 3개 (비대칭): w = (0.6, 0.8), b = −2.4, M = 1, A (3,2) α 0.5 (y +1), B1 (5,−2) α 0.3 (y −1), B2 (−3,4) α 0.2 (y −1). 논문 기준 = ‖w‖ = 1 기준 (M = 1).
    시소: 경계를 따라 A 에서 B1 은 4, B2 는 6 떨어져 0.3 × 4 = 0.2 × 6"""
    R = np.array([[3, 2], [4, 3], [5, 2], [4, 4]]); B = np.array([[5, -2], [-3, 4], [0, 0], [1, -1]])
    w, b, m = np.array([0.6, 0.8]), -2.4, 1.0
    M = SVC(kernel="linear", C=1e6, tol=1e-10).fit(np.r_[R, B], np.r_[np.ones(4), -np.ones(4)])
    al = dict(zip(map(tuple, M.support_vectors_.astype(int).tolist()), M.dual_coef_[0].round(3)))   # α_k y_k
    assert np.allclose(M.coef_[0], w) and np.isclose(M.intercept_[0], b) and al == {(5, -2): -0.3, (-3, 4): -0.2, (3, 2): 0.5}
    d = np.array([-w[1], w[0]]); pa, p1, p2 = np.array([3, 2]) @ d, np.array([5, -2]) @ d, np.array([-3, 4]) @ d
    assert np.isclose(abs(p1 - pa), 4) and np.isclose(abs(p2 - pa), 6)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    t = np.linspace(-6, 9, 2); ln = lambda c: (c - b - w[0] * t) / w[1]
    ax.fill_between(t, ln(-m), ln(m), color=GOLD, alpha=0.18, lw=0)
    ax.plot(t, ln(0), color="black", lw=2.5); ax.plot(t, ln(m), "--", color=RED, lw=1.5); ax.plot(t, ln(-m), "--", color=BLUE, lw=1.5)
    ax.scatter(R[:, 0], R[:, 1], s=100, color=RED, zorder=4); ax.scatter(B[:, 0], B[:, 1], s=100, color=BLUE, marker="s", zorder=4)
    lab = {(3, 2): ("A", (-2.9, 0.4)), (5, -2): ("B1", (-1.6, -0.75)), (-3, 4): ("B2", (0.3, 0.3))}
    for p, a in al.items():
        nm, off = lab[p]
        ax.scatter(*p, s=420, facecolors="none", edgecolors=GOLD, lw=2.5, zorder=5)
        ax.text(p[0] + off[0], p[1] + off[1], f"{nm} ({p[0]}, {p[1]})  α = {abs(a):.1f}".replace("-", "−"), fontsize=13, fontweight="bold", color=RED if a > 0 else BLUE, zorder=7,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.scatter(6, 2, marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=6); ax.text(5.5, 2.5, "P (6, 2)", fontsize=13, fontweight="bold", color="#2B8A3E")
    ax.set_xlim(-4.2, 7.2); ax.set_ylim(-3.2, 5.6); ax.set_aspect("equal"); ax.set_xticks(range(-4, 8, 2)); ax.set_yticks(range(-2, 6, 2)); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$", fontsize=14); ax.set_ylabel("$x_2$", fontsize=14, rotation=0, labelpad=10)
    ax.set_title("서포트 벡터 3개: 경계 $0.6x_1 + 0.8x_2 - 2.4 = 0$", fontsize=14)
    mf.save(fig, "kn_quizkid3.png")


def fig_dual():
    """지난 시간 점 20개: 서포트 벡터 3개만 α ≠ 0, w* = Σ α_k y_k x_k (꼬리 물기, 논문 식 (16))"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    SV = [np.array(s) for s in mf.SV]; AL = [mf.ALPHA[tuple(s)] * (1 if tuple(s) in mf.RED_PTS else -1) for s in mf.SV]   # α_k y_k
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    ax = axs[0]; mf.pts(ax, np.zeros((0, 2)), np.zeros(0)); mf.street_m(ax, WA, BA, MA, np.array(SV))
    ax.scatter(X[y > 0, 0], X[y > 0, 1], s=80, color=RED, alpha=0.28, zorder=3)          # α = 0 인 점은 흐리게
    ax.scatter(X[y < 0, 0], X[y < 0, 1], s=80, color=BLUE, alpha=0.28, marker="s", zorder=3)
    for s, a in zip(SV, AL):
        ax.scatter([s[0]], [s[1]], s=80, color=RED if a > 0 else BLUE, marker="o" if a > 0 else "s", zorder=5)
    off = {0: (0.2, -0.35), 1: (0.2, 0.2), 2: (0.2, -0.45)}
    for k, (s, a) in enumerate(zip(SV, AL)):
        dx, dy = off[k]
        ax.text(s[0] + dx, s[1] + dy, f"$\\alpha$ = {abs(a):g}", fontsize=14, fontweight="bold", color=RED if a > 0 else BLUE, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(0.25, 2.1, "흐린 점 17개: $\\alpha$ = 0", fontsize=13, color="#555555", va="top", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    nx = np.array(kd.NEW_PT)
    ax.scatter([nx[0]], [nx[1]], marker="*", s=420, color="#2B8A3E", edgecolor="black", lw=1, zorder=7)
    ax.text(nx[0] + 0.2, nx[1] - 0.05, f"새 점 ({nx[0]:g}, {nx[1]:g})\nD = {kd.NEW_F:+.2f} → 빨강".replace("-", "−"), fontsize=13, fontweight="bold",
            color="#2B8A3E", va="top", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_title("① 서포트 벡터 3개만 $\\alpha \\neq 0$", fontsize=16)
    ax = axs[1]
    cur = np.zeros(2); cols = [RED if a > 0 else BLUE for a in AL]
    for k, (s, a, c) in enumerate(zip(SV, AL, cols)):
        v = a * s
        ax.annotate("", cur + v, cur, arrowprops=dict(arrowstyle="-|>", lw=2.5, color=c, mutation_scale=18))
        mid = cur + v / 2
        lab = f"{a:+g} · ({s[0]:g}, {s[1]:g})".replace("-", "−")
        ax.text(mid[0] + (0.06 if k != 1 else 0.06), mid[1] + (-0.1 if k == 0 else 0.0), lab, fontsize=13, color=c, ha="left", va="center")
        cur = cur + v
    ax.annotate("", cur, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color="black", mutation_scale=20))
    ax.text(cur[0] / 2 - 0.08, cur[1] / 2 + 0.08, f"합 = $\\mathbf{{w}}^*$ = ({mf.W_STAR[0]:.2f}, {mf.W_STAR[1]:.2f})", fontsize=14, fontweight="bold", ha="right", va="bottom", rotation=45, rotation_mode="anchor")
    ax.scatter([0], [0], color="black", s=20)
    ax.set_xlim(-0.1, 1.4); ax.set_ylim(-0.1, 1.4); ax.set_aspect("equal"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=11); ax.grid(alpha=0.3)
    ax.set_title("② $\\alpha_k y_k \\mathbf{x}_k$ 를 이어 더하면 $\\mathbf{w}^*$", fontsize=16)
    mf.save(fig, "kn_dual.png")
    return cur.round(3)


def fig_alpha():
    """J 학습의 걸음마다 α: 서포트 벡터 3개만 남고 나머지 17개는 0 으로 (kd.train_dual)"""
    ts = kd.AL_STEPS; H = np.array([kd.AL_HIST[t][0] for t in ts]); X20 = kd._X20
    sv = [X20.tolist().index(list(s)) for s in mf.SV]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i in range(20):
        if i not in sv:
            ax.plot(ts, H[:, i], color=GRAY, lw=1, alpha=0.6, zorder=1)
    for i, s in zip(sv, mf.SV):
        c = RED if kd._y20[i] > 0 else BLUE
        ax.plot(ts, H[:, i], color=c, lw=3, marker="o" if c == RED else "s", ms=5, zorder=3)
        ax.text(ts[-1] * 1.25, H[-1, i] + (0.06 if s == mf.SV[2] else -0.06 if s == mf.SV[1] else 0), f"({s[0]:g}, {s[1]:g})  α = {H[-1, i]:.2f}".replace("-", "−"), color=c, fontsize=14, va="center", fontweight="bold")
    ax.text(ts[-1] * 1.25, 0.0, "나머지 17개  α = 0", color="#555555", fontsize=14, va="center")
    ax.plot([-0.25, ts[-1]], [0, 0], color="black", lw=0.8); ax.set_ylim(-0.06, 1.28)
    ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.25, ts[-1] * 9); ax.set_xticks([0, 1, 10, 100, 1000, 10000]); ax.set_xticklabels(["0", "1", "10", "100", "1000", "1만"]); ax.minorticks_off()
    ax.set_xlabel("걸음", fontsize=15); ax.set_ylabel("$\\alpha_k$", fontsize=16, rotation=0, labelpad=14); ax.tick_params(labelsize=12)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    mf.save(fig, "kn_alpha.png")


def fig_nails():
    """비유: 못 박힌 판 위에서 스프링이 두 자를 벌린다. 못 = 점 (고정), 두 자 사이 = 길 폭 2M, 자가 닿은 못 = 서포트 벡터, 못이 버티는 힘 = α.
    ① 좁다 (아무 못에도 안 닿음) ② 못에 걸림 (이 각도의 최대, 돌리면 더) ③ SVM: 폭 1.34, 못 3개가 α = 1.1 · 0.55 · 0.55 로 버팀"""
    X, y, WA, BA, MA = mf.X, mf.y, mf.WA, mf.BA, mf.MA
    wa = WA / np.linalg.norm(WA)
    th = np.radians(68); w2 = np.array([np.cos(th), np.sin(th)])
    rm = (X[y > 0] @ w2).min(); bm = (X[y < 0] @ w2).max(); M2, b2 = (rm - bm) / 2, -(rm + bm) / 2
    assert 0 < M2 < MA - 0.05 and int((y * (X @ w2 + b2) < M2 + 1e-6).sum()) == 2, (M2, b2)
    cases = ((wa, BA / np.linalg.norm(WA), 0.28, "① 스프링이 두 자를 벌린다"), (w2, b2, M2, "② 못에 걸렸다: 돌리면 더 벌어진다"), (wa, BA / np.linalg.norm(WA), MA, "③ 어느 쪽으로도 더는 못 벌어진다"))
    RULER, EDGE = "#d9a441", "#7a5a1e"
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.2))
    for k, (ax, (w, b, M, ttl)) in enumerate(zip(axs, cases)):
        d = np.array([-w[1], w[0]]); P = lambda c, t: (c - b) * w + t * d          # w·x + b = c 위, 선을 따라 t
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        road = np.array([P(M, -5), P(M, 5), P(-M, 5), P(-M, -5)]); ax.fill(road[:, 0], road[:, 1], color="#dde6f6", zorder=1)   # 길 안쪽만 칠, 바깥은 흰색
        for sg in (1, -1):                                                            # 자: 바깥 모서리가 ±M, 두께는 안쪽으로
            quad = np.array([P(sg * M, -5), P(sg * M, 5), P(sg * (M - 0.13), 5), P(sg * (M - 0.13), -5)])
            ax.fill(quad[:, 0], quad[:, 1], color=RULER, ec=EDGE, lw=1.2, zorder=2)
            for t in np.arange(-3.9, 4, 0.2):
                ax.plot(*np.c_[P(sg * M, t), P(sg * (M - (0.07 if round(t * 5) % 5 == 0 else 0.04)), t)], color=EDGE, lw=0.9, zorder=3)
        ax.plot(*np.c_[P(0, -5), P(0, 5)], "--", color="black", lw=1.6, zorder=3)
        t0 = 1.35; inner = M - 0.13; n = 11                                           # 스프링 (두 자 안쪽 사이)
        zz = [P(-inner, t0)] + [P(-inner + 2 * inner * (i + 0.5) / n, t0 + (0.13 if i % 2 == 0 else -0.13)) for i in range(n)] + [P(inner, t0)]
        zz = np.array(zz); ax.plot(zz[:, 0], zz[:, 1], color="#444444", lw=2.2, zorder=4)
        if k < 2:
            for sg in (1, -1):
                ax.annotate("", P(sg * (M + 0.36), -1.45), P(sg * (M - 0.05), -1.45), arrowprops=dict(arrowstyle="-|>", color="#2B8A3E", lw=3.5, mutation_scale=22), zorder=7)
        v = y * (X @ w + b); touch = v < M + 2e-3 if k else np.zeros(len(y), bool)
        for p, yy, tc in zip(X, y, touch):                                           # 못: 머리 + 가운데 점
            c = RED if yy > 0 else BLUE
            ax.scatter(*p, s=170, color=c, marker="o" if yy > 0 else "s", edgecolor="black", lw=0.8, zorder=6, alpha=1.0 if (tc or k < 2) else 0.35)
            ax.scatter(*p, s=18, color="white", zorder=6.5)
            if tc:
                ax.scatter(*p, s=520, facecolors="none", edgecolors="black", lw=2.4, zorder=6)
        if k == 2:
            for sv in mf.SV:
                p = np.array(sv); yy = 1 if tuple(sv) in mf.RED_PTS else -1; al = mf.ALPHA[tuple(sv)]
                start = p + yy * w * 0.02; end = p - yy * w * (0.30 + 0.42 * al)
                ax.annotate("", end, start, arrowprops=dict(arrowstyle="-|>", color=RED if yy > 0 else BLUE, lw=2.5 + 3.2 * al, mutation_scale=26, shrinkA=7, shrinkB=0), zorder=8)
                off = np.array([0.22, 0.28]) if yy > 0 else (np.array([-0.62, 0.1]) if sv[0] < 0 else np.array([0.55, -0.18]))
                ax.text(*(p + off), f"α {al:g}", fontsize=18, fontweight="bold", color=RED if yy > 0 else BLUE, ha="center", va="center", zorder=9,
                        bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
        ax.set_title(ttl, fontsize=17, fontweight="bold" if k == 2 else "normal")
        ax.set_xlabel(f"두 자 사이 {2 * M:.2f}", fontsize=19, fontweight="bold" if k == 2 else "normal", labelpad=8)
    mf.save(fig, "kn_nails.png")
    return dict(M2=round(float(M2), 3))


def fig_overlap():
    """오늘의 첫 질문: 두 색이 경계 근처에서 뒤섞인 데이터 (예시, 난수) → 어떤 직선을 그어도 몇 점은 틀린다. 회색 테두리 = 이 직선이 틀리는 점"""
    rng = np.random.default_rng(3)
    R = rng.normal((0.5, 0.5), 0.55, (30, 2)); B = rng.normal((-0.5, -0.5), 0.55, (30, 2))
    A = np.r_[R, B]; t = np.r_[np.ones(30), -np.ones(30)]
    m = SVC(kernel="linear", C=1).fit(A, t); bad = m.predict(A) != t
    assert 3 <= bad.sum() <= 10 and SVC(kernel="linear", C=1e4).fit(A, t).score(A, t) < 1
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(*R.T, s=70, color=RED, zorder=3); ax.scatter(*B.T, s=70, color=BLUE, marker="s", zorder=3)
    w, b = m.coef_[0], m.intercept_[0]; g = np.linspace(-2.2, 2.2, 2); ax.plot(g, -(b + w[0] * g) / w[1], color="black", lw=2.5)
    ax.scatter(*A[bad].T, s=360, facecolors="none", edgecolors="#8a8a8a", lw=2.8, zorder=5)                 # 반대쪽에 남은 점: 회색 테두리
    lo, hi = A.min() - 0.15, A.max() + 0.15; ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    mf.save(fig, "kn_overlap.png")
    return int(bad.sum())


def fig_saddle():
    """안장점 예 (점 두 개 x = ±1, y = ±1, b = 0, 두 점의 α 는 같다): ½w² 최소, 조건 w ≥ 1  →  L(w, α) = ½w² − 2α(w − 1). 답 w = 1, α = 0.5, L = 0.5.
    α = 0.5 에서 w 방향 = 골짜기 (최소). 각 α 에서 w 를 최소로 고른 값 (w = 2α) = 2α − 2α² 은 α = 0.5 에서 봉우리 (최대) = J"""
    L = lambda w, a: 0.5 * w ** 2 - 2 * a * (w - 1)
    g = np.linspace(-0.2, 2.2, 481)
    h = np.linspace(-0.1, 1.1, 481); J = 2 * h - 2 * h ** 2
    assert abs(g[np.argmin(L(g, 0.5))] - 1) < 1e-9 and abs(h[np.argmax(J)] - 0.5) < 1e-9 and L(1, 0.5) == 0.5
    assert np.allclose(L(2 * h, h), J)
    X2, Y2 = np.array([1.0, -1.0]), np.array([1.0, -1.0]); assert np.sum(0.5 * Y2 * X2) == 1.0        # w = Σ α y x = 1
    BL, RD = "#2E3192", "#d62728"
    fig = plt.figure(figsize=(15, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0], projection="3d", computed_zorder=False)
    W, A = np.meshgrid(np.linspace(-0.2, 2.2, 60), np.linspace(-0.1, 1.1, 60))
    ax.plot_surface(W, A, L(W, A), cmap="YlOrBr", alpha=0.55, linewidth=0, zorder=1)
    ax.plot(g, np.full_like(g, 0.5), L(g, 0.5), color=BL, lw=3.5, zorder=3); ax.plot(2 * h, h, J, color=RD, lw=3.5, zorder=3)
    ax.scatter([1], [0.5], [0.5], s=140, color="black", zorder=5)
    ax.set_xlabel("w", fontsize=15); ax.set_ylabel("α", fontsize=15); ax.set_zlabel("L", fontsize=15); ax.view_init(elev=24, azim=-58)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 1, 2]); ax.set_title("말안장 모양, 검은 점 = 안장점", fontsize=16)
    for k, (col, x, y, xl, ttl, lab) in enumerate((
            (BL, g, L(g, 0.5), "w", "① α = 0.5 에서 w 방향: 골짜기 = 최소", "w = 1"),
            (RD, h, J, "α", "② w 를 늘 바닥에 놓은 L = J(α): 봉우리 = 최대", "α = 0.5"))):
        a2 = fig.add_subplot(gs[k + 1]); a2.plot(x, y, color=col, lw=3.5); px = 1 if k == 0 else 0.5; a2.scatter([px], [0.5], s=140, color="black", zorder=5)
        a2.annotate(f"{lab}, L = 0.5", (px, 0.5), (px, 1.05 if k == 0 else 0.0), ha="center", fontsize=15, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.5))
        a2.set_xlabel(xl, fontsize=17); a2.set_ylabel("L", fontsize=15, rotation=0, labelpad=10); a2.set_title(ttl, fontsize=15, color=col, fontweight="bold")
        a2.set_xlim(*((-0.2, 2.2) if k == 0 else (-0.1, 1.1))); a2.set_ylim(-0.3, 1.6); a2.set_xticks([0, 1, 2] if k == 0 else [0, 0.5, 1]); a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12)
    fig.savefig(mf.O + "kn_saddle.png"); plt.close(fig)


def fig_meet():
    """양쪽에서 좁혀 가기: 예 ½w² 최소, 조건 w ≥ 1. 위 = 조건 지키는 w 의 ½w², 아래 = J(α) = 2α − 2α² (점 두 개, w = 2α).
    세 번의 시도 (w, α) = (2, 0) → (1.5, 0.25) → (1, 0.5) 에서 틈 2 → 0.75 → 0"""
    J = lambda a: 2 * a - 2 * a ** 2; P = lambda w: 0.5 * w ** 2
    ws, als = (2, 1.5, 1), (0, 0.25, 0.5)
    up, lo = [P(w) for w in ws], [J(a) for a in als]
    assert up == [2.0, 1.125, 0.5] and lo == [0.0, 0.375, 0.5] and [u - l for u, l in zip(up, lo)] == [2.0, 0.75, 0.0]
    assert max(J(np.linspace(0, 3, 301))) <= min(P(np.linspace(1, 3, 201))) == 0.5
    X, y = mf.X, mf.y; a = np.array([mf.ALPHA.get(tuple(p), 0.0) for p in X.tolist()]); w = (a * y) @ X
    assert abs(0.5 * w @ w - 1.11) < 0.02 and abs(a.sum() - 0.5 * w @ w - 1.11) < 0.02          # 점 20개: 최소 = 최대
    BL, RD, GD = "#2E3192", "#d62728", "#f2b705"
    fig, ax = plt.subplots(figsize=(12, 5.2)); x = [0, 1, 2]
    ax.fill_between(x, lo, up, color="#e9e9ef", zorder=1)
    ax.plot(x, up, "o-", color=BL, lw=5, ms=16, zorder=3); ax.plot(x, lo, "o-", color=RD, lw=5, ms=16, zorder=3)
    for k in (0, 1):
        ax.annotate("", (k, up[k] - 0.09), (k, lo[k] + 0.09), arrowprops=dict(arrowstyle="<->", color="#555555", lw=2))
        ax.text(k + 0.05, (up[k] + lo[k]) / 2, f"틈 {up[k] - lo[k]:g}", fontsize=19, color="#444444", va="center")
        ax.text(k + 0.22 * k, up[k] + 0.2 + 0.08 * k, f"w = {ws[k]:g}  →  {up[k]:g}", color=BL, fontsize=19, fontweight="bold", ha="center")
        ax.text(k, lo[k] - 0.26, f"α = {als[k]:g}  →  {lo[k]:g}", color=RD, fontsize=19, fontweight="bold", ha="center")
    ax.scatter([2], [0.5], s=900, marker="*", color=GD, edgecolor="black", lw=1.5, zorder=6)
    ax.text(2, 1.0, "틈 0 = 답\nw = 1,  α = 0.5\n값 0.5", fontsize=19, fontweight="bold", ha="center", va="bottom", linespacing=1.4)
    ax.text(-0.42, 1.2, "위\n½w²\n줄이기", color=BL, fontsize=19, fontweight="bold", ha="center", va="center", linespacing=1.3)
    ax.text(-0.42, -0.02, "아래\nJ\n키우기", color=RD, fontsize=19, fontweight="bold", ha="center", va="center", linespacing=1.3)
    ax.set_xlim(-0.65, 2.4); ax.set_ylim(-0.48, 2.45); ax.axis("off")
    fig.tight_layout(); fig.savefig(mf.O + "kn_meet.png"); plt.close(fig)


def fig_states():
    """말안장 위의 점마다 어떤 상태인가 (임베딩 지도처럼): w = 0.5 · 1 · 2 (가로), α = 0.25 · 0.5 · 1 (세로) 의 9칸.
    칸 = 점 두 개 (x = ±1) 와 두 자 (±M = ±1/w), 금색 원 = α, 숫자 = L = ½w² − 2α(w − 1). 파랑 테두리 = α = 0.5 에서 w 방향 골짜기, 빨강 = w = 2α (J)"""
    L = lambda w, a: 0.5 * w ** 2 - 2 * a * (w - 1)
    WS, AS = (0.5, 1, 2), (0.25, 0.5, 1)
    tab = {a: [L(w, a) for w in WS] for a in AS}
    assert tab == {0.25: [0.375, 0.5, 1.5], 0.5: [0.625, 0.5, 1.0], 1: [1.125, 0.5, 0.0]}
    assert [L(2 * a, a) for a in AS] == [0.375, 0.5, 0.0] and min(tab[0.5]) == tab[0.5][1]
    BL, RD, GD, WOOD = "#2E3192", "#d62728", "#f2b705", "#d9a441"
    fig = plt.figure(figsize=(14, 6.6)); X0, Y0, CW, CH, GX, GY = 0.085, 0.085, 0.285, 0.272, 0.018, 0.022
    for j, a in enumerate(AS):
        for i, w in enumerate(WS):
            ax = fig.add_axes([X0 + i * (CW + GX), Y0 + j * (CH + GY), CW, CH]); M = 1 / w; v = L(w, a)
            ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.25, 1.25); ax.set_xticks([]); ax.set_yticks([])
            ax.axvspan(-M, M, ymin=0.2, ymax=0.68, color="#fdf1cf", zorder=1)
            for s in (-1, 1): ax.plot([s * M, s * M], [-0.75, 0.45], color=WOOD, lw=7, solid_capstyle="butt", zorder=3)
            ax.plot([0, 0], [-0.75, 0.45], color="black", ls="--", lw=1.3, zorder=2); ax.plot([-2.45, 2.45], [-0.15, -0.15], color="#999999", lw=1, zorder=1.5)
            for x, c, mk in ((1, RD, "o"), (-1, BL, "s")):
                ax.scatter([x], [-0.15], s=150 + 1500 * a, color=GD, alpha=0.4, zorder=3.5)
                ax.scatter([x], [-0.15], s=120, color=c, marker=mk, edgecolor="black", lw=0.8, zorder=5)
            state = "점이 길 안 (어김)" if w < 1 else ("딱 닿음" if w == 1 else "점이 길 밖 (여유)")
            ax.text(0, -1.04, state, ha="center", va="center", fontsize=14, color=RD if w < 1 else "black")
            ax.text(-2.45, 0.9, f"w = {w:g},  α = {a:g}", ha="left", va="center", fontsize=13.5, color="#444444")
            ax.text(2.45, 0.9, f"L = {v:g}", ha="right", va="center", fontsize=17, fontweight="bold")
            ridge, valley = w == 2 * a, a == 0.5
            col = "black" if ridge and valley else (RD if ridge else (BL if valley else "#cccccc"))
            for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(5 if col != "#cccccc" else 1.2); sp.set_visible(True)
            if ridge and valley: ax.set_facecolor("#fff8dc")
    for i, w in enumerate(WS): fig.text(X0 + i * (CW + GX) + CW / 2, 0.035, f"w = {w:g}", ha="center", fontsize=17, fontweight="bold")
    for j, a in enumerate(AS): fig.text(0.043, Y0 + j * (CH + GY) + CH / 2, f"α = {a:g}", ha="center", va="center", fontsize=17, fontweight="bold")
    fig.savefig(mf.O + "kn_states.png"); plt.close(fig)


def fig_parab():
    """점 두 개 (x = ±1) 의 점수 J(α) = 2α − 2α²: 꼭대기 α = 0.5 → w = 2α = 1, 길 반폭 1"""
    J = lambda a: 2 * a - 2 * a ** 2
    g = np.linspace(0, 1, 201)
    assert abs(g[J(g).argmax()] - 0.5) < 1e-9 and [J(a) for a in (0.25, 0.5, 1)] == [0.375, 0.5, 0]
    BL, RD, GD = "#2E3192", "#d62728", "#f2b705"
    fig, ax = plt.subplots(figsize=(11, 5.0)); ax.plot(g, J(g), color=BL, lw=5)
    for a, c, txt, xy, ha in ((0.25, "#777777", "α = 0.25 :  $\\mathbf{w}$ = 0.5\n힘이 약하다: 점이 길 안", (0.0, 0.46), "left"),
                              (1.0, RD, "α = 1 :  $\\mathbf{w}$ = 2\n너무 세다: 길이 좁다", (0.98, 0.27), "right")):
        ax.scatter([a], [J(a)], s=150, color=c, zorder=4); ax.annotate(txt, (a, J(a)), xy, fontsize=17, color=c, ha=ha, va="center", linespacing=1.4, arrowprops=dict(arrowstyle="->", color=c, lw=1.6, shrinkB=8))
    ax.scatter([0.5], [0.5], s=700, marker="*", color=GD, edgecolor="black", lw=1.4, zorder=5)
    ax.text(0.5, 0.62, "꼭대기  α = 0.5 :  $\\mathbf{w}$ = 1,  점이 가장자리에 딱  =  답", fontsize=18, fontweight="bold", ha="center")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 0.68); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5])
    ax.set_xlabel("α  (두 점이 당기는 힘)", fontsize=17); ax.set_ylabel("점수 J", fontsize=17); ax.tick_params(labelsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(mf.O + "kn_parab.png"); plt.close(fig)


def fig_onearrow():
    """w = S × (빨강 중심 − 파랑 중심): 화살표 하나. 중심 = α 로 무게를 준 서포트 벡터의 평균. 경계는 화살표 한가운데를 수직으로, 화살표 길이 = 길 폭 2M"""
    X, y = mf.X, mf.y; a = np.array([mf.ALPHA.get(tuple(p), 0.0) for p in X.tolist()]); w = (a * y) @ X
    S = a[y > 0].sum(); R = a[y > 0] @ X[y > 0] / S; B = a[y < 0] @ X[y < 0] / a[y < 0].sum()
    assert abs(S - a[y < 0].sum()) < 0.01 and np.allclose(R, (0.7, 0.2)) and np.allclose(B, (-0.25, -0.75), atol=0.01)
    assert np.allclose(R - B, (0.95, 0.95), atol=0.01) and np.allclose(S * (R - B), mf.WA / mf.MA, atol=0.02)
    M = mf.MA; wa, ba = mf.WA / M, mf.BA / M                                     # 이 덱 기준: 가장자리 D = ±1
    assert abs(wa @ ((R + B) / 2) + ba) < 0.01 and abs(np.linalg.norm(R - B) - 2 * M) < 0.01
    BL, RD, GD = "#2E3192", "#d62728", "#f2b705"
    fig, ax = plt.subplots(figsize=(10.5, 6.2)); gx = np.linspace(-2.3, 2.3, 2)
    for c, col, ls in ((0, "black", "--"), (1, RD, ":"), (-1, BL, ":")): ax.plot(gx, (c - ba - wa[0] * gx) / wa[1], color=col, ls=ls, lw=2)
    ax.fill_between(gx, (-1 - ba - wa[0] * gx) / wa[1], (1 - ba - wa[0] * gx) / wa[1], color="#fdf1cf", zorder=0)
    for p, yy, al in zip(X, y, a):
        ax.scatter(*p, s=150, color=RD if yy > 0 else BL, marker="o" if yy > 0 else "s", alpha=1.0 if al > 0 else 0.25, zorder=4)
    b1, b2 = X[(y < 0) & (a > 0)]; ax.plot(*np.c_[b1, b2], color=BL, lw=1.5, alpha=0.6, zorder=3)
    ax.scatter(*B, s=230, marker="X", color=BL, edgecolor="white", lw=1.5, zorder=6)
    ax.scatter(*R, s=420, facecolors="none", edgecolors=RD, lw=3, zorder=5)
    ax.annotate("", R, B, arrowprops=dict(arrowstyle="-|>", color="black", lw=5, mutation_scale=34, shrinkA=8, shrinkB=10), zorder=7)
    ax.annotate("빨강 중심\n(0.7, 0.2)", R, (1.75, -0.35), fontsize=17, color=RD, fontweight="bold", ha="center", arrowprops=dict(arrowstyle="->", color=RD, lw=1.4, shrinkB=14))
    ax.annotate("파랑 중심 (두 점의 가운데)\n(−0.25, −0.75)", B, (0.95, -1.75), fontsize=17, color=BL, fontweight="bold", ha="center", arrowprops=dict(arrowstyle="->", color=BL, lw=1.4, shrinkB=12))
    ax.text(-0.35, 0.18, "(0.95, 0.95)\n길이 1.34 = 길 폭", fontsize=17, fontweight="bold", ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.9))
    ax.annotate("", (R + B) / 2 + np.array([-0.04, 0.03]), (-0.37, 0.13), arrowprops=dict(arrowstyle="-", color="black", lw=1))
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.05, 1.75); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); mf.save(fig, "kn_onearrow.png")


def fig_tilt():
    """벌점이 그릇을 기울여 바닥을 옮긴다: 점 하나 (x = 1, y = +1, b = 0) → D = w, L(w) = ½w² − α(w − 1). 바닥 w = α"""
    L = lambda w, a: 0.5 * w ** 2 - a * (w - 1)
    g = np.linspace(-0.4, 2.6, 601); AS = [(0, "#9a9a9a"), (0.5, "#c8a24a"), (1, "#2E3192"), (2, "#d62728")]
    for a, _ in AS: assert abs(g[L(g, a).argmin()] - a) < 1e-9
    assert [L(a, a) for a, _ in AS] == [0, 0.375, 0.5, 0] and all(abs(L(1, a) - 0.5) < 1e-12 for a, _ in AS)
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.axvspan(-0.4, 1, color="#fde2e2", zorder=0); ax.text(0.3, 2.35, "규칙 어김 (w < 1)\n여유 −  →  벌점", ha="center", va="top", fontsize=16, color="#c0392b", fontweight="bold")
    ax.axvline(1, color="#555555", ls="--", lw=1.5)
    for a, c in AS:
        ax.plot(g, L(g, a), color=c, lw=4 if a == 1 else 2.6, label=f"α = {a:g}  →  바닥 w = {a:g}")
        ax.scatter([a], [L(a, a)], s=260 if a == 1 else 110, color=c, edgecolor="black", lw=1.2, zorder=5, marker="*" if a == 1 else "o")
    ax.annotate("α = 1: 바닥이 딱 경계에", (1, 0.5), (1.45, -0.35), fontsize=16, fontweight="bold", color="#2E3192", arrowprops=dict(arrowstyle="->", color="#2E3192", lw=1.6))
    ax.set_xlim(-0.4, 2.6); ax.set_ylim(-0.6, 2.45); ax.set_xlabel("w", fontsize=17); ax.set_ylabel("L", fontsize=17, rotation=0, labelpad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=15, frameon=False); ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(mf.O + "kn_tilt.png"); plt.close(fig)


def fig_rope():
    """외줄타기 비유: 몸 (= 선의 방향) 이 기울면 균형봉 (= α) 으로 바로잡는다. 각도는 실제 학습의 방향 54° · 42° · 45° (답 45°), 그림은 기울기를 4배로 과장.
    흐린 회색 = 똑바로 선 자세, 휘어진 화살표 = 휘청이는 방향, 빨간 화살표 = 봉으로 바로잡는 방향"""
    from matplotlib.patches import Circle as Cc, FancyArrowPatch
    cases = ((54, "① 오른쪽으로 휘청", "봉을 왼쪽으로 → α 를 고친다"), (42, "② 너무 당겨 왼쪽으로 휘청", "다시 조금 되돌린다"), (45, "③ 똑바로: 균형", "두 힘이 같다 → 멈춤"))
    def person(ax, lean, col, alpha, lw=1.0):
        up = np.array([np.sin(lean), np.cos(lean)]); side = np.array([np.cos(lean), -np.sin(lean)])
        foot = np.array([0.0, 0.0]); hip = foot + 1.1 * up; neck = foot + 2.3 * up; head = foot + 2.75 * up; hand = neck - 0.45 * up
        for d in (-0.15, 0.15): ax.plot(*np.c_[foot + d * np.array([1, 0]), hip], color=col, lw=4 * lw, alpha=alpha, solid_capstyle="round")
        ax.plot(*np.c_[hip, neck], color=col, lw=5 * lw, alpha=alpha, solid_capstyle="round"); ax.add_patch(Cc(head, 0.32, color=col, alpha=alpha))
        for d in (-0.4, 0.4): ax.plot(*np.c_[neck, hand + d * side], color=col, lw=3 * lw, alpha=alpha)
        return up, side, hand, head
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, (deg, ttl, sub) in zip(axs, cases):
        lean = np.radians(4 * (deg - 45))
        ax.plot([-3.2, 3.2], [0, 0], color="#8a6a3a", lw=3, zorder=1)
        for x in (-3.2, 3.2): ax.plot([x, x], [-1.3, 0.2], color="#555555", lw=5)
        if deg != 45: person(ax, 0.0, "#bbbbbb", 0.6)                                                # 똑바로 선 자세 (흐리게)
        up, side, hand, head = person(ax, lean, "black", 1.0)
        pc = hand; pa, pb = pc - 1.9 * side, pc + 1.9 * side                                          # 봉은 몸과 같이 기운다
        ax.plot(*np.c_[pa, pb], color="#d62728", lw=6, solid_capstyle="round", zorder=4)
        for e in (pa, pb): ax.scatter(*e, s=200, color="#d62728", zorder=5)
        ax.plot([0, 0], [0, 3.3], color="#999999", ls="--", lw=1.3, zorder=0)
        if deg != 45:
            s = 1 if deg > 45 else -1
            ax.add_patch(FancyArrowPatch((0.15 * s, 3.35), (head[0] + 0.35 * s, head[1] + 0.25), connectionstyle=f"arc3,rad={-0.35 * s}", arrowstyle="-|>", mutation_scale=22, color="#555555", lw=2.5))
            ax.annotate("", (pc[0] - 1.6 * s, pc[1] - 0.75), (pc[0] - 0.2 * s, pc[1] - 0.75), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=3.5, mutation_scale=24))
        ax.set_title(f"{ttl}\n선의 방향 {deg}°", fontsize=17, fontweight="bold", linespacing=1.4)
        ax.text(0, -0.8, sub, ha="center", fontsize=16, color="#d62728" if deg != 45 else "#2E3192", fontweight="bold")
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1.15, 3.8); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=1.0); fig.savefig(mf.O + "kn_rope.png"); plt.close(fig)


def fig_history():
    """α 의 역사를 그림으로: ① 1788 철사에 꿴 구슬 (철사가 붙잡는 모르는 힘 λ) ② 1951 벽 (밀기만, 안 닿으면 0) ③ 1992 SVM 못과 자 (못이 버티는 힘 α)"""
    from matplotlib.patches import Circle as Cc, Rectangle
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    RED, BL, GR = "#d62728", "#2E3192", "#555555"
    def arrow(ax, a, b, c, lw=3.5): ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=24))
    # ① 구슬과 철사
    ax = axs[0]; th = np.linspace(np.pi * 1.1, np.pi * 1.9, 100); cx, cy, r = 0, 2.2, 2.0
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="#8a6a3a", lw=4)
    p = np.array([cx + r * np.cos(np.pi * 1.32), cy + r * np.sin(np.pi * 1.32)]); n = (np.array([cx, cy]) - p) / r
    ax.add_patch(Cc(p, 0.17, color="black", zorder=5))
    arrow(ax, p, p + np.array([0, -1.0]), GR); ax.text(p[0] + 0.12, p[1] - 0.95, "무게", fontsize=14, color=GR)
    arrow(ax, p, p + 0.95 * n, RED); ax.text(-1.45, 1.75, "철사가 붙잡는 힘\n= 모르는 수 λ", fontsize=14, color=RED, fontweight="bold", ha="center")
    ax.set_title("1788 라그랑주\n철사에 꿴 구슬", fontsize=17, fontweight="bold", linespacing=1.4)
    # ② 벽
    ax = axs[1]; ax.add_patch(Rectangle((1.2, -0.6), 0.35, 2.6, color="#8a8a8a"))
    for y_, x_, lab, touch in ((1.35, 0.97, "닿음: 벽이 민다 (α > 0)", True), (0.1, 0.1, "안 닿음: 힘 0 (α = 0)", False)):
        ax.add_patch(Cc((x_, y_), 0.22, color="black", zorder=5))
        arrow(ax, (x_ - 0.95, y_), (x_ - 0.25, y_), GR, 3)
        if touch: arrow(ax, (1.2, y_), (0.55, y_), RED)
        ax.text(0.1, y_ - 0.48, lab, fontsize=14, color=RED if touch else GR, fontweight="bold", ha="center")
    ax.text(-0.95, 1.65, "밀어붙이는 힘", fontsize=13, color=GR)
    ax.set_title("1939 · 1951 카루시, 쿤 · 터커\n벽은 밀기만 한다", fontsize=17, fontweight="bold", linespacing=1.4)
    # ③ 못과 자
    ax = axs[2]; M = 0.75
    for s in (-1, 1): ax.add_patch(Rectangle((s * M - (0.12 if s > 0 else 0), -0.6), 0.12, 2.6, color="#b98a2e"))
    zz = np.array([(-M + 0.0, 1.6)] + [(-M + 2 * M * (i + 0.5) / 10, 1.6 + (0.12 if i % 2 == 0 else -0.12)) for i in range(10)] + [(M, 1.6)]); ax.plot(*zz.T, color="#333333", lw=2)
    arrow(ax, (-0.2, 1.95), (-0.75, 1.95), GR, 2.5); arrow(ax, (0.2, 1.95), (0.75, 1.95), GR, 2.5); ax.text(0, 2.12, "스프링이 벌린다", fontsize=13, color=GR, ha="center")
    for (x_, y_, c, a_) in ((M + 0.17, 0.6, RED, "α 1.11"), (-M - 0.17, 0.95, BL, "α 0.55"), (-M - 0.17, 0.05, BL, "α 0.55")):
        ax.add_patch(Cc((x_, y_), 0.14, color=c, zorder=5))
        s = 1 if x_ > 0 else -1; arrow(ax, (s * M, y_), (s * (M - 0.55), y_), c)
        ax.text(x_ + s * 0.25, y_, a_, fontsize=14, color=c, fontweight="bold", ha="left" if s > 0 else "right", va="center")
    ax.set_title("1992 SVM\n못이 자를 버틴다", fontsize=17, fontweight="bold", linespacing=1.4)
    for ax in axs: ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.9, 2.5); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=1.5); fig.savefig(mf.O + "kn_history.png"); plt.close(fig)


def fig_wspring():
    """w 공간의 용수철: 원점 (w = 0, 끝없이 넓은 길) 과 w* 를 k = 1 용수철로 잇는다. 용수철 힘 −w*, 서포트 벡터의 힘 α y x 셋을 이어 붙이면 +w* → 평형 w* = Σ α y x"""
    sv = [(np.array(p), 1 if tuple(p) in mf.RED_PTS else -1, mf.ALPHA[tuple(p)]) for p in mf.SV]
    W = sum(al * yy * p for p, yy, al in sv); assert np.allclose(W, (1.045, 1.045), atol=0.01)
    RED, BL, GR = "#d62728", "#2E3192", "#444444"
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.axhline(0, color="#dddddd", lw=1); ax.axvline(0, color="#dddddd", lw=1)
    ax.scatter([0], [0], s=200, marker="s", color=GR, zorder=5); ax.text(0.0, -0.16, "원점 w = 0 = 끝없이 넓은 길", fontsize=16, ha="left", va="top", color=GR)
    n = 14; d = W / np.linalg.norm(W); perp = np.array([-d[1], d[0]]); L_ = np.linalg.norm(W)
    zz = np.array([np.zeros(2)] + [d * L_ * (0.08 + 0.84 * (i + 0.5) / n) + perp * (0.06 if i % 2 == 0 else -0.06) for i in range(n)] + [W])
    ax.plot(*zz.T, color=GR, lw=2.4, zorder=3)
    ax.scatter(*W, s=260, color="black", zorder=6); ax.text(W[0] + 0.1, W[1] - 0.08, "w*", fontsize=20, fontweight="bold", va="top")
    o = -perp * 0.28
    ax.annotate("", W - 0.75 * W + o, W + o, arrowprops=dict(arrowstyle="-|>", color=GR, lw=4.5, mutation_scale=28), zorder=7)
    ax.text(0.95, 0.12, "용수철 힘 −w\n(원점 쪽으로)", fontsize=17, color=GR, fontweight="bold", ha="left", va="center")
    cur = W.copy(); labs = []
    for p, yy, al in sv:
        f = al * yy * p; c = RED if yy > 0 else BL
        ax.annotate("", cur + f, cur, arrowprops=dict(arrowstyle="-|>", color=c, lw=4.5, mutation_scale=26), zorder=7)
        labs.append((cur + f / 2, c, f"{'빨강' if yy > 0 else '파랑'} α = {al:g}")); cur = cur + f
    for (m, c, txt), off in zip(labs, ((0.05, -0.2), (0.12, -0.12), (0.1, 0.0))):
        ax.text(m[0] + off[0], m[1] + off[1], txt, fontsize=15, color=c, fontweight="bold")
    assert np.allclose(cur, 2 * W)
    ax.text(cur[0] - 0.05, cur[1] + 0.12, "못들의 힘 합 = +w", fontsize=17, fontweight="bold", ha="center")
    ax.set_xlim(-0.35, 2.6); ax.set_ylim(-0.75, 2.5); ax.set_aspect("equal"); ax.set_xlabel("$w_1$", fontsize=17); ax.set_ylabel("$w_2$", fontsize=17, rotation=0, labelpad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=12)
    fig.tight_layout(); fig.savefig(mf.O + "kn_wspring.png"); plt.close(fig)


def fig_penalty():
    """벌금 게임: L(w, α) = ½w² − α(w − 1). ① α 마다 곡선 (모두 (1, 0.5) 를 지남) ② α 를 가장 크게 → w < 1 은 ∞, 최소는 w = 1
    ③ w 를 가장 작게 → α − ½α², 최대는 α = 1. ② 의 최소 = ③ 의 최대 = 0.5"""
    L = lambda w, a: 0.5 * w ** 2 - a * (w - 1)
    AL = [(0, "#9a9a9a"), (0.5, "#c8a24a"), (1, "#2E3192"), (2, "#d62728")]
    g = np.linspace(-0.2, 2.4, 521)
    assert all(abs(L(1, a) - 0.5) < 1e-12 and abs(g[np.argmin(L(g, a))] - a) < 1e-9 for a, _ in AL)
    assert [L(a, a) for a, _ in AL] == [0, 0.375, 0.5, 0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    ax = axs[0]
    for a, c in AL:
        ax.plot(g, L(g, a), color=c, lw=3, label=f"α = {a:g}"); ax.scatter([a], [L(a, a)], s=70, color=c, zorder=5)
    ax.scatter([1], [0.5], s=170, color="black", zorder=6)
    ax.annotate("w = 1 에서는\nα 가 얼마든 0.5", (1, 0.5), (1.05, 2.05), ha="center", fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.legend(loc="lower left", fontsize=12.5, frameon=False, ncol=2); ax.set_title("① 감시원이 α 를 바꾸면", fontsize=17, fontweight="bold")
    ax = axs[1]
    ax.axvspan(-0.2, 1, color="#d62728", alpha=0.13)
    ax.text(0.4, 1.55, "규칙 어김\n(w < 1)\nα ↑ → 점수 ∞", ha="center", va="center", fontsize=15, color="#d62728", fontweight="bold")
    ax.annotate("", (0.4, 2.5), (0.4, 2.05), arrowprops=dict(arrowstyle="-|>", lw=3, color="#d62728"))
    x = g[g >= 1]; ax.plot(x, 0.5 * x ** 2, color="#2E3192", lw=3.5); ax.text(1.68, 2.05, "½w²", fontsize=16, color="#2E3192", ha="right")
    ax.scatter([1], [0.5], s=170, color="black", zorder=6)
    ax.annotate("가장 작은 곳\nw = 1, 0.5", (1, 0.5), (1.65, 0.0), ha="center", fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("② α 를 가장 크게 → w 최소", fontsize=17, fontweight="bold")
    ax = axs[2]
    ax.plot(g, g - 0.5 * g ** 2, color="black", lw=3)
    for a, c in AL:
        ax.scatter([a], [L(a, a)], s=110, color=c, zorder=5)
    ax.annotate("가장 큰 곳\nα = 1, 0.5", (1, 0.5), (1, 1.45), ha="center", fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("③ w 를 가장 작게 → α 최대", fontsize=17, fontweight="bold")
    for k, ax in enumerate(axs):
        ax.set_xlim(-0.2, 2.4); ax.set_ylim(-0.75, 2.5); ax.set_xticks([0, 0.5, 1, 1.5, 2]); ax.tick_params(labelsize=12)
        ax.set_xlabel("α" if k == 2 else "w", fontsize=17); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("점수 L", fontsize=15)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(mf.O + "kn_penalty.png", dpi=160); plt.close(fig)


def fig_equiv():
    """SVM 에서 min ½‖w‖² (조건 y D ≥ 1) = min max L. b = b* 로 두고 w 평면: 점 하나 = 직선 규칙 하나.
    ① 규칙을 하나라도 어기면 max_α L = ∞, 모두 지키면 ½‖w‖²  ② 답에서 α y x 화살표 3개의 합 = w*"""
    X, y = np.array(mf.X, float), np.array(mf.y, float); b = kd._d2.B_STAR; wst = np.array(kd._d2.W_STAR, float)
    ok = lambda w: bool((y * (X @ w + b)).min() >= 1 - 2e-3)
    assert ok(wst) and ok(np.array([1.6, 1.6])) and not ok(np.array([0.7, 0.7]))
    sv = [(s, kd._AX[s], kd._YK[s]) for s in kd._d2.SV]
    parts = [a * yy * np.array(s) for s, a, yy in sv]
    assert np.allclose(sum(parts), wst, atol=3e-3) and abs(0.5 * wst @ wst - 1.108) < 3e-3
    lo, hi = -0.15, 2.35
    g = np.linspace(lo, hi, 901); W1, W2 = np.meshgrid(g, g)
    feas = np.ones_like(W1, bool)
    for p, yy in zip(X, y):
        feas &= yy * (W1 * p[0] + W2 * p[1] + b) >= 1
    # 격자에서 지키는 w 중 ½‖w‖² 가 가장 작은 곳 = w*
    J = np.where(feas, 0.5 * (W1 ** 2 + W2 ** 2), np.inf); i = np.unravel_index(np.argmin(J), J.shape)
    assert np.hypot(W1[i] - wst[0], W2[i] - wst[1]) < 0.02
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    ax = axs[0]
    ax.contourf(W1, W2, (~feas).astype(float), levels=[0.5, 1.5], colors=["#d62728"], alpha=0.13)
    for p, yy in zip(X, y):                                        # 규칙 20개 = 직선 20개
        on = any(np.allclose(p, s) for s, _, _ in sv)
        Z = yy * (W1 * p[0] + W2 * p[1] + b) - 1
        if Z.min() < 0 < Z.max():
            ax.contour(W1, W2, Z, levels=[0], colors=[RED if yy > 0 else BLUE], linewidths=2.8 if on else 0.8, alpha=1 if on else 0.35)
    for r in (np.linalg.norm(wst), 1.9, 2.263):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#555555", lw=1.4, ls="--"))
    ax.scatter(*wst, s=190, color="black", zorder=6); ax.scatter(1.6, 1.6, s=110, color="#2b8a3e", zorder=6); ax.scatter(0.7, 0.7, s=110, color="#d62728", marker="X", zorder=6)
    box = dict(fc="white", ec="none", alpha=0.92, pad=3)
    ax.annotate("어김 → max L = ∞", (0.7, 0.7), (0.42, 0.28), ha="center", fontsize=15, color="#d62728", fontweight="bold", bbox=box, arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5), zorder=7)
    ax.annotate("지킴 → max L\n= ½‖w‖² = 2.56", (1.6, 1.6), (1.75, 2.08), ha="center", fontsize=15, color="#2b8a3e", fontweight="bold", bbox=box, arrowprops=dict(arrowstyle="->", color="#2b8a3e", lw=1.5), zorder=7)
    ax.annotate("w* : ½‖w‖² = 1.11\n(지키는 곳 중 최소)", wst, (1.78, 0.42), ha="center", fontsize=15, fontweight="bold", bbox=box, arrowprops=dict(arrowstyle="->", lw=1.5), zorder=7)
    ax.set_title("① 점 하나 = 직선 규칙 하나 (b = 0.05)", fontsize=17, fontweight="bold")
    ax = axs[1]
    ax.add_patch(plt.Circle((0, 0), np.linalg.norm(wst), fill=False, color="#555555", lw=1.4, ls="--"))
    at = np.zeros(2); names = ["1.1 r", "−0.55 b1", "−0.55 b2"]; offs = [(0.12, -0.13), (0.22, -0.07), (0.3, 0.0)]
    for v, (s, a, yy), nm, off in zip(parts, sv, names, offs):
        c = RED if yy > 0 else BLUE
        ax.annotate("", at + v, at, arrowprops=dict(arrowstyle="-|>", lw=3.5, color=c, shrinkA=0, shrinkB=0, mutation_scale=22), zorder=5)
        ax.text(*(at + v / 2 + np.array(off)), nm, color=c, fontsize=16, fontweight="bold", ha="center")
        at = at + v
    ax.annotate("", wst, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=3.5, color="black", shrinkA=0, shrinkB=0, mutation_scale=22), zorder=4)
    ax.text(0.28, 0.62, "w*", fontsize=19, fontweight="bold")
    ax.scatter(*wst, s=120, color="black", zorder=6)
    ax.set_title("② 닿은 규칙 3개의 α 화살표 합 = w*", fontsize=17, fontweight="bold")
    for k, ax in enumerate(axs):
        ax.set_xlim(*((lo, hi) if k == 0 else (-0.25, 1.65))); ax.set_ylim(*((lo, hi) if k == 0 else (-0.25, 1.65)))
        ax.set_aspect("equal"); ax.set_xlabel("$w_1$", fontsize=17); ax.set_ylabel("$w_2$", fontsize=17); ax.tick_params(labelsize=12)
    fig.tight_layout(w_pad=3)
    fig.savefig(mf.O + "kn_equiv.png", dpi=160); plt.close(fig)


def fig_learn():
    """α 학습 (J 를 키우는 두 줄) 이 진행되며 선이 자리를 잡는 모습. 점 둘레 원 크기 = α. kn_learn1..6.png"""
    X, y = np.array(mf.X, float), np.array(mf.y, float); STEPS = [0, 1, 10, 50, 100, 500]
    ts = sorted(set(STEPS) | set(range(21)) | set(int(round(v)) for v in np.logspace(0, np.log10(500), 60)))
    _, _, H = kd.train_dual(X @ X.T, y, T=500, snaps=tuple(ts), a0=kd.A0)
    isv = [X.tolist().index(list(p)) for p in kd._d2.SV]
    wf = (H[500][0] * y) @ X
    assert np.allclose(wf, kd._d2.W_STAR, atol=5e-3) and int((H[500][0] > 1e-6).sum()) == 3
    cur = np.array([[H[t][0][i] for i in isv] for t in ts]); allc = np.array([H[t][0] for t in ts]); rest = [i for i in range(len(y)) if i not in isv]; out = {}
    gx = np.linspace(-2.2, 2.2, 400); GX, GY = np.meshgrid(gx, gx)
    for k, t in enumerate(STEPS, 1):
        a, b = H[t]; w = (a * y) @ X; n = int((a > 1e-6).sum()); M = 1 / np.linalg.norm(w)
        fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
        ax = axs[0]; Z = w[0] * GX + w[1] * GY + b
        ax.contourf(GX, GY, Z, levels=[-1, 1], colors=[GOLD], alpha=0.13)
        ax.contour(GX, GY, Z, levels=[-1, 0, 1], colors=[BLUE, "black", RED], linestyles=["--", "-", "--"], linewidths=[1.4, 3, 1.4])
        if k > 1:                                                   # 바로 앞 장면의 선 (옅게)
            a0, b0 = H[STEPS[k - 2]]; w0 = (a0 * y) @ X
            ax.contour(GX, GY, w0[0] * GX + w0[1] * GY + b0, levels=[0], colors="#6f6f6f", linestyles=":", linewidths=3.2)
        on = a > 1e-6
        ax.scatter(X[on, 0], X[on, 1], s=170 + 1100 * a[on], facecolors=GOLD, alpha=0.45, edgecolors="none", zorder=3)
        ax.scatter(X[:, 0], X[:, 1], c=np.where(y > 0, RED, BLUE), s=45, zorder=4)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.set_xlabel("$x_1$", fontsize=15); ax.set_ylabel("$x_2$", fontsize=15)
        ax.set_title(f"{t} 걸음: α > 0 인 점 {n}개, M = {M:.2f}", fontsize=17, fontweight="bold")
        ax = axs[1]; m = np.array(ts) <= t
        for k2, i in enumerate(rest):                                # 나머지 17개: 가는 선
            ax.plot(np.array(ts)[m], allc[m, i], color=RED if y[i] > 0 else BLUE, lw=1.3, alpha=0.4, zorder=2)
        ax.scatter([t] * len(rest), a[rest], c=np.where(y[rest] > 0, RED, BLUE), s=22, alpha=0.5, zorder=3)
        nr = int((a[rest] > 1e-6).sum())
        ax.plot([], [], color="#8a8a8a", lw=1.3, label=f"나머지 17개: α > 0 인 것 {nr}개" + (f" (최대 {a[rest].max():.2f})" if nr else " → 모두 0"))
        for j, (lab, c, ls) in enumerate((("빨강 r", RED, "-"), ("파랑 b1", BLUE, "-"), ("파랑 b2", BLUE, "--"))):
            ax.plot(np.array(ts)[m], cur[m, j], color=c, ls=ls, lw=3, label=f"{lab}: α = {a[isv[j]]:.2f}")
            ax.scatter([t], [a[isv[j]]], color=c, s=70, zorder=5)
        ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.2, 620); ax.set_ylim(0, 1.3); ax.set_xticks(STEPS); ax.set_xticklabels([str(v) for v in STEPS]); ax.minorticks_off()
        ax.set_xlabel("걸음", fontsize=15); ax.set_title("점 20개의 α", fontsize=17, fontweight="bold"); ax.legend(loc="upper left", fontsize=14, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(w_pad=2.5); fig.savefig(mf.O + f"kn_learn{k}.png", dpi=150); plt.close(fig)
        out[t] = (n, round(float(M), 2), w.round(2).tolist(), round(float(b), 2))
    return out


def fig_rock():
    """실제 점 20개를 학습할 때 두 자가 도는 모습 (스프링 · 못 · 고무줄, 파란 칠 = 길 안), 한 쪽에 세 장면씩 3쪽: kn_rock9_1..3.png.
    고무줄 = 길 안의 점에서 자기 쪽 자까지, 굵기 · 금색 원 = α, 회색 점선 = 앞 장면의 경계"""
    X, y = np.array(mf.X, float), np.array(mf.y, float); STEPS = [0, 1, 2, 3, 13, 25, 50, 98, 500]
    _, _, H = kd.train_dual(X @ X.T, y, T=500, snaps=tuple(STEPS), a0=kd.A0)
    def info(t):
        a, b = H[t]; w = (a * y) @ X; return a, b, w, float(np.degrees(np.arctan2(w[1], w[0]))), float(1 / np.linalg.norm(w))
    got = [(t, round(info(t)[3]), round(info(t)[4], 2)) for t in STEPS]
    assert got == [(0, 48, 0.41), (1, 57, 2.83), (2, 51, 1.0), (3, 54, 1.32), (13, 53, 0.92), (25, 51, 0.87), (50, 46, 0.75), (98, 42, 0.69), (500, 45, 0.67)], got
    ins = [int((y * (X @ info(t)[2] + H[t][1]) < 1 - 1e-3).sum()) for t in (0, 1)]; assert ins == [0, 20], ins
    BAND, RULER = "#1B7F3B", "#6b6b6b"; g = np.array([-4.0, 4.0])
    line = lambda c, w_, b_: (c - b_ - w_[0] * g) / w_[1]
    for page in range(3):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.7))
        for k, ax in zip(range(3 * page, 3 * page + 3), axs):
            t = STEPS[k]
            a, b, w, th, M = info(t); D = X @ w + b; n2 = w @ w
            ax.fill_between(g, line(-1, w, b), line(1, w, b), color="#dde6f6", zorder=0.5)           # 길 안쪽
            nv = w / np.sqrt(n2); dv = np.array([-nv[1], nv[0]])                                     # 스프링: 두 자 사이, 점이 없는 왼쪽 위
            end = lambda c: (c - b) / np.sqrt(n2) * nv + 1.75 * dv
            p0, p1 = end(-1), end(1); nz = 12
            zz = np.array([p0] + [p0 + (p1 - p0) * (0.08 + 0.84 * (i + 0.5) / nz) + dv * (0.17 if i % 2 == 0 else -0.17) for i in range(nz)] + [p1])
            ax.plot(*zz.T, color="#333333", lw=2.2, zorder=2.5)
            for c in (1, -1): ax.plot(g, line(c, w, b), color=RULER, lw=5, solid_capstyle="butt", zorder=2)
            ax.plot(g, line(0, w, b), color="black", ls="--", lw=2, zorder=2)
            if k > 0:
                _, b0, w0, _, _ = info(STEPS[k - 1]); ax.plot(g, line(0, w0, b0), color="#bbbbbb", ls=":", lw=3, zorder=1.5)
            for pt, yy, d, al in zip(X, y, D, a):
                if al > 1e-6 and yy * d < 1 - 1e-3:                                  # 고무줄: 점 → 자기 쪽 자 (D = y)
                    foot = pt + (yy - d) / n2 * w
                    ax.plot(*np.c_[pt, foot], color=BAND, lw=1.2 + 6 * al, solid_capstyle="round", zorder=3)
            on = a > 1e-6
            ax.scatter(X[on, 0], X[on, 1], s=140 + 800 * a[on], facecolors="none", edgecolors=BAND, linewidths=2, zorder=3.5)
            ax.scatter(X[:, 0], X[:, 1], c=np.where(y > 0, RED, BLUE), s=60, edgecolor="black", lw=0.6, zorder=5)
            ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{t} 걸음\n방향 {th:.0f}°,  길 반폭 M = {M:.2f}", fontsize=18, fontweight="bold", linespacing=1.4)
        fig.tight_layout(w_pad=1.5); fig.savefig(mf.O + f"kn_rock9_{page + 1}.png", dpi=150); plt.close(fig)
    return got


def fig_pull():
    """왜 서포트 벡터만 남나: 길 안의 점은 고무줄로 가장자리 (자) 를 자기 쪽으로 당긴다 (α ↑), 길 밖이면 α ↓ 0.
    점 6개 (특징 = 높이 하나, 좌우 위치는 그림용) 를 실제로 학습한 α · 길 폭 (η = 0.01, 3 · 40 · 300걸음)."""
    R = [(1.0, 1.0), (2.3, 1.7), (3.6, 2.4)]; B = [(1.7, -1.0), (3.0, -1.7), (0.5, -2.4)]
    pts = R + B; h = np.array([p[1] for p in pts]); y = np.sign(h)
    _, _, H = kd.train_dual(np.outer(h, h), y, T=300, eta=0.01, etab=0.01, snaps=(3, 40, 300))
    STEPS = [(3, "① 3걸음", "6개 모두 당긴다"), (40, "② 40걸음", "길 밖의 점은 α 가 줄어든다"), (300, "③ 300걸음: 멈춤", "닿은 점만 남는다 = 서포트 벡터")]
    AL = {t: H[t][0] for t, _, _ in STEPS}; M = {t: 1 / float((H[t][0] * y) @ h) for t, _, _ in STEPS}
    assert all(abs(H[t][1]) < 1e-9 for t in AL)
    assert (AL[3] > 0).all() and [round(float(v), 2) for v in AL[40][:3]] == [0.21, 0.07, 0.0] and [round(float(v), 2) for v in AL[300]] == [0.5, 0, 0, 0.5, 0, 0]
    assert [round(M[t], 2) for t, _, _ in STEPS] == [3.99, 1.55, 1.0]
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 6.6))
    for ax, (t, ttl, sub) in zip(axs, STEPS):
        m = M[t]; al = AL[t]
        ax.axhspan(-m, m, color=GOLD, alpha=0.14); ax.axhline(0, color="black", lw=2, ls=(0, (6, 4)))
        for sgn in (1, -1):
            ax.plot([-0.15, 4.55], [sgn * m, sgn * m], color="#8a6d2f", lw=8, solid_capstyle="round", zorder=3)
        for k, (x, hh) in enumerate(pts):
            c, sgn = (RED, 1) if hh > 0 else (BLUE, -1)
            gap = m - abs(hh); touch = abs(gap) < 0.02; zero = al[k] < 0.005
            if gap > 0 and not touch:                              # 길 안: 가장자리를 점 쪽으로 당긴다
                ax.annotate("", (x, hh + sgn * 0.19), (x, sgn * m), arrowprops=dict(arrowstyle="-|>", lw=1.8 + 1.7 * gap, color=c, mutation_scale=19, shrinkA=0, shrinkB=0), zorder=4)
            if touch:
                ax.scatter(x, hh, s=900, facecolors=GOLD, alpha=0.55, edgecolors="none", zorder=4)
            ax.scatter(x, hh, s=190, color=c, alpha=0.35 if zero else 1, zorder=5)
            inside = gap > 0.02
            ty = hh - sgn * 0.12 if inside else hh + sgn * 0.42
            if touch: ty = hh - sgn * 0.5
            ax.text(x + (0.17 if inside else 0.08), ty, "α = 0" if zero else f"α = {al[k]:.2f}", color="#7a7a7a" if zero else c, fontsize=15.5 if not zero else 14,
                    fontweight="normal" if zero else "bold", va="center", ha="left" if inside else "center", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.set_title(f"{ttl},  길 반폭 M = {m:.1f}", fontsize=18, fontweight="bold", pad=10)
        ax.text(2.2, -4.75, sub, fontsize=17, ha="center", fontweight="bold")
        ax.set_xlim(-0.35, 4.85); ax.set_ylim(-5.05, 4.3); ax.axis("off")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(mf.O + "kn_pull.png", dpi=150); plt.close(fig)
    return {t: ([round(float(v), 2) for v in AL[t]], round(M[t], 2)) for t in AL}


FIGS = {"pull": fig_pull, "rock": fig_rock, "learn": fig_learn, "equiv": fig_equiv, "penalty": fig_penalty, "recap": fig_recap, "svarrow": fig_svarrow, "quiz1": fig_quiz1, "quizkid": fig_quizkid, "force": fig_force, "quizkid3": fig_quizkid3, "alpha": fig_alpha, "nails": fig_nails, "overlap": fig_overlap, "saddle": fig_saddle, "meet": fig_meet, "wspring": fig_wspring, "history": fig_history, "rope": fig_rope, "tilt": fig_tilt, "onearrow": fig_onearrow, "parab": fig_parab, "states": fig_states, "dual": fig_dual, "cos": mf.fig_cos, "soft": mf.fig_soft, "soft9": mf.fig_soft9}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
