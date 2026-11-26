"""SVM 3 (커널) 덱의 그림 원본. 실행: python make_figs.py [키]. 커널 그림 함수는 ../2_svm/make_figs.py, 숫자는 data.py (이 폴더)."""
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
    box(1.6, 2.5, 2.8, 1.7, f"2차원 점 두 개\n$\\mathbf{{x}}$ = {f(kd.K_X)}\n$\\mathbf{{x}}_k$ = {f(kd.K_XP)}", "black")
    box(6.5, 4.0, 4.0, 1.6, "3차원 점 두 개를 만든다\n$\\varphi(\\mathbf{x})$ = (1, $2\\sqrt{2}$, 4)\n$\\varphi(\\mathbf{x}_k)$ = (9, $3\\sqrt{2}$, 1)", BLUE)
    box(6.5, 1.0, 3.8, 0.9, f"$\\mathbf{{x}}\\cdot\\mathbf{{x}}_k$ = 3 + 2 = {kd.K_DOT:g}", RED, fs=18)
    box(11.4, 2.5, 1.6, 1.2, f"{kd.K_DOT ** 2:g}", "black", fs=30, bold=True)
    arr((3.0, 3.0), (4.5, 3.8), BLUE, "① $\\varphi$ 로 옮기기", (3.3, 3.85))
    arr((8.5, 3.8), (10.5, 2.95), BLUE, "② 점곱  $1{\\cdot}9 + 2\\sqrt{2}{\\cdot}3\\sqrt{2} + 4{\\cdot}1$\n$= 9 + 12 + 4$", (11.0, 4.3))
    arr((3.0, 2.0), (4.5, 1.2), RED, "① 점곱", (3.35, 1.25))
    arr((8.5, 1.2), (10.5, 2.05), RED, f"② 제곱 {kd.K_DOT:g}²", (10.0, 1.15))
    ax.text(0.05, 4.75, "먼 길: 옮기고 점곱", ha="left", va="top", fontsize=17, color=BLUE, fontweight="bold")
    ax.text(12.95, 0.15, "지름길 = 커널 $K(\\mathbf{x}, \\mathbf{x}_k) = (\\mathbf{x}\\cdot\\mathbf{x}_k)^2$", ha="right", fontsize=17, color=RED, fontweight="bold")
    mf.save(fig, "kn_ktrick.png")


def fig_nn():
    """원 논문 (Boser 1992) 4장의 시험 오류율 (%): 숫자는 논문 본문 · 표 그대로.
    DB1: 깨끗한 손글씨 1200장 (절반 학습), 직선 분류기끼리: 최대 마진 3.2 vs 평균제곱오차 (역전파 · 유사역행렬) 12.7.
    DB2: 우편물 숫자 7300 학습 · 2000 시험: 최대 마진 다항 4차 4.9 vs 5층 신경망 (LeCun 1990, 같은 데이터) 5.1"""
    cases = [("① 직선끼리 (DB1: 깨끗한 손글씨)", [("평균제곱오차 직선\n(역전파 · 유사역행렬)", 12.7, GRAY), ("최대 마진 직선", 3.2, RED)], 14.5),
             ("② 우편물 숫자 (DB2)", [("5층 신경망\n(LeCun 1990)", 5.1, GRAY), ("최대 마진 + 다항 4차", 4.9, RED)], 6.2)]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (ttl, bars, top) in zip(axs, cases):
        for i, (name, v, c) in enumerate(bars):
            ax.bar(i, v, color=c, width=0.6)
            ax.text(i, v + top * 0.02, f"{v}%", ha="center", va="bottom", fontsize=19, fontweight="bold", color=c if c == RED else "#555555")
        ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars], fontsize=14)
        ax.set_ylim(0, top); ax.set_yticks([]); ax.set_title(ttl, fontsize=16)
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
    mf.save(fig, "kn_nn.png")


def _extra(name, **kw):
    """보충 덱(../9_svm_extra/make_figs.py) 그림 함수를 ‖w‖ = 1 기준으로 불러 이 폴더 figs/ 에 그린다"""
    sp = importlib.util.spec_from_file_location("xfigs", os.path.join(HERE, "..", "9_svm_extra", "make_figs.py"))
    xm = importlib.util.module_from_spec(sp); sp.loader.exec_module(xm)
    os.chdir(HERE); mf.O = os.path.join(HERE, "figs") + "/"
    return getattr(xm, name)(**kw)


def fig_nails_unit():
    """첫 쪽 직관: 스프링이 벌린 두 자가 못 3개에 걸려 멈춘다 (α = ‖w‖ = 1 기준 0.74 · 0.37 · 0.37)"""
    return _extra("fig_nails", unit=True)


def fig_force_unit():
    """힘의 평형: 판이 안 밀리고 (0.74 = 0.37 + 0.37) 안 돈다 (빨강 발 양쪽 0.49 시소)"""
    return _extra("fig_force", unit=True)


def fig_dual_unit():
    """w = 못이 버틴 힘 화살표 α_k y_k x_k 의 합 = (0.71, 0.71), 새 점 D = w·x + b"""
    return _extra("fig_dual", unit=True)


def fig_lagrange():
    """라그랑주 승수법 예 (위에서 본 그림, 축 비율 1:1): 언덕 f = 4 − ½[(x−2)² + (y−2)²] 등고선 (꼭대기 (2, 2) 중심 동심원) + 조건 직선 x + y = 2,
    답 (1, 1) 에서 등고선 f = 3 이 직선에 맞닿고 ∇f = α ∇g (오르막 = 꼭대기 쪽, 직선에 수직)"""
    t = np.linspace(0, 2 * np.pi, 400); cx, cy = kd.LG_C
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for r in (0.6, 2.0, 2.6, 3.2):
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=GRAY, lw=1.1)
    rs = np.sqrt(2 * (kd.LG_TOP - kd.LG_F))
    ax.plot(cx + rs * np.cos(t), cy + rs * np.sin(t), color=GOLD, lw=2.8)
    xs = np.linspace(-1.0, 3.0, 10); ax.plot(xs, 2 - xs, color="black", lw=2.6)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    ax.text(-0.9, 3.25, "g = x + y − 2 = 0", fontsize=17, ha="left", va="center", bbox=box)
    ax.text(2.0, 3.6, "f = 3", color=GOLD, fontsize=17, ha="center", bbox=box)
    ax.scatter([cx], [cy], s=60, color="#666666", zorder=5); ax.text(cx + 0.12, cy + 0.12, "꼭대기 f = 4", fontsize=15, color="#555555", bbox=box)
    x0, y0 = kd.LG_X, kd.LG_Y
    PUR = "#7b3fbf"; dl = np.array([1, -1]) / np.sqrt(2)                       # 직선 방향
    for (px, py), sgn in (((-0.2, 2.2), 1), ((2.2, -0.2), -1)):                # 답이 아닌 점: ∇f 가 직선에 수직이 아니다 → 직선을 따라 남은 오르막
        gf = -np.array([px - cx, py - cy]) * 0.35
        ax.annotate("", xy=(px + gf[0], py + gf[1]), xytext=(px, py), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, mutation_scale=16, alpha=0.7))
        along = (gf @ dl) * dl
        ax.annotate("", xy=(px + along[0], py + along[1]), xytext=(px, py), arrowprops=dict(arrowstyle="-|>", color=PUR, lw=2.6, mutation_scale=16))
        ax.scatter([px], [py], s=45, color="black", zorder=5)
    ax.text(-1.1, 1.35, "남은 오르막\n→ 더 간다", color=PUR, fontsize=15, ha="left", va="center", bbox=box)
    ax.annotate("", xy=(x0 + 0.75, y0 + 0.75), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.8, mutation_scale=20))
    ax.text(x0 + 0.95, y0 + 0.2, "남은 오르막 0\n∇f = α ∇g (수직)", color=RED, fontsize=17, va="center", bbox=box)
    ax.scatter([x0], [y0], s=110, color=RED, zorder=5)
    ax.text(x0 - 0.15, y0 - 0.2, "(1, 1)", fontsize=17, ha="right", va="top", bbox=box)
    ax.set_xlim(-1.2, 5.2); ax.set_ylim(-1.2, 5.2); ax.set_aspect("equal")   # 1:1 이어야 ∇g 가 직선에 수직으로 보인다
    ax.set_xlabel("x", fontsize=17); ax.set_ylabel("y", fontsize=17)
    mf.save(fig, "kn_lagrange.png")


def fig_lagrange3d():
    """라그랑주 예를 3D 로 (흰 배경): 언덕 z = f, 바닥에 동심원 등고선 · 조건 직선 x + y = 2,
    언덕 위 조건 곡선 (보라) 과 그 위 가장 높은 점 (주황), 조건 없을 때의 꼭대기 (회색). ∇ 화살표는 원근 때문에 수직이 안 보여 2D 그림에만"""
    SURF, PUR, ORG = "#2f7fc1", "#7b3fbf", "#e69500"
    cx0, cy0 = kd.LG_C; top = kd.LG_TOP
    R, T = np.meshgrid(np.linspace(0, 2.7, 40), np.linspace(0, 2 * np.pi, 72))
    gx, gy = cx0 + R * np.cos(T), cy0 + R * np.sin(T); F = top - 0.5 * R ** 2
    z0 = -0.6
    fig = plt.figure(figsize=(7.2, 6.4)); ax = fig.add_subplot(projection="3d")
    ax.plot_surface(gx, gy, F, color=SURF, alpha=0.22, linewidth=0, antialiased=True, shade=False)
    ax.plot_wireframe(gx, gy, F, rstride=4, cstride=4, color=SURF, linewidth=0.6, alpha=0.5)
    t = np.linspace(0, 2 * np.pi, 200)
    for r in (0.6, 1.0, 2.0, 2.6):
        ax.plot(cx0 + r * np.cos(t), cy0 + r * np.sin(t), np.full_like(t, z0), color=SURF, lw=1.0, alpha=0.7)
        ax.plot(cx0 + r * np.cos(t), cy0 + r * np.sin(t), np.full_like(t, top - 0.5 * r * r), color=SURF, lw=1.0, alpha=0.5)
    rs = np.sqrt(2 * (top - kd.LG_F))
    ax.plot(cx0 + rs * np.cos(t), cy0 + rs * np.sin(t), np.full_like(t, z0), color=ORG, lw=2.2)
    ax.plot(cx0 + rs * np.cos(t), cy0 + rs * np.sin(t), np.full_like(t, kd.LG_F), color=ORG, lw=2.2)
    h = np.sqrt(2.7 ** 2 - 2) / np.sqrt(2); u = np.linspace(1 - h, 1 + h, 200); cx, cy = u, 2 - u
    cz = top - 0.5 * ((cx - cx0) ** 2 + (cy - cy0) ** 2)
    ax.plot(cx, cy, np.full_like(u, z0), color=PUR, lw=2.2, ls="--")
    ax.plot(cx, cy, cz, color=PUR, lw=3.4)
    x0, y0, f0 = kd.LG_X, kd.LG_Y, kd.LG_F
    ax.plot([x0, x0], [y0, y0], [z0, f0], color=ORG, lw=1.4, ls=":")
    for sz, al in ((650, 0.12), (300, 0.25), (110, 0.5), (40, 1.0)):
        ax.scatter([x0], [y0], [f0], s=sz, color=ORG, alpha=al, depthshade=False, edgecolors="none")
        ax.scatter([x0], [y0], [z0], s=sz * 0.6, color=ORG, alpha=al, depthshade=False, edgecolors="none")
    ax.scatter([cx0], [cy0], [top], s=50, color="#555555", depthshade=False)
    for us, sgn in ((-0.1, 1), (2.1, -1)):                                     # 등산로 위 걷는 사람: 오르막 쪽으로 화살표
        px, py = us, 2 - us; pz = top - 0.5 * ((px - cx0) ** 2 + (py - cy0) ** 2)
        d = np.array([1.0, -1.0, 0.0]) * sgn; d[2] = -((px - cx0) * d[0] + (py - cy0) * d[1]); d = d / np.linalg.norm(d) * 1.1
        ax.scatter([px], [py], [pz], s=45, color="black", depthshade=False)
        ax.quiver(px, py, pz, *d, color=PUR, linewidth=3.6, arrow_length_ratio=0.3)
    ax.text2D(0.03, 0.40, "오르막이 남아 있다\n→ 더 간다", transform=ax.transAxes, color=PUR, fontsize=17, ha="left")
    ax.view_init(elev=26, azim=170); ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.75))
    ax.set_xlim(cx0 - 2.7, cx0 + 2.7); ax.set_ylim(cy0 - 2.7, cy0 + 2.7); ax.set_zlim(z0, top + 0.3)
    ax.text2D(0.03, 0.93, "f = 4 − ½[(x−2)² + (y−2)²]", transform=ax.transAxes, color=SURF, fontsize=19, ha="left")
    ax.text2D(0.56, 0.72, "조건 없으면 꼭대기", transform=ax.transAxes, color="#555555", fontsize=17, ha="left")
    ax.text2D(0.80, 0.50, "더 오를 곳 없다\n= 가장 높은 점", transform=ax.transAxes, color=ORG, fontsize=19, ha="left")
    ax.text2D(0.80, 0.20, "x + y = 2", transform=ax.transAxes, color=PUR, fontsize=19, ha="left")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(mf.O + "kn_lagrange3d.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close(fig)


def fig_lagrange_svm():
    """SVM 의 L 을 w 평면에서 (b = b* 고정, y D = 1 약속, 축 비율 1:1):
    ① 점 하나 = 조건 직선 하나, ½‖w‖² 원이 지키는 곳에 처음 닿는 점 = w*
    ② w* 확대: 닿은 직선 3개 각각에 수직 = ∇g_k = y_k x_k, 원에 수직 = ∇(½‖w‖²) = w* (직각 표시)
    ③ w* = 닿은 3개의 α_k y_k x_k 합 (= Σ α_k ∇g_k, 앞 예의 ∇f = α∇g 를 조건 여러 개로)"""
    d2 = kd._d2
    X = np.array(d2.RED_PTS + d2.BLUE_PTS, float); y = np.r_[np.ones(len(d2.RED_PTS)), -np.ones(len(d2.BLUE_PTS))]
    b, wst = d2.B_STAR, np.array(d2.W_STAR, float)
    sv = [(np.array(p_), d2.ALPHA[p_], 1.0 if p_ in d2.RED_PTS else -1.0) for p_ in d2.SV]
    parts = [a * yy * p_ for p_, a, yy in sv]
    assert np.allclose(sum(parts), wst, atol=0.02) and (y * (X @ wst + b)).min() > 1 - 2e-3
    lo, hi = -0.15, 2.35
    g = np.linspace(lo, hi, 901); W1, W2 = np.meshgrid(g, g)
    feas = np.ones_like(W1, bool)
    for p_, yy in zip(X, y):
        feas &= yy * (W1 * p_[0] + W2 * p_[1] + b) >= 1
    rs = np.linalg.norm(wst); box = dict(fc="white", ec="none", alpha=0.92, pad=3)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.4))
    for k in (0, 1):
        ax = axs[k]
        ax.contourf(W1, W2, (~feas).astype(float), levels=[0.5, 1.5], colors=[GRAY], alpha=0.18)
        for p_, yy in zip(X, y):
            on = any(np.allclose(p_, s_) for s_, _, _ in sv)
            if k == 1 and not on:
                continue
            Z = yy * (W1 * p_[0] + W2 * p_[1] + b) - 1
            if Z.min() < 0 < Z.max():
                ax.contour(W1, W2, Z, levels=[0], colors=[RED if yy > 0 else BLUE], linewidths=2.8 if on else 0.8, alpha=1 if on else 0.35)
        for r in ((0.8, rs, 1.9) if k == 0 else (rs,)):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GOLD if r == rs else "#777777", lw=2.6 if r == rs else 1.2))
        ax.scatter(*wst, s=150, color="black", zorder=6)
    ax = axs[0]
    ax.annotate("w* : 원이 처음 닿는 곳", wst, (1.75, 0.35), ha="center", fontsize=16, fontweight="bold", bbox=box, arrowprops=dict(arrowstyle="->", lw=1.5), zorder=7)
    ax.text(0.55, 1.95, "회색 = 조건을 어긴 w", fontsize=15, color="#555555", bbox=box, ha="center")
    ax.text(0.15, 0.62, "½‖w‖² 등고선", fontsize=14, color="#555555", bbox=box)
    ax.add_patch(plt.Rectangle((0.75, 0.75), 0.6, 0.6, fill=False, ls="--", lw=1.2, color="black"))
    ax.set_title("① b = b* 로 자른 단면: 점 하나 = 직선 하나", fontsize=17, fontweight="bold")
    ax = axs[1]                                                       # 확대
    L, e = 0.20, 0.03
    def _rt(n, c):
        n = n / np.linalg.norm(n); d = np.array([-n[1], n[0]])
        q = np.array([wst + e * d, wst + e * d + e * n, wst + e * n])
        ax.plot(q[:, 0], q[:, 1], color=c, lw=1.6, zorder=7)
    lab = {0: ("y x (빨강)", (0.0, -0.03)), 1: ("y x (파랑 1)", (0.0, 0.0)), 2: ("y x (파랑 2)", (0.03, -0.06))}
    for k, (p_, a, yy) in enumerate(sv):
        n = yy * p_; c = RED if yy > 0 else BLUE; u = n / np.linalg.norm(n)
        ax.annotate("", wst + L * u, wst, arrowprops=dict(arrowstyle="-|>", lw=2.8, color=c, shrinkA=0, shrinkB=0, mutation_scale=18), zorder=8)
        _rt(n, c)
        ax.text(*(wst + (L + 0.02) * u + np.array(lab[k][1])), lab[k][0], color=c, fontsize=15, fontweight="bold", ha="left", va="center", bbox=box, zorder=9)
    u = wst / rs
    ax.annotate("", wst + L * u, wst, arrowprops=dict(arrowstyle="-|>", lw=2.8, color="black", shrinkA=0, shrinkB=0, mutation_scale=18), zorder=8)
    _rt(wst, GOLD)
    ax.text(*(wst + (L + 0.02) * u + np.array([-0.06, 0.02])), "w (원에 수직)", fontsize=15, fontweight="bold", ha="right", va="bottom", bbox=box, zorder=9)
    ax.set_title("② 확대: 직선마다 수직 = y x", fontsize=17, fontweight="bold")
    ax = axs[2]
    ax.add_patch(plt.Circle((0, 0), rs, fill=False, color=GOLD, lw=2.0))
    at = np.zeros(2); names = ["α y x (빨강)", "α y x (파랑 1)", "α y x (파랑 2)"]; offs = [(0.12, -0.15), (0.32, -0.07), (0.38, 0.0)]
    for v, (s_, a, yy), nm, off in zip(parts, sv, names, offs):
        c = RED if yy > 0 else BLUE
        ax.annotate("", at + v, at, arrowprops=dict(arrowstyle="-|>", lw=3.5, color=c, shrinkA=0, shrinkB=0, mutation_scale=22), zorder=5)
        ax.text(*(at + v / 2 + np.array(off)), nm, color=c, fontsize=15, fontweight="bold", ha="center")
        at = at + v
    ax.annotate("", wst, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=3.5, color="black", shrinkA=0, shrinkB=0, mutation_scale=22), zorder=4)
    ax.text(-0.2, 1.3, "∇(½‖w‖²) = w*", fontsize=17, fontweight="bold")
    ax.scatter(*wst, s=120, color="black", zorder=6)
    ax.set_title("③ w* = 닿은 3개의 α y x 합", fontsize=17, fontweight="bold")
    lims = [(lo, hi), (0.75, 1.35), (-0.25, 1.65)]
    lims[1] = (0.75, 1.35)
    for k, ax in enumerate(axs):
        a0, a1 = lims[k]
        ax.set_xlim(a0, a1); ax.set_ylim(a0, a1)
        ax.set_aspect("equal"); ax.set_xlabel("$w_1$", fontsize=17); ax.set_ylabel("$w_2$", fontsize=17); ax.tick_params(labelsize=12)
    fig.tight_layout(w_pad=3)
    mf.save(fig, "kn_lagrange_svm.png")

def fig_lagrange_svm3d(azim=-60, elev=22, out="kn_lagrange_svm3d.png"):
    """SVM 을 (w1, w2, b) 3D 로: 서포트 벡터 3개의 조건 y_k(w·x_k + b) = 1 = 평면 3장, 답 (w*, b*) = 세 평면이 만나는 모서리,
    ½‖w‖² 등고선 = b 와 무관한 원기둥 (금색), b = b* 수평면으로 자르면 직선 3개 + 원 (= 다음 쪽 w 평면 그림)"""
    d2 = kd._d2
    b0, wst = d2.B_STAR, np.array(d2.W_STAR, float); rs = np.linalg.norm(wst)
    sv = [(np.array(p_), 1.0 if p_ in d2.RED_PTS else -1.0) for p_ in d2.SV]
    lo, hi, blo, bhi = 0.0, 1.7, -0.55, 0.65
    fig = plt.figure(figsize=(7.6, 6.0)); ax = fig.add_subplot(projection="3d")
    g = np.linspace(lo, hi, 60); W1, W2 = np.meshgrid(g, g)
    for (p_, yy), c in zip(sv, [RED, BLUE, "#5b6bd6"]):
        B = yy - (W1 * p_[0] + W2 * p_[1])                                   # y(w·x + b) = 1  →  b = y − w·x
        B = np.where((B >= blo) & (B <= bhi), B, np.nan)
        ax.plot_surface(W1, W2, B, color=RED if yy > 0 else BLUE, alpha=0.07, linewidth=0, shade=False)
        ax.plot_wireframe(W1, W2, B, rstride=20, cstride=20, color=RED if yy > 0 else BLUE, linewidth=0.5, alpha=0.18)
        t = np.linspace(lo, hi, 300)                                         # b = b* 에서 자른 선
        if abs(p_[1]) > 1e-9:
            w1 = t; w2 = (yy - b0 - w1 * p_[0]) / p_[1]
        else:
            w2 = t; w1 = (yy - b0 - w2 * p_[1]) / p_[0]
        m = (w1 >= lo) & (w1 <= hi) & (w2 >= lo) & (w2 <= hi)
        ax.plot(w1[m], w2[m], np.full(m.sum(), b0), color=RED if yy > 0 else BLUE, lw=3.0, zorder=10)
    th = np.linspace(0, np.pi / 2, 120); bz = np.linspace(blo, bhi, 2); TH, BZ = np.meshgrid(th, bz)
    CX, CY = rs * np.cos(TH), rs * np.sin(TH); mk = (CX >= lo) & (CX <= hi) & (CY >= lo) & (CY <= hi)
    ax.plot_surface(np.where(mk, CX, np.nan), np.where(mk, CY, np.nan), BZ, color=GOLD, alpha=0.35, linewidth=0, shade=False)
    for bb in (blo, bhi):                                                    # 원통 위 · 아래 테두리
        ax.plot(rs * np.cos(th), rs * np.sin(th), np.full_like(th, bb), color=GOLD, lw=2.4)
    for a_ in np.linspace(0, np.pi / 2, 7):                                  # 세로 줄
        ax.plot([rs * np.cos(a_)] * 2, [rs * np.sin(a_)] * 2, [blo, bhi], color=GOLD, lw=1.2, alpha=0.8)
    cm = (rs * np.cos(th) >= lo) & (rs * np.sin(th) >= lo) & (rs * np.cos(th) <= hi) & (rs * np.sin(th) <= hi)
    ax.plot(rs * np.cos(th)[cm], rs * np.sin(th)[cm], np.full(cm.sum(), b0), color=GOLD, lw=2.6)
    ax.plot_surface(*np.meshgrid([lo, hi], [lo, hi]), np.full((2, 2), b0), color="#888888", alpha=0.18, linewidth=0, shade=False)
    ax.plot([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo], [b0] * 5, color="#666666", lw=1.0)
    ax.scatter(*wst, b0, s=90, color="black", depthshade=False, zorder=12)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(blo, bhi)
    ax.set_xlabel("$w_1$", fontsize=16); ax.set_ylabel("$w_2$", fontsize=16); ax.set_zlabel("$b$", fontsize=16)
    ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 0.5, 1.0, 1.5]); ax.set_zticks([-0.5, 0, 0.5]); ax.tick_params(labelsize=10)
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out if "/" in out else mf.O + out, dpi=200, facecolor="white", bbox_inches=fig.bbox_inches.from_bounds(0.45, 0.15, 6.9, 5.75), pad_inches=0); plt.close(fig)


def fig_lagrange_svm_grow():
    """원통 ½‖w‖² = 반지름 r 을 키우다가 조건을 다 지키는 영역에 처음 닿는 곳 = (w*, b*). 두 줄로 나란히:
    위 = 서포트 벡터 3개 (우리 점 20개): 평면 3장이 깎은 뿔, 뿔 끝 (꼭짓점) 에 닿는다. r = 0.7 · 1.1 · 1.49
    아래 = 서포트 벡터 2개 (지난 시간 작은 예 점 8개): 평면 2장이 만든 지붕, 모서리 선 (b = 0, w1 + w2 = 2) 에 옆으로 닿는다. r = 0.6 · 1.0 · 1.41"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d2 = kd._d2
    # 3개
    V3 = np.r_[np.array(d2.W_STAR, float), d2.B_STAR]
    sv3 = [(np.array(p_), 1.0 if p_ in d2.RED_PTS else -1.0) for p_ in d2.SV]
    N3 = [np.r_[yy * p_, yy] for p_, yy in sv3]
    assert all(abs(n @ V3 - 1) < 2e-3 for n in N3)
    E = {}
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j; d = np.cross(N3[i], N3[j]); d = d if N3[k] @ d > 0 else -d
            E[(i, j)] = d / np.linalg.norm(d)
    L = 1.1
    faces3 = [(sv3[k][1], [V3, V3 + L * es[0], V3 + L * es[1]]) for k in range(3) for es in [[E[e] for e in E if k in e]]]
    edges3 = [(V3, V3 + L * e) for e in E.values()]
    # 2개
    V2 = np.r_[np.array(kd.TOY2_W, float), kd.TOY2_B]
    sv2 = [(np.array(p_), 1.0 if p_ in kd.TOY2_RED else -1.0) for p_ in kd.TOY2_SV]
    N2 = [np.r_[yy * p_, yy] for p_, yy in sv2]
    assert all(abs(n @ V2 - 1) < 2e-3 for n in N2)
    e = np.cross(N2[0], N2[1]); e = e / np.linalg.norm(e); T = 1.3; L2 = 0.9
    faces2, edges2 = [], [(V2 - T * e, V2 + T * e)]
    for k in range(2):
        dk = np.cross(e, N2[k]); dk = dk / np.linalg.norm(dk); dk = dk if N2[1 - k] @ dk > 0 else -dk
        faces2.append((sv2[k][1], [V2 - T * e, V2 + T * e, V2 + T * e + L2 * dk, V2 - T * e + L2 * dk]))
    cols = [(faces3, edges3, V3, "서포트 벡터 3개: 뿔 끝 (꼭짓점) 에 닿음"), (faces2, edges2, V2, "서포트 벡터 2개: 지붕 모서리 (선) 에 닿음")]
    fig = plt.figure(figsize=(16, 7.4))
    th = np.linspace(0, np.pi / 2, 120); lo, hi, blo, bhi = 0.0, 2.0, -0.9, 1.0
    for cl, (faces, edges, V, head) in enumerate(cols):
        ax = fig.add_subplot(1, 2, cl + 1, projection="3d")
        rstar = np.linalg.norm(V[:2])
        for yy, f in faces:
            col = RED if yy > 0 else BLUE
            ax.add_collection3d(Poly3DCollection([f], facecolor=col, alpha=0.18, edgecolor=col, linewidths=0.8))
        for a_, b_ in edges:
            ax.plot(*np.c_[a_, b_], color="#333333", lw=1.8)
        for r in (0.4, 0.8, 1.2):                                         # 키우는 중의 작은 원통 (점선 테두리)
            for bb in (blo, bhi):
                ax.plot(r * np.cos(th), r * np.sin(th), np.full_like(th, bb), color=GOLD, lw=1.3, ls="--", alpha=0.7)
        TH, BZ = np.meshgrid(th, np.linspace(blo, bhi, 2))
        ax.plot_surface(rstar * np.cos(TH), rstar * np.sin(TH), BZ, color=GOLD, alpha=0.35, linewidth=0, shade=False)
        for bb in (blo, bhi):
            ax.plot(rstar * np.cos(th), rstar * np.sin(th), np.full_like(th, bb), color=GOLD, lw=2.6)
        for a_ in np.linspace(0, np.pi / 2, 5):
            ax.plot([rstar * np.cos(a_)] * 2, [rstar * np.sin(a_)] * 2, [blo, bhi], color=GOLD, lw=1.0, alpha=0.8)
        a_ = np.pi / 4; ax.quiver(0.3 * np.cos(a_), 0.3 * np.sin(a_), bhi, (rstar - 0.45) * np.cos(a_), (rstar - 0.45) * np.sin(a_), 0, color=GOLD, linewidth=2.6, arrow_length_ratio=0.15)
        ax.text2D(0.5, 0.86, f"원통을 키운다 (점선): r = 0.4 → 0.8 → 1.2 → {rstar:.2f}", transform=ax.transAxes, color=GOLD, fontsize=15, fontweight="bold", ha="center")
        ax.scatter(*V, s=90, color="black", depthshade=False)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(blo, bhi)
        ax.set_xlabel("$w_1$", fontsize=15); ax.set_ylabel("$w_2$", fontsize=15); ax.set_zlabel("$b$", fontsize=15)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.set_zticks([-0.5, 0, 0.5]); ax.tick_params(labelsize=10)
        ax.view_init(elev=22, azim=-60)
        ax.set_title(head, fontsize=19, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.0, right=0.98, bottom=0.0, top=0.95, wspace=0.0)
    fig.savefig(mf.O + "kn_lagrange_svm_grow.png", dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

FIGS = {"nails_unit": fig_nails_unit, "force_unit": fig_force_unit, "dual_unit": fig_dual_unit, "q": fig_q, "lift1d": fig_lift1d, "ktrick": fig_ktrick, "nn": fig_nn, "lagrange": fig_lagrange, "lagrange3d": fig_lagrange3d, "lagrange_svm": fig_lagrange_svm, "lagrange_svm3d": fig_lagrange_svm3d, "lagrange_svm_grow": fig_lagrange_svm_grow, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree, "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
