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
    """라그랑주 승수법 예 (축 비율 1:1): f = x² + 2y 등고선 (포물선) + 조건 직선 3x + 2y + 1 = 0, 답에서 두 선이 맞닿고 ∇f = α ∇g (평행)"""
    xs = np.linspace(-2.5, 5.5, 400); gx, gy = np.meshgrid(xs, np.linspace(-6, 1.5, 400))
    F = gx ** 2 + 2 * gy
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.contour(gx, gy, F, levels=[-9, -6.5, -1, 1.5, 4], colors=GRAY, linewidths=1.0, linestyles="solid")
    ax.contour(gx, gy, F, levels=[kd.LG_F], colors=GOLD, linewidths=2.8, linestyles="solid")
    ax.plot(xs, (-1 - 3 * xs) / 2, color="black", lw=2.6)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    ax.text(-0.7, 1.0, "g = 3x + 2y + 1 = 0", fontsize=17, ha="left", va="center", bbox=box)
    ax.text(-2.3, -3.9, f"f = x² + 2y = {kd.LG_F:g}", color=GOLD, fontsize=17, ha="left", bbox=box)
    x0, y0 = kd.LG_X, kd.LG_Y
    u = np.array([3, 2]) / np.hypot(3, 2) * 1.1
    ax.annotate("", xy=(x0 + u[0], y0 + u[1]), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6, mutation_scale=18))
    ax.text(x0 + u[0] + 0.15, y0 + u[1], "∇f = α ∇g\n(2x, 2) = 1·(3, 2)", color=RED, fontsize=17, va="center", bbox=box)
    ax.scatter([x0], [y0], s=110, color=RED, zorder=5)
    ax.text(x0 - 0.25, y0 - 0.25, "(x, y) = (3/2, −11/4)", fontsize=17, ha="right", va="top", bbox=box)
    ax.set_xlim(-2.5, 5.5); ax.set_ylim(-6, 1.5); ax.set_aspect("equal")   # 1:1 이어야 ∇g 가 직선에 수직으로 보인다
    ax.set_xlabel("x", fontsize=17); ax.set_ylabel("y", fontsize=17)
    mf.save(fig, "kn_lagrange.png")

def fig_lagrange3d():
    """라그랑주 예를 3D 로 (어두운 배경): 곡면 z = f = x² + 2y, 조건 직선 위를 따라가는 곡선 (보라), 그 위 가장 낮은 점 (노랑 빛), 바닥에 등고선 · 조건 직선 (∇ 화살표는 원근 때문에 수직이 안 보여 2D 그림 kn_lagrange 에만)"""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    BG, SURF, PUR, YEL, RD, TXT = "#0f1117", "#4fb3e8", "#c58cff", "#ffe66d", "#ff5a4e", "#e8e8e8"
    xs, ys = np.linspace(-1.5, 4.0, 60), np.linspace(-5.5, 1.0, 60); gx, gy = np.meshgrid(xs, ys); F = gx ** 2 + 2 * gy
    z0 = -14.0                                                          # 바닥 높이
    fig = plt.figure(figsize=(7.2, 6.0), facecolor=BG); ax = fig.add_subplot(projection="3d", facecolor=BG)
    ax.plot_surface(gx, gy, F, color=SURF, alpha=0.28, linewidth=0, antialiased=True, shade=False)
    ax.plot_wireframe(gx, gy, F, rstride=4, cstride=4, color=SURF, linewidth=0.5, alpha=0.45)
    ax.contour(gx, gy, F, levels=[-9, -6.5, kd.LG_F, -1, 1.5, 4, 6.5, 9], zdir="z", offset=z0, colors=SURF, linewidths=0.9, alpha=0.6, linestyles="solid")
    t = np.linspace(-1.0, 3.3, 200); cy = (-1 - 3 * t) / 2; cz = t ** 2 + 2 * cy   # 곡면 범위 (y ≤ 1, y ≥ −5.5) 안만
    ax.plot(t, cy, np.full_like(t, z0), color=PUR, lw=2.2, ls="--", alpha=0.9)               # 바닥의 조건 직선
    ax.plot(t, cy, cz, color=PUR, lw=3.2)                                                   # 곡면 위로 올린 조건 곡선
    x0, y0, f0 = kd.LG_X, kd.LG_Y, kd.LG_F
    ax.plot([x0, x0], [y0, y0], [z0, f0], color=YEL, lw=1.2, ls=":", alpha=0.9)
    for sz, al in ((700, 0.10), (320, 0.22), (120, 0.45), (40, 1.0)):
        ax.scatter([x0], [y0], [f0], s=sz, color=YEL, alpha=al, depthshade=False, edgecolors="none")
        ax.scatter([x0], [y0], [z0], s=sz * 0.6, color=YEL, alpha=al, depthshade=False, edgecolors="none")
    ax.set_xlim(-1.5, 4.0); ax.set_ylim(-5.5, 1.0); ax.set_zlim(z0, 14)
    ax.view_init(elev=24, azim=-62); ax.set_axis_off()
    ax.text(x0 + 0.3, y0 - 0.4, f0 + 1.0, "조건 위에서 가장 낮은 점", color=YEL, fontsize=20, zorder=10)
    ax.text(-1.0, 0.9, z0 + 1.0, "g = 3x + 2y + 1 = 0", color=PUR, fontsize=20, zorder=10, ha="right")
    ax.text(-1.4, 1.0, 13, "f = x² + 2y", color=SURF, fontsize=20, zorder=10)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(mf.O + "kn_lagrange3d.png", dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.1); plt.close(fig)


def fig_lagrange_svm():
    """SVM 의 L 을 w 평면에서 (b = b* 고정, y D = 1 약속): 점 하나 = 조건 직선 하나, ½‖w‖² 원이 지키는 곳에 처음 닿는 점 = w*,
    거기서 ∇(½‖w‖²) = w* = 닿은 3개의 α_k y_k x_k 합 (= Σ α_k ∇g_k, 앞 예의 ∇f = α∇g 를 조건 여러 개로). 축 비율 1:1"""
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
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    ax = axs[0]
    ax.contourf(W1, W2, (~feas).astype(float), levels=[0.5, 1.5], colors=[GRAY], alpha=0.18)
    for p_, yy in zip(X, y):
        on = any(np.allclose(p_, s_) for s_, _, _ in sv)
        Z = yy * (W1 * p_[0] + W2 * p_[1] + b) - 1
        if Z.min() < 0 < Z.max():
            ax.contour(W1, W2, Z, levels=[0], colors=[RED if yy > 0 else BLUE], linewidths=2.8 if on else 0.8, alpha=1 if on else 0.35)
    for r in (0.8, np.linalg.norm(wst), 1.9):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GOLD if abs(r - np.linalg.norm(wst)) < 1e-9 else "#777777", lw=2.6 if abs(r - np.linalg.norm(wst)) < 1e-9 else 1.2))
    ax.scatter(*wst, s=190, color="black", zorder=6)
    box = dict(fc="white", ec="none", alpha=0.92, pad=3)
    ax.annotate("w* : 원이 처음 닿는 곳", wst, (1.75, 0.35), ha="center", fontsize=16, fontweight="bold", bbox=box, arrowprops=dict(arrowstyle="->", lw=1.5), zorder=7)
    ax.text(0.55, 1.95, "회색 = 조건을 어긴 w", fontsize=15, color="#555555", bbox=box, ha="center")
    ax.text(0.15, 0.62, "½‖w‖² 등고선", fontsize=14, color="#555555", bbox=box)
    ax.set_title("① 점 하나 = 조건 직선 하나 (b = b* 고정)", fontsize=17, fontweight="bold")
    ax = axs[1]
    ax.add_patch(plt.Circle((0, 0), np.linalg.norm(wst), fill=False, color=GOLD, lw=2.0))
    at = np.zeros(2); names = ["α y x (빨강)", "α y x (파랑 1)", "α y x (파랑 2)"]; offs = [(0.12, -0.15), (0.32, -0.07), (0.38, 0.0)]
    for v, (s_, a, yy), nm, off in zip(parts, sv, names, offs):
        c = RED if yy > 0 else BLUE
        ax.annotate("", at + v, at, arrowprops=dict(arrowstyle="-|>", lw=3.5, color=c, shrinkA=0, shrinkB=0, mutation_scale=22), zorder=5)
        ax.text(*(at + v / 2 + np.array(off)), nm, color=c, fontsize=15, fontweight="bold", ha="center")
        at = at + v
    ax.annotate("", wst, (0, 0), arrowprops=dict(arrowstyle="-|>", lw=3.5, color="black", shrinkA=0, shrinkB=0, mutation_scale=22), zorder=4)
    ax.text(-0.2, 1.3, "∇(½‖w‖²) = w*", fontsize=17, fontweight="bold")
    ax.scatter(*wst, s=120, color="black", zorder=6)
    ax.set_title("② w* = 닿은 3개의 α y x 합", fontsize=17, fontweight="bold")
    for k, ax in enumerate(axs):
        ax.set_xlim(*((lo, hi) if k == 0 else (-0.25, 1.65))); ax.set_ylim(*((lo, hi) if k == 0 else (-0.25, 1.65)))
        ax.set_aspect("equal"); ax.set_xlabel("$w_1$", fontsize=17); ax.set_ylabel("$w_2$", fontsize=17); ax.tick_params(labelsize=12)
    fig.tight_layout(w_pad=3)
    mf.save(fig, "kn_lagrange_svm.png")


FIGS = {"nails_unit": fig_nails_unit, "force_unit": fig_force_unit, "dual_unit": fig_dual_unit, "q": fig_q, "lift1d": fig_lift1d, "ktrick": fig_ktrick, "nn": fig_nn, "lagrange": fig_lagrange, "lagrange3d": fig_lagrange3d, "lagrange_svm": fig_lagrange_svm, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree, "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
