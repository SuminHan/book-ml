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
    box(6.5, 4.0, 4.0, 1.6, f"3차원 점 두 개를 만든다\n$\\varphi(\\mathbf{{x}})$ = {f(p)}\n$\\varphi(\\mathbf{{x}}_i)$ = {f(q)}", BLUE)
    box(6.5, 1.0, 3.8, 0.9, f"$\\mathbf{{x}}\\cdot\\mathbf{{x}}_i$ = 3 + 2 = {kd.K_DOT:g}", RED, fs=18)
    box(11.4, 2.5, 1.6, 1.2, f"{kd.K_DOT ** 2:g}", "black", fs=30, bold=True)
    arr((3.0, 3.0), (4.5, 3.8), BLUE, "① $\\varphi$ 로 옮기기", (3.3, 3.85))
    arr((8.5, 3.8), (10.5, 2.95), BLUE, f"② 점곱\n{pr[0]:g} + {pr[1]:g} + {pr[2]:g}", (10.0, 3.95))
    arr((3.0, 2.0), (4.5, 1.2), RED, "① 점곱", (3.35, 1.25))
    arr((8.5, 1.2), (10.5, 2.05), RED, f"② 제곱 {kd.K_DOT:g}²", (10.0, 1.15))
    ax.text(12.95, 4.75, "먼 길: 옮기고 점곱", ha="right", va="top", fontsize=17, color=BLUE, fontweight="bold")
    ax.text(12.95, 0.15, "지름길 = 커널 $K(\\mathbf{x}, \\mathbf{x}_i) = (\\mathbf{x}\\cdot\\mathbf{x}_i)^2$", ha="right", fontsize=17, color=RED, fontweight="bold")
    mf.save(fig, "kn_ktrick.png")

FIGS = {"ktrick": fig_ktrick, "dual": fig_dual, "q": fig_q, "lift1d": fig_lift1d, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree,
        "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
