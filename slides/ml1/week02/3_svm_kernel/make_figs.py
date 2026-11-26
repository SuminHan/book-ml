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


def fig_dual_unit():
    """w = 못이 버틴 힘 화살표 α_k y_k x_k 의 합 = (0.71, 0.71), 새 점 D = w·x + b"""
    return _extra("fig_dual", unit=True)


FIGS = {"nails_unit": fig_nails_unit, "dual_unit": fig_dual_unit, "q": fig_q, "lift1d": fig_lift1d, "ktrick": fig_ktrick, "nn": fig_nn, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree, "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
