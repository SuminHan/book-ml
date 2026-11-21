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


FIGS = {"q": fig_q, "lift1d": fig_lift1d, "kernel": mf.fig_kernel, "rbf": mf.fig_rbf, "rbfdeg": mf.fig_rbf_degree,
        "hills": mf.fig_hills, "hillcount": mf.fig_hill_count}

if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if a in FIGS] or list(FIGS)
    for k in pick:
        r = FIGS[k](); print("ok", k, "" if r is None else r)
