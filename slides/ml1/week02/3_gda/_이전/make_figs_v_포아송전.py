"""GDA 덱 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png)
숫자는 data.py (assert) 와 같은 함수로 계산한다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import data as D

plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
GREEN, BLUE, GOLD, GRAY, RED = "#2a9d5c", "#2E3192", "#A9976B", "#888888", "#d62728"
COL = (GREEN, BLUE)
O = "figs/"
os.makedirs(O, exist_ok=True)
XL = (2.8, 7.2)


def rug(ax, x, y, base=-0.12, step=0.07, seed=0):
    rng = np.random.default_rng(seed)
    for k in (0, 1):
        m = y == k; ax.scatter(x[m], base - step * k + rng.uniform(-0.015, 0.015, m.sum()), s=16, color=COL[k], alpha=0.6, edgecolors="none")


def fig_1d(x, y, M, s, xq=D.X_Q):
    """종마다 꽃잎 길이 히스토그램 + 공유 표준편차 가우시안, 4.9 cm 의 두 높이."""
    xs = np.linspace(*XL, 400); bins = np.arange(2.8, 7.3, 0.2)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k in (0, 1):
        ax.hist(x[y == k], bins=bins, density=True, color=COL[k], alpha=0.25)
        ax.plot(xs, D.gauss(xs, M[k], s), color=COL[k], lw=3.5, label=f"{D.NAMES[k]}: 평균 {M[k]:.2f}")
    h = [D.gauss(xq, M[k], s) for k in (0, 1)]
    ax.axvline(xq, color=GRAY, ls=":")
    for k in (0, 1):
        ax.plot(xq, h[k], "o", color=COL[k], ms=9)
    ax.annotate(f"{xq} cm: 높이 {h[0]:.3f} vs {h[1]:.3f}", (xq, h[0]), (5.3, 0.95), fontsize=14, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("밀도"); ax.set_xlim(*XL); ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "gda_1d.png"); plt.close(fig)


def fig_post(x, y, wg, wl):
    """GDA 의 사후확률을 그리면 S 자 + 로지스틱회귀 곡선."""
    xs = np.linspace(*XL, 400); sg = lambda w: 1 / (1 + np.exp(-(w[0] + w[1] * xs)))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(xs, sg(wg), color=BLUE, lw=3.5, label="GDA: 종마다 가우시안 + 베이즈")
    ax.plot(xs, sg(wl), color=GOLD, lw=3, ls="--", label="로지스틱회귀 (경사하강법)")
    rug(ax, x, y)
    for w, c, dy in ((wg, BLUE, 0.12), (wl, GOLD, -0.12)):
        p = 1 / (1 + np.exp(-(w[0] + w[1] * D.X_Q))); ax.plot(D.X_Q, p, "o", color=c, ms=9)
        ax.annotate(f"{p:.0%}", (D.X_Q, p), (D.X_Q + 0.25, p + dy), fontsize=15, color=c, fontweight="bold")
    ax.axvline(D.X_Q, color=GRAY, ls=":")
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("virginica 일 확률"); ax.set_xlim(*XL); ax.set_ylim(-0.25, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "gda_post.png"); plt.close(fig)


def ellipse(ax, mu, S, color, ns=(1, 2)):
    val, vec = np.linalg.eigh(S); ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    for n in ns:
        ax.add_patch(Ellipse(mu, 2 * n * np.sqrt(val[1]), 2 * n * np.sqrt(val[0]), angle=ang, fill=False, color=color, lw=2.5, alpha=0.9 if n == 1 else 0.5))


def scatter2d(ax, X, y):
    for k in (0, 1):
        m = y == k; ax.scatter(X[m, 0], X[m, 1], s=28, color=COL[k], alpha=0.65, edgecolors="none", label=D.NAMES[k])
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("꽃잎 너비 (cm)"); ax.set_xlim(2.8, 7.2); ax.set_ylim(0.8, 2.7)


def fig_sigma():
    """공분산 행렬 세 가지: 둥근 · 길쭉 · 비스듬."""
    cases = [((1, 0), (0, 1), "$\\Sigma = \\begin{bmatrix}1&0\\\\0&1\\end{bmatrix}$"), ((2.5, 0), (0, 0.4), "$\\begin{bmatrix}2.5&0\\\\0&0.4\\end{bmatrix}$"),
             ((1, 0.8), (0.8, 1), "$\\begin{bmatrix}1&0.8\\\\0.8&1\\end{bmatrix}$")]
    titles = ["둥근 종", "가로로 넓게", "함께 커진다 (비스듬)"]
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    for a, (r0, r1, lab), t in zip(ax, cases, titles):
        S = np.array([r0, r1], float); P = rng.multivariate_normal([0, 0], S, 300)
        a.scatter(P[:, 0], P[:, 1], s=10, color=GRAY, alpha=0.4, edgecolors="none"); ellipse(a, (0, 0), S, BLUE)
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_aspect("equal"); a.set_title(t, fontsize=16, fontweight="bold")
        a.text(0, -3.6, {0: "대각 = 분산 1, 1 · 비대각 0", 1: "대각 = 분산 2.5, 0.4", 2: "비대각 0.8 = 같이 움직임"}[list(ax).index(a)], ha="center", fontsize=13)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(O + "gda_sigma.png"); plt.close(fig)


def fig_2d(X, y, M, S, w0, w):
    """2차원 GDA: 같은 모양 (공유 공분산) 타원 두 개, 경계는 직선."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    scatter2d(ax, X, y)
    for k in (0, 1):
        ellipse(ax, M[k], S, COL[k]); ax.plot(*M[k], "X", color=COL[k], ms=13, mec="white")
    xs = np.linspace(2.8, 7.2, 50); ax.plot(xs, -(w0 + w[0] * xs) / w[1], color=GOLD, lw=3.5, label="경계 (직선)")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "gda_2d.png"); plt.close(fig)


def fig_qda(X, y):
    """공분산을 같게 (직선) vs 종마다 다르게 (곡선)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
    g0, g1 = np.meshgrid(np.linspace(2.8, 7.2, 400), np.linspace(0.8, 2.7, 300)); G = np.c_[g0.ravel(), g1.ravel()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, m, t in zip(ax, (LDA(store_covariance=True), QDA(store_covariance=True)), ("공분산 하나를 같이 쓰면: 직선", "종마다 따로 쓰면: 곡선")):
        m.fit(X, y); P = m.predict_proba(G)[:, 1].reshape(g0.shape)
        scatter2d(a, X, y); a.contour(g0, g1, P, levels=[0.5], colors=GOLD, linewidths=3.5)
        covs = [m.covariance_] * 2 if hasattr(m, "covariance_") and not isinstance(m.covariance_, list) else m.covariance_
        for k in (0, 1):
            ellipse(a, m.means_[k], covs[k], COL[k], ns=(1,))
        a.set_title(f"{t} (정확도 {m.score(X, y):.0%})", fontsize=15, fontweight="bold")
    fig.tight_layout(); fig.savefig(O + "gda_qda.png"); plt.close(fig)


def fig_small(sm):
    ns = list(sm); g = [100 * sm[n][0] for n in ns]; l = [100 * sm[n][1] for n in ns]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ns, g, "o-", color=BLUE, lw=3, ms=9, label="GDA (생성)"); ax.plot(ns, l, "s--", color=GOLD, lw=3, ms=8, label="로지스틱회귀 (판별)")
    for n, a, b in zip(ns, g, l):
        ax.annotate(f"{a:.1f}", (n, a), (0, 9), textcoords="offset points", ha="center", fontsize=12, color=BLUE)
        ax.annotate(f"{b:.1f}", (n, b), (0, -18), textcoords="offset points", ha="center", fontsize=12, color="#6b5b2e")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(ns, [str(n) for n in ns]); ax.set_xlabel("학습에 쓴 꽃 수 (종마다 절반)")
    ax.set_ylabel("시험 정확도 (%)"); ax.set_ylim(80, 97); ax.legend(frameon=False, loc="lower right", fontsize=13)
    fig.tight_layout(); fig.savefig(O + "gda_small.png"); plt.close(fig)


def fig_sample(X, y, M, S, seed=7):
    """생성 모델은 꽃을 '만들' 수 있다: 진짜 100송이 vs GDA 가 뽑은 100송이."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    scatter2d(ax[0], X, y); ax[0].set_title("진짜 꽃 100송이", fontsize=16, fontweight="bold")
    Xs = np.vstack([rng.multivariate_normal(M[k], S, 50) for k in (0, 1)]); ys = np.r_[np.zeros(50, int), np.ones(50, int)]
    scatter2d(ax[1], Xs, ys); ax[1].set_title("GDA 가 만든 꽃 100송이", fontsize=16, fontweight="bold")
    ax[0].legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "gda_sample.png"); plt.close(fig)


if __name__ == "__main__":
    X, y = D.iris2(); x = X[:, 0]
    phi, M1, S1 = D.fit_gda(x, y); s = float(np.sqrt(S1[0, 0]))
    wg0, wg = D.gda_w(phi, M1, S1); wl0, wl = D.fit_logistic(x, y)
    fig_1d(x, y, M1[:, 0], s); fig_post(x, y, (wg0, wg[0]), (wl0, wl[0]))
    phi2, M2, S2 = D.fit_gda(X, y); w0, w = D.gda_w(phi2, M2, S2)
    fig_sigma(); fig_2d(X, y, M2, S2, w0, w); fig_qda(X, y); fig_small(D.SMALL); fig_sample(X, y, M2, S2)
    print("figs ok")
