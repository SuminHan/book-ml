"""p14g1: 원래 6개 점 vs (4,3)->(0.5,0.5) 이동 후 새 경계."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.svm import SVC

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
RED = "#C0392B"
TAN = "#D6CBB1"

pos0 = np.array([[3, 3], [4, 3], [3, 4]])
neg0 = np.array([[-3, -3], [-4, -3], [-3, -4]])

pos1 = np.array([[3, 3], [0.5, 0.5], [3, 4]])
neg1 = neg0.copy()

def fit(pos, neg):
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(len(pos)), -np.ones(len(neg))]
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    return clf, X, y

fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))
x1 = np.linspace(-6, 6, 100)

for ax, pos, neg, title in [
    (axes[0], pos0, neg0, "원래 6개 점 --- $w=(1/6,1/6)$, $b=0$"),
    (axes[1], pos1, neg1, "$(4,3)\\to(0.5,0.5)$ 이동 후 --- 새 경계"),
]:
    clf, X, y = fit(pos, neg)
    w, b = clf.coef_[0], clf.intercept_[0]
    sv = X[clf.support_]
    f = lambda c: (-w[0] * x1 - b + c) / w[1]
    ax.fill_between(x1, f(1), f(-1), color=TAN, alpha=0.4, zorder=0)
    ax.plot(x1, f(0), color="black", lw=2.2, zorder=1)
    ax.plot(x1, f(1), color=NAVY, lw=1.2, ls="--", zorder=1)
    ax.plot(x1, f(-1), color=NAVY, lw=1.2, ls="--", zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=NAVY, s=90, zorder=3)
    ax.scatter(neg[:, 0], neg[:, 1], c=RED, s=90, marker="s", zorder=3)
    ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none",
               edgecolors="#E67E22", lw=2.5, zorder=4)
    for p in np.vstack([pos, neg]):
        ax.annotate(f"({p[0]:g},{p[1]:g})", p, textcoords="offset points",
                    xytext=(8, 6), fontsize=9.5)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal")
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.tick_params(labelsize=9.5)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_1_handcalc_points.png", dpi=200, facecolor="white")

clf, X, y = fit(pos1, neg1)
print("new w,b:", clf.coef_[0], clf.intercept_[0])
