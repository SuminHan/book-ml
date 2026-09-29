"""p04g1: '무수히 많은 분할 직선' vs '마진이 가장 넓은 직선'."""
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
TAN = "#D6CBB1"
RED = "#C0392B"

rng = np.random.default_rng(7)
Xp = rng.normal([5, 5], 0.8, (10, 2))
Xn = rng.normal([-5, -5], 0.8, (10, 2))
X = np.vstack([Xp, Xn])
y = np.r_[np.ones(10), -np.ones(10)]

clf = SVC(kernel="linear", C=1e6).fit(X, y)
w, b = clf.coef_[0], clf.intercept_[0]
sv = X[clf.support_]

fig, ax = plt.subplots(figsize=(7.2, 6.2))

ax.scatter(Xp[:, 0], Xp[:, 1], c=NAVY, s=70, zorder=3, label="양성 ($y=+1$)")
ax.scatter(Xn[:, 0], Xn[:, 1], c=RED, s=70, marker="s", zorder=3, label="음성 ($y=-1$)")

x1 = np.linspace(-8, 8, 100)

# several plausible-but-worse separating lines (thin gray)
candidates = [
    (1.0, 0.3, "gray"),
    (0.5, -1.0, "gray"),
    (1.6, 1.5, "gray"),
    (0.7, -2.2, "gray"),
]
for slope, intercept, color in candidates:
    ax.plot(x1, slope * x1 + intercept, color=color, lw=1.3, alpha=0.7, zorder=1)

# max-margin boundary (from w, b): w0*x1 + w1*x2 + b = c
f = lambda c: (-w[0] * x1 - b + c) / w[1]
ax.plot(x1, f(0), color=NAVY, lw=3, zorder=2, label="최대 마진 경계")
ax.plot(x1, f(1), color=NAVY, lw=1.4, ls="--", zorder=2)
ax.plot(x1, f(-1), color=NAVY, lw=1.4, ls="--", zorder=2)
ax.fill_between(x1, f(1), f(-1), color=TAN, alpha=0.35, zorder=0)

ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors="none",
           edgecolors="#E67E22", lw=2.5, zorder=4, label="서포트 벡터")

ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("무수히 많은 분할 직선 중, 마진이 가장 넓은 하나", fontsize=15, pad=10)
ax.legend(loc="lower right", fontsize=10.5, framealpha=0.9)
ax.set_aspect("equal")
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_1_candidate_lines.png", dpi=200, facecolor="white")
print("done", w, b)
