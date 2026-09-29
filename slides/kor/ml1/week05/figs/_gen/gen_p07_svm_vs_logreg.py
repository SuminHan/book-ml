"""p07g1: SVM(서포트 벡터만) vs 로지스틱회귀(모든 점) 대비."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"
GRAY = "#9B9B9B"

rng = np.random.default_rng(3)
Xp = rng.normal([5, 5], 1.4, (14, 2))
Xn = rng.normal([-5, -5], 1.4, (14, 2))
X = np.vstack([Xp, Xn])
y = np.r_[np.ones(14), -np.ones(14)]

svm = SVC(kernel="linear", C=1.0).fit(X, y)
w, b = svm.coef_[0], svm.intercept_[0]
sv_idx = set(svm.support_)

lr = LogisticRegression().fit(X, y)
wl, bl = lr.coef_[0], lr.intercept_[0]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
x1 = np.linspace(-9, 9, 100)

# --- SVM panel ---
ax = axes[0]
for i, (pt, lbl) in enumerate(zip(X, y)):
    is_sv = i in sv_idx
    color = NAVY if lbl == 1 else RED
    marker = "o" if lbl == 1 else "s"
    if is_sv:
        ax.scatter(*pt, c=color, marker=marker, s=140, zorder=4,
                   edgecolors="#E67E22", linewidths=2.5)
    else:
        ax.scatter(*pt, c=color, marker=marker, s=45, zorder=2, alpha=0.35)
f = lambda c: (-w[0] * x1 - b + c) / w[1]
ax.plot(x1, f(0), color="black", lw=2.4, zorder=3)
ax.plot(x1, f(1), color="black", lw=1.1, ls="--", zorder=3)
ax.plot(x1, f(-1), color="black", lw=1.1, ls="--", zorder=3)
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal")
ax.set_title("SVM --- 진한 원: 서포트 벡터만 경계를 결정", fontsize=12.5)
ax.tick_params(labelsize=10)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

# --- Logistic regression panel ---
ax = axes[1]
for i, (pt, lbl) in enumerate(zip(X, y)):
    color = NAVY if lbl == 1 else RED
    marker = "o" if lbl == 1 else "s"
    ax.scatter(*pt, c=color, marker=marker, s=110, zorder=3,
               edgecolors="#E67E22", linewidths=1.6)
fl = lambda c: (-wl[0] * x1 - bl + c) / wl[1]
ax.plot(x1, fl(0), color="black", lw=2.4, zorder=2)
ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal")
ax.set_title("로지스틱회귀 --- 모든 점이 손실에 기여", fontsize=12.5)
ax.tick_params(labelsize=10)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_1_svm_vs_logreg.png", dpi=200, facecolor="white")
print("done")
