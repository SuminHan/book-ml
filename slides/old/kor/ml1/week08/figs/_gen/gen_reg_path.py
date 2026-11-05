import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#D6CBB1"
RED = "#C0392B"
COLORS = [BLUE, RED, "#2E8B57", "#D6A800", "#7B4F9E"]

X, y = make_regression(n_samples=200, n_features=5, n_informative=3, noise=8.0, random_state=7)
X = StandardScaler().fit_transform(X)
y = (y - y.mean()) / y.std()

alphas = np.logspace(-2, 2.3, 60)
ridge_coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
lasso_coefs = np.array([Lasso(alpha=a/40, max_iter=20000).fit(X, y).coef_ for a in alphas])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=200, sharey=True)

ax = axes[0]
for j in range(5):
    ax.plot(alphas, ridge_coefs[:, j], color=COLORS[j], linewidth=2.2)
ax.set_xscale("log")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Ridge: $\\lambda \\uparrow$ 이면 0에\n가까워지지만 정확히 0은 아님", fontsize=14)
ax.set_xlabel("$\\lambda$", fontsize=13)
ax.set_ylabel("계수 $w_j$", fontsize=13)

ax = axes[1]
for j in range(5):
    ax.plot(alphas, lasso_coefs[:, j], color=COLORS[j], linewidth=2.2)
ax.set_xscale("log")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Lasso: $\\lambda \\uparrow$ 이면 계수가\n정확히 0에 도달(켄)", fontsize=14)
ax.set_xlabel("$\\lambda$", fontsize=13)

for ax in axes:
    ax.tick_params(axis='both', labelsize=11.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("두 항의 구조: 단순성 항이 $\\|w\\|^2$(L2)냐 $\\|w\\|_1$(L1)이냐", fontsize=16, y=1.04)
fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_3_reg_path.png",
            bbox_inches="tight", facecolor="white")
print("saved")
