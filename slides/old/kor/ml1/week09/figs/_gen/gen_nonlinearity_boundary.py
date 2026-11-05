import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
KSA_TAN = '#D6CBB1'
RED = '#C0392B'

rng = np.random.RandomState(0)
X, y = make_moons(n_samples=200, noise=0.15, random_state=0)

lin = LogisticRegression().fit(X, y)                 # 활성함수 없는 선형 누적과 동치
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu',
                     max_iter=3000, random_state=0).fit(X, y)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                      np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200)
for ax, model, title in zip(
        axes, [lin, mlp],
        ['활성함수 없음(또는 로지스틱회귀)\n$\\Rightarrow$ 직선 경계',
         '활성함수(ReLU) 있는 2층 MLP\n$\\Rightarrow$ 곡선 경계']):
    Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5], colors=KSA_BLUE, linewidths=2.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=KSA_TAN, edgecolor='k', s=25, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=RED, edgecolor='k', s=25, zorder=3)
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks([]); ax.set_yticks([])

fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_2_nonlinearity_boundary.png', dpi=200, facecolor='white')
print('saved')
