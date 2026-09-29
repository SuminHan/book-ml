import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
RED = '#C0392B'

# 좁고 긴 골짜기 형태의 손실 (anisotropic quadratic)
def loss(x, y):
    return 0.02 * x ** 2 + 2.0 * y ** 2

xs = np.linspace(-6, 6, 300)
ys = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(xs, ys)
Z = loss(X, Y)

def grad(p):
    x, y = p
    return np.array([0.04 * x, 4.0 * y])

def run(lr, steps, start):
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(p)
        path.append(p.copy())
    return np.array(path)

big_path = run(lr=0.45, steps=12, start=[-5.5, 1.7])
small_path = run(lr=0.06, steps=40, start=[-5.5, 1.7])

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
cs = ax.contour(X, Y, Z, levels=18, cmap='Greys', linewidths=0.8)
ax.plot(big_path[:, 0], big_path[:, 1], 'o-', color=RED, lw=1.8, ms=4,
        label='큰 lr: 골짜기 벽 사이에서 진동')
ax.plot(small_path[:, 0], small_path[:, 1], 'o-', color=KSA_BLUE, lw=1.4, ms=2.5,
        label='작은 lr: 느리지만 바닥까지 도달')
ax.scatter([0], [0], marker='*', color='gold', edgecolor='k', s=260, zorder=5,
           label='최솟값')

ax.set_xlabel('$w_1$', fontsize=13)
ax.set_ylabel('$w_2$', fontsize=13)
ax.set_title('좁은 골짜기에서 학습률의 두 얼굴', fontsize=14)
ax.legend(fontsize=10.5, frameon=False, loc='upper right')
ax.tick_params(labelsize=11)
fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_3_lr_valley.png', dpi=200, facecolor='white')
print('saved')
