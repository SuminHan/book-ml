import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
RED = '#C0392B'

epochs = np.linspace(0, 100, 200)

# 죽은 뉴런 비율: ReLU 52% -> 97% (S자형 상승), Leaky ReLU 항상 0%
dead_relu = 52 + (97 - 52) / (1 + np.exp(-(epochs - 40) / 12))
dead_leaky = np.zeros_like(epochs)

# 손실: ReLU는 초반 감소 후 0.26에서 고착, Leaky는 0.095까지 계속 감소
loss_relu = 0.26 + 0.5 * np.exp(-epochs / 8)
loss_leaky = 0.095 + 0.6 * np.exp(-epochs / 20)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)

ax = axes[0]
ax.plot(epochs, dead_relu, color=RED, lw=2.5, label='ReLU')
ax.plot(epochs, dead_leaky, color=KSA_BLUE, lw=2.5, label='Leaky ReLU')
ax.set_xlabel('에폭', fontsize=12.5)
ax.set_ylabel('죽은 뉴런 비율 (%)', fontsize=12.5)
ax.set_title('학습률 0.45: 죽은 뉴런 비율', fontsize=13)
ax.legend(fontsize=11, frameon=False)
ax.set_ylim(-5, 105)

ax = axes[1]
ax.plot(epochs, loss_relu, color=RED, lw=2.5, label='ReLU (0.26 고착)')
ax.plot(epochs, loss_leaky, color=KSA_BLUE, lw=2.5, label='Leaky ReLU (0.095까지)')
ax.set_xlabel('에폭', fontsize=12.5)
ax.set_ylabel('학습 손실', fontsize=12.5)
ax.set_title('같은 실험의 손실 곡선', fontsize=13)
ax.legend(fontsize=11, frameon=False)

for ax in axes:
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_2_dying_relu.png', dpi=200, facecolor='white')
print('saved')
