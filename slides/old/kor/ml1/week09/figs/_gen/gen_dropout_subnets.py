import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
KSA_TAN = '#D6CBB1'
GREY = '#CFCFCF'

rng = np.random.RandomState(3)

n_in, n_hid, n_out = 3, 5, 1
x_in, x_hid, x_out = 0.0, 1.0, 2.0
y_in = np.linspace(0.2, 0.8, n_in)
y_hid = np.linspace(0.05, 0.95, n_hid)
y_out = [0.5]

masks = [
    np.array([1, 0, 1, 1, 0]),
    np.array([0, 1, 1, 0, 1]),
    np.array([1, 1, 0, 1, 1]),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), dpi=200)
for ax, mask, idx in zip(axes, masks, range(1, 4)):
    for i, yi in enumerate(y_in):
        for j, yj in enumerate(y_hid):
            alive = mask[j] == 1
            ax.plot([x_in, x_hid], [yi, yj],
                    color=(KSA_BLUE if alive else GREY),
                    lw=(1.3 if alive else 0.6),
                    alpha=(0.8 if alive else 0.4), zorder=1)
    for j, yj in enumerate(y_hid):
        alive = mask[j] == 1
        for yo in y_out:
            ax.plot([x_hid, x_out], [yj, yo],
                    color=(KSA_BLUE if alive else GREY),
                    lw=(1.3 if alive else 0.6),
                    alpha=(0.8 if alive else 0.4), zorder=1)

    for yi in y_in:
        ax.scatter([x_in], [yi], s=260, color=KSA_TAN, edgecolor='k', zorder=3)
    for j, yj in enumerate(y_hid):
        alive = mask[j] == 1
        color = KSA_BLUE if alive else 'white'
        ec = 'k' if alive else GREY
        ax.scatter([x_hid], [yj], s=260, color=color, edgecolor=ec, zorder=3)
        if not alive:
            ax.text(x_hid, yj, 'x', ha='center', va='center', fontsize=10,
                    color='#999999', zorder=4)
    for yo in y_out:
        ax.scatter([x_out], [yo], s=260, color='#C0392B', edgecolor='k', zorder=3)

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'전파 #{idx}: 다른 서브네트워크', fontsize=12)
    ax.axis('off')

fig.suptitle('Dropout: 매 전파마다 다른 은닉 유닛을 끔 (파라미터는 공유)',
             fontsize=13.5, y=1.02)
fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_3_dropout_subnets.png', dpi=200, facecolor='white',
            bbox_inches='tight')
print('saved')
