import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
KSA_TAN = '#D6CBB1'
RED = '#C0392B'
GREY = '#888888'

T = 300
t = np.arange(0, T + 1)

# fixed lr
fixed = np.full_like(t, 0.01, dtype=float)

# StepLR: alpha0=0.01, gamma=0.5, step_size=50
step_size, gamma, a0 = 50, 0.5, 0.01
step = a0 * (gamma ** (t // step_size))

# cosine annealing: alpha_min=0, alpha_max=0.01, T_max=300
a_min, a_max = 0.0, 0.01
cosine = a_min + 0.5 * (a_max - a_min) * (1 + np.cos(np.pi * t / T))

fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
ax.plot(t, fixed, color=GREY, lw=2.5, label='고정 lr = 0.01')
ax.step(t, step, where='post', color=KSA_BLUE, lw=2.5, label='StepLR (step=50, $\\gamma$=0.5)')
ax.plot(t, cosine, color=RED, lw=2.5, label='코사인 (0.01 → 0, 300에폭)')

ax.set_xlabel('에폭 (epoch)', fontsize=13)
ax.set_ylabel('학습률 $\\alpha$', fontsize=13)
ax.set_title('학습률 스케줄: 고정 / StepLR / 코사인', fontsize=14)
ax.set_xlim(0, T)
ax.set_ylim(-0.0005, 0.0105)
ax.legend(fontsize=11, frameon=False, loc='upper right')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_3_lr_schedules.png', dpi=200, facecolor='white')
print('saved')
