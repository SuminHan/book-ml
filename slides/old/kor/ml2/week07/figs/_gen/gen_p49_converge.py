import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
RED = '#C0392B'

alpha, gamma = 0.5, 0.9
QS, QX = 0.0, 0.0
QS_hist, QX_hist = [0.0], [0.0]
# reproduce: iter0: real S->X (Q stays via QX=0), real X->T => QX=0.5
QX = alpha * (1 + gamma * 0 - QX)  # 0.5
QX_hist.append(QX); QS_hist.append(QS)
# subsequent planning steps alternate: plan (S,a)-> uses QX ; occasionally plan (X,a)-> keeps QX near 1
for k in range(12):
    QS = QS + alpha * (0 + gamma * QX - QS)
    QX = QX + alpha * (1 + gamma * 0 - QX)
    QS_hist.append(QS); QX_hist.append(QX)

steps = list(range(len(QS_hist)))

fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=200)
ax.plot(steps, QX_hist, 'o-', color=RED, linewidth=2, markersize=6, label='$Q(X)$ (목표 $\\to 1$)')
ax.plot(steps, QS_hist, 's-', color=BLUE, linewidth=2, markersize=6, label='$Q(S)$ (목표 $\\to 0.9$)')
ax.axhline(1.0, color=RED, linestyle=':', linewidth=1.2)
ax.axhline(0.9, color=BLUE, linestyle=':', linewidth=1.2)
ax.text(steps[-1]*0.72, 1.03, '참값 $Q(X)=1$', color=RED, fontsize=10)
ax.text(steps[-1]*0.72, 0.82, '참값 $Q(S)=0.9$', color=BLUE, fontsize=10)
ax.set_xlabel('계획(planning) 반복 횟수', fontsize=12)
ax.set_ylabel('$Q$ 값', fontsize=12)
ax.set_title('반복할수록 $Q(S)$, $Q(X)$가 참값에 수렴', fontsize=13, fontweight='bold', color=BLUE)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/tmp/gen_out_p49.png', facecolor='white', bbox_inches='tight')
print('done', QS_hist[-1], QX_hist[-1])
