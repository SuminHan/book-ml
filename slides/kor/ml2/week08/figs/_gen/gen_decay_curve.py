"""ch08_3_gamma_decay.png -- gamma^n 기하급수 감소 곡선 (p51 숫자 감각 보강)"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=200)

n = np.arange(0, 61)
for gamma, color, lw, label in [
    (0.99, TAN, 2.6, r'$\gamma=0.99$'),
    (0.9, NAVY, 3.2, r'$\gamma=0.9$ (본문 예제)'),
    (0.5, '#8C8C8C', 2.2, r'$\gamma=0.5$'),
]:
    ax.plot(n, gamma ** n, color=color, lw=lw, label=label)

# markers for n=5,10,20,50 on gamma=0.9 curve (table in p51)
marks_n = [5, 10, 20, 50]
for m in marks_n:
    y = 0.9 ** m
    ax.plot([m], [y], 'o', color=NAVY, ms=6, zorder=5)
    ax.annotate(f'n={m}\n{y:.3f}', (m, y), textcoords='offset points',
                xytext=(6, 10), fontsize=10.5, color=NAVY)

# 1% threshold line
n_star = np.log(0.01) / np.log(0.9)
ax.axhline(0.01, color=RED, lw=1.4, ls='--')
ax.axvline(n_star, color=RED, lw=1.4, ls='--')
ax.plot([n_star], [0.01], 's', color=RED, ms=7, zorder=6)
ax.annotate(f'$\\gamma^n<0.01$\n$n>{n_star:.0f}$', (n_star, 0.01),
            textcoords='offset points', xytext=(8, 22), fontsize=11.5, color=RED,
            fontweight='bold')

ax.set_xlabel('반복 횟수 $n$', fontsize=13)
ax.set_ylabel(r'남은 오차 비율 $\gamma^n$', fontsize=13)
ax.set_title(r'축소 사상의 기하급수 감소: $\|V_n-V^*\|\leq\gamma^n\|V_0-V^*\|$', fontsize=13.5)
ax.set_xlim(0, 60)
ax.set_ylim(-0.03, 1.03)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11.5, loc='upper right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

plt.tight_layout(pad=0.4)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_3_gamma_decay.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
