import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
KSA_TAN = '#D6CBB1'

N, C = 6, 8  # 배치(행) x 채널(열) 격자로 단순화 (H,W는 각 칸 내부에 접혀있다고 가정)

def draw_grid(ax, highlight_fn, title):
    for n in range(N):
        for c in range(C):
            color = KSA_BLUE if highlight_fn(n, c) else '#EDEDED'
            ax.add_patch(Rectangle((c, N - 1 - n), 0.92, 0.92,
                                    facecolor=color, edgecolor='white', lw=0.6))
    ax.set_xlim(0, C)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12.5)

fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), dpi=200)

# 열(column)=채널, 행(row)=배치 샘플
draw_grid(axes[0], lambda n, c: c == 0, 'BatchNorm\n(같은 채널, 배치 전체)')
draw_grid(axes[1], lambda n, c: n == 0, 'LayerNorm\n(같은 샘플, 채널 전체)')
draw_grid(axes[2], lambda n, c: n == 0 and c == 0, 'InstanceNorm\n(샘플·채널 하나씩)')
draw_grid(axes[3], lambda n, c: n == 0 and (c // (C // 4) == 0), 'GroupNorm\n(같은 샘플, 채널 그룹)')

fig.suptitle('정규화 축 비교 — 어떤 칸들을 함께 평균·분산 내는가 (파란 칸)', fontsize=13.5)
fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_3_norm_axes.png', dpi=200, facecolor='white', bbox_inches='tight')
print('saved')
