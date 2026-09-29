"""p41: T^pi (기대) vs T^* (최적) backup diagram.
고전적인 강화학습 backup diagram: 위쪽 흰 원 = 상태, 아래 검은 점 = 행동,
그 아래 흰 원들 = 다음 상태(전이확률로 갈라짐).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

ACTIONS = [(-1.4, '$a_1$'), (0.0, '$a_2$'), (1.4, '$a_3$')]
NEXT_OFFSETS = [-0.55, 0.0, 0.55]


def draw_state(ax, x, y, r=0.16, fc='white', ec='black', lw=1.6, z=5):
    circ = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=z)
    ax.add_patch(circ)


def draw_action(ax, x, y, r=0.055, fc='black', z=5):
    circ = plt.Circle((x, y), r, facecolor=fc, edgecolor=fc, zorder=z)
    ax.add_patch(circ)


def panel(ax, mode):
    top_y, act_y, bot_y = 2.0, 1.0, 0.0
    draw_state(ax, 0, top_y, r=0.2)
    ax.text(0, top_y + 0.34, "$s$", ha='center', fontsize=15)

    for i, (dx, label) in enumerate(ACTIONS):
        ax.plot([0, dx], [top_y - 0.2, act_y + 0.055], color=KSA_GRAY, lw=1.3, zorder=1)
        is_best = (mode == 'star' and i == 2)
        draw_action(ax, dx, act_y, fc=(KSA_RED if is_best else 'black'))
        ax.text(dx, act_y - 0.30, label, ha='center', fontsize=12.5,
                color=(KSA_RED if is_best else 'black'))
        for off in NEXT_OFFSETS:
            nx = dx + off * 0.85
            active = (mode == 'expect') or (mode == 'star' and i == 2)
            ax.plot([dx, nx], [act_y - 0.055, bot_y + 0.2], color=(KSA_BLUE if active else '#cfcfcf'),
                    lw=(1.6 if active else 1.0), zorder=1)
            draw_state(ax, nx, bot_y, r=0.14,
                       ec=(KSA_BLUE if active else '#cfcfcf'),
                       lw=(1.6 if active else 1.0))

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.6, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')


panel(axes[0], 'expect')
axes[0].set_title("$T^\\pi$: 기대(expectation)\n모든 행동을 $\\pi(a|s)$ 가중합",
                   fontsize=13, color=KSA_BLUE_DK, pad=6)

panel(axes[1], 'star')
axes[1].set_title("$T^{*}$: 최적(optimal)\n최선의 행동 하나만 선택 ($\\max_a$, 빨강)",
                   fontsize=13, color=KSA_BLUE_DK, pad=6)

fig.suptitle("Backup diagram: 기대 열 vs 최적 열의 차이", fontsize=14.5, color=KSA_BLUE_DK, y=1.02)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'backup_diagram_expect_vs_optimal.png')
savefig(fig, out)
print("saved", out)
