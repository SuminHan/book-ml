"""p20: 잔디깎는 기 MDP에서 위쪽/아래쪽 루프를 타는 에피소드의 방문 빈도 차이.
왼쪽: 두 대표 루프(위/아래)를 화살표로 표시, 각 루프에서 (0,2)/(0,0)이 몇 번
방문되는지 주석. 오른쪽: p14 표의 실제 방문 횟수(20만 에피소드, 시드 42)를
막대그래프로 재현.
"""
import matplotlib
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
SAND = '#D6CBB1'
RED = '#C0392B'

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=200)

# ---- Left: grid with two loops ----
ax = axes[0]
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(-0.6, 2.6)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.axis('off')

coords = {(r, c): (c, r) for r in range(3) for c in range(3)}
for (r, c), (x, y) in coords.items():
    is_term = (r, c) == (1, 1)
    face = SAND if is_term else 'white'
    rect = mpatches.FancyBboxPatch((x - 0.42, y - 0.42), 0.84, 0.84,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.6, edgecolor=NAVY, facecolor=face)
    ax.add_patch(rect)
    label = "터미널" if is_term else f"({r},{c})"
    ax.text(x, y, label, ha='center', va='center', fontsize=12,
            color=NAVY, fontweight='bold' if is_term else 'normal')

def arrow(p0, p1, color, rad, lw=2.4):
    (r0, c0), (r1, c1) = p0, p1
    x0, y0 = coords[(r0, c0)]
    x1, y1 = coords[(r1, c1)]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=22, shrinkB=22))

# 위쪽 루프: (0,0) -> (0,1) -> (0,2) -> (1,2) -> 터미널, 미끄러지면 (0,2) 재방문
arrow((0, 0), (0, 1), NAVY, 0.15)
arrow((0, 1), (0, 2), NAVY, 0.15)
arrow((0, 2), (1, 2), NAVY, 0.15)
arrow((1, 2), (1, 1), NAVY, 0.15)
arrow((1, 2), (0, 2), NAVY, -0.35)  # 미끄러짐: (1,2)->터미널 시도가 (0,2)로 되돌아감

# 아래쪽 루프: (2,0) -> (2,1) -> (2,2) -> (1,2)... 상태 (0,0)은 이 루프에 없음
arrow((2, 0), (2, 1), RED, -0.15)
arrow((2, 1), (2, 2), RED, -0.15)
arrow((2, 2), (1, 2), RED, -0.15)

ax.text(1.0, -0.62, "위쪽 루프: (0,0)→(0,1)→(0,2)→(1,2)→종료\n(미끄러지면 (0,2) 재방문 → 2회 이상)",
        ha='center', va='bottom', fontsize=10.5, color=NAVY)
ax.text(1.0, 2.62, "아래쪽 루프: (2,0)→(2,1)→(2,2)→(1,2)→종료\n(이 루프는 (0,0)을 전혀 지나지 않음)",
        ha='center', va='top', fontsize=10.5, color=RED)

ax.set_title("두 루프가 상태마다 다른 빈도로 지나간다", fontsize=13, color=NAVY, pad=28)

# ---- Right: actual visit counts from p14 (200k episodes, seed 42) ----
ax2 = axes[1]
states = ["(0,0)", "(0,1)", "(0,2)", "(1,2)", "(2,0)", "(2,2)"]
counts = [50546, 50546, 100155, 100155, 99845, 49967]
colors = [NAVY if c < 80000 else RED for c in counts]
y = np.arange(len(states))
ax2.barh(y, counts, color=colors, edgecolor='black', linewidth=0.6, height=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(states, fontsize=12)
ax2.invert_yaxis()
ax2.set_xlabel("실제 방문 횟수 (20만 에피소드, 시드 42)", fontsize=11)
for yi, c in zip(y, counts):
    ax2.text(c + 1500, yi, f"{c:,}", va='center', fontsize=10.5)
ax2.set_xlim(0, 120000)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title("(0,2)/(1,2)/(2,0) 이 (0,0)/(0,1)/(2,2) 의 거의 2배",
              fontsize=12.5, color=NAVY)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_1_loop_visits.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
