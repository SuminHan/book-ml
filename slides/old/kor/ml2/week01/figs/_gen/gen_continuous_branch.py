"""p33g1: 표 기반 -> 상태/행동 연속일 때의 두 확장 경로 흐름도"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(-0.8, 10); ax.axis("off")

def box(xy, w, h, text, fc=KSA_TAN, ec=KSA_BLUE, fontsize=12):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        fc=fc, ec=ec, lw=1.8)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(p0, p1, color=KSA_BLUE, lw=2.0, connectionstyle=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                         color=color, lw=lw, connectionstyle=connectionstyle)
    ax.add_patch(a)

# 중앙 상단: 표 기반 출발점
box((2.5, 8.0), 5, 1.4, "표 기반 Q-테이블\n(상태 × 행동)", fc=KSA_TAN, fontsize=13)

# 왼쪽 갈래: 상태 연속
arrow((3.6, 8.0), (1.9, 6.6), color=KSA_BLUE)
box((0.3, 5.2), 3.6, 1.4, "상태가 연속이면", fc="white", fontsize=11.5)
arrow((2.1, 5.2), (2.1, 3.9), color=KSA_BLUE)
box((0.3, 2.5), 3.6, 1.4, "함수근사(신경망)로\n$Q(s,a)$를 표현", fc=KSA_TAN, fontsize=11)
arrow((2.1, 2.5), (2.1, 1.2), color=KSA_BLUE)
box((0.3, -0.15+0.3), 3.6, 1.2, "9주차: DQN", fc=ACCENT_RED, ec=ACCENT_RED, fontsize=12.5)

# 오른쪽 갈래: 행동 연속
arrow((6.4, 8.0), (8.1, 6.6), color=KSA_BLUE)
box((6.1, 5.2), 3.6, 1.4, "행동이 연속이면", fc="white", fontsize=11.5)
arrow((7.9, 5.2), (7.9, 3.9), color=KSA_BLUE)
box((6.1, 2.5), 3.6, 1.4, "정책 $\\pi_\\theta(a|s)$를\n직접 학습", fc=KSA_TAN, fontsize=11)
arrow((7.9, 2.5), (7.9, 1.2), color=KSA_BLUE)
box((6.1, 0.15), 3.6, 1.2, "10~11주차: REINFORCE/PPO", fc=ACCENT_RED, ec=ACCENT_RED, fontsize=11)

ax.text(5, -0.35, "두 확장 모두 필요: 연속 상태 + 연속 행동 + 물리 법칙 (로봇)",
        ha="center", fontsize=11.5, color=GREY)

# fix overlapping white-box texts
for t in ax.texts:
    pass

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "continuous_extension_branch.png"))
