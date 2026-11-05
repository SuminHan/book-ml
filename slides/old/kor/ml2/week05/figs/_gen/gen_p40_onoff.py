"""p40: on-policy vs off-policy 데이터 수집/평가 정책 구조 비교 다이어그램."""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
SAND = '#D6CBB1'
RED = '#C0392B'

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=200)


def box(ax, xy, w, h, text, color, fontsize=12, fontweight='bold', textcolor=None):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=color,
                                    facecolor=color, alpha=0.15)
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor or color, fontweight=fontweight)
    return cx, cy


def arrow(ax, p0, p1, color='#555555', lw=2.0, text=None, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.18, text, ha='center', fontsize=9.5, color=color)


# ---- Left: on-policy ----
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title("On-policy", fontsize=15, color=NAVY, fontweight='bold')

pc = box(ax, (1.6, 3.2), 2.8, 1.0, r"정책 $\pi$" + "\n(행동 + 평가 대상)", NAVY)
env = box(ax, (1.6, 0.6), 2.8, 1.0, "환경 (에피소드)", '#555555', textcolor='#333333')
arrow(ax, (2.2, 3.2), (2.2, 1.6), color=NAVY, text="행동")
arrow(ax, (3.6, 1.6), (3.6, 3.2), color=NAVY, rad=0.0, text="데이터 → 그대로 평균")
ax.text(3.0, 4.7, r"같은 정책 $\pi$ 로 행동하고 그 $\pi$ 를 평가", ha='center',
        fontsize=10.5, color=NAVY)

# ---- Right: off-policy ----
ax = axes[1]
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title("Off-policy", fontsize=15, color=RED, fontweight='bold')

bx = box(ax, (0.2, 3.2), 2.4, 1.0, r"행동 정책 $b$" + "\n(데이터 수집)", RED)
tx = box(ax, (3.4, 3.2), 2.4, 1.0, r"목표 정책 $\pi$" + "\n(가치 평가 대상)", NAVY)
env2 = box(ax, (1.8, 0.6), 2.4, 1.0, "환경 (에피소드)", '#555555', textcolor='#333333')

arrow(ax, (1.4, 3.2), (2.6, 1.6), color=RED, text="")
arrow(ax, (3.0, 1.6), (3.6, 3.2), color=RED, rad=0.25,
      text=r"데이터는 $b$ 로부터")
arrow(ax, (2.6, 3.7), (3.4, 3.7), color='#888888', lw=1.6)
ax.text(3.0, 3.95, r"$\rho$ 로 보정", ha='center', fontsize=9.5, color='#555555')
ax.text(3.0, 4.7, r"$b \neq \pi$: 그냥 평균내면 $V^b$ 를 추정 (편향)", ha='center',
        fontsize=10.5, color=RED)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_3_onoff_policy.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
