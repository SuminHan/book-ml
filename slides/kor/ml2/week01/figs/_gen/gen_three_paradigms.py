"""p03g1: 지도/비지도/강화학습의 데이터 흐름 비교 다이어그램"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

def box(ax, xy, w, h, text, fc=KSA_TAN, ec=KSA_BLUE, fontsize=12.5):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        fc=fc, ec=ec, lw=1.8)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color="black", wrap=True)

def arrow(ax, p0, p1, color=KSA_BLUE, style="-|>", lw=2.0, connectionstyle=None, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                         color=color, lw=lw, connectionstyle=connectionstyle, linestyle=ls)
    ax.add_patch(a)

# ---- Panel 1: 지도학습 ----
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("지도학습", fontsize=15, fontweight="bold", color=KSA_BLUE, pad=12)
box(ax, (0.5, 7.2), 9, 1.7, "데이터 $(x, y)$\n(정답 라벨 있음)")
arrow(ax, (5, 7.2), (5, 5.6))
box(ax, (0.5, 3.9), 9, 1.7, "모델 $\\hat{y}=f_\\theta(x)$")
arrow(ax, (5, 3.9), (5, 2.3))
box(ax, (0.5, 0.6), 9, 1.7, "예측 $\\hat{y}$ vs 정답 $y$\n즉시 비교, 매 샘플 업데이트", fc="white")
arrow(ax, (0.4, 1.45), (9.6, 4.6), color=ACCENT_RED, ls="--",
      connectionstyle="arc3,rad=0.35")
ax.text(9.8, 3.0, "손실\n피드백", fontsize=10.5, color=ACCENT_RED, ha="left")

# ---- Panel 2: 비지도학습 ----
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("비지도학습", fontsize=15, fontweight="bold", color=KSA_BLUE, pad=12)
box(ax, (0.5, 7.2), 9, 1.7, "데이터 $x$만\n(정답 라벨 없음)")
arrow(ax, (5, 7.2), (5, 5.6))
box(ax, (0.5, 3.9), 9, 1.7, "모델: 구조 탐색")
arrow(ax, (5, 3.9), (5, 2.3))
box(ax, (0.5, 0.6), 9, 1.7, "군집 / 잠재 구조 출력\n(맞았는지 비교할 대상 없음)", fc="white")

# ---- Panel 3: 강화학습 ----
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("강화학습", fontsize=15, fontweight="bold", color=KSA_BLUE, pad=12)
box(ax, (0.3, 5.7), 3.3, 2.0, "에이전트\n$\\pi_\\theta(a|s)$", fc=KSA_TAN)
box(ax, (6.4, 5.7), 3.3, 2.0, "환경", fc="white")
arrow(ax, (3.6, 7.15), (6.4, 7.15), color=KSA_BLUE)
ax.text(5.0, 7.5, "행동 $a$", fontsize=10.5, ha="center")
arrow(ax, (6.4, 6.25), (3.6, 6.25), color=ACCENT_RED)
ax.text(5.0, 5.9, "관측+보상 $R$", fontsize=10, ha="center", color=ACCENT_RED)
box(ax, (0.3, 0.6), 9.4, 3.8,
    "데이터는 에이전트가 스스로 생성\n"
    "보상은 나중에(에피소드 끝) 옴\n"
    "정답과 비교할 대상이 없음",
    fc="white")
arrow(ax, (1.9, 5.7), (1.9, 4.4), color=KSA_BLUE, ls=":", connectionstyle="arc3,rad=-0.2")

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "three_paradigms_dataflow.png"))
