"""p67g1: CartPole 구조도 -- 카트+막대, 힘, 각도, 좌표"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
import numpy as np

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.set_xlim(-4.8, 4.8)
ax.set_ylim(-1.2, 4.2)
ax.set_aspect("equal")
ax.axis("off")

# 레일
ax.plot([-4.8, 4.8], [0, 0], color="black", lw=2.5)
for xt in [-4.8, -2.4, 0, 2.4, 4.8]:
    ax.plot([xt, xt], [-0.15, 0], color="black", lw=1.5)
    ax.text(xt, -0.55, f"{xt:+.1f}" if xt != 0 else "0", ha="center", fontsize=9.5, color=GREY)
ax.text(0, -1.0, "카트 위치 $x$ (m), 허용 범위 $|x|\\leq 2.4$", ha="center", fontsize=11, color=GREY)

# 카트
cart_x, cart_w, cart_h = -0.6, 1.6, 0.65
cart = Rectangle((cart_x, 0), cart_w, cart_h, fc=KSA_TAN, ec=KSA_BLUE, lw=2)
ax.add_patch(cart)
pivot = (cart_x + cart_w/2, cart_h)

# 막대 (기울어짐, theta from vertical)
theta = np.deg2rad(18)
pole_len = 2.6
pole_end = (pivot[0] + pole_len*np.sin(theta), pivot[1] + pole_len*np.cos(theta))
ax.plot([pivot[0], pole_end[0]], [pivot[1], pole_end[1]], color=ACCENT_RED, lw=5, solid_capstyle="round")
ax.scatter([pivot[0]], [pivot[1]], s=70, color=KSA_BLUE, zorder=5)

# 수직 기준선 + 각도 표시
ax.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] + pole_len], color=GREY, lw=1.2, ls="--")
arc = Arc(pivot, 1.6, 1.6, angle=0, theta1=90 - np.rad2deg(theta), theta2=90, color=KSA_BLUE, lw=1.6)
ax.add_patch(arc)
ax.text(pivot[0] + 0.55, pivot[1] + 1.05, r"$\theta$", fontsize=15, color=KSA_BLUE)
ax.annotate("막대(pole)", xy=(pole_end[0] - 0.2, pole_end[1] - 0.3), xytext=(2.1, 3.7),
            fontsize=11.5, color=ACCENT_RED,
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.3))

# 힘 화살표 (좌우 두 방향)
fa = FancyArrowPatch((cart_x - 0.15, cart_h/2), (cart_x - 1.4, cart_h/2),
                      arrowstyle="-|>", mutation_scale=20, color=KSA_BLUE, lw=2.2)
ax.add_patch(fa)
ax.text(cart_x - 1.6, cart_h/2, "행동 0\n(왼쪽으로 밀기)", ha="right", va="center", fontsize=10)
fa2 = FancyArrowPatch((cart_x + cart_w + 0.15, cart_h/2), (cart_x + cart_w + 1.4, cart_h/2),
                       arrowstyle="-|>", mutation_scale=20, color=KSA_BLUE, lw=2.2)
ax.add_patch(fa2)
ax.text(cart_x + cart_w + 1.6, cart_h/2, "행동 1\n(오른쪽으로 밀기)", ha="left", va="center", fontsize=10)

ax.text(cart_x + cart_w/2, cart_h/2, "카트", ha="center", va="center", fontsize=11, fontweight="bold")

ax.set_title("CartPole: 카트를 좌우로 밀어 막대의 균형을 잡는다\n목표: $|\\theta| < 0.2094$ rad(약 12도) 유지",
              fontsize=13, color=KSA_BLUE, pad=8)

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "cartpole_diagram.png"))
