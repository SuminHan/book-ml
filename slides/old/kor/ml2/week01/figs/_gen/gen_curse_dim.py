"""p32g1: 차원의 저주 -- 0.01 단위 그라데이션(차원당 10^4칸) 기준 표의 행 수"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
import numpy as np

dims = np.array([1, 2, 3, 4])
bins_per_dim = 1e4  # 0.01 단위 그라데이션 -> 차원당 10^4개 구간
rows = bins_per_dim ** dims

fig, ax = plt.subplots(figsize=(8.6, 5.4))
colors = [KSA_TAN, KSA_TAN, KSA_TAN, ACCENT_RED]
bars = ax.bar(dims, rows, color=colors, edgecolor=KSA_BLUE, linewidth=1.6, width=0.55)
ax.set_yscale("log")
ax.set_ylim(1, 1e18)
ax.set_xticks(dims)
ax.set_xticklabels([f"{d}차원" for d in dims], fontsize=12.5)
ax.set_ylabel("표에 필요한 행(칸) 수 (로그 스케일)", fontsize=12)
ax.set_xlabel("연속 상태의 차원 수", fontsize=12.5)
ax.set_title("차원의 저주: 0.01 단위로 쪼갠 표 기반 Q-테이블의 크기",
              fontsize=13.5, color=KSA_BLUE, pad=14)

for d, r in zip(dims, rows):
    label = f"$10^{{{int(np.log10(r))}}}$"
    ax.text(d, r * 2.2, label, ha="center", fontsize=13, fontweight="bold",
            color=(ACCENT_RED if d == 4 else "black"))

ax.annotate("CartPole: 4차원 연속 상태\n$(x,\\dot{x},\\theta,\\dot{\\theta})$ → 약 $10^{16}$개의 행",
            xy=(4, rows[-1]), xytext=(1.15, 3e14),
            fontsize=11.5, color=ACCENT_RED,
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.6))

ax.grid(axis="y", which="major", color=GREY, alpha=0.25, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "curse_of_dimensionality.png"))
