"""p49g1: 시그모이드 vs ReLU 함수와 도함수 -- 사라지는 그래디언트 대비"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-6, 6, 400)
sig = 1 / (1 + np.exp(-z))
sig_d = sig * (1 - sig)
relu = np.maximum(0, z)
relu_d = (z > 0).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))

ax = axes[0]
ax.plot(z, sig, color=KSA_BLUE, lw=2.4, label="$\\sigma(z)$")
ax.plot(z, sig_d, color=ACCENT_RED, lw=2.4, label="$\\sigma'(z)$")
ax.axvspan(-6, -2.5, color=GREY, alpha=0.15)
ax.axvspan(2.5, 6, color=GREY, alpha=0.15)
ax.text(-4.3, 0.85, "도함수 ≈ 0\n(사라짐)", fontsize=10, color=GREY, ha="center")
ax.text(4.3, 0.85, "도함수 ≈ 0\n(사라짐)", fontsize=10, color=GREY, ha="center")
ax.set_title("시그모이드: 양 끝에서 그래디언트 소실", fontsize=12.5, color=KSA_BLUE)
ax.set_xlim(-6, 6); ax.set_ylim(-0.1, 1.15)
ax.legend(loc="center left", fontsize=11.5, frameon=False)
ax.axhline(0, color="black", lw=0.6)

ax = axes[1]
ax.plot(z, relu, color=KSA_BLUE, lw=2.4, label="ReLU$(z)$")
ax.plot(z, relu_d, color=ACCENT_RED, lw=2.4, label="ReLU$'(z)$")
ax.set_title("ReLU: $z>0$에서 도함수 항상 1", fontsize=12.5, color=KSA_BLUE)
ax.set_xlim(-6, 6); ax.set_ylim(-0.4, 6.3)
ax.legend(loc="upper left", fontsize=11.5, frameon=False)
ax.axhline(0, color="black", lw=0.6)
ax.annotate("$z\\leq 0$: 도함수 0\n(죽은 뉴런)", xy=(-2, 0), xytext=(-5.6, 3.2),
            fontsize=10, color=GREY,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))

for ax in axes:
    ax.set_xlabel("$z$", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "sigmoid_relu_gradient.png"))
