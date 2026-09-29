"""p26g1: 힌지 손실 L(f)=max(0,1-f) vs 교차 엔트로피 모양."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
RED = "#C0392B"
TAN = "#D6CBB1"

f = np.linspace(-2, 3, 400)
hinge = np.maximum(0, 1 - f)
# cross-entropy-like: -log(sigmoid(f)), rescaled/shifted so shapes are visually comparable near f=1
ce = np.log1p(np.exp(-f))
ce = ce / ce[np.argmin(np.abs(f - (-2)))] * hinge[0]  # scale so it starts at same height at f=-2

fig, ax = plt.subplots(figsize=(8.0, 5.8))
ax.axvspan(-2, 1, color=TAN, alpha=0.35, zorder=0)
ax.plot(f, hinge, color=NAVY, lw=3, label=r"힌지: $L(f)=\max(0,1-f)$", zorder=3)
ax.plot(f, ce, color=RED, lw=2.4, ls="--", label="교차 엔트로피 (형태 비교용)", zorder=2)
ax.axvline(1, color="black", lw=1.1, ls=":", zorder=1)
ax.scatter([1], [0], color=NAVY, s=70, zorder=4)
ax.annotate("꺾이는 점 $f=1$\n(\"접힌\" kink)", xy=(1, 0), xytext=(1.35, 1.3),
            fontsize=11, color=NAVY,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
ax.text(-1.85, 2.55, "$f<1$: 마진 위반\n(손실 $>0$)", fontsize=10.5, color="#7a4a10")
ax.text(1.5, 0.25, "$f\\geq1$: 손실 정확히 0", fontsize=10.5, color=NAVY)

ax.set_xlim(-2, 3); ax.set_ylim(-0.3, 3.2)
ax.set_xlabel(r"$f = y^{(i)}(w^\top x^{(i)}+b)$", fontsize=12.5)
ax.set_ylabel("손실 $L(f)$", fontsize=12.5)
ax.set_title("힌지 손실의 모양: $f\\geq1$이면 손실이 아예 0", fontsize=14, pad=10)
ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92)
ax.tick_params(labelsize=10.5)
ax.axhline(0, color="#999999", lw=0.8)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_2_hinge_loss.png", dpi=200, facecolor="white")
print("done")
