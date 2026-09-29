import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = "#2E3192"
KSA_TAN = "#D6CBB1"
RED = "#C0392B"

OUT = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week17/slides/kor/ml1/week17/figs"

# ---------------------------------------------------------------
# D1 (p06g1): BPE merge progress -- token count down, vocab size up
# ---------------------------------------------------------------
merges = [0, 1, 2, 3, 4, 5, 6]
tokens = [25, 23, 21, 19, 17, 15, 13]
vocab = [13, 14, 15, 16, 17, 18, 19]
merge_labels = ["처음", "서+울", "서울+의", "겨+울", "겨울+은", "겨울은+춥", "겨울은춥+다"]

fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=200)
ax2 = ax1.twinx()

l1, = ax1.plot(merges, tokens, "-o", color=KSA_BLUE, linewidth=3, markersize=9, label="corpus 토큰 수")
l2, = ax2.plot(merges, vocab, "-s", color=RED, linewidth=3, markersize=9, label="어휘 사전 크기")

for x, y in zip(merges, tokens):
    ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=13, color=KSA_BLUE, fontweight="bold")
for x, y in zip(merges, vocab):
    ax2.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, -20),
                 ha="center", fontsize=13, color=RED, fontweight="bold")

ax1.set_xticks(merges)
ax1.set_xticklabels(merge_labels, fontsize=11, rotation=18, ha="right")
ax1.set_ylabel("corpus 안의 토큰 수 (개)", fontsize=14, color=KSA_BLUE)
ax2.set_ylabel("어휘 사전 크기 (종)", fontsize=14, color=RED)
ax1.set_xlabel("병합 순서", fontsize=14)
ax1.set_ylim(10, 27)
ax2.set_ylim(11, 21)
ax1.tick_params(axis="y", labelcolor=KSA_BLUE, labelsize=12)
ax2.tick_params(axis="y", labelcolor=RED, labelsize=12)
ax1.tick_params(axis="x", labelsize=11)
ax1.set_title("BPE 6회 병합: 토큰 수는 줄고, 어휘 크기는 늘어난다", fontsize=15, fontweight="bold", pad=14)
ax1.grid(True, alpha=0.25)
ax1.legend(handles=[l1, l2], loc="center right", fontsize=12, frameon=True)
fig.tight_layout()
fig.savefig(f"{OUT}/bpe_merge_progress.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# D2 (p09g1): left = bigram distribution for "서울의", right = PPL unigram vs bigram
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

ax = axes[0]
words = ["겨울은", "봄은"]
probs = [0.5, 0.5]
bars = ax.bar(words, probs, color=[KSA_BLUE, KSA_TAN], edgecolor="black", linewidth=1.2, width=0.55)
for b, p in zip(bars, probs):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p:.1f}", ha="center", fontsize=16, fontweight="bold")
ax.set_ylim(0, 0.75)
ax.set_ylabel("확률", fontsize=14)
ax.set_title('bigram: "서울의" 다음 토큰', fontsize=14, fontweight="bold")
ax.tick_params(labelsize=13)
ax.grid(True, axis="y", alpha=0.25)

ax = axes[1]
models = ["unigram\n(n=1)", "bigram\n(n=2)"]
ppl = [5.67, 1.19]
bars = ax.bar(models, ppl, color=[KSA_TAN, KSA_BLUE], edgecolor="black", linewidth=1.2, width=0.5)
for b, p in zip(bars, ppl):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.15, f"{p:.2f}", ha="center", fontsize=16, fontweight="bold")
ax.axhline(1.0, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(1.45, 1.05, "PPL=1 (완전 확신)", color=RED, fontsize=10, ha="right")
ax.set_ylim(0, 6.5)
ax.set_ylabel("퍼플렉시티 (PPL)", fontsize=14)
ax.set_title("문맥을 볼수록 PPL이 낮아진다", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=13)
ax.grid(True, axis="y", alpha=0.25)

fig.suptitle("n-gram 손계산: bigram이 unigram보다 확신 있고 정확하다", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/ngram_bigram_ppl.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# D3 (p13g1): cross-entropy loss curve  -log(p) vs p
# ---------------------------------------------------------------
p = np.linspace(0.001, 1.0, 2000)
loss = -np.log(p)

fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=200)
ax.plot(p, loss, color=KSA_BLUE, linewidth=3)

points = [(0.9, -np.log(0.9)), (0.5, -np.log(0.5)), (0.001, -np.log(0.001))]
labels = ["$p=0.9$\n손실 $\\approx 0.105$\n(거의 벌 없음)",
          "$p=0.5$\n손실 $\\approx 0.693$\n(중간 벌)",
          "$p=0.001$\n손실 $\\approx 6.908$\n(큰 벌)"]
offsets = [(30, 40), (40, 40), (-10, -55)]

for (x, y), lab, (dx, dy) in zip(points, labels, offsets):
    ax.plot(x, y, "o", color=RED, markersize=10, zorder=5)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=12, ha="left" if dx >= 0 else "right",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

ax.set_xlabel("모델이 정답 토큰에 준 확률 $p_j$", fontsize=14)
ax.set_ylabel("토큰 손실 $-\\log p_j$", fontsize=14)
ax.set_title("교차 엔트로피 손실: 확신이 낮을수록 벌이 급격히 커진다", fontsize=14.5, fontweight="bold", pad=12)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 7.5)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/cross_entropy_curve.png", facecolor="white")
plt.close(fig)

print("done: bpe_merge_progress.png, ngram_bigram_ppl.png, cross_entropy_curve.png")
