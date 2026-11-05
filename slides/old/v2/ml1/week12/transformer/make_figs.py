"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def box(ax, x, y, text, color, w=1.0, h=0.6, fc="white", fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)


# 1. RNN: 차례로 vs 트랜스포머: 한 번에
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
xs = np.arange(5) * 1.3
for ax, tf in zip(axs, (False, True)):
    ax.set_xlim(-0.8, 6.0); ax.set_ylim(-0.7, 2.2); ax.axis("off")
    for x, t in zip(xs, TOKENS):
        box(ax, x, 0, t, BLUE if not tf else GOLD, fc="#e8e9f6" if not tf else "#f3f0e8")
    if not tf:
        for k in range(4):
            ax.annotate("", (xs[k + 1] - 0.55, 0), (xs[k] + 0.55, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
        for k, x in enumerate(xs): ax.text(x, 0.55, f"{k + 1}", ha="center", color=BLUE, fontsize=14)
        ax.set_title("RNN: 하나씩 차례로 (5걸음)", color=BLUE, fontsize=16)
    else:
        for i in range(5):
            for j in range(i + 1, 5):
                ax.annotate("", (xs[j], 0.32), (xs[i], 0.32), arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.6, connectionstyle=f"arc3,rad=-{0.5 if j - i == 1 else 0.45}"))
        ax.set_title("트랜스포머: 모든 쌍을 한 번에 (1걸음)", color="#7a6a40", fontsize=16)
save(fig, "tf_seq.png")

# 2. 위치 정보 (사인 · 코사인)
pos = np.arange(64)[:, None]; i = np.arange(64)[None, :]
ang = pos / 10000 ** (2 * (i // 2) / 128); PE = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
im = axs[0].imshow(PE.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
axs[0].set_xlabel("position"); axs[0].set_ylabel("dimension (first 64)"); axs[0].set_title("PE(pos, i)", fontsize=15)
for k, c in zip((0, 20, 40), (BLUE, GOLD, RED)):
    axs[1].plot(PE[:, k], color=c, lw=2, label=f"dim {k}")
axs[1].set_xlabel("position"); axs[1].legend(frameon=False, fontsize=12, loc="upper center", ncol=3); axs[1].set_ylim(-1.2, 1.8)
save(fig, "tf_pe.png")

# 3. 블록 그림
fig, ax = plt.subplots(figsize=(5.2, 6.4)); ax.set_xlim(-2.8, 2.6); ax.set_ylim(-0.6, 7.3); ax.axis("off")
ys = dict(x=0, ln1=1.1, att=2.1, add1=3.1, ln2=4.1, ff=5.1, add2=6.1)
ax.text(0, -0.3, "단어 벡터 + 위치", ha="center", fontsize=14)
box(ax, 0, ys["ln1"], "LayerNorm", GRAY, w=2.4, h=0.5, fs=13)
box(ax, 0, ys["att"], "멀티헤드 어텐션", GOLD, w=2.8, h=0.65, fc="#f3f0e8", fs=15)
box(ax, 0, ys["add1"], "+", BLUE, w=0.5, h=0.5, fs=17)
box(ax, 0, ys["ln2"], "LayerNorm", GRAY, w=2.4, h=0.5, fs=13)
box(ax, 0, ys["ff"], "피드포워드", RED, w=2.8, h=0.65, fc="#fbeaea", fs=15)
box(ax, 0, ys["add2"], "+", BLUE, w=0.5, h=0.5, fs=17)
for a, b in [(0, ys["ln1"] - 0.25), (ys["ln1"] + 0.25, ys["att"] - 0.33), (ys["att"] + 0.33, ys["add1"] - 0.25), (ys["add1"] + 0.25, ys["ln2"] - 0.25),
             (ys["ln2"] + 0.25, ys["ff"] - 0.33), (ys["ff"] + 0.33, ys["add2"] - 0.25), (ys["add2"] + 0.25, 6.9)]:
    ax.annotate("", (0, b), (0, a), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
for y0, y1 in [(0.5, ys["add1"]), (ys["add1"] + 0.45, ys["add2"])]:
    ax.plot([0, 1.8, 1.8], [y0, y0, y1], color=BLUE, lw=1.6); ax.annotate("", (0.27, y1), (1.8, y1), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
ax.text(1.95, 1.8, "지름길", color=BLUE, fontsize=13, rotation=90, va="center")
ax.text(1.95, 4.8, "지름길", color=BLUE, fontsize=13, rotation=90, va="center")
ax.text(-2.2, ys["att"], "단어끼리\n주고받기", ha="center", va="center", fontsize=12, color="#7a6a40")
ax.text(-2.2, ys["ff"], "단어마다\n다듬기", ha="center", va="center", fontsize=12, color=RED)
ax.add_patch(FancyBboxPatch((-1.6, 0.7), 3.8, 5.75, boxstyle="round,pad=0.05", fc="none", ec=GRAY, lw=1.2, ls="--"))
ax.text(2.0, 6.85, "× N", fontsize=17, color=GRAY, ha="center")
save(fig, "tf_block.png")

# 4. 앞만 보기 (causal mask)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
M = np.tril(np.ones((5, 5)))
ax.imshow(M, cmap="Blues", vmin=0, vmax=1.6)
for r in range(5):
    for c in range(5):
        ax.text(c, r, "O" if M[r, c] else "X", ha="center", va="center", fontsize=15, color=BLUE if M[r, c] else RED)
ax.set_xticks(range(5)); ax.set_xticklabels(TOKENS); ax.set_yticks(range(5)); ax.set_yticklabels(TOKENS)
ax.set_xlabel("key (보이는 단어)"); ax.set_ylabel("query (지금 단어)")
for s in ax.spines.values(): s.set_visible(False)
save(fig, "tf_mask.png")

# 5. 학습 곡선
fig, ax = plt.subplots(figsize=(7.5, 4.3))
sm = lambda c: np.convolve(c, np.ones(10) / 10, mode="valid")
for k, c, lab in [("rnn", GRAY, "RNN"), ("lstm", BLUE, "LSTM"), ("tf_nopos", "#d9a3a3", "transformer, no position"), ("tf", GOLD, "transformer")]:
    y = sm(J[k]["curve"]); ax.plot(np.arange(len(y)) * 10, y, color=c, lw=2.2, label=f"{lab} (val {VAL[k]})")
ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_ylim(1.2, 3.0); ax.legend(frameon=False, fontsize=12)
save(fig, "tf_curve.png")
print("figs ok")
