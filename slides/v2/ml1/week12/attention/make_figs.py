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


def box(ax, x, y, text, color, w=1.0, h=0.7, fc="white", fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)


# 1. Seq2Seq vs 어텐션: 디코더가 무엇을 보나
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
src = ["I", "love", "you"]; alpha = [0.1, 0.8, 0.1]
for ax, attn in zip(axs, (False, True)):
    ax.set_xlim(-0.8, 6.2); ax.set_ylim(-0.9, 3.1); ax.axis("off")
    for j, w in enumerate(src):
        box(ax, j * 1.4, 0.4, f"$h_{j + 1}$", BLUE, fc="#e8e9f6"); ax.text(j * 1.4, -0.55, w, ha="center", fontsize=15)
    box(ax, 4.9, 2.3, "$s$ (aime)", RED, w=1.6, fc="#fbeaea")
    if attn:
        for j, a in enumerate(alpha):
            ax.annotate("", (4.15, 2.15), (j * 1.4, 1.2), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1 + 9 * a, alpha=0.9))
            ax.text(j * 1.4, 0.95, f"{a:.1f}", ha="center", fontsize=14, color="#7a6a40", weight="bold")
        ax.set_title("어텐션: 모든 칸을 비율대로", color=GOLD, fontsize=16)
    else:
        ax.annotate("", (4.15, 2.15), (2 * 1.4 + 0.5, 0.6), arrowprops=dict(arrowstyle="->", color=GRAY, lw=3))
        ax.set_title("Seq2Seq: 마지막 칸 하나만", color=GRAY, fontsize=16)
save(fig, "at_idea.png")

# 2. '그것은' 이 보는 비율
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cols = [BLUE, GRAY, GRAY, GOLD]
b = ax.bar(WORDS, W[IT] * 100, color=cols)
for x, v in zip(range(4), W[IT] * 100):
    ax.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=15)
ax.set_ylabel("weight (%)"); ax.set_ylim(0, 70); ax.set_title("'그것은' 이 보는 비율", fontsize=16)
save(fig, "at_it.png")

# 3. 길이별 베끼기
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.axvspan(32.5, 52, color="#f3f0e8"); ax.text(41.5, 50, "학습 때\n없던 길이", ha="center", color="#7a6a40", fontsize=13)
ax.plot(LENS, [ACC["plain_fwd"][n] for n in LENS], "o-", color=GRAY, lw=2.5, label="Seq2Seq")
ax.plot(LENS, [ACC["attn_fwd"][n] for n in LENS], "o-", color=GOLD, lw=2.5, label="Seq2Seq + 어텐션")
for n in (5, 30, 50):
    ax.annotate(f"{ACC['plain_fwd'][n]}%", (n, ACC["plain_fwd"][n]), (6, 6 if n == 50 else -18), textcoords="offset points", color=GRAY, fontsize=13)
    ax.annotate(f"{ACC['attn_fwd'][n]}%", (n, ACC["attn_fwd"][n]), (-10, 9), textcoords="offset points", color="#7a6a40", fontsize=13)
ax.axhline(10, ls=":", color=RED); ax.text(5, 12, "찍기 10%", color=RED, fontsize=12)
ax.set_xlabel("sequence length (digits)"); ax.set_ylabel("digits copied correctly (%)"); ax.set_ylim(0, 108); ax.set_xticks(LENS)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.62, 0.75))
save(fig, "at_copy.png")

# 4. 어텐션 지도 (베끼기, 입력 그대로)
fig, ax = plt.subplots(figsize=(5.6, 5.0))
A = np.array(MAP["attn"]); ax.imshow(A, cmap="YlOrBr", vmin=0, vmax=1)
ax.set_xticks(range(len(MAP["src"]))); ax.set_xticklabels(MAP["src"]); ax.set_yticks(range(len(MAP["out"]))); ax.set_yticklabels(MAP["out"])
ax.set_xlabel("input digit (key)"); ax.set_ylabel("output digit (query)")
for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "at_map.png")
print("figs ok")
