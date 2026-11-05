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


# 1. 빈칸 확률
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, top, t, c in [(axs[0], BREAD, "... to buy some bread.", BLUE), (axs[1], DOCTOR, "... to see a doctor.", RED)]:
    w = [x for x, _ in top[:4]][::-1]; v = [p for _, p in top[:4]][::-1]
    ax.barh(w, v, color=[GRAY] * 3 + [c]); ax.set_xlim(0, 100); ax.set_title(f"I went to the ___ {t}", fontsize=14)
    for y, p in enumerate(v): ax.text(p + 1.5, y, f"{p}%", va="center", fontsize=13)
    ax.set_xlabel("probability (%)")
save(fig, "bert_fill.png")

# 2. GPT vs BERT 가 보는 쪽
toks = ["the", "cat", "sat", "on", "the", "[MASK]", "and", "slept"]
fig, axs = plt.subplots(2, 1, figsize=(10.5, 4.6))
for ax, bi in zip(axs, (False, True)):
    ax.set_xlim(-0.7, 7.7); ax.set_ylim(-0.45, 2.0); ax.axis("off")
    for k, t in enumerate(toks):
        seen = (k < 5) or (bi and k > 5)
        c = RED if k == 5 else (BLUE if seen else "#cccccc")
        ax.add_patch(FancyBboxPatch((k - 0.42, -0.3), 0.84, 0.6, boxstyle="round,pad=0.03", fc="white", ec=c, lw=2))
        ax.text(k, 0, t, ha="center", va="center", fontsize=13, color=c if k == 5 or seen else "#bbbbbb")
        if seen: ax.annotate("", (5, 0.33), (k, 0.33), arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4, connectionstyle=f"arc3,rad={-0.22 if k < 5 else 0.3}"))
    ax.text(-0.65, 1.75, "BERT: 앞뒤를 모두 본다" if bi else "GPT: 앞만 보고 다음 단어", fontsize=14, color=BLUE if bi else GRAY)
save(fig, "bert_dir.png")

# 3. bank 유사도
lab = ["money at the bank", "bank approved my loan", "bank of the river", "river bank ... grass"]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.imshow(BANK_SIM, cmap="YlOrBr", vmin=0.3, vmax=1)
for i in range(4):
    for j in range(4): ax.text(j, i, f"{BANK_SIM[i, j]:.2f}", ha="center", va="center", fontsize=14, color="white" if BANK_SIM[i, j] > 0.8 else "k")
ax.set_xticks(range(4)); ax.set_xticklabels([f"#{k + 1}" for k in range(4)]); ax.set_yticks(range(4)); ax.set_yticklabels([f"#{k + 1} {l}" for k, l in enumerate(lab)], fontsize=12)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "bert_bank.png")
print("figs ok")
