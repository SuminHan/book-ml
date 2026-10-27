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


def box(ax, x, y, text, color, w=1.0, h=0.8, fc="white", fs=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05", fc=fc, ec=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color=GRAY):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=1.8))


# 1. 인코더-디코더
def fig_encdec(src, tgt, name):
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(-0.8, 11.4); ax.set_ylim(-0.9, 2.9); ax.axis("off")
    xs = np.arange(len(src)) * 1.5
    for k, (x, w) in enumerate(zip(xs, src)):
        box(ax, x, 1, "enc", BLUE, fc="#e8e9f6")
        ax.text(x, -0.35, w, ha="center", fontsize=17); arrow(ax, x, -0.1, x, 0.58)
        if k: arrow(ax, x - 1.5 + 0.55, 1, x - 0.55, 1, BLUE)
    vx = xs[-1] + 1.6
    ax.add_patch(FancyBboxPatch((vx - 0.35, 0.45), 0.7, 1.1, boxstyle="round,pad=0.05", fc=GOLD, ec=GOLD))
    ax.text(vx, 2.0, "vector", ha="center", fontsize=14, color="#7a6a40"); arrow(ax, xs[-1] + 0.55, 1, vx - 0.4, 1, BLUE)
    dx = vx + 1.6 + np.arange(len(tgt)) * 1.5
    prev = ["<s>"] + tgt[:-1]
    for k, (x, w, p) in enumerate(zip(dx, tgt, prev)):
        box(ax, x, 1, "dec", RED, fc="#fbeaea")
        ax.text(x, -0.35, p, ha="center", fontsize=14, color=GRAY); arrow(ax, x, -0.1, x, 0.58)
        ax.text(x, 2.25, w, ha="center", fontsize=17, color=RED, weight="bold"); arrow(ax, x, 1.42, x, 2.15, RED)
        arrow(ax, (vx + 0.4) if k == 0 else (x - 1.5 + 0.55), 1, x - 0.55, 1, RED)
    ax.text(xs.mean(), 2.55, "인코더: 읽기", ha="center", color=BLUE, fontsize=15)
    ax.text(dx.mean(), -0.85, "디코더: 한 단어씩 쓰기 (앞에서 쓴 단어를 다시 넣는다)", ha="center", color=RED, fontsize=13)
    save(fig, name)

fig_encdec(["I", "love", "you"], ["je", "t'", "aime", "</s>"], "s2s_encdec.png")

# 2. 길이별 베끼기 정답률
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(LENS, [COPY[n] for n in LENS], "o-", color=GRAY, lw=2.5, label="입력 그대로")
ax.plot(LENS, [COPY_REV[n] for n in LENS], "o-", color=BLUE, lw=2.5, label="입력 거꾸로")
for n in (5, 10, 30):
    ax.annotate(f"{COPY[n]}%", (n, COPY[n]), (8, -22), textcoords="offset points", color=GRAY, fontsize=13)
    ax.annotate(f"{COPY_REV[n]}%", (n, COPY_REV[n]), (6, 6), textcoords="offset points", color=BLUE, fontsize=13)
ax.axhline(10, ls=":", color=RED); ax.text(30, 12, "찍기 10%", color=RED, ha="right", fontsize=12)
ax.set_xlabel("sequence length (digits)"); ax.set_ylabel("digits copied correctly (%)"); ax.set_ylim(0, 105); ax.set_xticks(LENS)
ax.legend(frameon=False)
save(fig, "s2s_copy.png")

# 3. 거꾸로 넣기: 첫 단어끼리의 거리
fig, axs = plt.subplots(2, 1, figsize=(8.5, 3.6))
for ax, src, title in [(axs[0], ["a", "b", "c"], "그대로"), (axs[1], ["c", "b", "a"], "거꾸로")]:
    ax.set_xlim(-1.4, 9.4); ax.set_ylim(-0.5, 1.1); ax.axis("off")
    for k, w in enumerate(src): box(ax, k, 0, w, BLUE, w=0.7, h=0.7, fs=17)
    box(ax, 3.2, 0, "→", GOLD, w=0.5, h=0.7, fs=17)
    for k, w in enumerate(["α", "β", "γ"]): box(ax, 4.2 + k * 0.9, 0, w, RED, w=0.7, h=0.7, fs=17)
    ia = src.index("a")
    ax.annotate("", (4.2, 0.4), (ia, 0.4), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2, connectionstyle="arc3,rad=-0.3"))
    ax.text(-0.45, 0, title, ha="right", va="center", fontsize=15)
    ax.text(6.7, 0, f"a 다음 α 까지 {len(src) - ia}걸음", ha="left", va="center", fontsize=13, color=GREEN)
save(fig, "s2s_reverse.png")
print("figs ok")
