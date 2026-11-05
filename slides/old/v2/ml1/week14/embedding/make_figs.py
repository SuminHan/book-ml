"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py.
figs/schopenhauer_1859.jpg 는 쇼츠 lessons/ml1/embedding/src/assets 에서 복사 (퍼블릭 도메인)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"
COL = dict(cat=BLUE, dog=GOLD, tiger="#e67e22", leopard="#8e44ad", lion="#16a085", orange=RED, apple="#c0392b", cloud=GRAY)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 사진 예시
fig, axs = plt.subplots(1, 8, figsize=(12, 2.0))
for ax, n in zip(axs, NAMES):
    ax.imshow(Z["img_" + n][1]); ax.set_title(KO[n], fontsize=14); ax.axis("off")
save(fig, "emb_samples.png")

# 2. 고양이 · 개 CNN 의 64칸 → PCA 2D
p = PCA(2).fit(REF)
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for n in ("cat", "dog"):
    P = p.transform(Z[n][:400]); ax.scatter(*P.T, s=6, color=COL[n], alpha=0.35)
for n in ("tiger", "leopard", "lion", "orange"):
    P = p.transform(Z[n]); ax.scatter(*P.T, s=16, color=COL[n], marker="^" if n != "orange" else "s", alpha=0.8, label=f"{KO[n]} (고양이 쪽 {CAT_SIDE[n]}%)")
for n in ("cat", "dog"):
    m = p.transform(Z[n].mean(0)[None])[0]; ax.text(*m, KO[n], fontsize=18, weight="bold", color=COL[n], ha="center", bbox=dict(fc="white", ec=COL[n], alpha=0.85))
ax.legend(frameon=False, fontsize=12, loc="upper right"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
save(fig, "emb_cnn_map.png")

# 3. CLIP 유사도 표
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(SIM, cmap="YlOrBr", vmin=0.85, vmax=1.0)
ax.set_xticks(range(8)); ax.set_xticklabels([KO[n] for n in NAMES], rotation=45); ax.set_yticks(range(8)); ax.set_yticklabels([KO[n] for n in NAMES])
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{SIM[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white" if SIM[i, j] > 0.965 else "k")
for s in ax.spines.values(): s.set_visible(False)
save(fig, "emb_clip_sim.png")
print("figs ok")
