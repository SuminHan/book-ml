"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"; CM = plt.get_cmap("tab10")


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def latent(ax, M, title, circle=True):
    ax.scatter(*M.T, c=CM(Y), s=5)
    for k in range(10):
        m = np.median(M[Y == k], 0); ax.text(*m, str(k), fontsize=14, weight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=CM(k)))
    if circle: ax.add_patch(Circle((0, 0), 2, fc="none", ec=RED, lw=2, ls="--"))
    ax.set_title(title, fontsize=15); ax.set_aspect("equal")


# 1. 잠재 지도: AE vs VAE (같은 눈금)
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
latent(axs[0], Z["ae_mu"], f"AE: 평균 ({AE['lat_mean'][0]:.1f}, {AE['lat_mean'][1]:.1f}), 퍼짐 {np.mean(AE['lat_std']):.1f}")
latent(axs[1], Z["vae_mu"], f"VAE: 평균 (0, 0), 퍼짐 {np.mean(VAE['lat_std']):.2f}")
lim = np.abs(Z["ae_mu"]).max() * 1.05
for ax in axs: ax.set_xlim(-lim * 0.6, lim); ax.set_ylim(-lim * 0.6, lim)
axs[1].set_xlim(-3.5, 3.5); axs[1].set_ylim(-3.5, 3.5)
axs[0].text(0, -4.0, "N(0,1)", color=RED, fontsize=12, ha="center")
save(fig, "vae_maps.png")

# 2. N(0,1) 에서 뽑아 되살리기
def tiles(ax, imgs, title):
    big = np.ones((4 * 9 - 1, 4 * 9 - 1))
    for k, im in enumerate(imgs[:16]):
        r, c = divmod(k, 4); big[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = 1 - im.reshape(8, 8)
    ax.imshow(big, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(title, fontsize=14)
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 1.2, 1, 1.2]))
for (a_img, a_bar), key, name in [((axs[0], axs[1]), "ae", "AE"), ((axs[2], axs[3]), "vae", "VAE")]:
    tiles(a_img, Z[f"{key}_n01"], f"{name}: N(0,1) 에서 16장")
    h = np.array(S[key]["n01_hist"]) * 100; a_bar.bar(range(10), h, color=[CM(k) for k in range(10)])
    a_bar.set_xticks(range(10)); a_bar.set_ylim(0, 36); a_bar.set_xlabel("predicted digit"); a_bar.set_ylabel("%"); a_bar.set_title(f"2,000장: {S[key]['n01_kinds']}종류", fontsize=14)
save(fig, "vae_samples.png")

# 3. VAE 지도 격자를 되살린 그림
G = Z["vae_grid"]; n = int(np.sqrt(len(G)))
big = np.ones((n * 9 - 1, n * 9 - 1))
for k, im in enumerate(G):
    r, c = divmod(k, n); big[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = 1 - im.reshape(8, 8)
fig, ax = plt.subplots(figsize=(5.2, 5.2)); ax.imshow(big, cmap="gray"); ax.axis("off")
save(fig, "vae_grid.png")

# 4. 0 → 1 걷기 (쇼츠 VAE)
fig, axs = plt.subplots(1, 5, figsize=(8, 1.9))
for ax, im in zip(axs, SH["walkimg"]): ax.imshow(im.reshape(8, 8), cmap="gray_r"); ax.axis("off")
save(fig, "vae_walk.png")

# 5. 점 vs 구름
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, cloud in zip(axs, (False, True)):
    ax.set_xlim(-3, 3); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for (x, y_), c in [((-1.2, 0.6), CM(0)), ((1.1, -0.3), CM(1)), ((0.2, 1.4), CM(7))]:
        if cloud: ax.add_patch(Ellipse((x, y_), 1.6, 1.1, fc=c, alpha=0.25, ec=c))
        ax.plot(x, y_, "o", color=c, ms=9)
    if cloud: ax.add_patch(Circle((0, 0), 2, fc="none", ec=RED, ls="--", lw=1.8)); ax.text(0, -2.25, "N(0,1)", color=RED, ha="center", fontsize=12)
    ax.set_title("AE: 그림 → 점 하나" if not cloud else "VAE: 그림 → 구름 (μ, σ)", fontsize=15)
save(fig, "vae_cloud.png")
print("figs ok")
