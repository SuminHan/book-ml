"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py.
Stable Diffusion 그림은 assets/sd/ (쇼츠에서 복사) 를 이어 붙인다."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"; SD = "assets/sd/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. Stable Diffusion: 앞으로 (노이즈 섞기)
fig, axs = plt.subplots(1, 6, figsize=(13, 2.6))
for ax, t in zip(axs, [0, 200, 400, 600, 800, 999]):
    ax.imshow(Image.open(f"{SD}fwd_{t:03d}.png")); ax.axis("off"); ax.set_title(f"t = {t}", fontsize=13)
save(fig, "df_forward.png")

# 2. Stable Diffusion: 뒤로 (조금씩 지우기)
fig, axs = plt.subplots(1, 6, figsize=(13, 2.6))
for ax, i in zip(axs, META["saved"]):
    ax.imshow(Image.open(f"{SD}latent_{i:02d}.png")); ax.axis("off"); ax.set_title(f"{i + 1}/50단계", fontsize=13)
save(fig, "df_reverse.png")

# 3. 숫자: 앞으로 섞기
fig, axs = plt.subplots(1, 6, figsize=(10, 2.0))
for ax, im, t in zip(axs, Z["fwd"], Z["fwd_t"]):
    ax.imshow(im.reshape(8, 8), cmap="gray_r", vmin=-1.5, vmax=1.5); ax.axis("off"); ax.set_title(f"t = {t}", fontsize=12)
save(fig, "df_digit_fwd.png")

# 4. 숫자: 생성 16장 × 방법 3개
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, key, t in zip(axs, ("ddpm1000", "ddim50", "ddim10"), ("1,000단계", "50단계 (DDIM)", "10단계 (DDIM)")):
    g = Z[f"img_{key}"]; big = np.ones((4 * 9 - 1, 4 * 9 - 1))
    for k in range(16):
        r, c = divmod(k, 4); big[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = (1 - g[k].reshape(8, 8)) / 2
    ax.imshow(big, cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(f"{t}: 확신 {J[key]['conf']:.0%}", fontsize=14)
save(fig, "df_digits.png")

# 5. 신호 비율 √ᾱ
fig, ax = plt.subplots(figsize=(6.6, 3.6))
beta = np.linspace(1e-4, 0.02, 1000); abar = np.cumprod(1 - beta); t = np.arange(1, 1001)
ax.plot(t, np.sqrt(abar), color=BLUE, lw=2.5, label="signal  $\\sqrt{\\bar\\alpha_t}$"); ax.plot(t, np.sqrt(1 - abar), color=RED, lw=2.5, label="noise  $\\sqrt{1-\\bar\\alpha_t}$")
ax.axvline(500, ls=":", color=GRAY); ax.text(510, 0.5, "t = 500\n0.28 : 0.96", fontsize=12)
ax.set_xlabel("step t"); ax.legend(frameon=False, fontsize=12)
save(fig, "df_schedule.png")
print("figs ok")
