"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py.
고양이 사진 figs/cat_*.png 는 쇼츠에서 복사한 것 (AI 생성, Stable Diffusion v1.5)."""
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
O = "figs/"
KO = {"flip": "좌우 뒤집기", "rotl": "왼쪽으로 돌리기", "rot": "오른쪽으로 돌리기", "crop": "일부 자르기", "face": "얼굴만", "bright": "밝게", "dark": "어둡게", "gray": "흑백", "color": "색 바꾸기"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 고양이 증강 격자
fig, axs = plt.subplots(2, 5, figsize=(13, 5.6))
axs = axs.ravel()
axs[0].imshow(Image.open("figs/cat_orig.png")); axs[0].set_title("원본", fontsize=14, color=RED)
for ax, k in zip(axs[1:], CATS):
    ax.imshow(Image.open(f"figs/cat_{k}.png")); ax.set_title(KO[k], fontsize=13)
for ax in axs: ax.axis("off")
save(fig, "da_cats.png")

# 2. 숫자 증강 예
import torch
from torchvision import datasets, transforms as T
tr = datasets.MNIST(os.path.join("..", "..", "week02", "sgd", "lab", "data"), train=True, download=True)
x = tr.data[0].float().div(255).unsqueeze(0)
aug = T.RandomAffine(degrees=10, translate=(2 / 28, 2 / 28)); torch.manual_seed(0)
fig, axs = plt.subplots(1, 8, figsize=(13, 2.2))
axs[0].imshow(x[0], cmap="gray_r"); axs[0].set_title("원본", fontsize=13, color=RED)
for ax in axs[1:]: ax.imshow(aug(x)[0], cmap="gray_r")
for ax in axs: ax.axis("off")
save(fig, "da_digits.png")

# 3. 결과
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(5); w = 0.35
ax.bar(xs - w / 2, BASE_ALL, w, color=GRAY, label=f"증강 없음 (평균 {BASE}%)"); ax.bar(xs + w / 2, AUG_ALL, w, color=GREEN, label=f"증강 (평균 {AUG}%)")
ax.set_xticks(xs); ax.set_xticklabels([f"학습 세트 {i + 1}" for i in xs]); ax.set_ylim(70, 95); ax.set_ylabel("시험 정답률 (%)"); ax.legend(frameon=False, loc="upper left")
save(fig, "da_result.png")

# 4. 6 과 9
six = tr.data[(tr.targets == 6).nonzero()[0, 0]].numpy()
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.8))
axs[0].imshow(six, cmap="gray_r"); axs[0].set_title("6", fontsize=16); axs[1].imshow(np.rot90(six, 2), cmap="gray_r"); axs[1].set_title("180도 돌리면 → 9 ?", fontsize=16, color=RED)
for ax in axs: ax.axis("off")
save(fig, "da_six.png")
print("ok")
