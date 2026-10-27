"""이 덱에만 있는 그림 원본 (LayerNorm · 정규화 가족). 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py
나머지 그림은 원본 덱 slides/v2/ml1/week09/batchnorm/figs/.
표 숫자는 쇼츠 lessons/ml1/layernorm (src/norm_exp.py, PyTorch BatchNorm1d · LayerNorm, γ=1 β=0) 과 같은 예시 값."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 14, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#999999"
X = np.array([[1.0, 1.4, 0.6, -0.3, 0.3, -0.5],     # 샘플 A
              [1.1, 3.0, 0.3, 0.1, 1.7, 1.5],      # B
              [1.2, -0.4, 1.0, 2.0, -1.0, 0.3],    # C
              [-1.9, -0.9, -1.8, 0.6, -0.9, 1.4]])  # D
Y = np.array([[-0.7, -1.4, -6.0, -2.1, -1.1, -0.8],  # A 옆에 바꿔 넣을 샘플 3개
              [-4.1, -2.0, -3.0, -2.6, 1.1, -2.6],
              [-1.1, 0.8, -2.2, -1.2, -0.8, -0.9]])


def bn(T):
    return (T - T.mean(0)) / T.std(0)                 # 열마다 (배치 안 샘플끼리)


def ln(T):
    return (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)   # 행마다 (샘플 하나 안 특징끼리)


def table(ax, T, title, box, color):
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(3.5, -0.9); ax.axis("off")
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f"{T[i, j]:.1f}", ha="center", va="center", fontsize=14)
    for j in range(6): ax.text(j, -0.75, f"특징{j + 1}", ha="center", fontsize=11, color=GRAY)
    for i, s in enumerate("ABCD"): ax.text(-0.75, i, s, ha="center", va="center", fontsize=13, color=GRAY)
    ax.add_patch(Rectangle(box[0], box[1], box[2], fill=False, ec=color, lw=3))
    ax.set_title(title, color=color, fontsize=15, pad=16)


fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
table(axs[0], X, "BatchNorm: 열마다 (샘플 4개끼리)", ((-0.45, -0.45), 0.9, 3.9), BLUE)
table(axs[1], X, "LayerNorm: 행마다 (특징 6개끼리)", ((-0.45, -0.45), 5.9, 0.9), RED)
fig.tight_layout(); fig.savefig("figs/norm_dirs.png"); plt.close(fig)

# 정규화 가족: 그림 데이터 (배치 N × 채널 C, 칸 하나 = 그 채널의 H × W 픽셀들). 같은 색 = 함께 평균 · 분산
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
names = [("BatchNorm", "채널 하나, 배치 전체"), ("LayerNorm", "샘플 하나, 채널 전체"), ("InstanceNorm", "샘플 하나, 채널 하나"), ("GroupNorm", "샘플 하나, 채널 몇 개")]
for ax, (n, sub), sel in zip(axs, names, [lambda i, j: j == 1, lambda i, j: i == 1, lambda i, j: i == 1 and j == 1, lambda i, j: i == 1 and j in (0, 1)]):
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(4.8, -1.2); ax.axis("off"); ax.set_aspect("equal")
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fc=GOLD if sel(i, j) else "#eeeeee", ec="white"))
    ax.text(1.5, -0.95, n, ha="center", fontsize=15, fontweight="bold", color=BLUE)
    ax.text(1.5, 4.25, sub, ha="center", fontsize=12)
    ax.text(-0.8, 1.5, "샘플 N", rotation=90, va="center", ha="center", fontsize=11, color=GRAY); ax.text(1.5, -0.6, "채널 C →", ha="center", fontsize=11, color=GRAY)
fig.tight_layout(); fig.savefig("figs/norm_family.png"); plt.close(fig)

A_bn, A_ln = bn(X)[0, 0], ln(X)[0, 0]
X2 = np.vstack([X[:1], Y])
print("A first value: BN", round(A_bn, 2), "->", round(bn(X2)[0, 0], 2), "| LN", round(A_ln, 2), "->", round(ln(X2)[0, 0], 2))
print("A mean", round(X[0].mean(), 2), "std", round(X[0].std(), 2), "LN(A)", ln(X)[0].round(2))
print("col1 mean", round(X[:, 0].mean(), 2), "std", round(X[:, 0].std(), 2))
