"""p47g1: XOR 패턴과 동심원 -- 노이즈 없어도 직선으로 원리적으로 못 나뉨."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.datasets import make_circles

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
RED = "#C0392B"

rng = np.random.default_rng(11)

# XOR-like: four gaussian clusters at (+-1,+-1), class = sign(x1*x2)
centers = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
Xs, ys = [], []
for cx, cy in centers:
    pts = rng.normal([cx, cy], 0.28, (25, 2))
    Xs.append(pts)
    ys.append(np.full(25, 1 if cx * cy > 0 else -1))
X_xor = np.vstack(Xs)
y_xor = np.concatenate(ys)

X_circ, y_circ = make_circles(n_samples=200, factor=0.45, noise=0.06, random_state=3)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6))

ax = axes[0]
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c=NAVY, s=45, label="클래스 A (1·3사분면)")
ax.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c=RED, s=45, marker="s", label="클래스 B (2·4사분면)")
ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color="#999999", lw=0.8)
ax.set_title("XOR --- 대각선은 직선으로 못 가름", fontsize=13)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9.5)
ax.tick_params(labelsize=9.5)

ax = axes[1]
ax.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], c=RED, s=40, marker="s", label="바깥쪽 원 (클래스 0)")
ax.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], c=NAVY, s=40, label="안쪽 원 (클래스 1)")
ax.set_title("동심원 --- 원으로만 나뉨", fontsize=13)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9.5)
ax.tick_params(labelsize=9.5)

fig.suptitle("노이즈가 없어도 직선(초평면)으로는 원리적으로 못 나뉜다", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_3_xor_circles_intro.png", dpi=200, facecolor="white", bbox_inches="tight")
print("done")
