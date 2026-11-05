"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
COLS = [BLUE, RED, GREEN]
O = "figs/"
X = np.array(HOUSES)
G = kmeans(X, INIT); B = kmeans(X, BAD_INIT)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def town(ax, C=None, lab=None, lines=False):
    cs = [GRAY] * len(X) if lab is None else [COLS[l] for l in lab]
    ax.scatter(X[:, 0], X[:, 1], s=60, c=cs, zorder=3)
    if C is not None:
        if lines:
            for x, l in zip(X, lab): ax.plot([x[0], C[l][0]], [x[1], C[l][1]], color=COLS[l], lw=0.8, alpha=0.6)
        for j, c in enumerate(C): ax.scatter(*c, s=420, marker="*", color=COLS[j], edgecolor="black", zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# 1. 질문
fig, ax = plt.subplots(figsize=(5.4, 5.4)); town(ax)
ax.set_title("집 30곳: 배달 기지 3곳은 어디에?", fontsize=16)
save(fig, "km_hook.png")

# 1b. 질문 (점 대신 집 아이콘; v2_cs 덱이 쓴다)
from matplotlib.path import Path
HOUSE = Path([(-1, -1), (1, -1), (1, 0.2), (0, 1.2), (-1, 0.2), (-1, -1)],
             [Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY])
fig, ax = plt.subplots(figsize=(5.4, 5.4)); town(ax)
ax.collections[0].remove()
ax.scatter(X[:, 0], X[:, 1], s=260, marker=HOUSE, c=GRAY, edgecolor="#555555", lw=1, zorder=3)
ax.set_title("집 30곳: 배달 기지 3곳은 어디에?", fontsize=16)
save(fig, "km_hook_house.png")

# 2. 할당
C0, L0, s0 = G[0]
fig, ax = plt.subplots(figsize=(5.4, 5.4)); town(ax, C0, L0, lines=True)
ax.set_title(f"① 할당: 가장 가까운 기지 (SSE {round(s0)})", fontsize=15)
save(fig, "km_assign.png")

# 3. 갱신
C1 = G[1][0]
fig, ax = plt.subplots(figsize=(5.4, 5.4)); town(ax, None, L0)
for j in range(3):
    ax.scatter(*C0[j], s=300, marker="*", color="white", edgecolor=COLS[j], lw=2, zorder=4)
    ax.annotate("", C1[j], C0[j], arrowprops=dict(arrowstyle="->", lw=2.5, color=COLS[j]))
    ax.scatter(*C1[j], s=420, marker="*", color=COLS[j], edgecolor="black", zorder=5)
ax.set_title("② 갱신: 자기 집들의 평균으로 이동", fontsize=15)
save(fig, "km_update.png")

# 4. 반복
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
for ax, (C, L, s), i in zip(axs, G, range(4)):
    town(ax, C, L, lines=True); ax.set_title(f"{i + 1}번째: SSE {round(s)}", fontsize=15)
save(fig, "km_iter.png")

# 5. 나쁜 시작
fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
town(axs[0], np.array(BAD_INIT, float), B[0][1], lines=True); axs[0].set_title("나쁜 시작 (두 곳이 아래 동네에)", fontsize=15)
town(axs[1], B[-1][0], B[-1][1], lines=True); axs[1].set_title(f"멈춘 곳: SSE {round(B[-1][2])} (좋은 답 29)", fontsize=15)
save(fig, "km_bad.png")

# 6. 엘보우
fig, ax = plt.subplots(figsize=(7.6, 4.3))
ks = range(1, 7)
ax.plot(ks, ELBOW, "o-", color=BLUE, lw=2.5); ax.scatter([3], [ELBOW[2]], s=200, color=RED, zorder=4)
for k, v in zip(ks, ELBOW): ax.text(k + 0.08, v + 8, str(v), fontsize=13)
ax.annotate("꺾이는 곳 k = 3", (3, ELBOW[2]), (3.6, 140), fontsize=16, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("기지 수 k"); ax.set_ylabel("SSE (거리² 합)")
save(fig, "km_elbow.png")

# 7. 한계: 반달
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
Xm, ym = make_moons(200, noise=0.06, random_state=0)
lab = KMeans(2, n_init=10, random_state=0).fit_predict(Xm)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
axs[0].scatter(Xm[:, 0], Xm[:, 1], c=[COLS[i] for i in ym], s=25); axs[0].set_title("진짜 무리: 반달 두 개", fontsize=15)
axs[1].scatter(Xm[:, 0], Xm[:, 1], c=[COLS[i] for i in lab], s=25); axs[1].set_title("k-means (k=2): 직선으로 자른다", fontsize=15)
for ax in axs: ax.set_xticks([]); ax.set_yticks([])
save(fig, "km_moons.png")
print("ok")
