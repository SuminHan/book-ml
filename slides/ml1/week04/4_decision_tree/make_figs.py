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
SP = {"setosa": BLUE, "versicolor": GREEN, "virginica": RED}
O = "figs/"
X, y = iris()


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def flowers(ax, legend=True):
    for s, c in SP.items():
        m = y == s; ax.scatter(X[m, 0], X[m, 1], s=30, color=c, label=s, alpha=0.8)
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("꽃잎 너비 (cm)"); ax.set_xlim(0.8, 7.2); ax.set_ylim(0, 2.7)
    if legend: ax.legend(frameon=False, loc="upper left", fontsize=12)


# 1. 질문
fig, ax = plt.subplots(figsize=(7.5, 4.6)); flowers(ax)
save(fig, "dt_hook.png")

# 2. 후보 질문 둘
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (f, t, title) in zip(axs, ((1, 1.75, "꽃잎 너비 < 1.75 ?"), (0, 2.45, "꽃잎 길이 < 2.45 ?"))):
    flowers(ax, legend=False)
    (ax.axhline if f == 1 else ax.axvline)(t, color="black", lw=2.5)
    _, _, w = split_gini(X, y, f, t); ax.set_title(f"{title}  →  가중 지니 {w:.3f}", fontsize=15)
save(fig, "dt_candidates.png")

# 3. 모든 기준값 훑기
fig, ax = plt.subplots(figsize=(8.5, 4.0))
for f, c, lab in ((0, BLUE, "꽃잎 길이"), (1, RED, "꽃잎 너비")):
    v = np.unique(X[:, f]); th = (v[:-1] + v[1:]) / 2
    ax.plot(th, [split_gini(X, y, f, t)[2] for t in th], "o-", ms=3, color=c, lw=2, label=lab)
ax.axhline(ROOT, color=GRAY, ls="--"); ax.text(5.2, ROOT + 0.01, f"나누기 전 {ROOT}", color="#777777", fontsize=13)
ax.annotate("길이 2.45: 0.333", (2.45, 0.333), (3.2, 0.2), fontsize=14, arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("기준값 (cm)"); ax.set_ylabel("나눈 뒤 가중 지니"); ax.legend(frameon=False); ax.set_ylim(0.1, 0.72)
save(fig, "dt_sweep.png")

# 4. 트리 그림 (깊이 2)
fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 5)
def node(x, yy, txt, c="black", w=3.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, yy - 0.45), w, 0.9, boxstyle="round,pad=0.05", fc="white", ec=c, lw=2))
    ax.text(x, yy, txt, ha="center", va="center", fontsize=13)
def edge(a, b, t):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", lw=1.8)); ax.text((a[0] + b[0]) / 2 + (0.25 if b[0] > a[0] else -0.55), (a[1] + b[1]) / 2, t, fontsize=12, color="#555555")
node(5, 4.3, "꽃잎 길이 < 2.45 ?\n150송이 · 지니 0.667", w=3.6)
node(2.2, 2.6, "setosa 50\n지니 0", SP["setosa"]); node(7.2, 2.6, "꽃잎 너비 < 1.75 ?\n100송이 · 지니 0.5", w=3.6)
node(5.0, 0.8, "versicolor 49 + virginica 5\n지니 0.168", SP["versicolor"], w=3.5); node(8.9, 0.8, "virginica 45 + versicolor 1\n지니 0.043", SP["virginica"], w=3.5)
edge((4.2, 3.85), (2.6, 3.05), "예"); edge((5.8, 3.85), (6.8, 3.05), "아니오"); edge((6.6, 2.15), (5.4, 1.25), "예"); edge((7.8, 2.15), (8.6, 1.25), "아니오")
save(fig, "dt_tree.png")

# 5. 경계: 깊이 2 vs 끝까지
from sklearn.tree import DecisionTreeClassifier
gx, gy = np.meshgrid(np.linspace(0.8, 7.2, 400), np.linspace(0, 2.7, 300))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
lab2num = {s: i for i, s in enumerate(SP)}
for ax, d in zip(axs, (2, None)):
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
    Z = np.vectorize(lab2num.get)(m.predict(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=list(SP.values()), alpha=0.15)
    flowers(ax, legend=False); ax.set_title(f"깊이 {m.get_depth()}, 잎 {m.get_n_leaves()}개, 학습 정확도 {m.score(X, y):.0%}", fontsize=14)
save(fig, "dt_boundary.png")

# 6. Pima: 깊이별 학습 vs 교차 검증
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
d = pd.read_csv("assets/pima.csv", header=None); Xp, yp = d.iloc[:, :8].to_numpy(), d[8].to_numpy()
deps = list(PIMA_SWEEP); tr = [DecisionTreeClassifier(max_depth=k, random_state=0).fit(Xp, yp).score(Xp, yp) for k in deps]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(deps, np.array(tr) * 100, "o-", color=BLUE, lw=2.5, label="학습 정확도"); ax.plot(deps, np.array(list(PIMA_SWEEP.values())) * 100, "s-", color=RED, lw=2.5, label="5겹 교차 검증")
ax.annotate("깊이 2: 75.4%", (2, 75.4), (3.5, 62), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("최대 깊이"); ax.set_ylabel("정확도 (%)"); ax.legend(frameon=False); ax.set_ylim(55, 102)
save(fig, "dt_depth.png")
print("ok")
