"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
SP = {"setosa": BLUE, "versicolor": GREEN, "virginica": RED}
O = "figs/"
Xi, yi = iris()


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 평균 오차: 독립 vs 상관
n = np.arange(1, 801)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for rho, c, t in ((0.0, BLUE, "서로 독립인 실수"), (0.3, GOLD, "실수가 30% 닮음"), (0.7, RED, "실수가 70% 닮음")):
    ax.plot(n, avg_error_sd(n, rho), color=c, lw=3, label=t)
ax.set_xscale("log"); ax.set_xlabel("모은 답의 수 (로그 눈금)"); ax.set_ylabel("평균의 오차 (한 명 = 1)"); ax.legend(frameon=False); ax.set_ylim(0, 1.05)
save(fig, "rf_crowd.png")

# 2. 부트스트랩으로 키운 나무 둘
gx, gy = np.meshgrid(np.linspace(0.8, 7.2, 300), np.linspace(0, 2.7, 220))
num = {s: i for i, s in enumerate(SP)}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, seed in zip(axs[:2], (1, 2)):
    idx = np.random.default_rng(seed).integers(0, 150, 150)
    t = DecisionTreeClassifier(random_state=0).fit(Xi[idx], yi[idx])
    Z = np.vectorize(num.get)(t.predict(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=list(SP.values()), alpha=0.18)
    for s, c in SP.items():
        m = yi[idx] == s; ax.scatter(Xi[idx][m, 0], Xi[idx][m, 1], s=14, color=c)
    ax.set_title(f"다시 뽑은 데이터 {seed}로 키운 나무", fontsize=14); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(2.8, 7.2); ax.set_ylim(0.8, 2.6)
rf = RandomForestClassifier(100, random_state=0).fit(Xi, yi)
P = rf.predict_proba(np.c_[gx.ravel(), gy.ravel()])
ax = axs[2]; Z = np.vectorize(num.get)(rf.predict(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=list(SP.values()), alpha=0.18)
for s, c in SP.items():
    m = yi == s; ax.scatter(Xi[m, 0], Xi[m, 1], s=14, color=c)
ax.scatter(*FLOWER, marker="*", s=350, color=GOLD, edgecolor="black", zorder=5)
ax.set_title(f"100그루 다수결: ★ 는 {VOTES[0]} 대 {VOTES[1]}", fontsize=14); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(2.8, 7.2); ax.set_ylim(0.8, 2.6)
save(fig, "rf_trees.png")

# 3. 나무 수
fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(NTREES); ax.plot(ks, np.array(list(NTREES.values())) * 100, "o-", color=GREEN, lw=2.5, label="랜덤 포레스트")
ax.axhline(PIMA["tree"] * 100, color=RED, ls="--", lw=2); ax.text(1.1, PIMA["tree"] * 100 + 0.4, f"나무 한 그루 (끝까지) {PIMA['tree']:.1%}", color=RED, fontsize=13)
ax.axhline(PIMA["tree_d2"] * 100, color=GRAY, ls=":", lw=2); ax.text(1.1, PIMA["tree_d2"] * 100 + 0.4, f"깊이 2 로 다듬은 한 그루 {PIMA['tree_d2']:.1%}", color="#777777", fontsize=12)
ax.set_xscale("log"); ax.set_xlabel("나무 수 (로그 눈금)"); ax.set_ylabel("5겹 교차 검증 정확도 (%)"); ax.set_ylim(66, 79)
save(fig, "rf_ntrees.png")

# 4. 특징 중요도
X, y = pima(); imp = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X, y).feature_importances_
o = np.argsort(imp)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(np.array(PIMA_COLS)[o], imp[o], color=[GREEN if i in o[-3:] else GRAY for i in o])
for k, i in enumerate(o): ax.text(imp[i] + 0.004, k, f"{imp[i]:.3f}", va="center", fontsize=12)
ax.set_xlabel("불순도 감소 기준 중요도 (합 = 1)")
save(fig, "rf_importance.png")
print("ok")
