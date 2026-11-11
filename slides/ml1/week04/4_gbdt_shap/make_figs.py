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
O = "figs/"
KO = {"sex": "성별", "class": "등급", "rest": "나머지 (나이·요금·가족)"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 부스팅 장난감 예
from sklearn.ensemble import GradientBoostingRegressor
x, t = toy(); xs = np.linspace(0, 10, 400)
g = GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.3, random_state=0).fit(x[:, None], t)
stages = list(g.staged_predict(xs[:, None]))
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, n in zip(axs, (1, 3, 10, 50)):
    ax.scatter(x, t, s=18, color="black"); ax.plot(xs, stages[n - 1], color=RED, lw=2.5)
    ax.set_title(f"나무 {n}그루: MSE {TOY_MSE[n]}", fontsize=14); ax.set_xticks([])
save(fig, "gb_toy.png")

# 2. 잔차를 고친다 (1 → 2 그루)
p1 = list(g.staged_predict(x[:, None]))[0]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].scatter(x, t, s=18, color="black"); axs[0].plot(xs, stages[0], color=RED, lw=2.5)
for xi, ti, pi in zip(x, t, p1): axs[0].plot([xi, xi], [ti, pi], color=GOLD, lw=1.2)
axs[0].set_title("1그루의 예측과 남은 오차 (노랑)", fontsize=14)
axs[1].scatter(x, t - p1, s=18, color=GOLD); axs[1].axhline(0, color="black", lw=0.8)
h2 = np.array(stages[1]) - np.array(stages[0]); axs[1].plot(xs, h2 / 0.3, color=BLUE, lw=2.5)
axs[1].set_title("2번째 나무는 남은 오차를 맞힌다", fontsize=14)
for ax in axs: ax.set_xticks([])
save(fig, "gb_residual.png")

# 3. 두 승객의 몫 (폭포)
def waterfall(ax, phi, title, color):
    cur = BASE; ax.barh(4, BASE, color=GRAY); ax.text(BASE + 1, 4, f"평균 {BASE:.1f}%", va="center", fontsize=13)
    for k, (g_, v) in enumerate(phi.items()):
        yv = 3 - k; ax.barh(yv, v, left=cur, color=BLUE if v > 0 else RED)
        ax.text(max(cur, cur + v) + 1, yv, f"{KO[g_]} {v:+.1f}", va="center", fontsize=13); cur += v
    ax.barh(0, cur, color=color); ax.text(cur + 1, 0, f"= {cur:.1f}%", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 125); ax.set_yticks([]); ax.set_title(title, fontsize=15); ax.set_xlabel("생존 확률 (%)")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.0))
waterfall(axs[0], PHI_A, "26살 · 1등석 · 여성", GREEN); waterfall(axs[1], PHI_B, "26살 · 3등석 · 남성", RED)
save(fig, "gb_waterfall.png")

# 4. 전체 중요도 + 나이 SHAP
import shap
X, y, age_ok = titanic(); M = gbdt(X, y)
ex = shap.TreeExplainer(M, data=shap.sample(X, 200, random_state=0), model_output="probability"); sv = ex.shap_values(X) * 100
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
names = {"female": "성별", "pclass": "등급", "age": "나이", "fare": "요금", "family": "가족 수"}
o = sorted(IMP, key=IMP.get)
axs[0].barh([names[c] for c in o], [IMP[c] for c in o], color=GREEN); axs[0].set_xlabel("평균 |몫| (%p)")
for k, c in enumerate(o): axs[0].text(IMP[c] + 0.3, k, str(IMP[c]), va="center", fontsize=12)
ia = list(X.columns).index("age")
axs[1].scatter(X["age"][age_ok], sv[age_ok, ia], s=8, color=BLUE, alpha=0.5); axs[1].axhline(0, color="black", lw=0.8)
axs[1].set_xlabel("나이"); axs[1].set_ylabel("나이의 몫 (%p)"); axs[1].set_title("어린아이일수록 생존 쪽으로", fontsize=14)
save(fig, "gb_global.png")

# 5. 표 데이터: 배달비 계단 (나무 vs 신경망, 그리고 구간 one-hot)
import warnings; warnings.filterwarnings("ignore")
won = fee(DIST).astype(float); grid = np.linspace(0, 6, 601); C = step_curves(DIST, won, grid)
def step_panel(ax, keys, title):
    ax.step(grid, fee(grid), where="post", color=GRAY, lw=1.2, ls=":")
    ax.scatter(DIST, won, s=22, color="black", zorder=3)
    for k in keys:
        col, lab = {"tree": (GREEN, "나무 (GBDT)"), "mlp": (RED, "신경망"), "mlp_bin": (BLUE, "신경망 + 구간 one-hot")}[k]
        ax.plot(grid, C[k], color=col, lw=2.6, label=f"{lab}: 평균 오차 {STEP_MAE[k]}원")
    ax.set_xlabel("거리 (km)"); ax.set_ylabel("배달비 (원)"); ax.set_title(title, fontsize=15); ax.legend(fontsize=12, loc="upper left", frameon=False)
fig, ax = plt.subplots(figsize=(10, 4.4)); step_panel(ax, ("tree", "mlp"), "배달비 30건: 나무는 계단을 자르고, 신경망은 뭉갠다"); save(fig, "gb_tab_step.png")
fig, ax = plt.subplots(figsize=(10, 4.4)); step_panel(ax, ("mlp", "mlp_bin"), "거리를 0.3km 칸 20개로 쪼개 one-hot 으로 넣으면"); save(fig, "gb_tab_bins.png")

# 6. 쓸모없는 열을 섞으면
ks = list(NOISE_MAE)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(ks, [NOISE_MAE[k][0] for k in ks], "o-", color=GREEN, lw=2.6, label="나무 (GBDT)")
ax.plot(ks, [NOISE_MAE[k][1] for k in ks], "o-", color=RED, lw=2.6, label="신경망")
for k in ks:
    ax.text(k, NOISE_MAE[k][0] + 25, str(NOISE_MAE[k][0]), ha="center", fontsize=12, color=GREEN)
    ax.text(k, NOISE_MAE[k][1] + 25, str(NOISE_MAE[k][1]), ha="center", fontsize=12, color=RED)
ax.set_xlabel("섞은 쓸모없는 열 (무작위 숫자) 개수"); ax.set_ylabel("시험 평균 오차 (원)"); ax.set_xticks(ks)
ax.set_title("주문 400건, 쓸모 있는 열 3개 (거리 · 시각 · 비) + 쓸모없는 열", fontsize=15); ax.legend(fontsize=13, frameon=False)
save(fig, "gb_tab_noise.png")

# 7. 열을 섞으면 (회전): 나무는 축을 따라서만 자른다
from sklearn.tree import DecisionTreeClassifier
r = np.random.default_rng(3); P = r.uniform(-1, 1, (200, 2)); lab = (P[:, 0] > 0.1).astype(int)   # 정답: 첫 열만 보고 한 번에 갈림
th = np.deg2rad(45); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
gx, gy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300)); G = np.c_[gx.ravel(), gy.ravel()]
for ax, Q, title in ((axs[0], P, "원래 열: 나이 > 30 같은 기준"), (axs[1], P @ R.T, "두 열을 섞으면 (45° 회전)")):
    t = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Q, lab)
    ax.contourf(gx, gy, t.predict(G).reshape(gx.shape), levels=[-0.5, 0.5, 1.5], colors=["#f6d5d5", "#d5d6ee"])
    ax.scatter(Q[:, 0], Q[:, 1], c=np.where(lab, BLUE, RED), s=12)
    ax.set_title(f"{title}\n나무 잎 {t.get_n_leaves()}개", fontsize=14); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
save(fig, "gb_tab_rotate.png")


# 8. XGBoost: 기울기 + 곡률, 잎 값 벌점
def leaf_curves(w, p, y):
    F0 = np.log(p / (1 - p))
    true = np.array([np.sum(np.logaddexp(0, -(2 * y - 1) * (F0 + v))) for v in w])
    G, H = (p - y).sum(), (p * (1 - p) * np.ones_like(y)).sum()
    return true, true[np.argmin(np.abs(w))] + G * w + 0.5 * H * w ** 2, G, H


w = np.linspace(-0.4, 2.0, 400)
true, quad, G, H = leaf_curves(w, LEAF_P, LEAF_Y)
fig, ax = plt.subplots(figsize=(6.6, 5.0))
ax.plot(w, true, color="black", lw=2.6, label="실제 손실")
ax.plot(w, quad, color=BLUE, lw=2.6, label="기울기 + 곡률 근사")
ax.plot(w, quad + 0.5 * w ** 2, color=RED, lw=2.6, ls="--", label=r"근사 + 잎 값 벌점 ($\lambda=1$)")
for lam, col in ((0, BLUE), (1, RED)):
    v = -G / (H + lam); val = quad[np.argmin(np.abs(w - v))] + 0.5 * lam * v ** 2
    ax.scatter([v], [val], color=col, s=90, zorder=4)
    ax.annotate(f"$w$ = {LEAF_W[lam]}", (v, val), xytext=(0, -26), textcoords="offset points", ha="center", color=col, fontsize=15, fontweight="bold")
ax.set_xlabel("이 잎이 더할 값 $w$"); ax.set_ylabel("승객 셋의 손실 합"); ax.set_ylim(1.4, 2.6); ax.legend(fontsize=13, frameon=False, loc="upper center")
save(fig, "gb_xgb_leaf.png")


# 9. LightGBM: 구간으로 묶기 + 층별 / 잎별 나무
def draw_tree(ax, edges, leaves, splits, title):
    pos = {**leaves, **splits}
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=GRAY, lw=2, zorder=1)
    ax.scatter(*zip(*leaves.values()), s=420, color=GREEN, marker="s", zorder=2)
    for k, (px, py) in splits.items():
        ax.scatter([px], [py], s=780, color=BLUE, zorder=2)
        ax.text(px, py, k[1:], color="white", ha="center", va="center", fontsize=16, fontweight="bold", zorder=3)
    ax.set_title(title, fontsize=15); ax.set_xlim(-0.4, 4.4); ax.set_ylim(-0.5, 3.5); ax.axis("off")


ages = titanic()[0]["age"].to_numpy()
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
axs[0].hist(ages, bins=AGE_BINS, color=GOLD, edgecolor="white")
axs[0].set_xlabel("타이타닉 승객 나이"); axs[0].set_ylabel("승객 수")
axs[0].set_title(f"고유값 {AGE_UNIQUE}개 → 구간 {AGE_BINS}개: 막대 사이만 확인", fontsize=15)
draw_tree(axs[1], [("s1", "s2"), ("s1", "s3"), ("s2", "a"), ("s2", "b"), ("s3", "c"), ("s3", "d")],
          {"a": (0.2, 0), "b": (1.4, 0), "c": (2.6, 0), "d": (3.8, 0)}, {"s1": (2, 3), "s2": (0.8, 1.5), "s3": (3.2, 1.5)}, "층별: 한 층을 다 채운다")
draw_tree(axs[2], [("s1", "a"), ("s1", "s2"), ("s2", "b"), ("s2", "s3"), ("s3", "c"), ("s3", "d")],
          {"a": (0.6, 1.5), "b": (1.6, 0.75), "c": (2.4, 0), "d": (3.8, 0)}, {"s1": (1.6, 3), "s2": (2.4, 2), "s3": (3.1, 1)}, "잎별: 오차가 큰 잎부터 판다")
save(fig, "gb_lgbm.png")
print("ok")
