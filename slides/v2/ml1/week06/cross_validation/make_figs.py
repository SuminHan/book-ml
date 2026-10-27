"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py.
iris (assets/iris.csv, seaborn-data 사본) 는 '정렬된 데이터를 안 섞으면' 장면에만 쓴다."""
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
X, y = make_data(4)
r = np.random.default_rng(4); r.normal(0, 1, (N, D)); r.normal(0, 3, N)
Xt = r.normal(0, 1, (20000, D)); yt = 3 * Xt[:, 0] - 2 * Xt[:, 1] + 1.5 * Xt[:, 2] + r.normal(0, 3, 20000)
T = np.array([mse(fit_ridge(X, y, l), Xt, yt) for l in LAMS])
E = np.array([cv_errors(X, y, l) for l in LAMS]); mu, sd = E.mean(1), E.std(1)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 질문: 실제 오차 곡선은 볼 수 없다
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.plot(LAMS, T, ls="--", color=GRAY, lw=2.5); ax.set_xscale("log")
ax.text(0.4, 18.5, "새 데이터에서의 실제 오차\n(현실에서는 볼 수 없다)", color="#777777", fontsize=14)
ax.text(12, 9.6, "?", fontsize=34, color=RED, fontweight="bold", ha="center")
ax.set_xlabel("λ (로그 눈금)"); ax.set_ylabel("오차 (MSE)"); ax.set_ylim(8, 25)
save(fig, "cv_hook.png")

# 2. 한 번만 나누면
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.scatter(SINGLE, np.arange(10), s=90, color=RED, zorder=3)
for i, v in enumerate(SINGLE): ax.text(v * 1.12, i, f"{v:g}", va="center", fontsize=11)
ax.axvline(LAM_BEST, color=BLUE, ls="--", lw=2); ax.text(LAM_BEST * 1.1, 9.6, f"실제로 가장 좋은 λ = {LAM_BEST}", color=BLUE, fontsize=13)
ax.set_xscale("log"); ax.set_xlim(0.5, 900); ax.set_yticks(range(10)); ax.set_yticklabels([f"{i + 1}번째" for i in range(10)], fontsize=10)
ax.set_xlabel("검증 12명으로 고른 λ (로그 눈금)")
save(fig, "cv_single.png")

# 3. 5겹 교차 검증 곡선
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for j in range(K): ax.plot(LAMS, E[:, j], ":", color=GRAY, lw=1)
ax.fill_between(LAMS, mu - sd, mu + sd, color=GOLD, alpha=0.2)
ax.plot(LAMS, mu, "o-", color=GOLD, lw=2.5, ms=3, label="5겹 교차 검증 평균")
ax.plot(LAMS, T, "--", color=BLUE, lw=2, label="실제 오차 (예시라서 잴 수 있다)")
b = int(mu.argmin()); ax.axvline(LAMS[b], color=RED, lw=1.5)
ax.text(LAMS[b] * 1.15, 33, f"λ* = {LAM_BEST}", color=RED, fontsize=15, fontweight="bold")
ax.plot([], [], ":", color=GRAY, label="조각별 오차"); ax.set_xscale("log"); ax.set_ylim(5, 40)
ax.set_xlabel("λ (로그 눈금)"); ax.set_ylabel("MSE"); ax.legend(frameon=False, loc="upper left", fontsize=12)
save(fig, "cv_curve.png")

# 4. 데이터가 바뀌면 λ* 도 흔들린다
fig, ax = plt.subplots(figsize=(8.5, 2.6))
ax.scatter(SEEDS, np.zeros(10) + np.linspace(-0.3, 0.3, 10), s=110, color=GOLD, edgecolor="black", zorder=3)
ax.set_xscale("log"); ax.set_xlim(1, 300); ax.set_yticks([]); ax.set_ylim(-1, 1); ax.spines["left"].set_visible(False)
ax.set_xlabel("새로 뽑은 데이터 10개에서 5겹 CV 가 고른 λ* (로그 눈금)")
save(fig, "cv_seeds.png")

# 5. 정렬된 데이터를 안 섞으면 (iris)
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("assets/iris.csv"); Xi, yi = df.iloc[:, :4].to_numpy(), df["species"].to_numpy()
a = cross_val_score(LogisticRegression(max_iter=1000), Xi, yi, cv=KFold(3))
c = cross_val_score(LogisticRegression(max_iter=1000), Xi, yi, cv=StratifiedKFold(3, shuffle=True, random_state=0))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {"setosa": BLUE, "versicolor": GREEN, "virginica": RED}
for i, s in enumerate(yi): axs[0].add_patch(plt.Rectangle((i, 0), 1, 1, color=cols[s]))
for f in (50, 100): axs[0].axvline(f, color="black", lw=2)
for f, t in enumerate(("1번째 조각", "2번째", "3번째")): axs[0].text(25 + 50 * f, 1.15, t, ha="center", fontsize=13)
axs[0].set_xlim(0, 150); axs[0].set_ylim(0, 1.4); axs[0].axis("off"); axs[0].set_title("붓꽃 150송이: 종 순서대로 저장돼 있다", fontsize=14)
xb = np.arange(3); axs[1].bar(xb - 0.2, a, 0.4, color=RED, label="안 섞고 3조각"); axs[1].bar(xb + 0.2, c, 0.4, color=BLUE, label="섞고 종 비율 맞춤")
axs[1].set_xticks(xb); axs[1].set_xticklabels(["1번째", "2번째", "3번째"]); axs[1].set_ylabel("검증 정확도"); axs[1].set_ylim(0, 1.15); axs[1].legend(frameon=False, fontsize=12)
save(fig, "cv_sorted.png")
print("ok", a.round(2), c.round(2))
