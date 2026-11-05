"""
ml1-week02 enrichment: 6 new concept diagrams for p26, p28, p55, p62, p77, p84.
Run with figenv python. Outputs PNG into ../ (figs/).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

KSA_BLUE = "#2E3192"
KSA_TAN = "#D6CBB1"
RED = "#C0392B"

OUT = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week02/slides/kor/ml1/week02/figs"

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# Shared toy dataset from the book: X=[1,2,3], y=[3,5,7], true w*=(1,2)
# ---------------------------------------------------------------
x_toy = np.array([1.0, 2.0, 3.0])
y_toy = np.array([3.0, 5.0, 7.0])
m = 3
Xb = np.column_stack([np.ones(m), x_toy])  # bias column


def cost(w):
    err = Xb @ w - y_toy
    return (err @ err) / (2 * m)


def grad(w):
    err = Xb @ w - y_toy
    return (Xb.T @ err) / m


def run_gd(alpha, n_steps, w0=np.array([0.0, 0.0])):
    ws = [w0.copy()]
    w = w0.copy()
    for _ in range(n_steps):
        w = w - alpha * grad(w)
        ws.append(w.copy())
    return np.array(ws)


eigvals = np.linalg.eigvalsh((Xb.T @ Xb) / m)
lam_min, lam_max = eigvals[0], eigvals[1]
alpha_star = 2.0 / lam_max
print("eigvals", eigvals, "alpha*", alpha_star)

# ---------------------------------------------------------------
# Fig 1 (p26): 발산 궤적 -- alpha=0.5 (> alpha*=0.361), 8 steps, w-space
# ---------------------------------------------------------------
alpha_div = 0.5
ws = run_gd(alpha_div, 5)
print("p26 path w:", ws)

pad0 = (ws[:, 0].max() - ws[:, 0].min()) * 0.15 + 0.5
pad1 = (ws[:, 1].max() - ws[:, 1].min()) * 0.15 + 0.5
w0g = np.linspace(ws[:, 0].min() - pad0, ws[:, 0].max() + pad0, 220)
w1g = np.linspace(ws[:, 1].min() - pad1, ws[:, 1].max() + pad1, 220)
W0, W1 = np.meshgrid(w0g, w1g)
J = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        J[i, j] = cost(np.array([W0[i, j], W1[i, j]]))

fig, ax = plt.subplots(figsize=(8, 6.4), dpi=200)
levels = np.geomspace(1.0, J.max(), 7)
cs = ax.contour(W0, W1, J, levels=levels, colors="#7FB37F", linewidths=1.3, alpha=0.8)
ax.plot(ws[:, 0], ws[:, 1], "-o", color=RED, lw=2, ms=6,
        label=f"경사하강법 ($\\alpha={alpha_div}$, {len(ws)-1}스텝)")
ax.plot(ws[0, 0], ws[0, 1], "v", color=RED, ms=16, mec="black")
ax.plot(1, 2, "*", color="black", ms=20, label="최솟값 $w^{*}=(1,2)$")
ax.set_xlim(w0g.min(), w0g.max())
ax.set_ylim(w1g.min(), w1g.max())
ax.set_xlabel("$w_0$ (절편)")
ax.set_ylabel("$w_1$ (기울기)")
ax.set_title(f"발산: $\\alpha={alpha_div} > \\alpha^{{*}}\\approx{alpha_star:.3f}$ 이면 매 스텝 더 크게 튕겨나감")
ax.legend(loc="upper left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/ch02_divergence_trajectory.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 2 (p28): 느리게 발산 -- alpha=0.4 (barely > alpha*), cost vs step, oscillating growth
# ---------------------------------------------------------------
alpha_slow = 0.4
ws2 = run_gd(alpha_slow, 25)
Js = np.array([cost(w) for w in ws2])
print("p28 costs:", Js[:10], "...", Js[-3:])

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
ax = axes[0]
ax.plot(range(len(Js)), Js, "-o", color=KSA_BLUE, ms=4, lw=1.6)
ax.set_xlabel("스텝")
ax.set_ylabel("비용 $J$ (선형 눈금)")
ax.set_title("(a) 선형 눈금: 처음 8스텝은 거의 안 보임")
ax.set_xlim(0, len(Js) - 1)
ax.axvspan(0, 8, color=KSA_TAN, alpha=0.5)
ax.annotate("여기까지는 ``거의 그대로''로 보인다", xy=(8, Js[8]), xytext=(9, Js[-1]*0.35),
            fontsize=9.5, arrowprops=dict(arrowstyle="->", color="black"))

ax = axes[1]
ax.plot(range(len(Js)), Js, "-o", color=RED, ms=4, lw=1.6,
        label=f"$\\alpha={alpha_slow}$ ($\\alpha^{{*}}\\approx{alpha_star:.3f}$ 를 살짝 넘음)")
ax.set_yscale("log")
ax.set_xlabel("스텝")
ax.set_ylabel("비용 $J$ (로그 눈금)")
ax.set_title("(b) 로그 눈금: 사실 매 스텝 일정 비율로 커지는 중")
ax.legend(loc="lower right", fontsize=9.5)

fig.suptitle("같은 데이터, 같은 경로 -- 눈금만 다르다", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/ch02_slow_divergence_curve.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 3 (p55): 회귀 출력(무한 범위) vs 시그모이드 출력(0~1)
# ---------------------------------------------------------------
z = np.linspace(-6, 6, 400)
lin_out = 1.5 * z  # unbounded linear output w^T x
sig_out = 1 / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200)
ax = axes[0]
ax.plot(z, lin_out, color=KSA_BLUE, lw=2.5)
ax.set_title("선형회귀: $h_w(x)=w^Tx$\n출력 범위 $(-\\infty, +\\infty)$")
ax.set_xlabel("$w^Tx$")
ax.set_ylabel("$h_w(x)$")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylim(-10, 10)

ax = axes[1]
ax.plot(z, sig_out, color=RED, lw=2.5)
ax.axhspan(0, 1, color=KSA_TAN, alpha=0.35)
ax.axhline(0, color="black", lw=1, ls=":")
ax.axhline(1, color="black", lw=1, ls=":")
ax.set_title("로지스틱회귀: $h_w(x)=\\sigma(w^Tx)$\n출력 범위 $[0, 1]$")
ax.set_xlabel("$w^Tx$")
ax.set_ylabel("$h_w(x)$")
ax.set_ylim(-0.3, 1.3)

fig.suptitle("같은 $w^Tx$ 를 통과시켜도, 출력 범위가 다르다", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/ch03_output_range_comparison.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 4 (p62): 결정 경계는 직선 -- synthetic 2-class scatter + logistic decision boundary
# ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

n_each = 90
c0 = rng.normal(loc=[-1.3, -0.6], scale=[1.0, 1.1], size=(n_each, 2))
c1 = rng.normal(loc=[1.3, 0.9], scale=[1.0, 1.1], size=(n_each, 2))
Xc = np.vstack([c0, c1])
yc = np.array([0] * n_each + [1] * n_each)

clf = LogisticRegression().fit(Xc, yc)
xx = np.linspace(Xc[:, 0].min() - 1, Xc[:, 0].max() + 1, 200)
w = clf.coef_[0]
b = clf.intercept_[0]
yy = -(w[0] * xx + b) / w[1]

fig, ax = plt.subplots(figsize=(7.2, 6), dpi=200)
ax.scatter(c0[:, 0], c0[:, 1], color=KSA_BLUE, alpha=0.8, label="클래스 0", edgecolor="white", s=45)
ax.scatter(c1[:, 0], c1[:, 1], color=RED, alpha=0.8, label="클래스 1", edgecolor="white", s=45)
ax.plot(xx, yy, "-", color="black", lw=2.5, label="결정 경계 $w^Tx=0$")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(Xc[:, 1].min() - 1, Xc[:, 1].max() + 1)
ax.set_xlabel("특징 $x_1$")
ax.set_ylabel("특징 $x_2$")
ax.set_title("$h_w(x) \\geq 0.5$ 인 영역과 아닌 영역을 가르는 경계는 직선")
ax.legend(loc="upper left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/ch03_decision_boundary_linear.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 5 (p77): 스팸 필터 두 특징 산점도
# x1 = 스팸 단어 빈도, ham 대략 낮은 범위, spam 10~36
# x2 = 발신자 미등록 여부 (0/1), spam 70% clear + 30% sneaky(overlap with ham)
# 500통 중 스팸 100통(20%)
# ---------------------------------------------------------------
n_total = 500
n_spam = 100
n_ham = n_total - n_spam
n_clear = int(round(n_spam * 0.7))
n_sneaky = n_spam - n_clear

ham_x1 = rng.uniform(0, 14, n_ham)
ham_x2 = rng.choice([0, 1], size=n_ham, p=[0.85, 0.15])

clear_x1 = rng.uniform(18, 36, n_clear)
clear_x2 = rng.choice([0, 1], size=n_clear, p=[0.15, 0.85])

sneaky_x1 = rng.uniform(8, 18, n_sneaky)
sneaky_x2 = rng.choice([0, 1], size=n_sneaky, p=[0.6, 0.4])

spam_x1 = np.concatenate([clear_x1, sneaky_x1])
spam_x2 = np.concatenate([clear_x2, sneaky_x2])

jit = lambda a: a + rng.normal(0, 0.05, size=a.shape)

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(ham_x1, jit(ham_x2), color=KSA_BLUE, alpha=0.55, s=35, label=f"정규 메일 ({n_ham}통)")
ax.scatter(spam_x1[:n_clear], jit(spam_x2[:n_clear]), color=RED, alpha=0.75, s=35,
           label=f"``분명한'' 스팸 ({n_clear}통)")
ax.scatter(spam_x1[n_clear:], jit(spam_x2[n_clear:]), color="#E67E22", alpha=0.85, s=35,
           marker="^", label=f"``교묘한'' 스팸 ({n_sneaky}통)")
ax.set_xlabel("$x_1$ = 스팸 단어 빈도 (개)")
ax.set_ylabel("$x_2$ = 발신자 미등록 여부 (0/1, 지터 표시)")
ax.set_title(f"스팸 필터 데이터: 전체 {n_total}통 중 스팸 {n_spam}통 (20\\%)")
ax.legend(loc="center right", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/ch03_spam_features_scatter.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# Fig 6 (p84): 스케일링 없이 로지스틱 경사하강법 경로 왜곡
# feature 1 raw scale ~ [10,36], feature 2 in {0,1}; 30x scale gap
# ---------------------------------------------------------------
n = 200
x1_raw = rng.uniform(10, 36, n)
x2_raw = rng.integers(0, 2, n).astype(float)
true_w = np.array([-4.0, 0.15, 2.5])  # bias, w1, w2
z_true = true_w[0] + true_w[1] * x1_raw + true_w[2] * x2_raw
p_true = 1 / (1 + np.exp(-z_true))
y_lab = (rng.uniform(size=n) < p_true).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_path(X2, y, alpha, n_steps):
    m_, _ = X2.shape
    w = np.zeros(3)
    path = [w.copy()]
    for _ in range(n_steps):
        z_ = X2 @ w
        h_ = sigmoid(z_)
        g = X2.T @ (h_ - y) / m_
        w = w - alpha * g
        path.append(w.copy())
    return np.array(path)

X_raw2 = np.column_stack([np.ones(n), x1_raw, x2_raw])
mu1, sd1 = x1_raw.mean(), x1_raw.std()
x1_std = (x1_raw - mu1) / sd1
X_std2 = np.column_stack([np.ones(n), x1_std, x2_raw])

n_show = 60
path_raw = logreg_path(X_raw2, y_lab, alpha=0.001, n_steps=n_show)
path_std = logreg_path(X_std2, y_lab, alpha=0.5, n_steps=n_show)
# well-converged reference solution (standardized features, many more steps)
path_std_long = logreg_path(X_std2, y_lab, alpha=0.5, n_steps=3000)
w_std_star = path_std_long[-1]
w1_raw_star = w_std_star[1] / sd1   # same slope in raw-x1 units
w2_star = w_std_star[2]
print("w_std_star", w_std_star, "-> raw-scale (w1,w2) target:", w1_raw_star, w2_star)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
ax = axes[0]
ax.plot(path_raw[:, 1], path_raw[:, 2], "-o", color=RED, ms=3, lw=1.3,
        label=f"{n_show}스텝 경로")
ax.plot(path_raw[0, 1], path_raw[0, 2], "v", color=RED, ms=12, mec="black")
ax.plot(w1_raw_star, w2_star, "*", color="black", ms=18, label="수렴 해 (참고용)")
ax.set_xlabel("$w_1$ (원본 스케일 특징의 가중치)")
ax.set_ylabel("$w_2$")
ax.set_title(f"(a) 스케일링 없음 ($x_1{{\\in}}[10,36]$)\n{n_show}스텝 후에도 수렴 해와 거리가 멀다")
ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
ax.plot(path_std[:, 1], path_std[:, 2], "-o", color=KSA_BLUE, ms=3, lw=1.3,
        label=f"{n_show}스텝 경로")
ax.plot(path_std[0, 1], path_std[0, 2], "v", color=KSA_BLUE, ms=12, mec="black")
ax.plot(w_std_star[1], w_std_star[2], "*", color="black", ms=18, label="수렴 해 (참고용)")
ax.set_xlabel("$w_1$ (표준화된 특징의 가중치)")
ax.set_ylabel("$w_2$")
ax.set_title(f"(b) 표준화 후\n같은 {n_show}스텝만에 수렴 해 근처까지 이동")
ax.legend(fontsize=9, loc="lower right")

fig.suptitle("같은 60스텝, 같은 데이터 -- 스케일만 다르다", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/ch03_scaling_path_distortion.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

print("done")
