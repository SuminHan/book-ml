"""
ml1-week03 (생성 모델 관점의 분류: 나이브베이즈와 GDA) enrichment 도식 생성 스크립트.
KSA 팔레트: 주색 #2E3192, 보조 #D6CBB1, 강조 빨강 #C0392B
출력: ../ 에 PNG (dpi 200, 흰 배경)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(__file__), '..')

NAVY = '#2E3192'
SAND = '#D6CBB1'
RED = '#C0392B'
GRAY = '#555555'

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


# ------------------------------------------------------------------
# 1. p07g1 — 뒤집기 문제(inversion problem): P(x|y) <-> P(y|x) 다리
# ------------------------------------------------------------------
def fig_bridge():
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, 5)
    ax.axis('off')

    box1 = FancyBboxPatch((0.4, 1.6), 3.8, 2.0, boxstyle='round,pad=0.15,rounding_size=0.2',
                           linewidth=2.5, edgecolor=NAVY, facecolor='#EEF0FA')
    box2 = FancyBboxPatch((5.8, 1.6), 3.8, 2.0, boxstyle='round,pad=0.15,rounding_size=0.2',
                           linewidth=2.5, edgecolor=RED, facecolor='#FBEDEC')
    ax.add_patch(box1)
    ax.add_patch(box2)

    ax.text(2.3, 3.15, r'$P(x\,|\,y)$', ha='center', va='center', fontsize=22, color=NAVY, fontweight='bold')
    ax.text(2.3, 2.45, '학습 가능한 방향', ha='center', va='center', fontsize=13, color=GRAY)
    ax.text(2.3, 2.0, '(스팸 메일 뭉치에서\n단어 빈도를 센다)', ha='center', va='center', fontsize=11.5, color=GRAY)

    ax.text(7.7, 3.15, r'$P(y\,|\,x)$', ha='center', va='center', fontsize=22, color=RED, fontweight='bold')
    ax.text(7.7, 2.45, '실제로 필요한 방향', ha='center', va='center', fontsize=13, color=GRAY)
    ax.text(7.7, 2.0, '(이 메일이 스팸일 확률)', ha='center', va='center', fontsize=11.5, color=GRAY)

    arrow = FancyArrowPatch((4.3, 3.4), (5.7, 3.4), connectionstyle='arc3,rad=-0.35',
                             arrowstyle='-|>', mutation_scale=26, linewidth=2.6, color=NAVY)
    ax.add_patch(arrow)
    arrow2 = FancyArrowPatch((5.7, 2.6), (4.3, 2.6), connectionstyle='arc3,rad=-0.35',
                              arrowstyle='-|>', mutation_scale=26, linewidth=2.6, color=NAVY)
    ax.add_patch(arrow2)
    ax.text(5.0, 4.15, '베이즈 정리', ha='center', va='center', fontsize=15, color=NAVY, fontweight='bold')
    ax.text(5.0, 1.15, '뒤집기 문제 (inversion problem)', ha='center', va='center', fontsize=12.5,
            color=GRAY, style='italic')
    ax.text(5.0, 0.05, r'$P(y|x)=\frac{P(x|y)P(y)}{P(x)}$', ha='center', va='center',
            fontsize=17, color='black')
    savefig(fig, 'ch03_bridge_diagram.png')


# ------------------------------------------------------------------
# 2. p28 — w 의 방향: 마할라노비스 보정 (기울어진 공유 공분산 vs 유클리드 수직이등분선)
# ------------------------------------------------------------------
def fig_mahalanobis():
    rng = np.random.default_rng(7)
    Sigma = np.array([[2.4, 1.7], [1.7, 1.4]])
    mu0 = np.array([1.0, 4.0])
    mu1 = np.array([5.0, 1.5])
    X0 = rng.multivariate_normal(mu0, Sigma, 220)
    X1 = rng.multivariate_normal(mu1, Sigma, 220)

    Sinv = np.linalg.inv(Sigma)
    w = Sinv @ (mu1 - mu0)
    mid = (mu0 + mu1) / 2
    b = -w @ mid

    # naive (Euclidean) perpendicular bisector: normal = mu1-mu0, through mid
    n_naive = (mu1 - mu0)
    b_naive = -n_naive @ mid

    xs = np.linspace(-3, 9, 200)

    def line_ys(normal, bb, xs):
        # normal[0]*x + normal[1]*y + bb = 0
        return -(normal[0] * xs + bb) / normal[1]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(X0[:, 0], X0[:, 1], s=14, color=NAVY, alpha=0.55, label='클래스 0')
    ax.scatter(X1[:, 0], X1[:, 1], s=14, color=RED, alpha=0.55, label='클래스 1')
    ax.scatter(*mu0, marker='X', s=180, color=NAVY, edgecolor='black', linewidth=1.2, zorder=5)
    ax.scatter(*mu1, marker='X', s=180, color=RED, edgecolor='black', linewidth=1.2, zorder=5)
    ax.plot([mu0[0], mu1[0]], [mu0[1], mu1[1]], color='black', linewidth=1.2, linestyle=':', alpha=0.6)

    ys_correct = line_ys(w, b, xs)
    ys_naive = line_ys(n_naive, b_naive, xs)
    ax.plot(xs, ys_correct, color='black', linewidth=3.0, label=r'정확한 경계 ($\Sigma^{-1}$ 보정)')
    ax.plot(xs, ys_naive, color=GRAY, linewidth=2.4, linestyle='--',
            label='단순 수직이등분선 (틀린 방향)')

    ax.set_xlim(-3, 9)
    ax.set_ylim(-4.5, 9.5)
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title('공유 공분산이 기울어져 있으면 $w=\\Sigma^{-1}(\\mu_1-\\mu_0)$ 도 함께 기운다',
                 fontsize=13.5)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95)
    ax.set_aspect('equal')
    savefig(fig, 'ch03_mahalanobis_boundary.png')


# ------------------------------------------------------------------
# 3. p29 — phi 는 위치만, 기울기는 그대로 (사전확률에 따라 평행 이동하는 경계)
# ------------------------------------------------------------------
def fig_prior_shift():
    rng = np.random.default_rng(3)
    Sigma = np.array([[1.4, 0.5], [0.5, 1.0]])
    mu0 = np.array([1.0, 1.0])
    mu1 = np.array([5.0, 4.5])
    X0 = rng.multivariate_normal(mu0, Sigma, 200)
    X1 = rng.multivariate_normal(mu1, Sigma, 200)

    Sinv = np.linalg.inv(Sigma)
    w = Sinv @ (mu1 - mu0)
    b0 = -0.5 * (mu1 - mu0) @ Sinv @ (mu1 + mu0)

    xs = np.linspace(-2, 8, 200)
    phis = [0.1, 0.5, 0.9]
    colors = ['#7C8FE0', NAVY, RED]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(X0[:, 0], X0[:, 1], s=13, color=NAVY, alpha=0.35)
    ax.scatter(X1[:, 0], X1[:, 1], s=13, color=RED, alpha=0.35)

    for phi, c in zip(phis, colors):
        b = b0 + np.log(phi / (1 - phi))
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color=c, linewidth=2.8,
                label=fr'$\phi={phi}$')

    ax.annotate('', xy=(6.6, 6.7), xytext=(6.6, 1.4),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
    ax.text(6.9, 4.0, '같은 기울기\n위치만 이동', fontsize=11.5, color='black', va='center')

    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(r'$\phi$ 는 $\log\frac{\phi}{1-\phi}$ 항으로 절편 $b$ 만 바꾼다 --- 기울기 $w$ 는 불변',
                 fontsize=13)
    ax.legend(loc='lower right', fontsize=11.5, framealpha=0.9)
    ax.set_aspect('equal')
    savefig(fig, 'ch03_prior_shift.png')


# ------------------------------------------------------------------
# 4. p36 — 경계의 물리적 이동 = 기저율 오류 (50:50 vs 80:20 두 패널)
# ------------------------------------------------------------------
def fig_baserate_boundary():
    rng = np.random.default_rng(11)
    Sigma = np.array([[1.1, 0.3], [0.3, 0.9]])
    mu0 = np.array([1.0, 1.0])
    mu1 = np.array([4.5, 4.0])
    Sinv = np.linalg.inv(Sigma)
    w = Sinv @ (mu1 - mu0)
    b0 = -0.5 * (mu1 - mu0) @ Sinv @ (mu1 + mu0)
    xs = np.linspace(-2, 7, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True, sharey=True)
    settings = [(0.5, 0.5, '사전확률 50 : 50'), (0.8, 0.2, '사전확률 불합격 80 : 합격 20')]

    for ax, (phi0, phi1, title) in zip(axes, settings):
        n0, n1 = int(400 * phi0), int(400 * phi1)
        X0 = rng.multivariate_normal(mu0, Sigma, max(n0, 30))
        X1 = rng.multivariate_normal(mu1, Sigma, max(n1, 30))
        ax.scatter(X0[:, 0], X0[:, 1], s=11, color=NAVY, alpha=0.45, label='클래스 0 (불합격)')
        ax.scatter(X1[:, 0], X1[:, 1], s=11, color=RED, alpha=0.45, label='클래스 1 (합격)')

        phi = phi1
        b = b0 + np.log(phi / (1 - phi))
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color='black', linewidth=3.0)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-2, 7)
        ax.set_ylim(-2, 7)
        ax.set_xlabel(r'$x_1$', fontsize=12)
        ax.set_aspect('equal')

    axes[0].set_ylabel(r'$x_2$', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[1].annotate('경계가 소수 클래스\n(합격) 쪽으로 밀림', xy=(4.0, 3.6), xytext=(0.0, 5.8),
                      fontsize=11.5, color=RED,
                      arrowprops=dict(arrowstyle='->', color=RED, lw=1.8))
    fig.suptitle('클래스 비율이 기울면 결정 경계가 같은 기울기로 평행 이동한다', fontsize=14, y=1.02)
    savefig(fig, 'ch03_baserate_boundary.png')


# ------------------------------------------------------------------
# 5. p37 — 공유 공분산(GDA, 직선) vs 클래스별 공분산(QDA, 곡선)
# ------------------------------------------------------------------
def fig_gda_vs_qda():
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                                 QuadraticDiscriminantAnalysis)
    rng = np.random.default_rng(5)

    # 왼쪽: 공유 공분산 (GDA 가정이 맞는 경우)
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu0, mu1 = np.array([1.0, 1.0]), np.array([4.5, 4.0])
    X0 = rng.multivariate_normal(mu0, Sigma, 200)
    X1 = rng.multivariate_normal(mu1, Sigma, 200)
    Xg = np.vstack([X0, X1])
    yg = np.array([0] * 200 + [1] * 200)
    lda = LinearDiscriminantAnalysis().fit(Xg, yg)

    # 오른쪽: 클래스마다 다른 공분산 (실제로 GDA 가정이 깨지는 경우) -> QDA
    Sigma0 = np.array([[0.35, 0.0], [0.0, 0.35]])
    Sigma1 = np.array([[2.6, 1.8], [1.8, 2.6]])
    Y0 = rng.multivariate_normal(mu0, Sigma0, 200)
    Y1 = rng.multivariate_normal(mu1, Sigma1, 200)
    Xq = np.vstack([Y0, Y1])
    yq = np.array([0] * 200 + [1] * 200)
    qda = QuadraticDiscriminantAnalysis().fit(Xq, yq)

    xx, yy = np.meshgrid(np.linspace(-2, 8, 400), np.linspace(-2, 8, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8))

    zz = (lda.decision_function(grid)).reshape(xx.shape)
    axes[0].scatter(X0[:, 0], X0[:, 1], s=11, color=NAVY, alpha=0.4)
    axes[0].scatter(X1[:, 0], X1[:, 1], s=11, color=RED, alpha=0.4)
    axes[0].contour(xx, yy, zz, levels=[0], colors='black', linewidths=3.0)
    axes[0].set_title('GDA: 공유 공분산 $\\Rightarrow$ 경계 = 직선', fontsize=13)

    zz2 = (qda.decision_function(grid)).reshape(xx.shape)
    axes[1].scatter(Y0[:, 0], Y0[:, 1], s=11, color=NAVY, alpha=0.4)
    axes[1].scatter(Y1[:, 0], Y1[:, 1], s=11, color=RED, alpha=0.4)
    axes[1].contour(xx, yy, zz2, levels=[0], colors='black', linewidths=3.0)
    axes[1].set_title('QDA: 클래스별 공분산 $\\Rightarrow$ 경계 = 곡선', fontsize=13)

    for ax in axes:
        ax.set_xlim(-2, 8)
        ax.set_ylim(-2, 8)
        ax.set_xlabel(r'$x_1$', fontsize=12)
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$x_2$', fontsize=12)
    savefig(fig, 'ch03_gda_vs_qda.png')


# ------------------------------------------------------------------
# 6. p20g1 — 중심극한정리: 균등분포 합이 가우시안으로 수렴
# ------------------------------------------------------------------
def fig_clt():
    rng = np.random.default_rng(1)
    ns = [1, 4, 30]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    for ax, n in zip(axes, ns):
        samples = rng.uniform(0, 1, size=(20000, n)).sum(axis=1)
        ax.hist(samples, bins=40, color=NAVY, alpha=0.75, density=True, edgecolor='white')
        ax.set_title(f'균등분포 {n}개 합' if n > 1 else '균등분포 1개', fontsize=13)
        ax.set_yticks([])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
    fig.suptitle('작은 효과 여러 개를 더하면 --- 중심극한정리로 가우시안 모양에 가까워진다',
                 fontsize=14, y=1.03)
    savefig(fig, 'ch03_clt_illustration.png')


# ------------------------------------------------------------------
# 7. p60 — 왜 로그를 더하는가: 확률 곱은 0으로 꺼지고, 로그 합은 유한
# ------------------------------------------------------------------
def fig_underflow():
    Vs = np.arange(1, 260001, 500)
    per_word = 0.98
    log_prob = Vs * np.log(per_word)          # log P(x|y), 항상 유한
    prob = np.exp(np.clip(log_prob, -745, 0))  # float64 underflow 시 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].plot(Vs, prob, color=RED, linewidth=2.8)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title('확률의 곱 $P(x|y)$', fontsize=13)
    axes[0].set_xlabel('어휘(단어) 수 $V$', fontsize=11.5)
    axes[0].set_ylabel('확률값', fontsize=11.5)
    axes[0].annotate('$V\\!\\approx\\!200{,}000$ 근처부터\n부동소수점 0.0으로 붕괴',
                      xy=(200000, 0), xytext=(60000, 0.35),
                      fontsize=10.5, color=RED,
                      arrowprops=dict(arrowstyle='->', color=RED, lw=1.6))

    axes[1].plot(Vs, log_prob, color=NAVY, linewidth=2.8)
    axes[1].set_title(r'로그 공간의 합 $\sum_j \log P(x_j|y)$', fontsize=13)
    axes[1].set_xlabel('어휘(단어) 수 $V$', fontsize=11.5)
    axes[1].set_ylabel('로그 점수 (nat)', fontsize=11.5)
    axes[1].annotate('$V\\!=\\!200{,}000$: $\\approx-4041$\n(유한, 비교 가능)',
                      xy=(200000, -4041), xytext=(60000, -3300),
                      fontsize=10.5, color=NAVY,
                      arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.6))

    for ax in axes:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    fig.suptitle('단어당 가능도 0.98을 $V$ 번 곱하면: 확률은 0.0으로 꺼지지만 로그 합은 유한하다',
                 fontsize=13.5, y=1.03)
    savefig(fig, 'ch03_underflow.png')


# ------------------------------------------------------------------
if __name__ == '__main__':
    fig_bridge()
    fig_mahalanobis()
    fig_prior_shift()
    fig_baserate_boundary()
    fig_gda_vs_qda()
    fig_clt()
    fig_underflow()
