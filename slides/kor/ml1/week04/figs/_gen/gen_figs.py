"""
ml1-week04 (kNN / k-means / GMM-EM) enrichment figures.
Run with the figenv python. Writes PNGs to ../ (figs/).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

BLUE = '#2E3192'
GOLD = '#D6CBB1'
GOLDDK = '#A9976B'
RED = '#C0392B'
GRAY = '#5B5B5B'

OUT = os.path.join(os.path.dirname(__file__), '..')
rng = np.random.default_rng(7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------
# p03: 가장 가까운 k=5 다수결 (2D scatter, query point, 5-NN circled)
# ------------------------------------------------------------------
def fig_p03():
    n0 = rng.normal(loc=[2, 2], scale=0.9, size=(14, 2))
    n1 = rng.normal(loc=[5.5, 5.2], scale=0.9, size=(14, 2))
    q = np.array([3.8, 3.6])
    X = np.vstack([n0, n1])
    y = np.array([0] * 14 + [1] * 14)
    d = np.linalg.norm(X - q, axis=1)
    order = np.argsort(d)
    k = 5
    nn_idx = order[:k]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=90, c=GOLDDK, edgecolor='white',
               linewidth=0.8, label='클래스 A', zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=90, c=BLUE, edgecolor='white',
               linewidth=0.8, label='클래스 B', zorder=3)
    # circle radius = distance to 5th neighbor
    r = d[order[k - 1]] + 0.05
    ax.add_patch(Circle(q, r, fill=False, ec=RED, lw=2, ls='--', zorder=2))
    for i in nn_idx:
        ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color=RED, lw=1.1, alpha=0.65, zorder=1)
        ax.add_patch(Circle(X[i], 0.22, fill=False, ec='black', lw=1.4, zorder=4))
    ax.scatter(*q, s=230, marker='*', c=RED, edgecolor='black', linewidth=0.6,
               label='새 점 (미분류)', zorder=4)
    n_a = int((y[nn_idx] == 0).sum())
    n_b = int((y[nn_idx] == 1).sum())
    ax.set_title(f'5-최근접이웃: A {n_a}표 vs B {n_b}표 $\\Rightarrow$ 다수결로 '
                 f'{"A" if n_a > n_b else "B"}', fontsize=13, color=BLUE)
    ax.legend(loc='upper left', fontsize=10.5, frameon=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'gen_knn_5nn_vote.png')


# ------------------------------------------------------------------
# p14: 잘못 찍힌 한 점 -> k=1 vs k=5 결정경계 (분산 얘기)
# ------------------------------------------------------------------
def fig_p14():
    rng2 = np.random.default_rng(3)
    n = 90
    X = rng2.uniform(0, 10, size=(n, 2))
    y = (X[:, 1] > X[:, 0]).astype(int)
    # flip one point near the boundary
    diffs = np.abs(X[:, 1] - X[:, 0])
    flip_i = np.argsort(diffs)[3]
    y_flip = y.copy()
    y_flip[flip_i] = 1 - y_flip[flip_i]

    def knn_grid(k, yy):
        xx1, xx2 = np.meshgrid(np.linspace(0, 10, 160), np.linspace(0, 10, 160))
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        d2 = ((grid[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, :k]
        votes = yy[idx].mean(axis=1)
        return xx1, xx2, (votes > 0.5).astype(int).reshape(xx1.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    for ax, k, title in zip(axes, [1, 5], ['$k=1$ --- 한 점에 경계 전체가 출렁',
                                             '$k=5$ --- 4표가 노이즈 1표를 이김']):
        xx1, xx2, zz = knn_grid(k, y_flip)
        ax.contourf(xx1, xx2, zz, levels=[-0.5, 0.5, 1.5],
                    colors=[GOLD, BLUE], alpha=0.45)
        ax.scatter(X[y_flip == 0, 0], X[y_flip == 0, 1], s=32, c=GOLDDK, edgecolor='white', linewidth=0.4)
        ax.scatter(X[y_flip == 1, 0], X[y_flip == 1, 1], s=32, c=BLUE, edgecolor='white', linewidth=0.4)
        ax.scatter(*X[flip_i], s=180, marker='X', c=RED, edgecolor='black',
                   linewidth=1.0, zorder=5, label='잘못 찍힌 점')
        ax.set_title(title, fontsize=12.5, color=BLUE)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(loc='lower right', fontsize=10)
    save(fig, 'gen_knn_variance_flip.png')


# ------------------------------------------------------------------
# p17: 보로노이 다이어그램 (k=1) vs 매끄러워진 k=5 경계
# ------------------------------------------------------------------
def fig_p17():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    rng3 = np.random.default_rng(11)
    n0 = rng3.normal([3, 6], 1.1, size=(10, 2))
    n1 = rng3.normal([7, 3.5], 1.1, size=(10, 2))
    X = np.vstack([n0, n1])
    y = np.array([0] * 10 + [1] * 10)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    ax = axes[0]
    vor = Voronoi(X)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors=GRAY,
                     line_width=1.0, point_size=0)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=70, c=GOLDDK, edgecolor='white', linewidth=0.6, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=70, c=BLUE, edgecolor='white', linewidth=0.6, zorder=3)
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 10)
    ax.set_title('$k=1$: 보로노이 셀 경계 = 결정경계', fontsize=12.5, color=BLUE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    xx1, xx2 = np.meshgrid(np.linspace(-1, 11, 200), np.linspace(-1, 10, 200))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    d2 = ((grid[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, :5]
    votes = y[idx].mean(axis=1)
    zz = (votes > 0.5).astype(int).reshape(xx1.shape)
    ax.contourf(xx1, xx2, zz, levels=[-0.5, 0.5, 1.5], colors=[GOLD, BLUE], alpha=0.45)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=70, c=GOLDDK, edgecolor='white', linewidth=0.6, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=70, c=BLUE, edgecolor='white', linewidth=0.6, zorder=3)
    ax.set_xlim(-1, 11); ax.set_ylim(-1, 10)
    ax.set_title('$k=5$: $k$-보로노이 $\\Rightarrow$ 매끄러운 경계', fontsize=12.5, color=BLUE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'gen_voronoi_k1_k5.png')


# ------------------------------------------------------------------
# p34: 훈련 정확도 vs 검증 정확도 곡선 (over k) -- overfitting at k=1
# ------------------------------------------------------------------
def fig_p34():
    k_vals = np.arange(1, 26)
    train_acc = 1.0 - 0.003 * (k_vals - 1) - 0.0006 * (k_vals - 1) ** 1.3
    train_acc = np.clip(train_acc, 0.80, 1.0)
    train_acc[0] = 1.0
    val_acc = 0.955 - 0.00055 * (k_vals - 9) ** 2
    val_acc = np.clip(val_acc, 0.70, 0.97)
    best_k = k_vals[np.argmax(val_acc)]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(k_vals, train_acc, 'o-', color=GOLDDK, lw=2, ms=5, label='훈련 정확도')
    ax.plot(k_vals, val_acc, 's-', color=BLUE, lw=2.2, ms=5, label='검증 정확도')
    ax.axvline(1, color=RED, lw=1.3, ls=':')
    ax.annotate('k=1: 훈련 100%\n(자기 자신을 외움)', xy=(1, 1.0), xytext=(4.5, 0.97),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.axvline(best_k, color=BLUE, lw=1.1, ls='--', alpha=0.6)
    ax.annotate(f'검증 최고: k={best_k}', xy=(best_k, val_acc.max()),
                xytext=(best_k + 2, val_acc.max() - 0.10), fontsize=10.5, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.1))
    ax.set_xlabel('$k$'); ax.set_ylabel('정확도')
    ax.set_ylim(0.68, 1.02)
    ax.legend(loc='lower right', fontsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'gen_train_val_acc_vs_k.png')


# ------------------------------------------------------------------
# p34b: k-겹 교차검증 폴드 다이어그램
# ------------------------------------------------------------------
def fig_p34b():
    kf = 5
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for round_i in range(kf):
        for fold_j in range(kf):
            is_val = (fold_j == round_i)
            color = RED if is_val else BLUE
            alpha = 0.85 if is_val else 0.55
            ax.add_patch(plt.Rectangle((fold_j, kf - 1 - round_i), 0.94, 0.94,
                                         color=color, alpha=alpha))
        ax.text(-0.35, kf - 1 - round_i + 0.47, f'{round_i+1}회', ha='right', va='center', fontsize=11)
    for j in range(kf):
        ax.text(j + 0.47, kf + 0.25, f'fold {j+1}', ha='center', fontsize=10.5, color=GRAY)
    ax.set_xlim(-1.5, kf + 0.3)
    ax.set_ylim(-0.3, kf + 0.8)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((kf + 0.6, kf - 0.55), 0.5, 0.5, color=RED, alpha=0.85))
    ax.text(kf + 1.25, kf - 0.3, '검증', va='center', fontsize=11)
    ax.add_patch(plt.Rectangle((kf + 0.6, kf - 1.4), 0.5, 0.5, color=BLUE, alpha=0.55))
    ax.text(kf + 1.25, kf - 1.15, '학습', va='center', fontsize=11)
    ax.set_title('5-겹 교차검증: 매 회 다른 fold가 검증셋', fontsize=12.5, color=BLUE)
    save(fig, 'gen_kfold_cv.png')


# ------------------------------------------------------------------
# p54: k-means++ 의 D(x)^2 비례 확률 선택
# ------------------------------------------------------------------
def fig_p54():
    rng4 = np.random.default_rng(5)
    blob_a = rng4.normal([2, 6], 0.7, size=(18, 2))
    blob_b = rng4.normal([8, 7], 0.7, size=(18, 2))
    blob_c = rng4.normal([5, 1.5], 0.7, size=(18, 2))
    X = np.vstack([blob_a, blob_b, blob_c])
    c1 = np.array([2.1, 6.1])
    d2 = ((X - c1) ** 2).sum(1)
    prob = d2 / d2.sum()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sizes = 40 + 900 * prob
    sc = ax.scatter(X[:, 0], X[:, 1], s=sizes, c=prob, cmap='OrRd',
                     edgecolor=GRAY, linewidth=0.5, alpha=0.9)
    ax.scatter(*c1, marker='*', s=320, c=BLUE, edgecolor='black', linewidth=0.8,
               label='첫 중심점 $c_1$ (무작위)', zorder=5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label('다음 중심점으로 뽑힐 확률 $\\propto D(x)^2$', fontsize=10.5)
    ax.set_title('점이 클수록·진할수록 $c_1$에서 멀어 뽑힐 확률이 높다', fontsize=12, color=BLUE)
    ax.legend(loc='lower right', fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'gen_kmeanspp_prob.png')


# ------------------------------------------------------------------
# p56: 두 반달(moons) 데이터에서 k-means 직선 경계 실패
# ------------------------------------------------------------------
def fig_p56():
    from sklearn.datasets import make_moons
    from sklearn.cluster import KMeans
    X, y_true = make_moons(n_samples=220, noise=0.06, random_state=0)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    lbl = km.labels_
    cen = km.cluster_centers_

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    ax = axes[0]
    ax.scatter(X[y_true == 0, 0], X[y_true == 0, 1], s=28, c=GOLDDK, label='진짜 반달 A')
    ax.scatter(X[y_true == 1, 0], X[y_true == 1, 1], s=28, c=BLUE, label='진짜 반달 B')
    ax.set_title('진짜 구조: 두 개의 초승달', fontsize=12.5, color=BLUE)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    ax.scatter(X[lbl == 0, 0], X[lbl == 0, 1], s=28, c=GOLDDK, label='k-means 클러스터 1')
    ax.scatter(X[lbl == 1, 0], X[lbl == 1, 1], s=28, c=BLUE, label='k-means 클러스터 2')
    # perpendicular-bisector boundary
    mid = cen.mean(axis=0)
    d = cen[1] - cen[0]
    perp = np.array([-d[1], d[0]])
    perp = perp / np.linalg.norm(perp)
    p1 = mid - perp * 3
    p2 = mid + perp * 3
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=RED, lw=2.2, ls='--', label='k-means 직선 경계')
    ax.scatter(cen[:, 0], cen[:, 1], marker='X', s=140, c='black', zorder=5)
    ax.set_title('k-means: 각 반달을 반으로 자르는 직선 경계', fontsize=12.5, color=BLUE)
    ax.legend(loc='upper right', fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'gen_kmeans_moons.png')


# ------------------------------------------------------------------
# p60: 하드 할당(k-means) vs 소프트 할당(GMM)
# ------------------------------------------------------------------
def fig_p60():
    rng5 = np.random.default_rng(9)
    c1 = rng5.normal([3.2, 5], 1.5, size=(30, 2))
    c2 = rng5.normal([6.0, 5.3], 1.5, size=(30, 2))
    X = np.vstack([c1, c2])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax = axes[0]
    d1 = np.linalg.norm(X - [3.2, 5], axis=1)
    d2 = np.linalg.norm(X - [6.0, 5.3], axis=1)
    hard = (d2 < d1).astype(int)
    ax.scatter(X[hard == 0, 0], X[hard == 0, 1], s=45, c=GOLDDK, edgecolor='white', linewidth=0.4)
    ax.scatter(X[hard == 1, 0], X[hard == 1, 1], s=45, c=BLUE, edgecolor='white', linewidth=0.4)
    ax.set_title('k-means: 하드 할당 (한 점 = 한 클러스터)', fontsize=12, color=BLUE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    from scipy.stats import multivariate_normal
    p1 = multivariate_normal([3.2, 5], [[2.2, 0], [0, 2.2]]).pdf(X)
    p2 = multivariate_normal([6.0, 5.3], [[2.2, 0], [0, 2.2]]).pdf(X)
    resp = p2 / (p1 + p2)  # soft prob of cluster 2 (blue)
    colors = np.outer(1 - resp, np.array(matplotlib.colors.to_rgb(GOLDDK))) + \
             np.outer(resp, np.array(matplotlib.colors.to_rgb(BLUE)))
    ax.scatter(X[:, 0], X[:, 1], s=45, c=colors, edgecolor='white', linewidth=0.4)
    ax.set_title('GMM: 소프트 할당 (색이 섞인 점 = 애매한 소속)', fontsize=12, color=BLUE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'gen_hard_vs_soft.png')


if __name__ == '__main__':
    fig_p03()
    fig_p14()
    fig_p17()
    fig_p34()
    fig_p34b()
    fig_p54()
    fig_p56()
    fig_p60()
    print('done')
