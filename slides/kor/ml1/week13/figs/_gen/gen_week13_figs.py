"""
Week13 (ml1) enrichment: generate 7 concept diagrams.
Run with the figenv python (matplotlib/numpy/sklearn only, no networkx).
Outputs land in ../  (i.e. .../week13/figs/*.png)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Circle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

BLUE = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'
DARK = '#333333'

OUT = os.path.join(os.path.dirname(__file__), "..")


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------
# 0. shared data: Zachary karate club edges (exact list from book ch13 1.md)
# ---------------------------------------------------------------
KARATE_EDGES = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]
N_KARATE = 34
# standard Zachary ground-truth split (Mr. Hi / Officer), node0 vs node33 side
MR_HI = {0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21}


def spring_layout(edges, n, iterations=400, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n, 2))
    k = 1.2 / np.sqrt(n)
    for it in range(iterations):
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[dist < 1e-3] = 1e-3
            force = (k * k) / dist
            disp[i] += np.sum((delta.T * (force / dist)).T, axis=0)
        for a, b in edges:
            delta = pos[a] - pos[b]
            dist = np.linalg.norm(delta)
            dist = max(dist, 1e-3)
            f = dist * dist / k
            d = delta / dist * f
            disp[a] -= d
            disp[b] += d
        length = np.linalg.norm(disp, axis=1)
        length[length < 1e-3] = 1e-3
        t = 0.15 * (1 - it / iterations) + 0.01
        pos += (disp.T * np.minimum(length, t) / length).T
    # normalize
    pos -= pos.mean(axis=0)
    pos /= np.abs(pos).max()
    return pos


def degrees(edges, n):
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


POS_KARATE = spring_layout(KARATE_EDGES, N_KARATE, seed=7)


# ---------------------------------------------------------------
# 1. karate_club_network.png  (p13 -- Zachary's Karate Club 구조)
# ---------------------------------------------------------------
def fig_karate_network():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for a, b in KARATE_EDGES:
        ax.plot([POS_KARATE[a, 0], POS_KARATE[b, 0]],
                [POS_KARATE[a, 1], POS_KARATE[b, 1]],
                color='#bbbbbb', lw=1.0, zorder=1)
    for i in range(N_KARATE):
        color = BLUE if i in MR_HI else '#E08A2B'
        size = 520 if i in (0, 33) else 200
        edge = DARK if i in (0, 33) else 'white'
        lw = 2.2 if i in (0, 33) else 0.6
        ax.scatter(*POS_KARATE[i], s=size, c=color, edgecolors=edge,
                   linewidths=lw, zorder=3)
        if i in (0, 33):
            ax.annotate(str(i), POS_KARATE[i], color='white', fontsize=13,
                        fontweight='bold', ha='center', va='center', zorder=4)
    ax.scatter([], [], c=BLUE, s=200, label='감독파 (0)')
    ax.scatter([], [], c='#E08A2B', s=200, label='관장파 (33)')
    ax.legend(loc='lower center', ncol=2, frameon=False, fontsize=10)
    ax.set_title("Zachary's Karate Club --- 34개 노드, 78개 엣지", fontsize=12.5)
    ax.set_aspect('equal')
    ax.axis('off')
    savefig(fig, "karate_club_network.png")


# ---------------------------------------------------------------
# 2. karate_embedding_norms.png (p12 -- 허브 노드가 더 자주 "나타난다")
# ---------------------------------------------------------------
def fig_karate_degree_bar():
    deg = degrees(KARATE_EDGES, N_KARATE)
    order = np.argsort(-deg)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = []
    for idx in order:
        if idx in (0, 33):
            colors.append(RED)
        else:
            colors.append(TAN)
    bars = ax.bar(range(N_KARATE), deg[order], color=colors, edgecolor='white', linewidth=0.4)
    ax.set_xticks(range(N_KARATE))
    ax.set_xticklabels([str(i) for i in order], fontsize=7, rotation=90)
    ax.set_xlabel("노드 번호 (도 내림차순)", fontsize=11)
    ax.set_ylabel("도(degree)", fontsize=11)
    ax.set_title("도(degree) = 걷기 말뭉치 등장 빈도에 비례", fontsize=13)
    for idx_pos, idx in enumerate(order):
        if idx in (0, 33):
            ax.annotate(f"node {idx}\n도 {deg[idx]}", (idx_pos, deg[idx] + 0.4),
                        ha='center', fontsize=9, color=RED, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    savefig(fig, "karate_embedding_norms.png")


# ---------------------------------------------------------------
# 3. node2vec_pq_diagram.png (p22 -- p, q 이웃 탐색 다이어그램)
# ---------------------------------------------------------------
def fig_node2vec_pq():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    prev_pos = np.array([-2.6, 0.0])
    cur_pos = np.array([0.0, 0.0])
    back_pos = prev_pos
    local_pos = [np.array([1.1, 1.15]), np.array([1.1, -1.15])]
    far_pos = [np.array([2.7, 0.55]), np.array([2.7, -0.55])]

    def node(pos, label, color=BLUE, r=0.28):
        circ = Circle(pos, r, facecolor=color, edgecolor=DARK, linewidth=1.4, zorder=3)
        ax.add_patch(circ)
        ax.annotate(label, pos, color='white', ha='center', va='center',
                    fontsize=11, fontweight='bold', zorder=4)

    # already-walked edge
    ax.annotate("", xy=cur_pos, xytext=prev_pos,
                arrowprops=dict(arrowstyle='-', color=DARK, lw=2))
    node(prev_pos, r"$x_{t-1}$", color='#888888')
    node(cur_pos, r"$x_t$", color=BLUE)

    # back to x_{t-1}: curved arrow
    arr = FancyArrowPatch(cur_pos, back_pos, connectionstyle="arc3,rad=0.5",
                          arrowstyle='-|>', color=RED, lw=2, mutation_scale=18, zorder=2)
    ax.add_patch(arr)
    ax.annotate(r"$1/p$" + "\n(돌아가기)", (-1.5, 0.85), color=RED, fontsize=11, ha='center')

    for lp in local_pos:
        node(lp, "", color='#E08A2B', r=0.22)
        ax.annotate("", xy=lp, xytext=cur_pos,
                    arrowprops=dict(arrowstyle='-|>', color='#E08A2B', lw=2))
    ax.annotate(r"$1/q$" + "\n(인근 머물기)", (1.15, 0.0), color='#E08A2B',
                fontsize=11, ha='center', va='center')

    for fp in far_pos:
        node(fp, "", color=TAN, r=0.22)
        ax.annotate("", xy=fp, xytext=cur_pos,
                    arrowprops=dict(arrowstyle='-|>', color='#8a7a5c', lw=2))
    ax.annotate(r"$1$" + "\n(새 영토)", (3.35, 0.0), color='#8a7a5c',
                fontsize=11, ha='center', va='center')

    ax.set_xlim(-3.4, 4.0)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r"$x_{t-1}\to x_t$ 다음, 어느 이웃으로? (가중치 $\propto$ 그림의 값)",
                fontsize=12.5)
    savefig(fig, "node2vec_pq_diagram.png")


# ---------------------------------------------------------------
# 4. pagerank_equal_degree.png (p48 -- 겉모습(도)은 같아도 점수는 다르다)
# ---------------------------------------------------------------
def fig_pagerank_equal_degree():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    pos = {
        'D': np.array([-1.6, 1.2]),
        'A': np.array([0.0, 1.6]),
        'B': np.array([1.4, 0.0]),
        'C': np.array([0.0, -1.6]),
    }
    pr = {'D': 0.125, 'A': 0.321, 'B': 0.286, 'C': 0.268}
    edges = [('D', 'A'), ('A', 'B'), ('B', 'C'), ('C', 'A')]

    for u, v in edges:
        arr = FancyArrowPatch(pos[u], pos[v], arrowstyle='-|>', color=DARK,
                              lw=2, mutation_scale=22,
                              connectionstyle="arc3,rad=0.12",
                              shrinkA=26, shrinkB=26, zorder=2)
        ax.add_patch(arr)

    for name, p in pos.items():
        r = 0.35 + 0.55 * pr[name]
        color = BLUE if name != 'D' else '#888888'
        circ = Circle(p, r, facecolor=color, edgecolor=DARK, linewidth=1.4, zorder=3)
        ax.add_patch(circ)
        ax.annotate(f"{name}\nPR={pr[name]:.3f}", p, color='white', ha='center',
                    va='center', fontsize=10.5, fontweight='bold', zorder=4)
    ax.annotate("A, B, C 모두 도 2\n(들어옴 1 / 나감 1)\n하지만 점수는 다르다",
                (1.1, -1.7), fontsize=10.5, color=DARK, ha='left')
    ax.set_xlim(-2.6, 3.0)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r"$D\to A\to B\to C\to A$, $d=0.5$ 손계산 결과", fontsize=13)
    savefig(fig, "pagerank_equal_degree.png")


# ---------------------------------------------------------------
# 5. aml_layering_graph.png (p65 -- 레이어링: 정상 클러스터 + 3~4홉 경로)
# ---------------------------------------------------------------
def fig_aml_layering():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rng = np.random.default_rng(3)
    cluster_n = 9
    cluster_pos = rng.uniform(low=[-3.4, -1.1], high=[-1.6, 1.1], size=(cluster_n, 2))
    # random sparse edges within cluster
    for i in range(cluster_n):
        for j in range(i + 1, cluster_n):
            if rng.random() < 0.25:
                ax.plot([cluster_pos[i, 0], cluster_pos[j, 0]],
                        [cluster_pos[i, 1], cluster_pos[j, 1]],
                        color='#cccccc', lw=1.0, zorder=1)
    ax.scatter(cluster_pos[:, 0], cluster_pos[:, 1], s=160, c=TAN,
              edgecolors='white', linewidths=0.8, zorder=3)
    ax.annotate("정상 거래\n클러스터", (-2.5, 1.5), ha='center', fontsize=11, color=DARK)

    chain_labels = ["계좌 A", "계좌 B", "계좌 C", "해외 계좌"]
    chain_pos = [np.array([-0.4, 0.0]), np.array([1.1, 0.75]),
                 np.array([2.6, 0.0]), np.array([4.1, 0.75])]
    entry = cluster_pos[np.argmin(np.linalg.norm(cluster_pos - chain_pos[0], axis=1))]
    ax.annotate("", xy=chain_pos[0], xytext=entry,
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=2, ls='dashed'))
    for k in range(len(chain_pos) - 1):
        ax.annotate("", xy=chain_pos[k + 1], xytext=chain_pos[k],
                    arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.4))
    for p, lab in zip(chain_pos, chain_labels):
        circ = Circle(p, 0.42, facecolor=RED, edgecolor=DARK, linewidth=1.2, zorder=4)
        ax.add_patch(circ)
        ax.annotate(lab, (p[0], p[1] - 0.62), ha='center', fontsize=10, color=DARK)
    ax.annotate("레이어링 경로: 3~4홉 이상", (1.7, 1.6), ha='center',
                fontsize=11, color=RED, fontweight='bold')
    ax.set_xlim(-3.8, 5.0)
    ax.set_ylim(-1.9, 2.1)
    ax.set_aspect('equal')
    ax.axis('off')
    savefig(fig, "aml_layering_graph.png")


# ---------------------------------------------------------------
# 6. stgcn_structure.png (p70 -- 공간 x 시간 순환 구조)
# ---------------------------------------------------------------
def fig_stgcn():
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    n_road = 4
    times = [0, 1, 2]
    time_labels = [r"$t-2$", r"$t-1$", r"$t$"]
    xs = {t: 1.6 * i for i, t in enumerate(times)}
    ys = np.linspace(0, 1.5 * (n_road - 1), n_road)

    for t in times:
        for y in ys:
            circ = Circle((xs[t], y), 0.22, facecolor=BLUE, edgecolor=DARK,
                          linewidth=1.0, zorder=3)
            ax.add_patch(circ)
        # spatial edges (road graph: simple path 0-1-2-3)
        for k in range(n_road - 1):
            ax.plot([xs[t], xs[t]], [ys[k], ys[k + 1]], color=BLUE, lw=2.0, zorder=2)
        ax.annotate(time_labels[t], (xs[t], ys[-1] + 0.7), ha='center',
                    fontsize=12, color=DARK, fontweight='bold')

    for k, y in enumerate(ys):
        for i in range(len(times) - 1):
            ax.annotate("", xy=(xs[times[i + 1]] - 0.24, y), xytext=(xs[times[i]] + 0.24, y),
                        arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.6, ls='dashed'))

    ax.annotate("공간 방향\n(이웃 도로 합치기)", (-1.0, ys[-1] / 2), rotation=90,
                ha='center', va='center', fontsize=11, color=BLUE)
    ax.annotate(r"시간 방향 (과거 $\to$ 현재) $\rightarrow$", (xs[1], -1.0),
                ha='center', fontsize=11, color=RED)
    ax.set_xlim(-1.8, xs[2] + 1.0)
    ax.set_ylim(-1.6, ys[-1] + 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("STGCN: 매 단계마다 공간(파랑)+시간(빨강)을 함께 본다", fontsize=12.5)
    savefig(fig, "stgcn_structure.png")


# ---------------------------------------------------------------
# 7. bipartite_graph.png (p71b -- 사용자-아이템 이분 그래프)
# ---------------------------------------------------------------
def fig_bipartite():
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    n_user, n_item = 5, 5
    uy = np.linspace(0, 4, n_user)
    iy = np.linspace(0, 4, n_item)
    ux, ix = 0.0, 3.0
    rng = np.random.default_rng(5)
    edges = []
    for u in range(n_user):
        k = rng.integers(1, 3)
        items = rng.choice(n_item, size=k, replace=False)
        for it in items:
            edges.append((u, it))
    for u, it in edges:
        ax.plot([ux, ix], [uy[u], iy[it]], color='#bbbbbb', lw=1.2, zorder=1)
    for u in range(n_user):
        circ = Circle((ux, uy[u]), 0.28, facecolor=BLUE, edgecolor=DARK, linewidth=1.0, zorder=3)
        ax.add_patch(circ)
        ax.annotate(f"사용자{u+1}", (ux - 0.55, uy[u]), ha='right', va='center', fontsize=10)
    for it in range(n_item):
        circ = Circle((ix, iy[it]), 0.28, facecolor='#E08A2B', edgecolor=DARK, linewidth=1.0, zorder=3)
        ax.add_patch(circ)
        ax.annotate(f"아이템{it+1}", (ix + 0.55, iy[it]), ha='left', va='center', fontsize=10)
    ax.annotate("사용자끼리는 직접 연결 안 됨\n(이분 그래프)", (1.5, 4.5), ha='center',
                fontsize=10.5, color=DARK)
    ax.set_xlim(-1.7, 4.7)
    ax.set_ylim(-0.8, 5.0)
    ax.set_aspect('equal')
    ax.axis('off')
    savefig(fig, "bipartite_graph.png")


if __name__ == "__main__":
    fig_karate_network()
    fig_karate_degree_bar()
    fig_node2vec_pq()
    fig_pagerank_equal_degree()
    fig_aml_layering()
    fig_stgcn()
    fig_bipartite()
