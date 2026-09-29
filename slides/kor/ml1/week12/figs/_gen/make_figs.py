#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Week12 (어텐션/트랜스포머) 이미지 enrichment - 8개 개념 도식 생성 스크립트.
FIGPY 환경(matplotlib/numpy)에서 실행. 출력은 ../ (figs/) 에 저장.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon
import matplotlib.patches as mpatches
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
KSA_TAN = '#D6CBB1'
RED = '#C0392B'
GRAY = '#888888'
OUTDIR = os.path.join(os.path.dirname(__file__), '..')

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


# ---------------------------------------------------------------
# 1. p03: RNN 순차 처리 vs 어텐션 직접 접근 (경로 길이 n-1 vs 1)
# ---------------------------------------------------------------
def fig_rnn_seq():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    n = 5
    xs = np.arange(n)

    ax = axes[0]
    ax.set_title('RNN: 순차 처리 (경로 길이 = 4)', fontsize=13, color=KSA_BLUE, fontweight='bold')
    for i, x in enumerate(xs):
        ax.add_patch(Rectangle((x - 0.32, -0.32), 0.64, 0.64, facecolor=KSA_TAN,
                                edgecolor=KSA_BLUE, linewidth=1.6, zorder=3))
        ax.text(x, 0, f'$h_{{{i+1}}}$', ha='center', va='center', fontsize=13, zorder=4)
    for x in xs[:-1]:
        ax.annotate('', xy=(x + 0.68, 0), xytext=(x + 0.32, 0),
                     arrowprops=dict(arrowstyle='-|>', color=KSA_BLUE, lw=2))
    ax.text(xs[0], -0.85, '$w_1$', ha='center', fontsize=11, color=GRAY)
    ax.text(xs[-1], -0.85, '$w_5$', ha='center', fontsize=11, color=GRAY)
    ax.text(2, 0.85, '$w_5$가 $w_1$을 보려면 4단계를 거쳐야 함', ha='center', fontsize=10.5, color=RED)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-1.2, 1.3)
    ax.axis('off')

    ax = axes[1]
    ax.set_title('어텐션: 모든 단어 직접 연결 (경로 길이 = 1)', fontsize=13, color=KSA_BLUE, fontweight='bold')
    r = 1.0
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    pts = np.c_[r * np.cos(angles), r * np.sin(angles)]
    for i in range(n):
        for j in range(n):
            if i < j:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color=KSA_BLUE, lw=0.9, alpha=0.45, zorder=1)
    for i, (x, y) in enumerate(pts):
        ax.add_patch(plt.Circle((x, y), 0.16, facecolor=KSA_TAN, edgecolor=KSA_BLUE,
                                 linewidth=1.6, zorder=3))
        ax.text(x, y, f'$w_{{{i+1}}}$', ha='center', va='center', fontsize=11, zorder=4)
    ax.text(0, -1.55, '어떤 두 단어든 한 번의 내적으로 바로 연결', ha='center', fontsize=10.5, color=RED)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.75, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    savefig(fig, 'ch12_1_rnn_vs_attention_path.png')


# ---------------------------------------------------------------
# 2. p04: 어텐션 가중치 분포 ("그것은"의 가중치)
# ---------------------------------------------------------------
def fig_attn_weight_dist():
    words = ['동물', '도로', '건너지', '그것은\n(자기 자신)']
    weights = [0.56, 0.01, 0.01, 0.42]
    colors = [KSA_BLUE, KSA_TAN, KSA_TAN, KSA_BLUE]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(words))[::-1]
    bars = ax.barh(y, weights, color=colors, edgecolor='black', linewidth=0.8, height=0.55)
    for yi, w in zip(y, weights):
        ax.text(w + 0.015, yi, f'{w:.2f}', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(words, fontsize=12)
    ax.set_xlim(0, 0.68)
    ax.set_xlabel('어텐션 가중치 (합 = 1)', fontsize=11)
    ax.set_title('"그것은"이 문장의 각 단어에 주는 어텐션 가중치', fontsize=12.5,
                 color=KSA_BLUE, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, 'ch12_1_attention_weight_dist.png')


# ---------------------------------------------------------------
# 3. p18: 세 가지 어텐션 점수 패턴 (대각선 / 특정 열 / 균일)
# ---------------------------------------------------------------
def fig_three_patterns():
    labels = ['w1', 'w2', 'w3', 'w4']
    diag = np.array([[0.85, 0.05, 0.05, 0.05],
                      [0.05, 0.85, 0.05, 0.05],
                      [0.05, 0.05, 0.85, 0.05],
                      [0.05, 0.05, 0.05, 0.85]])
    col = np.array([[0.10, 0.75, 0.10, 0.05],
                     [0.05, 0.80, 0.10, 0.05],
                     [0.10, 0.70, 0.15, 0.05],
                     [0.05, 0.82, 0.08, 0.05]])
    uni = np.full((4, 4), 0.25)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    titles = ['(a) 대각선 압도적\n"자기 자신만 본다"',
              '(b) 특정 열 압도적\n"중심어가 끌어당긴다"',
              '(c) 거의 균일\n"집중 실패"']
    for ax, mat, title in zip(axes, [diag, col, uni], titles):
        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=10, color='white' if v > 0.5 else 'black')
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=11, color=KSA_BLUE, fontweight='bold')
        ax.set_xlabel('Key (열)', fontsize=9.5)
    axes[0].set_ylabel('Query (행)', fontsize=9.5)
    fig.suptitle('예시: 세 가지 대표적인 어텐션 점수 패턴', fontsize=13, color=KSA_BLUE, y=1.04)
    fig.tight_layout()
    savefig(fig, 'ch12_1_three_attn_patterns.png')


# ---------------------------------------------------------------
# 4. p26: 온도(temperature)에 따른 softmax 출력 비교
#    ("그것은" 행의 원점수를 역산: scaled = raw / sqrt(3))
# ---------------------------------------------------------------
def fig_temperature():
    # 12.2절 "실제로 재보기"(d_k=64, 스케일링 전 내적 범위 대략 -19~22)와
    # 같은 자릿수의 예시 원점수 벡터 (특정 문장이 아닌 일반적 예시)
    raw = np.array([15.0, 3.0, -8.0, 10.0])
    words = ['단어 A', '단어 B', '단어 C', '단어 D']

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    temps = {'$t=1$ (스케일링 없음)': 1.0,
             '$t=\\sqrt{d_k}=8$ ($d_k=64$, 책의 선택)': 8.0,
             '$t=d_k=64$ (과도하게 나눔)': 64.0}
    x = np.arange(len(words))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = [RED, KSA_BLUE, KSA_TAN]
    for i, (label, t) in enumerate(temps.items()):
        w = softmax(raw / t)
        ax.bar(x + (i - 1) * width, w, width=width, label=label,
               color=colors[i], edgecolor='black', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(words, fontsize=12)
    ax.set_ylabel('softmax 가중치', fontsize=11)
    ax.set_title('같은 원점수(예시, $d_k=64$ 규모)에 세 온도 $t$를 적용한 softmax',
                 fontsize=12, color=KSA_BLUE, fontweight='bold')
    ax.text(0.5, -0.22, '$t$ 작음 → 한 단어로 쏠림(one-hot)  /  $t$ 큼 → 균등분포에 가까워짐',
            transform=ax.transAxes, ha='center', fontsize=10, color='dimgray')
    ax.legend(fontsize=9.5, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, 'ch12_2_temperature_softmax.png')


# ---------------------------------------------------------------
# 5. p28: 어텐션 출력의 볼록성 (V 원행들의 볼록 껍질)
#    실제 예: E=[[3,0,0],[0,3,0],[0,0,3],[2.8,0.3,0]], 첫 두 성분만 2D 투영
# ---------------------------------------------------------------
def fig_convex_hull():
    from scipy.spatial import ConvexHull
    pts = np.array([[3, 0], [0, 3], [0, 0], [2.8, 0.3]])
    names = ['동물', '도로', '건너지', '그것은']
    hull = ConvexHull(pts)

    out = np.array([2.898, 0.133])   # "동물" 행 어텐션 출력의 (성분1,성분2) 투영
    avg = pts.mean(axis=0)           # 균등(1/4) 어텐션의 출력

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    hull_pts = pts[hull.vertices]
    ax.add_patch(Polygon(hull_pts, closed=True, facecolor=KSA_TAN, alpha=0.5,
                          edgecolor=KSA_BLUE, linewidth=1.8, zorder=1))
    ax.scatter(pts[:, 0], pts[:, 1], s=90, color=KSA_BLUE, zorder=3)
    label_offsets = {'동물': (10, 26), '도로': (10, 8), '건너지': (10, 8), '그것은': (-70, 14)}
    for (x, y), name in zip(pts, names):
        ax.annotate(f'{name} $V$', (x, y), textcoords='offset points',
                    xytext=label_offsets[name], fontsize=11)
    ax.scatter(*out, marker='*', s=320, color=RED, zorder=4, label='실제 출력 (가중치 0.582,0.003,0.003,0.412)')
    ax.scatter(*avg, marker='s', s=90, color='dimgray', zorder=4, label='균등 어텐션(1/4씩) 출력')
    ax.annotate('출력\n(2.898, 0.133)', out, textcoords='offset points', xytext=(-95, -8),
                fontsize=10, color=RED, fontweight='bold', ha='left')
    ax.annotate('균등 평균 (1.45, 0.825)', avg, textcoords='offset points', xytext=(10, 10),
                fontsize=10, color='dimgray')
    ax.set_xlim(-0.5, 4.3); ax.set_ylim(-0.5, 3.7)
    ax.set_xlabel('성분 1', fontsize=10.5); ax.set_ylabel('성분 2', fontsize=10.5)
    ax.set_title('어텐션 출력은 항상 $V$ 원행들의 볼록 껍질 안\n(2D 투영, 실제 4단어 예의 성분1·2)',
                 fontsize=12, color=KSA_BLUE, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout()
    savefig(fig, 'ch12_2_convex_hull.png')


# ---------------------------------------------------------------
# 6. p33: QK^T 행렬에서 행(Query)과 열(Key)의 방향
#    (실제 4단어 예의 어텐션 가중치 행렬 재사용)
# ---------------------------------------------------------------
def fig_row_col():
    words = ['동물', '도로', '건너지', '그것은']
    W = np.array([[0.582, 0.003, 0.003, 0.412],
                  [0.005, 0.980, 0.005, 0.009],
                  [0.005, 0.005, 0.984, 0.005],
                  [0.561, 0.007, 0.004, 0.427]])

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.imshow(W, cmap='Greys', vmin=0, vmax=1, alpha=0.25)
    # 행 강조: "그것은" (i=3)
    ax.add_patch(Rectangle((-0.5, 2.5), 4, 1, fill=True, facecolor=KSA_BLUE, alpha=0.22, zorder=1))
    ax.add_patch(Rectangle((-0.5, 2.5), 4, 1, fill=False, edgecolor=KSA_BLUE, linewidth=2.6, zorder=2))
    # 열 강조: "동물" (j=0)
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 4, fill=True, facecolor=RED, alpha=0.15, zorder=1))
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 4, fill=False, edgecolor=RED, linewidth=2.6, zorder=2))
    for i in range(4):
        for j in range(4):
            v = W[i, j]
            highlight = (i == 3 and j == 0)
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=12 if highlight else 10.5,
                    fontweight='bold' if highlight else 'normal',
                    color='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(words, fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel('Key ( 열 = "보고받는" 단어)', fontsize=11, color=RED)
    ax.set_ylabel('Query (행 = "보고 있는" 단어)', fontsize=11, color=KSA_BLUE)
    ax.set_title('$QK^T$(softmax 후)의 행과 열이 뜻하는 것', fontsize=12.5,
                 color=KSA_BLUE, fontweight='bold')
    ax.text(0, 4.35, '파란 테두리(행) = "그것은"이 각 단어에 주는 주의',
            fontsize=9.5, color=KSA_BLUE, ha='left')
    ax.text(0, 4.75, '빨간 테두리(열) = 각 단어가 "동물"에게 받는 주의',
            fontsize=9.5, color=RED, ha='left')
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(4.9, -0.5)
    fig.tight_layout()
    savefig(fig, 'ch12_2_row_col_direction.png')


# ---------------------------------------------------------------
# 7. p41: Multi-Head Attention 구조도
# ---------------------------------------------------------------
def fig_multihead_arch():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    ax.add_patch(FancyBboxPatch((0.3, 2.6), 1.3, 1.0, boxstyle='round,pad=0.06',
                                 facecolor=KSA_TAN, edgecolor=KSA_BLUE, linewidth=1.6))
    ax.text(0.95, 3.1, '입력 $X$', ha='center', va='center', fontsize=11)

    n_heads = 3
    head_x0 = 2.3
    head_w = 1.5
    gap = 0.35
    ys = [4.6, 2.9, 1.2]
    for i, y in enumerate(ys):
        x = head_x0
        ax.add_patch(FancyBboxPatch((x, y), head_w, 0.9, boxstyle='round,pad=0.05',
                                     facecolor='white', edgecolor=KSA_BLUE, linewidth=1.4))
        ax.text(x + head_w / 2, y + 0.45, f'$Q_{i+1},K_{i+1},V_{i+1}$', ha='center', va='center', fontsize=9.5)
        ax.annotate('', xy=(x, y + 0.45), xytext=(1.6, 3.1),
                     arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.2))
        x2 = x + head_w + gap
        ax.add_patch(FancyBboxPatch((x2, y), 1.9, 0.9, boxstyle='round,pad=0.05',
                                     facecolor=KSA_BLUE, edgecolor=KSA_BLUE, linewidth=1.4, alpha=0.85))
        ax.text(x2 + 0.95, y + 0.45, f'헤드 {i+1}\nAttention', ha='center', va='center',
                fontsize=9.5, color='white')
        ax.annotate('', xy=(x2, y + 0.45), xytext=(x + head_w, y + 0.45),
                     arrowprops=dict(arrowstyle='-|>', color=KSA_BLUE, lw=1.4))
        ax.annotate('', xy=(7.1, 3.1), xytext=(x2 + 1.9, y + 0.45),
                     arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.2))
    ax.text(head_x0 + head_w / 2, 5.7, f'... $h$ 개 헤드 병렬 ...', ha='center', fontsize=10.5, color=RED)

    ax.add_patch(FancyBboxPatch((7.1, 2.6), 1.1, 1.0, boxstyle='round,pad=0.06',
                                 facecolor=KSA_TAN, edgecolor=KSA_BLUE, linewidth=1.6))
    ax.text(7.65, 3.1, 'Concat', ha='center', va='center', fontsize=10.5)

    ax.annotate('', xy=(8.85, 3.1), xytext=(8.2, 3.1),
                arrowprops=dict(arrowstyle='-|>', color=KSA_BLUE, lw=1.6))
    ax.add_patch(FancyBboxPatch((8.85, 2.6), 1.0, 1.0, boxstyle='round,pad=0.06',
                                 facecolor=KSA_TAN, edgecolor=KSA_BLUE, linewidth=1.6))
    ax.text(9.35, 3.1, '$W_O$\n선형', ha='center', va='center', fontsize=9.5)

    fig.suptitle('Multi-Head Attention: $h$개 헤드 병렬 계산 후 Concat + 선형변환',
                 fontsize=12.5, color=KSA_BLUE, fontweight='bold', y=0.99)
    fig.tight_layout()
    savefig(fig, 'ch12_3_multihead_architecture.png')


# ---------------------------------------------------------------
# 8. p56: Causal (masked) attention — 상삼각 마스킹
# ---------------------------------------------------------------
def fig_causal_mask():
    words = ['동물', '도로', '건너지', '그것은']
    n = 4
    mask = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j > i:
                mask[i, j] = -1  # -inf 표시용

    weights = np.zeros((n, n))
    for i in range(n):
        allowed = i + 1
        weights[i, :allowed] = 1.0 / allowed

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))

    ax = axes[0]
    cmap_mask = matplotlib.colors.ListedColormap(['#4a4a4a', KSA_TAN])
    ax.imshow(mask, cmap=cmap_mask, vmin=-1, vmax=0)
    for i in range(n):
        for j in range(n):
            txt = '$-\\infty$' if mask[i, j] < 0 else '$0$'
            ax.text(j, i, txt, ha='center', va='center', fontsize=12,
                    color='white' if mask[i, j] < 0 else 'black')
    ax.set_xticks(range(n)); ax.set_xticklabels(words, fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(words, fontsize=10)
    ax.set_title('(a) 마스크: 상삼각(미래)에 $-\\infty$ 더함', fontsize=11.5, color=KSA_BLUE, fontweight='bold')
    ax.set_xlabel('Key (미래 방향 →)', fontsize=9.5)
    ax.set_ylabel('Query', fontsize=9.5)

    ax = axes[1]
    im = ax.imshow(weights, cmap='Blues', vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            v = weights[i, j]
            ax.text(j, i, f'{v:.2f}' if v > 0 else '0', ha='center', va='center',
                    fontsize=11, color='white' if v > 0.5 else 'black')
    ax.set_xticks(range(n)); ax.set_xticklabels(words, fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(words, fontsize=10)
    ax.set_title('(b) softmax 후: 미래 위치 가중치 = 정확히 0', fontsize=11.5,
                 color=KSA_BLUE, fontweight='bold')
    ax.set_xlabel('Key', fontsize=9.5)

    fig.suptitle('Causal(인과) 마스크 — 디코더는 "왼쪽"(과거)만 본다', fontsize=12.5,
                 color=KSA_BLUE, y=1.03)
    fig.tight_layout()
    savefig(fig, 'ch12_3_causal_mask.png')


if __name__ == '__main__':
    fig_rnn_seq()
    fig_attn_weight_dist()
    fig_three_patterns()
    fig_temperature()
    fig_convex_hull()
    fig_row_col()
    fig_multihead_arch()
    fig_causal_mask()
    print('done')
