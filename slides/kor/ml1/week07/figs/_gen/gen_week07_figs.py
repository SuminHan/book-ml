"""
ml1 week07 (트리 기반 모델) 보강용 개념 도식 생성 스크립트.
결과 PNG는 ../ (즉 DECK_DIR/figs/) 에 저장.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.path import Path
import matplotlib.patches as mpatches
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(__file__), "..")

PRIMARY = "#2E3192"
SECOND = "#D6CBB1"
RED = "#C0392B"
GRAY = "#9A9A9A"
LGRAY = "#DDDDDD"
DARK = "#222222"


def box(ax, cx, cy, w, h, text, fc="white", ec=DARK, tc=DARK, fs=13, lw=1.6, bold=False, zorder=3):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.02",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
             weight=weight, zorder=zorder + 1, linespacing=1.3)
    return b


def arrow(ax, x0, y0, x1, y1, color=DARK, lw=2.0, style="-|>", zorder=2, connectionstyle=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=zorder,
                         connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def new_ax(figsize, xlim=(0, 1), ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


# ------------------------------------------------------------------
# 1) p03: 스무고개 질문 흐름 -> 결정 트리 구조
# ------------------------------------------------------------------
def fig_p03():
    fig, ax = new_ax((12.6, 5.6), (0, 12.6), (0, 5.6))

    # left: dialogue flow
    ax.text(2.3, 5.3, "스무고개: 질문을 하나씩", ha="center", fontsize=15, weight="bold", color=PRIMARY)
    qs = ["“동물인가요?”\n→ 예", "“네 다리로 걷나요?”\n→ 예",
          "“야옹 소리를 내나요?”\n→ 예", "정답: 고양이"]
    ys = [4.45, 3.30, 2.15, 1.00]
    for i, (q, y) in enumerate(zip(qs, ys)):
        fc = SECOND if i < 3 else PRIMARY
        tc = DARK if i < 3 else "white"
        box(ax, 2.3, y, 3.7, 0.85, q, fc=fc, tc=tc, fs=12.5, bold=(i == 3))
        if i < 3:
            arrow(ax, 2.3, y - 0.43, 2.3, ys[i + 1] + 0.43, color=PRIMARY, lw=2.2)

    # middle arrow: 변환
    arrow(ax, 4.6, 2.75, 6.2, 2.75, color=RED, lw=3.0)
    ax.text(5.4, 3.1, "변환", ha="center", fontsize=13, color=RED, weight="bold")

    # right: tree. "예" backbone goes straight down at bx; "아니오" leaves stub
    # out to the left at each level -- keeps the whole tree inside xlim, no drift.
    ax.text(9.3, 5.3, "결정 트리: 같은 질문을 나무로", ha="center", fontsize=15, weight="bold", color=PRIMARY)
    bx = 10.5
    lx = 7.7
    root = (bx, 4.45)
    box(ax, *root, 2.8, 0.72, "동물인가요?", fc="white", ec=PRIMARY, fs=12)

    leaf_a1 = (lx, 4.45)
    box(ax, *leaf_a1, 2.0, 0.62, "식물/사물", fc=LGRAY, ec=GRAY, tc=GRAY, fs=11)
    arrow(ax, root[0] - 1.4, root[1], leaf_a1[0] + 1.0, leaf_a1[1], color=GRAY, lw=1.6)
    ax.text((root[0] - 1.4 + leaf_a1[0] + 1.0) / 2, root[1] + 0.22, "아니오", fontsize=9.5, color=GRAY, ha="center")

    n2 = (bx, 3.30)
    box(ax, *n2, 2.4, 0.72, "네 다리로\n걷나요?", fc="white", ec=PRIMARY, fs=11)
    arrow(ax, root[0], root[1] - 0.36, n2[0], n2[1] + 0.36, color=PRIMARY, lw=2.4)
    ax.text(bx + 0.35, (root[1] - 0.36 + n2[1] + 0.36) / 2, "예", fontsize=10.5, color=PRIMARY, weight="bold")

    leaf_b1 = (lx, 3.30)
    box(ax, *leaf_b1, 2.0, 0.62, "새/물고기 등", fc=LGRAY, ec=GRAY, tc=GRAY, fs=10.5)
    arrow(ax, n2[0] - 1.2, n2[1], leaf_b1[0] + 1.0, leaf_b1[1], color=GRAY, lw=1.6)
    ax.text((n2[0] - 1.2 + leaf_b1[0] + 1.0) / 2, n2[1] + 0.22, "아니오", fontsize=9.5, color=GRAY, ha="center")

    n3 = (bx, 2.15)
    box(ax, *n3, 2.4, 0.72, "야옹 소리를\n내나요?", fc="white", ec=PRIMARY, fs=11)
    arrow(ax, n2[0], n2[1] - 0.36, n3[0], n3[1] + 0.36, color=PRIMARY, lw=2.4)
    ax.text(bx + 0.35, (n2[1] - 0.36 + n3[1] + 0.36) / 2, "예", fontsize=10.5, color=PRIMARY, weight="bold")

    leaf_c1 = (lx, 2.15)
    box(ax, *leaf_c1, 1.8, 0.62, "개 등", fc=LGRAY, ec=GRAY, tc=GRAY, fs=11)
    arrow(ax, n3[0] - 1.2, n3[1], leaf_c1[0] + 0.9, leaf_c1[1], color=GRAY, lw=1.6)
    ax.text((n3[0] - 1.2 + leaf_c1[0] + 0.9) / 2, n3[1] + 0.22, "아니오", fontsize=9.5, color=GRAY, ha="center")

    leaf_cat = (bx, 1.00)
    box(ax, *leaf_cat, 1.9, 0.72, "고양이", fc=PRIMARY, tc="white", fs=13, bold=True)
    arrow(ax, n3[0], n3[1] - 0.36, leaf_cat[0], leaf_cat[1] + 0.36, color=RED, lw=2.6)
    ax.text(bx + 0.32, (n3[1] - 0.36 + leaf_cat[1] + 0.36) / 2, "예", fontsize=10.5, color=RED, weight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_1_20q_to_tree.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 2) p05: 경로 = 설명 (대출 심사 예시, 루트~리프 경로 강조)
# ------------------------------------------------------------------
def fig_p05():
    fig, ax = new_ax((8.6, 5.6), (0, 8.6), (0, 5.6))
    ax.text(4.3, 5.3, "경로 = 설명: 새 신청자 한 명을 따라가면", ha="center",
            fontsize=15, weight="bold", color=PRIMARY)

    root = (4.3, 4.35)
    box(ax, *root, 3.6, 0.85, "소득 $<$ 3000만원?", fc="white", ec=RED, lw=2.6, fs=13)

    leaf_r = (6.6, 3.1)
    box(ax, *leaf_r, 2.6, 0.75, "아니오 → 승인", fc=LGRAY, ec=GRAY, tc=GRAY, fs=11.5)
    a0 = (root[0] + 0.6, root[1] - 0.50)
    a1 = (leaf_r[0] - 0.1, leaf_r[1] + 0.45)
    arrow(ax, *a0, *a1, color=GRAY, lw=1.6)
    ax.text((a0[0] + a1[0]) / 2 + 0.05, (a0[1] + a1[1]) / 2 - 0.28, "아니오", fontsize=10, color=GRAY)

    n2 = (2.3, 3.1)
    box(ax, *n2, 3.0, 0.85, "연령 $>$ 45세?", fc="white", ec=RED, lw=2.6, fs=13)
    b0 = (root[0] - 0.7, root[1] - 0.50)
    b1 = (n2[0] + 0.4, n2[1] + 0.50)
    arrow(ax, *b0, *b1, color=RED, lw=3.0)
    ax.text((b0[0] + b1[0]) / 2 - 0.55, (b0[1] + b1[1]) / 2 - 0.05, "예", fontsize=12, color=RED, weight="bold")

    leaf2a = (0.95, 1.7)
    box(ax, *leaf2a, 1.8, 0.75, "아니오 → 승인", fc=LGRAY, ec=GRAY, tc=GRAY, fs=10.5)
    c0 = (n2[0] - 0.6, n2[1] - 0.50)
    c1 = (leaf2a[0] + 0.15, leaf2a[1] + 0.45)
    arrow(ax, *c0, *c1, color=GRAY, lw=1.6)
    ax.text((c0[0] + c1[0]) / 2 - 0.10, (c0[1] + c1[1]) / 2 - 0.28, "아니오", fontsize=9.5, color=GRAY)

    leaf2b = (3.15, 1.55)
    box(ax, *leaf2b, 1.9, 0.85, "예 → 거절", fc=RED, tc="white", fs=13, bold=True)
    d0 = (n2[0] + 0.35, n2[1] - 0.50)
    d1 = (leaf2b[0] - 0.15, leaf2b[1] + 0.50)
    arrow(ax, *d0, *d1, color=RED, lw=3.2)
    ax.text((d0[0] + d1[0]) / 2 + 0.35, (d0[1] + d1[1]) / 2 - 0.05, "예", fontsize=12, color=RED, weight="bold")

    ax.text(4.3, 0.45,
            "경로: 소득 $<$ 3000만원 → 연령 $>$ 45세 → 거절 --- 이 경로를 그대로 읽어주면 설명이 끝난다",
            ha="center", fontsize=11.5, color=DARK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_1_path_explanation.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 3) p24: 다수결 = 여러 트리의 평균이 안정적 (책 7.2절 실측)
# ------------------------------------------------------------------
def fig_p24():
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    trees = [0.871, 0.918, 0.895, 0.889, 0.883]
    mean_t = np.mean(trees)
    std_t = np.std(trees)
    rf, rf_std = 0.959, 0.0068

    xs = list(range(5))
    ax.bar(xs, trees, color=SECOND, edgecolor=DARK, width=0.62, zorder=3)
    ax.axhline(mean_t, color=GRAY, linestyle="--", lw=1.6, zorder=2)
    ax.text(2.0, mean_t + 0.006, f"5개 평균 {mean_t*100:.1f}% ($\\sigma$={std_t*100:.2f}%p)",
            fontsize=10.5, color=GRAY, ha="center", va="bottom")

    xrf = 6.1
    ax.bar([xrf], [rf], color=PRIMARY, edgecolor=DARK, width=0.62, zorder=3,
           yerr=[[rf_std], [rf_std]], capsize=6, ecolor=DARK)
    ax.text(xrf, rf + rf_std + 0.006, f"랜덤 포레스트\n(100개, 다수결)\n{rf*100:.1f}%",
            ha="center", va="bottom", fontsize=11, color=PRIMARY, weight="bold")

    for x, v in zip(xs, trees):
        ax.text(x, v - 0.015, f"{v*100:.1f}", ha="center", va="top", fontsize=10.5, color=DARK)

    ax.set_xticks(xs + [xrf])
    ax.set_xticklabels([f"tree{i}" for i in range(5)] + ["RF"], fontsize=11.5)
    ax.set_xlim(-0.65, 6.95)
    ax.set_ylabel("test 정확도", fontsize=12.5)
    ax.set_ylim(0.80, 1.0)
    ax.set_title("전문가 5명(개별 트리) vs 다수결(랜덤 포레스트) --- 유방암 데이터", fontsize=13.5, color=PRIMARY, pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_2_ensemble_vote.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 4) p25: 두 가지 무작위성 (배깅 + 특징 무작위화)
# ------------------------------------------------------------------
def fig_p25():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=200)
    fig.patch.set_facecolor("white")

    # --- left: bagging (bootstrap resampling of samples) ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(5, 5.3, "1. 배깅: 데이터를 복원추출", ha="center", fontsize=14, weight="bold", color=PRIMARY)
    orig = list(range(1, 9))
    for i, v in enumerate(orig):
        cx = 1.2 + i * 1.05
        box(ax, cx, 4.4, 0.82, 0.55, str(v), fc="white", ec=DARK, fs=11)
    ax.text(0.15, 4.4, "원본", fontsize=11, color=DARK, ha="right", va="center")

    rng = np.random.RandomState(0)
    samples = [sorted(rng.choice(orig, size=8, replace=True)) for _ in range(3)]
    labels = ["Tree 1 데이터", "Tree 2 데이터", "Tree 3 데이터"]
    for r, (samp, lab) in enumerate(zip(samples, labels)):
        y = 3.35 - r * 1.15
        ax.text(0.15, y, lab, fontsize=10.5, color=PRIMARY, ha="right", va="center")
        for i, v in enumerate(samp):
            cx = 1.2 + i * 1.05
            box(ax, cx, y, 0.82, 0.5, str(v), fc=SECOND, ec=PRIMARY, fs=10.5)
    ax.text(5, 0.25, "→ 트리마다 조금씩 다른 데이터를 봄 (중복 허용)", ha="center", fontsize=10.5, color=DARK)

    # --- right: feature randomization ---
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(5, 5.3, "2. 특징 무작위화: 분기마다 일부 특징만", ha="center", fontsize=14, weight="bold", color=PRIMARY)
    feats = [f"f{i}" for i in range(1, 7)]
    for i, f in enumerate(feats):
        cx = 1.3 + i * 1.35
        box(ax, cx, 4.4, 1.05, 0.55, f, fc="white", ec=DARK, fs=11.5)
    ax.text(0.15, 4.4, "전체\n특징", fontsize=10, color=DARK, ha="right", va="center")

    picks = [(1, 4), (0, 3), (2, 5)]
    labels2 = ["분기 A 후보", "분기 B 후보", "분기 C 후보"]
    for r, (pick, lab) in enumerate(zip(picks, labels2)):
        y = 3.35 - r * 1.15
        ax.text(0.15, y, lab, fontsize=10.5, color=PRIMARY, ha="right", va="center")
        for i, f in enumerate(feats):
            cx = 1.3 + i * 1.35
            if i in pick:
                box(ax, cx, y, 1.05, 0.5, f, fc=RED, tc="white", ec=RED, fs=11, bold=True)
            else:
                box(ax, cx, y, 1.05, 0.5, f, fc=LGRAY, ec=LGRAY, tc=GRAY, fs=10.5)
    ax.text(5, 0.25, "→ 같은 데이터를 봐도 매번 다른 질문을 탐색", ha="center", fontsize=10.5, color=DARK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_2_two_randomness.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 5) p33: OOB (out-of-bag) 다이어그램
# ------------------------------------------------------------------
def fig_p33():
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    sizes = [63, 37]
    colors = [SECOND, RED]
    startangle = 90
    wedges, _ = ax.pie(sizes, colors=colors, startangle=startangle, counterclock=False,
                        wedgeprops=dict(edgecolor="white", linewidth=3), radius=1.15)
    labels = ["63%\n이 트리가\n학습에 사용\n(in-bag)", "37%\nOOB\n(학습에\n안 쓰임)"]
    tcolors = [DARK, "white"]
    for w, lab, tc in zip(wedges, labels, tcolors):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        r = 0.62
        ax.text(r * np.cos(ang), r * np.sin(ang), lab, ha="center", va="center",
                fontsize=12, color=tc, weight="bold")
    ax.set_title("부트스트랩 샘플 1개 = in-bag 63% + OOB 37%", fontsize=13.5, color=PRIMARY, pad=16)
    w_oob = wedges[1]
    ang_oob = np.deg2rad((w_oob.theta1 + w_oob.theta2) / 2)
    ax.annotate("→ 이 37%로 즉시 검증\n(별도 validation set 불필요)",
                xy=(1.16 * np.cos(ang_oob), 1.16 * np.sin(ang_oob)),
                xytext=(-2.05, 0.05), fontsize=11.5, color=DARK, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8,
                                 connectionstyle="arc3,rad=0.0"))
    ax.set_xlim(-2.2, 1.6)
    ax.set_ylim(-1.5, 1.6)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_2_oob_diagram.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 6) p40: 랜덤 포레스트(병렬) vs GBDT(순차) 조직 구조 비교
# ------------------------------------------------------------------
def fig_p40():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), dpi=200)
    fig.patch.set_facecolor("white")

    # top: RF parallel
    ax = axes[0]
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.text(0.2, 2.85, "랜덤 포레스트 --- 병렬 / 독립", fontsize=14.5, weight="bold", color=PRIMARY, ha="left")
    data = (1.0, 1.5)
    box(ax, *data, 1.4, 0.9, "데이터", fc="white", ec=DARK, fs=12)
    tree_xs = [3.6, 5.6, 7.6]
    tree_labels = ["Tree 1", "Tree 2", "⋯ Tree 100"]
    for tx, tl in zip(tree_xs, tree_labels):
        box(ax, tx, 2.35, 1.55, 0.75, tl, fc=SECOND, ec=PRIMARY, fs=11)
        arrow(ax, data[0] + 0.6, data[1] + 0.3, tx - 0.7, 2.2, color=PRIMARY, lw=1.8,
              connectionstyle="arc3,rad=-0.15")
        arrow(ax, tx, 1.97, 9.6, 1.65, color=PRIMARY, lw=1.8, connectionstyle="arc3,rad=0.1")
    box(ax, 10.7, 1.5, 1.9, 0.9, "다수결", fc=PRIMARY, tc="white", fs=12.5, bold=True)
    ax.text(6, 0.5, "각 트리는 서로 독립적으로 전체를 처음부터 학습 → 마지막에 투표로 합침",
            ha="center", fontsize=11, color=DARK)

    # bottom: GBDT sequential
    ax = axes[1]
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.text(0.2, 2.85, "GBDT --- 순차 / 잔차 보정", fontsize=14.5, weight="bold", color=PRIMARY, ha="left")
    xs = [0.9, 3.0, 5.6, 8.2, 10.6]
    labels = ["데이터", "Tree 1", "잔차 → Tree 2", "잔차 → Tree 3", "⋯ 예측합"]
    widths = [1.3, 1.3, 2.0, 2.0, 1.7]
    for x, l, w in zip(xs, labels, widths):
        fc = RED if "Tree" in l or "예측" in l else "white"
        tc = "white" if fc == RED else DARK
        box(ax, x, 1.6, w, 0.85, l, fc=fc, ec=RED if fc == RED else DARK, tc=tc, fs=10.8)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + widths[i] / 2 + 0.05, 1.6, xs[i + 1] - widths[i + 1] / 2 - 0.05, 1.6,
              color=DARK, lw=2.2)
    ax.text(6, 0.5, "새 트리는 매번 “지금까지 못 맞힌 부분(잔차)”만 목표로 학습",
            ha="center", fontsize=11, color=DARK)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_3_rf_vs_gbdt_structure.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 7) p51: RF vs GBDT 학습 곡선 개념도 (early stopping)
# ------------------------------------------------------------------
def fig_p51():
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    x = np.linspace(1, 100, 400)
    rf = 0.75 / (x ** 0.55) + 0.10
    rise = np.where(x > 12, 0.00006 * np.clip(x - 12, 0, None) ** 1.6, 0.0)
    gbdt = 0.55 / (x ** 0.5) + 0.09 + rise

    ax.plot(x, rf, color=PRIMARY, lw=3, label="랜덤 포레스트 (RF) --- 정체만, 안 올라감")
    ax.plot(x, gbdt, color=RED, lw=3, label="GBDT --- 정점을 지나면 다시 상승")

    i_min = int(np.argmin(gbdt))
    ax.axvline(x[i_min], color=GRAY, linestyle="--", lw=1.6)
    ax.scatter([x[i_min]], [gbdt[i_min]], color=RED, zorder=5, s=60)
    ax.annotate("early stopping\n(정점에서 멈춤)", xy=(x[i_min], gbdt[i_min]),
                xytext=(x[i_min] + 22, gbdt[i_min] + 0.10), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))

    ax.set_xlabel("트리 개수 (n\\_estimators)", fontsize=12.5)
    ax.set_ylabel("검증 오차 (validation error)", fontsize=12.5)
    ax.set_title("개념도: RF는 평탄, GBDT는 정점 후 급상승", fontsize=13.5, color=PRIMARY, pad=12)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_ylim(0, max(rf.max(), gbdt.max()) * 1.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_3_rf_vs_gbdt_curve.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------
# 8) p52: SHAP 가산 분해 개념도 (일반 예시, 실제 수치 아님)
# ------------------------------------------------------------------
def fig_p52():
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    base = 0.30
    contribs = [("$f_A$", +0.15), ("$f_B$", -0.05), ("$f_C$", +0.08), ("$f_D$", +0.02)]
    cum = base
    xs = [0]
    vals = [base]
    ax.barh(0, base, left=0, color=LGRAY, edgecolor=DARK, height=0.55)
    ax.text(base / 2, 0, f"base\n{base:.2f}", ha="center", va="center", fontsize=10.5, color=DARK)

    y = 0
    left = base
    for i, (name, v) in enumerate(contribs):
        y = i + 1
        color = PRIMARY if v > 0 else RED
        ax.barh(y, abs(v), left=left if v > 0 else left + v, color=color, edgecolor=DARK, height=0.55)
        ax.text(left + v / 2, y, f"{name}\n{v:+.2f}", ha="center", va="center", fontsize=10.5, color="white")
        newleft = left + v
        ax.plot([left, left], [y - 0.5, y + 0.5], color=GRAY, lw=1.0, linestyle=":")
        left = newleft
        cum = left

    ax.barh(len(contribs) + 1, cum, left=0, color=PRIMARY, edgecolor=DARK, height=0.6)
    ax.text(cum / 2, len(contribs) + 1, f"f(x) 예측값\n{cum:.2f}", ha="center", va="center",
            fontsize=11, color="white", weight="bold")

    ax.set_yticks(range(len(contribs) + 2))
    ax.set_yticklabels(["기준값 E[f(x)]"] + [f"{n} 반영" for n, _ in contribs] + ["최종 예측"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(cum, base) + 0.15)
    ax.set_xlabel("모델 출력값", fontsize=12)
    ax.set_title("SHAP 개념도: 기준값 + 특징별 기여의 합 = 예측값 (일반 예시)", fontsize=12.8, color=PRIMARY, pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ch07_3_shap_concept.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_p03()
    fig_p05()
    fig_p24()
    fig_p25()
    fig_p33()
    fig_p40()
    fig_p51()
    fig_p52()
    print("done")
