"""
Week10 (CNN) 개념 도식 8종 생성 스크립트.
실행: {FIGPY} gen_figs.py
출력: ../<이름>.png (DECK_DIR/figs/)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#D6CBB1"
GOLDDK = "#A9976B"
RED = "#C0392B"
GRAY = "#5B5B5B"

OUTDIR = os.path.join(os.path.dirname(__file__), "..")


def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------
# p03g1: 이미지를 벡터로 펼치기 + 위치 이동 시 벡터가 완전히 달라짐
# ---------------------------------------------------------------
def fig_p03():
    fig, axes = plt.subplots(1, 1, figsize=(11, 4.6))
    ax = axes
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def draw_grid(x0, y0, cell, mat, highlight=None):
        n = mat.shape[0]
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                color = "white"
                if highlight is not None and highlight[i, j]:
                    color = GOLD
                x = x0 + j * cell
                y = y0 + (n - 1 - i) * cell
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                        edgecolor=BLUE, linewidth=1.2))

    n = 4
    cell = 0.5
    base = np.zeros((4, 4), dtype=int)
    base[:, 1] = 1  # 세로선이 열1 위치
    shifted = np.zeros((4, 4), dtype=int)
    shifted[:, 2] = 1  # 세로선이 열2로 한 칸 이동

    hi_base = (base == 1)
    hi_shift = (shifted == 1)

    # 왼쪽: 원본 이미지 -> 펼치기 -> 벡터
    gx, gy = 0.3, 2.5
    draw_grid(gx, gy, cell, base, hi_base)
    ax.text(gx + n * cell / 2, gy + n * cell + 0.25, "원본 이미지 (4$\\times$4)",
             ha="center", fontsize=13, color=BLUE, fontweight="bold")

    ax.annotate("", xy=(gx + n * cell + 0.9, gy + n * cell / 2),
                xytext=(gx + n * cell + 0.15, gy + n * cell / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax.text(gx + n * cell + 0.5, gy + n * cell / 2 + 0.28, "펼치기",
             ha="center", fontsize=11, color=GRAY)

    vec = base.flatten()
    vx0 = gx + n * cell + 1.1
    vy0 = gy + n * cell - 0.15
    for k, v in enumerate(vec):
        col = GOLD if v == 1 else "white"
        ax.add_patch(Rectangle((vx0 + k * cell, vy0 - cell), cell, cell,
                                facecolor=col, edgecolor=BLUE, linewidth=1.0))
    ax.text(vx0 + n * n * cell / 2, vy0 + 0.3, "길이 16 벡터", ha="center",
             fontsize=11, color=BLUE)

    # 오른쪽: 이동된 이미지 -> 펼치기 -> 벡터 (다른 위치가 붉게 강조)
    gx2, gy2 = 0.3, 0.1
    draw_grid(gx2, gy2, cell, shifted, hi_shift)
    ax.text(gx2 + n * cell / 2, gy2 + n * cell + 0.25,
             "1칸 오른쪽으로 이동", ha="center", fontsize=13, color=RED,
             fontweight="bold")

    ax.annotate("", xy=(gx2 + n * cell + 0.9, gy2 + n * cell / 2),
                xytext=(gx2 + n * cell + 0.15, gy2 + n * cell / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax.text(gx2 + n * cell + 0.5, gy2 + n * cell / 2 + 0.28, "펼치기",
             ha="center", fontsize=11, color=GRAY)

    vec2 = shifted.flatten()
    vy02 = gy2 + n * cell - 0.15
    for k, (v0, v1) in enumerate(zip(vec, vec2)):
        diff = (v0 != v1)
        col = RED if diff else ("white" if v1 == 0 else GOLD)
        ax.add_patch(Rectangle((vx0 + k * cell, vy02 - cell), cell, cell,
                                facecolor=col, edgecolor=BLUE, linewidth=1.0))
    ax.text(vx0 + n * n * cell / 2, vy02 + 0.3,
             "16개 중 8개 원소가 바뀜", ha="center",
             fontsize=11, color=RED)

    savefig(fig, "gen_flatten_shift.png")


# ---------------------------------------------------------------
# p04g1: 필터가 입력 위를 슬라이드하며 원소별 곱의 합을 계산
# ---------------------------------------------------------------
def fig_p04():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    cell = 0.62
    inp = np.array([
        [0, 1, 1, 0, 2],
        [1, 2, 0, 1, 0],
        [0, 0, 3, 1, 0],
        [1, 1, 0, 2, 1],
        [0, 2, 1, 0, 1],
    ])
    x0, y0 = 0.3, 0.5
    n = 5
    for i in range(n):
        for j in range(n):
            x = x0 + j * cell
            y = y0 + (n - 1 - i) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor="white",
                                    edgecolor=GRAY, linewidth=1.0))
            ax.text(x + cell / 2, y + cell / 2, str(inp[i, j]), ha="center",
                     va="center", fontsize=10, color=GRAY)

    # 필터 위치 (파란 테두리, 현재 위치는 행1 열1부터 3x3)
    fi, fj = 1, 1
    fx = x0 + fj * cell
    fy = y0 + (n - 1 - (fi + 2)) * cell
    ax.add_patch(Rectangle((fx, fy), 3 * cell, 3 * cell, facecolor="none",
                            edgecolor=BLUE, linewidth=3))
    ax.text(x0 + n * cell / 2, y0 + n * cell + 0.3, "입력 (5$\\times$5)",
             ha="center", fontsize=13, color=BLUE, fontweight="bold")

    # 화살표: 다른 위치들도 스캔함을 점선 사각형으로 표시
    for (di, dj) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        if (di, dj) == (1, 1):
            continue
        gx = x0 + dj * cell
        gy = y0 + (n - 1 - (di + 2)) * cell
        ax.add_patch(Rectangle((gx, gy), 3 * cell, 3 * cell, facecolor="none",
                                edgecolor=GOLDDK, linewidth=1.3, linestyle="--"))

    # 필터 3x3 박스 (오른쪽)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kx0 = x0 + n * cell + 1.3
    ky0 = y0 + 1.0
    kcell = 0.62
    for i in range(3):
        for j in range(3):
            x = kx0 + j * kcell
            y = ky0 + (2 - i) * kcell
            ax.add_patch(Rectangle((x, y), kcell, kcell, facecolor=GOLD,
                                    edgecolor=BLUE, linewidth=1.2))
            ax.text(x + kcell / 2, y + kcell / 2, str(kernel[i, j]),
                     ha="center", va="center", fontsize=10, color=BLUE)
    ax.text(kx0 + 1.5 * kcell, ky0 + 3 * kcell + 0.3, "필터 (3$\\times$3)",
             ha="center", fontsize=13, color=BLUE, fontweight="bold")

    # 연산 표시
    ax.text(kx0 + 1.5 * kcell, ky0 - 0.5,
             "$\\sum_{{d_i,d_j}} \\text{{image}}[i{+}d_i][j{+}d_j]\\cdot k[d_i][d_j]$",
             ha="center", fontsize=12, color=GRAY)

    # 화살표: 필터 -> 출력
    win = inp[fi:fi + 3, fj:fj + 3]
    val = int((win * kernel).sum())
    ox0 = kx0 + 3 * kcell + 1.1
    oy0 = ky0 + 0.9
    ax.annotate("", xy=(ox0 - 0.1, oy0 + 0.3), xytext=(kx0 + 3 * kcell + 0.15, ky0 + 0.9),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.add_patch(Rectangle((ox0, oy0), 0.9, 0.9, facecolor="white",
                            edgecolor=BLUE, linewidth=1.6))
    ax.text(ox0 + 0.45, oy0 + 0.45, str(val), ha="center", va="center",
             fontsize=14, color=BLUE, fontweight="bold")
    ax.text(ox0 + 0.45, oy0 + 1.25, "출력 지도의\n한 원소", ha="center",
             fontsize=10, color=GRAY)

    ax.text(6.2, 0.05,
             "파란 실선 = 현재 위치, 금색 점선 = 필터가 슬라이드할 다른 위치들",
             ha="center", fontsize=10, color=GRAY)

    savefig(fig, "gen_conv_slide.png")


# ---------------------------------------------------------------
# p08g1: 5x5 -> 3x3, 파라미터 공유(같은 필터가 모든 위치에서 재사용)
# ---------------------------------------------------------------
def fig_p08():
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    cell = 0.58
    n = 5
    inp = np.zeros((5, 5), dtype=int)
    inp[:, 2] = 1  # 열2에 세로선
    x0, y0 = 0.3, 0.6
    for i in range(n):
        for j in range(n):
            x = x0 + j * cell
            y = y0 + (n - 1 - i) * cell
            col = GOLD if inp[i, j] == 1 else "white"
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=col,
                                    edgecolor=GRAY, linewidth=1.0))
    ax.text(x0 + n * cell / 2, y0 + n * cell + 0.3, "입력 5$\\times$5 (열2에 세로선)",
             ha="center", fontsize=12.5, color=BLUE, fontweight="bold")

    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    out = np.zeros((3, 3), dtype=int)
    for oi in range(3):
        for oj in range(3):
            win = inp[oi:oi + 3, oj:oj + 3]
            out[oi, oj] = int((win * kernel).sum())

    # 필터 3곳 위치(대각선 세 곳)에 같은 색 테두리로 "같은 필터"를 표시
    positions = [(0, 0), (1, 1), (2, 2)]
    colors = [BLUE, RED, GOLDDK]
    for (fi, fj), c in zip(positions, colors):
        fx = x0 + fj * cell
        fy = y0 + (n - 1 - (fi + 2)) * cell
        ax.add_patch(Rectangle((fx, fy), 3 * cell, 3 * cell, facecolor="none",
                                edgecolor=c, linewidth=2.4))

    # 화살표: 세 위치가 모두 -> 같은 필터 박스로 (파라미터 공유)
    kx0 = x0 + n * cell + 1.5
    ky0 = y0 + 1.7
    kcell = 0.55
    for i in range(3):
        for j in range(3):
            x = kx0 + j * kcell
            y = ky0 + (2 - i) * kcell
            col = GOLD if kernel[i, j] else "white"
            ax.add_patch(Rectangle((x, y), kcell, kcell, facecolor=col,
                                    edgecolor=BLUE, linewidth=1.2))
    ax.add_patch(Rectangle((kx0 - 0.08, ky0 - 0.08), 3 * kcell + 0.16,
                            3 * kcell + 0.16, facecolor="none",
                            edgecolor="black", linewidth=1.0, linestyle=":"))
    ax.text(kx0 + 1.5 * kcell, ky0 + 3 * kcell + 0.35,
             "하나의 필터 (3$\\times$3)", ha="center", fontsize=12.5,
             color=BLUE, fontweight="bold")

    for (fi, fj), c in zip(positions, colors):
        fx = x0 + fj * cell + 1.5 * cell
        fy = y0 + (n - 1 - (fi + 2)) * cell + 1.5 * cell
        ax.annotate("", xy=(kx0 + 1.5 * kcell - 0.55, ky0 + 1.5 * kcell),
                     xytext=(fx, fy),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, alpha=0.85,
                                      connectionstyle="arc3,rad=0.15"))

    # 출력 3x3
    ox0 = kx0 + 3 * kcell + 1.5
    oy0 = y0 + 1.7
    for i in range(3):
        for j in range(3):
            x = ox0 + j * kcell
            y = oy0 + (2 - i) * kcell
            c = colors[i] if i == j else "white"
            ax.add_patch(Rectangle((x, y), kcell, kcell, facecolor=c if i == j else "white",
                                    edgecolor=BLUE, linewidth=1.2, alpha=0.9 if i == j else 1))
            ax.text(x + kcell / 2, y + kcell / 2, str(out[i, j]), ha="center",
                     va="center", fontsize=9,
                     color="white" if i == j else GRAY)
    ax.text(ox0 + 1.5 * kcell, oy0 + 3 * kcell + 0.35, "출력 3$\\times$3",
             ha="center", fontsize=12.5, color=BLUE, fontweight="bold")

    ax.text(6.3, 0.1,
             "세 위치(파랑·빨강·황토) 모두 같은 필터를 재사용 --- 위치마다 새로 배우지 않는다",
             ha="center", fontsize=10.5, color=GRAY)

    savefig(fig, "gen_shared_kernel.png")


# ---------------------------------------------------------------
# p25g1: 최대풀링 - 원본과 1픽셀 이동 버전, 풀링 결과 동일 (교재 수치 그대로)
# ---------------------------------------------------------------
def fig_p25():
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    orig = np.array([[1, 0, 2, 1], [1, 0, 3, 1], [1, 0, 4, 1], [1, 0, 2, 1]])
    shift = np.array([[1, 1, 0, 2], [1, 1, 0, 3], [1, 1, 0, 4], [1, 1, 0, 2]])
    pool_colors = [[BLUE, RED], [GOLDDK, "#3C7A54"]]

    def draw(mat, x0, y0, cell, title, title_color):
        n = 4
        for i in range(n):
            for j in range(n):
                x = x0 + j * cell
                y = y0 + (n - 1 - i) * cell
                pc = pool_colors[i // 2][j // 2]
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="white",
                                        edgecolor=pc, linewidth=2.2))
                ax.text(x + cell / 2, y + cell / 2, str(mat[i, j]), ha="center",
                         va="center", fontsize=13, color=GRAY)
        # 2x2 풀링 셀 경계 강조
        for pi in range(2):
            for pj in range(2):
                px = x0 + pj * 2 * cell
                py = y0 + (n - 1 - (pi * 2 + 1)) * cell
                ax.add_patch(Rectangle((px, py), 2 * cell, 2 * cell, facecolor="none",
                                        edgecolor=pool_colors[pi][pj], linewidth=3.2))
        ax.text(x0 + n * cell / 2, y0 + n * cell + 0.35, title, ha="center",
                 fontsize=13, color=title_color, fontweight="bold")

    cell = 0.62
    fig2, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    draw(orig, 0.3, 0.6, cell, "원본 (\"세로선\" 열1)", BLUE)
    draw(shift, 6.0, 0.6, cell, "1픽셀 오른쪽 이동 (열2)", RED)

    def draw_pool_out(mat, x0, y0, cell):
        pooled = np.array([[mat[0:2, 0:2].max(), mat[0:2, 2:4].max()],
                            [mat[2:4, 0:2].max(), mat[2:4, 2:4].max()]])
        for i in range(2):
            for j in range(2):
                x = x0 + j * cell
                y = y0 + (1 - i) * cell
                pc = pool_colors[i][j]
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor=pc,
                                        edgecolor=pc, linewidth=1.5, alpha=0.85))
                ax.text(x + cell / 2, y + cell / 2, str(pooled[i, j]), ha="center",
                         va="center", fontsize=13, color="white", fontweight="bold")
        return pooled

    ax.annotate("", xy=(3.15, 2.4), xytext=(2.65, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax.text(2.9, 4.55, "2$\\times$2\n최대풀링", ha="center", fontsize=10, color=GRAY)
    p1 = draw_pool_out(orig, 3.35, 1.55, cell)

    ax.annotate("", xy=(9.0, 2.4), xytext=(8.5, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax.text(8.75, 4.55, "2$\\times$2\n최대풀링", ha="center", fontsize=10, color=GRAY)
    draw_pool_out(shift, 9.2, 1.55, cell)

    ax.annotate("", xy=(11.0, 2.05), xytext=(10.55, 2.05),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.6))
    ax.text(10.75, 1.6, "동일", ha="center", fontsize=12, color="black", fontweight="bold")

    ax.text(6.4, 0.05,
             "네 풀링 영역의 최댓값 $(1,3,1,4)$ --- 이동 전후 완전히 동일",
             ha="center", fontsize=11, color=GRAY)

    plt.close(fig)
    savefig(fig2, "gen_maxpool_invariance.png")


# ---------------------------------------------------------------
# p27g1: 풀링 횟수에 따른 공간 크기·연산량 감소
# ---------------------------------------------------------------
def fig_p27():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    pools = [0, 1, 2, 3]
    spatial = [1, 1 / 2, 1 / 4, 1 / 8]
    flops = [1, 1 / 4, 1 / 16, 1 / 64]

    width = 0.32
    xs = np.arange(len(pools))
    ax.bar(xs - width / 2, spatial, width, label="공간 크기 비율 ($n/n_0$)",
           color=BLUE)
    ax.bar(xs + width / 2, flops, width, label="연산량(FLOPs) 비율",
           color=GOLD, edgecolor=GOLDDK)

    for x, s, f in zip(xs, spatial, flops):
        ax.text(x - width / 2, s + 0.02, f"{s:.3g}", ha="center", fontsize=10,
                 color=BLUE)
        ax.text(x + width / 2, f + 0.02, f"{f:.3g}", ha="center", fontsize=10,
                 color=GOLDDK)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p}회 풀링" for p in pools], fontsize=12)
    ax.set_ylabel("원본 대비 비율", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("2$\\times$2 풀링 반복 시 해상도·연산량 감소", fontsize=13.5,
                  color=BLUE, fontweight="bold")
    ax.legend(fontsize=10.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    savefig(fig, "gen_pooling_cost.png")


# ---------------------------------------------------------------
# p35g1: LeNet-5 C5(1x1x120) 를 펼쳐서 F6 입력으로
# ---------------------------------------------------------------
def fig_p35():
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.9, 4.5)
    ax.axis("off")

    # C5: 1x1x120 특징지도 (세로로 쌓인 얇은 막대들로 표현, 축약해서 20개만 그림)
    cx0, cy0 = 0.6, 0.6
    n_show = 20
    for k in range(n_show):
        y = cy0 + k * 0.155
        ax.add_patch(Rectangle((cx0, y), 1.0, 0.14, facecolor=BLUE,
                                edgecolor="white", linewidth=0.4, alpha=0.85))
    ax.text(cx0 + 0.5, cy0 + n_show * 0.155 + 0.35, "C5\n(1$\\times$1$\\times$120)",
             ha="center", fontsize=12, color=BLUE, fontweight="bold")
    ax.text(cx0 + 0.5, cy0 - 0.22, "공간 1$\\times$1, 채널 120개", ha="center",
             fontsize=9.5, color=GRAY)

    ax.annotate("", xy=(3.1, 2.4), xytext=(1.9, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.2))
    ax.text(2.5, 2.75, "펼치기\n(flatten)", ha="center", fontsize=10.5, color=GRAY)

    # 펼쳐진 벡터 (가로로 120개 중 20개만 표시 + 생략기호)
    vx0, vy0 = 3.3, 2.25
    for k in range(n_show):
        x = vx0 + k * 0.16
        ax.add_patch(Rectangle((x, vy0), 0.14, 0.32, facecolor=GOLD,
                                edgecolor=BLUE, linewidth=0.5))
    ax.text(vx0 + n_show * 0.16 + 0.35, vy0 + 0.16, "$\\cdots$", fontsize=14,
             color=GRAY, va="center")
    ax.text(vx0 + n_show * 0.08, vy0 + 0.75, "펼쳐진 벡터 (길이 120)", ha="center",
             fontsize=11.5, color=BLUE, fontweight="bold")

    ax.annotate("", xy=(9.4, 2.4), xytext=(8.5, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.2))
    ax.text(8.95, 2.75, "완전연결", ha="center", fontsize=10.5, color=GRAY)

    # F6 layer as small circles
    fx0, fy0 = 9.6, 1.4
    for k in range(6):
        cy = fy0 + k * 0.32
        ax.add_patch(plt.Circle((fx0, cy), 0.13, facecolor=BLUE, edgecolor="white"))
    ax.text(fx0, fy0 + 6 * 0.32 + 0.15, "F6 (84개)", ha="center", fontsize=11.5,
             color=BLUE, fontweight="bold")

    ax.text(6.2, -0.65,
             "채널을 크게 늘린 최신 네트워크는 이 벡터가 수만$\\sim$수십만 길이가 되어 "
             "FC 층이 파라미터 대부분을 차지한다", ha="center", fontsize=10, color=RED)

    savefig(fig, "gen_lenet_flatten.png")


# ---------------------------------------------------------------
# p51g1: 224x224 -> 7x7, 5번의 2x2 최대풀링
# ---------------------------------------------------------------
def fig_p51():
    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    sizes = [224, 112, 56, 28, 14, 7]
    # log-scale visual sizes for drawing (clip max)
    draw_sizes = [1.9, 1.55, 1.25, 1.0, 0.78, 0.6]
    xs = [0.9 + i * 2.25 for i in range(6)]
    y_mid = 2.0

    for i, (s, ds, x) in enumerate(zip(sizes, draw_sizes, xs)):
        ax.add_patch(Rectangle((x - ds / 2, y_mid - ds / 2), ds, ds,
                                facecolor=BLUE if i < 5 else RED,
                                edgecolor="white", alpha=0.85))
        ax.text(x, y_mid - ds / 2 - 0.28, f"{s}$\\times${s}", ha="center",
                 fontsize=11.5, color=BLUE if i < 5 else RED, fontweight="bold")
        if i < 5:
            ax.annotate("", xy=(xs[i + 1] - draw_sizes[i + 1] / 2 - 0.08, y_mid),
                         xytext=(x + ds / 2 + 0.08, y_mid),
                         arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8))
            ax.text((x + xs[i + 1]) / 2, y_mid + 0.35, "2$\\times$2\n풀링",
                     ha="center", fontsize=9, color=GRAY)

    ax.text(6.4, 3.25, "224$\\times$224 입력이 2$\\times$2 최대풀링 5회로 7$\\times$7 격자까지 줄어든다",
             ha="center", fontsize=12.5, color=BLUE, fontweight="bold")
    ax.text(xs[-1], y_mid + draw_sizes[-1] / 2 + 0.55, "채널 512\n(격자당 분류기)",
             ha="center", fontsize=9.5, color=RED)

    savefig(fig, "gen_vgg_grid_reduction.png")


# ---------------------------------------------------------------
# p52g1: U-Net 인코더-디코더 + 스킵 연결
# ---------------------------------------------------------------
def fig_p52():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    # 인코더 (왼쪽, 아래로 갈수록 작아짐) - 4단계
    enc_w = [2.2, 1.7, 1.3, 1.0]
    enc_h = [0.55, 0.5, 0.45, 0.4]
    enc_x = [0.6, 1.05, 1.45, 1.8]
    enc_y = [5.1, 4.15, 3.25, 2.4]
    dec_x = [6.0, 6.4, 6.75, 7.1]
    dec_y = enc_y[:]
    dec_w = enc_w[::-1]
    dec_h = enc_h[::-1]

    for i in range(4):
        ax.add_patch(Rectangle((enc_x[i], enc_y[i]), enc_w[i], enc_h[i],
                                facecolor=BLUE, edgecolor="white", alpha=0.88))
        ax.annotate("", xy=(enc_x[i] + enc_w[i] / 2 - 0.05, enc_y[i] - 0.28),
                     xytext=(enc_x[i] + enc_w[i] / 2 + 0.35, enc_y[i] + 0.02),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))

    # 바닥 (병목)
    ax.add_patch(Rectangle((2.9, 1.45), 3.4, 0.4, facecolor=GOLDDK,
                            edgecolor="white", alpha=0.9))
    ax.text(4.6, 1.15, "병목(bottleneck)", ha="center", fontsize=9.5, color=GRAY)

    for i in range(4):
        j = 3 - i
        ax.add_patch(Rectangle((dec_x[i], dec_y[j]), dec_w[i], dec_h[i],
                                facecolor=RED, edgecolor="white", alpha=0.85))

    # 인코더->디코더 스킵 연결 (같은 레벨끼리 점선 화살표)
    for i in range(4):
        j = 3 - i
        y1 = enc_y[i] + enc_h[i] / 2
        y2 = dec_y[j] + dec_h[i] / 2
        ax.annotate("", xy=(dec_x[i] - 0.15, y2), xytext=(enc_x[i] + enc_w[i] + 0.15, y1),
                     arrowprops=dict(arrowstyle="-|>", color=GOLDDK, lw=1.6,
                                      linestyle="--", connectionstyle="arc3,rad=-0.12"))

    ax.text(1.3, 5.6, "인코더\n(코더, 줄여가기)", ha="center", fontsize=11.5,
             color=BLUE, fontweight="bold")
    ax.text(6.7, 5.6, "디코더\n(늘려가기)", ha="center", fontsize=11.5,
             color=RED, fontweight="bold")
    ax.text(4.6, 0.55, "점선 화살표 = 스킵 연결 (같은 레벨의 위치 정보를 디코더로 직접 전달)",
             ha="center", fontsize=10, color=GOLDDK)
    ax.set_title("U-Net: 인코더--디코더 대칭 구조 + 스킵 연결", fontsize=13.5,
                  color=BLUE, fontweight="bold", y=0.99)

    savefig(fig, "gen_unet.png")


if __name__ == "__main__":
    fig_p03()
    fig_p04()
    fig_p08()
    fig_p25()
    fig_p27()
    fig_p35()
    fig_p51()
    fig_p52()
    print("ALL DONE")
