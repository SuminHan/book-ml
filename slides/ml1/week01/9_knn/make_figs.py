"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, ORANGE = "#2E3192", "#d62728", "#A9976B", "#888888", "#f28e2b"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def fruits(ax, sy=1.0, q=True, legend=True, qs=320):
    a = np.array(APPLE); t = np.array(TANG)
    ax.scatter(a[:, 0], a[:, 1] * sy, s=110, color=RED, label="사과", zorder=3)
    ax.scatter(t[:, 0], t[:, 1] * sy, s=110, marker="s", color=ORANGE, label="귤", zorder=3)
    ax.scatter([NOISE[0]], [NOISE[1] * sy], s=110, marker="s", color=ORANGE, edgecolor="black", lw=1.5, zorder=3)
    if q:
        ax.scatter([Q[0]], [Q[1] * sy], s=qs, marker="*", color=GOLD, edgecolor="black", zorder=4, label="처음 보는 과일")
    ax.set_xlabel("크기"); ax.set_ylabel("단맛" if sy == 1 else "단맛 (0~100 점)")
    if legend: ax.legend(frameon=False, loc="lower left", fontsize=13)


# 1. 질문
fig, ax = plt.subplots(figsize=(7.5, 5)); fruits(ax, qs=170)
ax.text(Q[0] + 0.25, Q[1] + 0.35, "?", fontsize=30, color=GOLD, fontweight="bold")
ax.set_xlim(1, 9.5); ax.set_ylim(2, 9.5)
save(fig, "kn_hook.png")

# 1-2. 회귀 (지난 시간 선형회귀 데이터) vs 분류 (오늘 과일)
import importlib.util
_s = importlib.util.spec_from_file_location("lrdata", "../2_linear_regression/data.py")
lr = importlib.util.module_from_spec(_s); _s.loader.exec_module(lr)


def fig_cls(hours, score, w0, w1):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axs[0]
    ax.scatter(hours, score, s=60, color=GRAY, zorder=2)
    xs = np.array([0, 10.5]); ax.plot(xs, w0 + w1 * xs, color=BLUE, lw=2.5)
    p = w0 + w1 * 7
    ax.plot([7, 7, 0], [35, p, p], ls="--", color=GOLD, lw=2)
    ax.scatter([7], [p], s=320, marker="*", color=GOLD, edgecolor="black", zorder=4)
    ax.text(0.2, p + 2, f"{p:.1f}점", fontsize=16, color=BLUE, fontweight="bold")
    ax.set_xlim(0, 10.5); ax.set_ylim(35, 105); ax.set_xlabel("공부 시간"); ax.set_ylabel("점수")
    ax.set_title("지난 시간: 회귀 → 숫자 맞히기", fontsize=18, color=BLUE)
    ax = axs[1]
    fruits(ax, qs=300)
    ax.text(Q[0] + 0.25, Q[1] + 0.35, "?", fontsize=28, color=GOLD, fontweight="bold")
    ax.set_xlim(1, 9.5); ax.set_ylim(2, 9.5)
    ax.set_title("오늘: 분류 → 종류 맞히기", fontsize=18, color=RED)
    save(fig, "kn_cls.png")


_h, _sc = lr.study()
_w1, _w0 = np.polyfit(_h, _sc, 1)
assert abs(_w0 - lr.W0) < 0.01 and abs(_w1 - lr.W1) < 0.01 and round(_w0 + 7 * _w1, 1) == lr.PRED7
fig_cls(_h, _sc, _w0, _w1)

# 2. k=3 이웃
fig, ax = plt.subplots(figsize=(7.5, 5)); fruits(ax)
r = NEAR3[-1][2]
ax.add_patch(Circle(Q, r + 0.05, fill=False, ls="--", lw=2, color=GOLD))
for p, l, d in NEAR3:
    ax.plot([Q[0], p[0]], [Q[1], p[1]], color=GOLD, lw=2)
    ax.text((Q[0] + p[0]) / 2 + 0.08, (Q[1] + p[1]) / 2 - 0.05, f"{d:.2f}", fontsize=13, color="#7a6a3f", fontweight="bold")
ax.set_xlim(4.5, 8); ax.set_ylim(3, 6.3); ax.set_aspect("equal")
save(fig, "kn_k3.png")

# 3. 결정 경계 k = 1, 5, 21
gx, gy = np.meshgrid(np.linspace(1, 9.5, 170), np.linspace(2, 9.5, 150))
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, k in zip(axs, (1, 5, 21)):
    Z = np.array([[knn(PTS, (x, y), k)[1] == "사과" for x, y in zip(rx, ry)] for rx, ry in zip(gx, gy)])
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=[ORANGE, RED], alpha=0.15)
    fruits(ax, legend=False); ax.set_title(f"k = {k}", fontsize=17); ax.set_xlim(1, 9.5); ax.set_ylim(2, 9.5)
save(fig, "kn_boundary.png")

# 4. leave-one-out
fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(LOO); v = [LOO[k] for k in ks]
b = ax.bar([str(k) for k in ks], v, color=[RED if x == max(v) else GRAY for x in v])
for r_, x in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x + 0.3, f"{x}/21", ha="center", fontsize=13)
ax.set_xlabel("k"); ax.set_ylabel("맞힌 과일 수 (21개 중)"); ax.set_ylim(0, 23)
save(fig, "kn_loo.png")

# 5. 단위: 단맛을 0~100 점으로
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, sy, t in ((axs[0], 1, "크기 0~10, 단맛 0~10"), (axs[1], 10, "크기 0~10, 단맛 0~100")):
    fruits(ax, sy=sy, legend=False)
    near, vote = knn(PTS, Q, 3, sy=sy)
    for p, l in near:
        ax.plot([Q[0], p[0]], [Q[1] * sy, p[1] * sy], color=GOLD, lw=2.5)
    ax.set_title(f"{t} → k=3 답: {vote}", fontsize=15, color=RED if vote == "사과" else ORANGE)
save(fig, "kn_scale.png")

# 6. 거리 비
ds = sorted(RATIO)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(ds, [RATIO[d] for d in ds], "o-", color=BLUE, lw=2.5); ax.set_xscale("log")
ax.set_xlabel("차원 (로그 눈금)"); ax.set_ylabel("가까운 거리 ÷ 먼 거리"); ax.set_ylim(0, 1)
ax.annotate(f"2차원 {RATIO[2]:.2f}", (2, RATIO[2]), (3, 0.25), fontsize=14, arrowprops=dict(arrowstyle="->"))
ax.annotate(f"1,000차원 {RATIO[1000]:.2f}", (1000, RATIO[1000]), (60, 0.93), fontsize=14, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "kn_ratio.png")

# 7. 잡음 특징
fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(NOISE_ACC); v = [NOISE_ACC[k] for k in ks]
b = ax.bar([f"{k:,}" for k in ks], v, color=[BLUE, GRAY, GRAY, RED])
for r_, x in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, x + 1.5, f"{x}%", ha="center", fontsize=15, fontweight="bold")
ax.axhline(50, ls="--", color=GRAY); ax.text(3.45, 51.5, "찍기 50%", color=GRAY, fontsize=12, ha="right")
ax.set_xlabel("더한 잡음 특징 수 (원래 특징 2개)"); ax.set_ylabel("kNN 정확도 (%)"); ax.set_ylim(0, 105)
save(fig, "kn_noise.png")
# 8. 손글씨 숫자 (MNIST): 질의 3장과 가장 가까운 5장 (exp_mnist.py). 칸 위치를 직접 정한다 (tight_layout 이 글자 때문에 그림을 줄임)
E = np.load("assets/mnist_examples.npz")
fig = plt.figure(figsize=(11, 5.6)); S, H = 0.105, 0.205                       # 칸 너비 · 높이 (그림 비율)
fig.text(0.075, 0.95, "처음 보는 글씨", fontsize=15, ha="center"); fig.text(0.53, 0.95, "가장 가까운 5장 (학습 60,000장 중)", fontsize=15, ha="center")
def cell(img, x, y0, col, ttl, tcol="black"):
    a = fig.add_axes([x, y0, S, H]); a.imshow(img, cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_color(col); s_.set_linewidth(3)
    a.set_title(ttl, fontsize=13, color=tcol, pad=3)
for r in range(3):
    y0 = 0.65 - r * 0.31
    cell(E["q"][r], 0.025, y0, GOLD, f"정답 {E['ql'][r]}")
    for j in range(5):
        good = E["nl"][r, j] == E["ql"][r]
        cell(E["nb"][r, j], 0.2 + j * 0.135, y0, BLUE if good else RED, f"{E['nl'][r, j]} · 거리 {E['nd'][r, j]:.1f}", "black" if good else RED)
fig.savefig(O + "kn_mnist.png"); plt.close(fig)
# 9. 학습 없이 가장 가까운 한 장 (1-NN): 처음 보는 7 · 6만 장 중 가장 가까운 장 · 숫자별 학습 그림 · 신경망과 비교
E1 = np.load("assets/mnist_one.npz"); G1 = E1["grid"]
fig = plt.figure(figsize=(11, 6.6))
def tile(img, rect, col, ttl):
    a = fig.add_axes(rect); a.imshow(img, cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_color(col); s_.set_linewidth(3)
    a.set_title(ttl, fontsize=14, pad=4)
tile(E1["q"], [0.13, 0.66, 0.15, 0.25], GOLD, "처음 보는 글씨")
tile(E1["nb"], [0.72, 0.66, 0.15, 0.25], BLUE, "가장 가까운 한 장")
fig.text(0.205, 0.625, f"정답 {E1['ql']}", ha="center", fontsize=14)
fig.text(0.795, 0.625, f"{E1['nl']} → 답 {E1['nl']}", ha="center", fontsize=14, color=BLUE, fontweight="bold")
fig.text(0.5, 0.80, "거리", ha="center", fontsize=14, color=GRAY)
fig.text(0.5, 0.735, f"{MNIST_ONE_D}", ha="center", fontsize=30, fontweight="bold")
fig.text(0.5, 0.69, "학습 60,000장 중 가장 작다", ha="center", fontsize=12.5, color=GRAY)
big = np.ones((10 * 30, G1.shape[1] * 30))
for d in range(10):
    for j in range(G1.shape[1]): big[d * 30 + 1:d * 30 + 29, j * 30 + 1:j * 30 + 29] = G1[d, j]
a = fig.add_axes([0.06, 0.2, 0.9, 0.38]); a.imshow(big, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
a.set_xticks([]); a.set_yticks(np.arange(10) * 30 + 15); a.set_yticklabels(range(10), fontsize=11)
for s_ in a.spines.values(): s_.set_visible(False)
r_, c_ = E1["pos"]; a.add_patch(plt.Rectangle((c_ * 30 - 0.5, r_ * 30 - 0.5), 30, 30, fill=False, ec=BLUE, lw=2.5))
a.set_title("학습 그림 (숫자마다 40장만 보임, 실제로는 모두 60,000장)", fontsize=12.5, loc="left", color=GRAY)
fig.text(0.3, 0.085, f"{MNIST_NN[1]}%", ha="center", fontsize=30, fontweight="bold", color=GRAY)
fig.text(0.3, 0.035, "신경망: 1에폭 (1,875걸음) 학습", ha="center", fontsize=13, color=GRAY)
fig.text(0.72, 0.085, f"{MNIST_ACC[1]}%", ha="center", fontsize=30, fontweight="bold", color=BLUE)
fig.text(0.72, 0.035, "가장 가까운 한 장: 학습 없음", ha="center", fontsize=13, color=BLUE)
fig.savefig(O + "kn_mnist_one.png"); plt.close(fig)
# 10. MNIST 에서 kNN 하는 법: 그림 → 픽셀 784개 = 점 하나, 두 그림의 거리 = 칸마다 차이를 제곱해 더하고 루트
fig = plt.figure(figsize=(11, 4.4))
dif = E1["q"] - E1["nb"]
for x0, img, cm, ttl, col in ((0.03, E1["q"], "gray_r", "처음 보는 글씨", GOLD), (0.30, E1["nb"], "gray_r", "학습 그림 한 장", BLUE),
                              (0.62, dif, "bwr", "칸마다 차이", GRAY)):
    a = fig.add_axes([x0, 0.22, 0.2, 0.6]); a.imshow(img, cmap=cm, vmin=-1 if cm == "bwr" else 0, vmax=1)
    a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=15)
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_color(col); s_.set_linewidth(2.5)
fig.text(0.265, 0.5, "vs", ha="center", va="center", fontsize=20); fig.text(0.56, 0.5, "→", ha="center", va="center", fontsize=26)
fig.text(0.13, 0.1, "28 × 28 = 784칸, 칸마다 진하기 0~1\n→ 그림 한 장 = 특징 784개인 점 하나", ha="center", fontsize=13)
fig.text(0.72, 0.1, "빨강: 처음 보는 글씨만 진함 · 파랑: 학습 그림만 진함", ha="center", fontsize=12, color=GRAY)
fig.text(0.92, 0.6, "제곱해서\n784개 더하고\n루트", ha="center", fontsize=14)
fig.text(0.92, 0.33, f"거리 {MNIST_ONE_D}", ha="center", fontsize=20, fontweight="bold")
fig.savefig(O + "kn_mnist_how.png"); plt.close(fig)
print("ok")
