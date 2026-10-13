"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py 와 같은 폴더의 것을 쓴다."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))          # 덱 폴더에서 실행 -> figs/ 에 저장
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"

# 1. 오늘의 질문: 같은 6만 장, 54% vs 96%
fig, ax = plt.subplots(figsize=(8.5, 4.4))
bars = ax.bar(["6만 장을 다 보고\n1걸음", f"{B_MNIST}장씩 보고\n{N_MNIST // B_MNIST:,}걸음"], [ACC_FULL, ACC_SGD], color=[RED, BLUE], width=0.5)
for b_, v in zip(bars, [ACC_FULL, ACC_SGD]):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 2, f"{v}%", ha="center", fontsize=24, fontweight="bold", color=b_.get_facecolor())
ax.set_ylim(0, 112); ax.set_ylabel("시험 정답률 (%)")
ax.set_title("손글씨 사진 6만 장을 똑같이 한 번씩 봤다", fontsize=17)
fig.tight_layout(); fig.savefig(O + "sgd_hook.png"); plt.close(fig)

# 2. 8명 점수와 각자의 기울기 (b = 40)
def number_line(ax, b, lab, col=GOLD):
    ax.set_xlim(35, 95); ax.set_ylim(-1.6, 1.4); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.axhline(0, color=GRAY, lw=1)
    ax.scatter(SCORES, [0] * M, s=140, color=BLUE, zorder=3)
    for y in SCORES: ax.text(y, 0.3, str(y), ha="center", fontsize=14, color=BLUE)
    ax.axvline(b, color=col, lw=3); ax.text(b, 1.1, lab, ha="center", fontsize=15, color=col, fontweight="bold")
    ax.set_xlabel("점수")
fig, ax = plt.subplots(figsize=(9.5, 3.8))
number_line(ax, B0, f"추측 b = {B0}")
for i, (y, g) in enumerate(zip(SCORES, GRADS)):
    ax.annotate("", xy=(y, -0.55 - 0.1 * (i % 2)), xytext=(B0, -0.55 - 0.1 * (i % 2)), arrowprops=dict(arrowstyle="->", color=RED, lw=1.4, alpha=0.7))
    ax.text(y, -1.05 - 0.25 * (i % 2), f"{g}", ha="center", fontsize=13, color=RED)
ax.text(36, -1.45, "빨간 숫자 = 학생별 기울기 b − y", fontsize=13, color=RED)
fig.tight_layout(); fig.savefig(O + "sgd_scores.png"); plt.close(fig)

# 3. 몇 명을 뽑느냐에 따라 미니배치 기울기의 흩어짐 (B = 1, 2, 4, 전부 경우를 센다)
fig, axs = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
for ax, k in zip(axs, (1, 2, 4)):
    gs = [B0 - np.mean(c) for c in itertools.combinations(SCORES, k)]
    vals, cnt = np.unique(gs, return_counts=True)
    ax.bar(vals, cnt, width=0.8 if k > 1 else 1.2, color=BLUE)
    ax.axvline(FULL_GRAD, color=GOLD, lw=3)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(0, max(cnt) * 1.7)
    ax.text(-52, max(cnt) * 1.35, f"B = {k}  ({len(gs)}가지)", fontsize=15, fontweight="bold")
    ax.text(-8, max(cnt) * 1.35, f"평균 {np.mean(gs):.0f} · 표준편차 {np.std(gs):.1f}", fontsize=13, ha="right")
axs[0].text(FULL_GRAD + 0.6, axs[0].get_ylim()[1] * 0.8, f"전체 기울기 {FULL_GRAD}", color=GOLD, fontsize=13, fontweight="bold")
axs[-1].set_xlabel("미니배치 기울기"); axs[-1].set_xlim(-52, -8)
fig.tight_layout(); fig.savefig(O + "sgd_spread.png"); plt.close(fig)

# 4. 28가지 쌍 (B = 2) 만
fig, ax = plt.subplots(figsize=(9, 3.6))
vals, cnt = np.unique(PAIR_GRADS, return_counts=True)
ax.bar(vals, cnt, width=0.8, color=BLUE)
ax.axvline(FULL_GRAD, color=GOLD, lw=3); ax.text(FULL_GRAD + 0.5, max(cnt) * 0.95, f"전체 기울기 {FULL_GRAD}", color=GOLD, fontsize=14, fontweight="bold")
ax.axvline(PAIR_GRAD, color=RED, lw=2, ls="--"); ax.text(PAIR_GRAD + 0.5, max(cnt) * 0.7, f"(50, 74) → {PAIR_GRAD}", color=RED, fontsize=13)
ax.set_xlabel("두 명으로 구한 기울기"); ax.set_ylabel("쌍의 수"); ax.set_yticks(range(0, max(cnt) + 1))
ax.set_title(f"28가지 쌍의 기울기를 평균하면 {np.mean(PAIR_GRADS):.0f}", fontsize=16)
fig.tight_layout(); fig.savefig(O + "sgd_pairs.png"); plt.close(fig)

# 5. 1에폭: 전체 배치 1걸음 vs 미니배치 4걸음 (한 걸음 = 한 줄)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
rows = [("전체 배치: 8명 다 보고", B0, GD_ONE, RED)] + [(f"SGD {k + 1}걸음: {p[0]}, {p[1]}", SGD_PATH[k], SGD_PATH[k + 1], BLUE) for k, p in enumerate(EPOCH_BATCHES)]
for r, (lab, s0, s1, col) in enumerate(rows):
    yy = -r - (0.6 if r else 0)
    ax.annotate("", xy=(s1, yy), xytext=(s0, yy), arrowprops=dict(arrowstyle="->", color=col, lw=2.4))
    ax.scatter([s0], [yy], s=40, color=col, zorder=3)
    ax.text(s1 + (0.6 if s1 > s0 else -0.6), yy, f"{s1:g}", va="center", ha="left" if s1 > s0 else "right", fontsize=15, color=col, fontweight="bold")
    ax.text(34.5, yy, lab, va="center", ha="right", fontsize=13, color=col)
ax.axvline(MEAN, color=GOLD, lw=2.5, ls="--"); ax.text(MEAN, 0.7, f"최적 {MEAN}", color=GOLD, fontsize=14, fontweight="bold", ha="center")
ax.axhline(-0.8, color=GRAY, lw=0.8, ls=":")
ax.set_xlim(35, 76); ax.set_ylim(-4.9, 1.1); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("b")
fig.tight_layout(); fig.savefig(O + "sgd_epoch.png", bbox_inches="tight"); plt.close(fig)

# 6. 2차원에서 보기: diabetes (BMI → 1년 뒤 진행 정도), 특징·정답 표준화, ŷ = w x + b
from sklearn.datasets import load_diabetes
d = load_diabetes(); x = d.data[:, 2]; yv = d.target
x = (x - x.mean()) / x.std(); yv = (yv - yv.mean()) / yv.std(); m_ = len(x)
def J(w, b): return 0.5 * np.mean((w * x[:, None, None] + b - yv[:, None, None]) ** 2, axis=0)
def gd(w, b, ids): r = w * x[ids] + b - yv[ids]; return np.mean(r * x[ids]), np.mean(r)
rng = np.random.default_rng(0); w, b, a_ = -1.2, 1.2, 0.05; path_gd = [(w, b)]
for _ in range(3):
    gw, gb = gd(w, b, np.arange(m_)); w, b = w - a_ * gw, b - a_ * gb; path_gd.append((w, b))
w, b = -1.2, 1.2; path_sgd = [(w, b)]
for ep in range(3):
    perm = rng.permutation(m_)
    for i in range(0, m_, 16):
        gw, gb = gd(w, b, perm[i:i + 16]); w, b = w - 0.05 * gw, b - 0.05 * gb; path_sgd.append((w, b))
W, Bb = np.meshgrid(np.linspace(-1.6, 1.5, 120), np.linspace(-1.0, 1.5, 120))
Z = J(W, Bb); wopt = np.polyfit(x, yv, 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, P, col, ttl in ((axs[0], path_gd, RED, f"전체 배치 ({m_}명) · 3에폭 = 3걸음 · α 0.05"), (axs[1], path_sgd, BLUE, f"미니배치 16명 · 3에폭 = {len(path_sgd) - 1}걸음 · α 0.05")):
    ax.contour(W, Bb, Z, levels=np.geomspace(0.39, 2.5, 9), colors=GRAY, linewidths=0.8)
    P = np.array(P); ax.plot(P[:, 0], P[:, 1], "-o", color=col, ms=3 if len(P) > 10 else 7, lw=1.6)
    ax.plot(*wopt, "*", color=GOLD, ms=18); ax.set_title(ttl, fontsize=14, color=col); ax.set_xlabel("w")
axs[0].set_ylabel("b")
fig.tight_layout(); fig.savefig(O + "sgd_paths.png"); plt.close(fig)
PATHS = dict(gd_end=path_gd[-1], sgd_end=path_sgd[-1], sgd_steps=len(path_sgd) - 1, J_gd=J(*path_gd[-1]).item(), J_sgd=J(*path_sgd[-1]).item(), J_opt=J(*wopt).item())
print("paths", PATHS)

# 7. MNIST 정답률 곡선 (본 사진 수)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot([c[0] for c in CURVE], [c[1] for c in CURVE], color=BLUE, lw=2.5, label=f"SGD ({B_MNIST}장씩)")
ax.plot([0, 60, 60], [ACC0, ACC0, ACC_FULL], color=RED, lw=2.5, label="전체 배치 (6만 장마다 1걸음)")
ax.scatter([60, 60], [ACC_SGD, ACC_FULL], color=[BLUE, RED], s=70, zorder=3)
ax.text(57, ACC_SGD + 3, f"{ACC_SGD}%", color=BLUE, fontsize=17, fontweight="bold", ha="right")
ax.text(57, ACC_FULL + 2, f"{ACC_FULL}%", color=RED, fontsize=17, fontweight="bold", ha="right")
ax.set_xlabel("본 사진 (천 장)"); ax.set_ylabel("시험 정답률 (%)"); ax.set_ylim(0, 105); ax.set_xlim(0, 63)
ax.legend(loc=(0.3, 0.3), frameon=False, fontsize=13)
fig.tight_layout(); fig.savefig(O + "sgd_mnist.png"); plt.close(fig)

# 8. 방향: 배치 크기별 미니배치 기울기 100개 vs 전체 기울기 (코사인 → 각도)
fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), subplot_kw=dict(aspect="equal"))
rng = np.random.default_rng(0)
for ax, k in zip(axs, (1, 32, 1024)):
    cs = np.array(COS[k]); ang = np.arccos(np.clip(cs, -1, 1)) * rng.choice([-1, 1], len(cs))
    th = np.linspace(-np.pi / 2, np.pi / 2, 50); ax.fill(np.r_[0, np.cos(th), 0], np.r_[0, np.sin(th), 0], color=GREEN, alpha=0.12)
    for a, c in zip(ang, cs): ax.plot([0, np.cos(a)], [0, np.sin(a)], color=BLUE if c > 0 else RED, lw=1, alpha=0.6)
    ax.annotate("", xy=(1.15, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=3))
    ax.set_xlim(-1.1, 1.3); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    neg = int((cs <= 0).sum())
    ax.set_title(f"B = {k:,}\n코사인 평균 {cs.mean():.2f} · 오르막 {neg}개", fontsize=14, color=RED if neg else BLUE, pad=12)
fig.text(0.5, 0.02, "금색 = 전체 기울기의 내리막 방향 · 초록 = 내리막 쪽 반 · 빨강 = 오르막을 가리킨 미니배치", ha="center", fontsize=12, color=GRAY)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(O + "sgd_fan.png"); plt.close(fig)

# 9. 배치 크기 표 → 막대 (1에폭 정답률) + 걸음 수
fig, ax = plt.subplots(figsize=(10, 4.4))
labs = [f"{b['B']:,}" for b in BATCH]; accs = [b["acc"] * 100 for b in BATCH]
bars = ax.bar(labs, accs, color=[BLUE if b["B"] < 60000 else RED for b in BATCH], width=0.6)
for b_, b in zip(bars, BATCH):
    ax.text(b_.get_x() + b_.get_width() / 2, b["acc"] * 100 + 1.5, f"{b['acc'] * 100:.1f}%", ha="center", fontsize=14, fontweight="bold")
    ax.text(b_.get_x() + b_.get_width() / 2, 6, f"{b['steps']:,}\n걸음", ha="center", fontsize=12, color="white", fontweight="bold")
ax.set_ylim(0, 110); ax.set_xlabel("배치 크기 B"); ax.set_ylabel("1에폭 뒤 시험 정답률 (%)")
fig.tight_layout(); fig.savefig(O + "sgd_batch.png"); plt.close(fig)
print("ok")
