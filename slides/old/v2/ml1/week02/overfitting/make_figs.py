"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"
vs = np.linspace(5, 105, 400)
M = {k: fit(SPEED, DIST, d) for k, d in DEGS.items()}
COL = {"직선": GOLD, "2차 곡선": GREEN, "구불구불 곡선": RED}


def base(ax, new=False):
    ax.scatter(SPEED, DIST, s=70, color=BLUE, zorder=3, label="측정 데이터")
    if new:
        ax.scatter(SPEED_NEW, DIST_NEW, s=90, marker="^", color=RED, zorder=3, label="새 데이터")
    ax.set_xlabel("속도 (km/h)"); ax.set_ylabel("제동거리 (m)"); ax.set_xlim(0, 108); ax.set_ylim(-15, 75)
    ax.axhline(0, color=GRAY, lw=0.8)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 질문: 모든 점을 지나는 곡선
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax)
ax.plot(vs, M["구불구불 곡선"](vs), color=RED, lw=3)
ax.text(3, 66, "측정 점 10개를 모두 지난다: 오차 0 m", fontsize=16, color=RED, fontweight="bold")
save(fig, "of_hook.png")

# 2. 데이터와 물리 법칙
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax)
ax.plot(vs, truth(vs), color=GRAY, lw=2, ls="--", label="물리 법칙 $d = v^2 / 2\\mu g$")
for v in (50, 100):
    ax.annotate(f"{v} km/h → {round(float(truth(v)))} m", (v, truth(v)), (v - 42, truth(v) + 10), fontsize=14, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY))
ax.legend(frameon=False, loc="upper left")
save(fig, "of_data.png")

# 3. 직선: 10 km/h 에서 -5 m
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax)
ax.plot(vs, M["직선"](vs), color=GOLD, lw=3)
for x, y in zip(SPEED, DIST):
    ax.plot([x, x], [y, M["직선"](x)[0]], color=RED, lw=1.5)
ax.scatter([10], [LINE_AT_10], s=160, color=GOLD, edgecolor="black", zorder=4)
ax.annotate(f"10 km/h → {LINE_AT_10} m ?", (10, LINE_AT_10), (22, -12), fontsize=16, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text(3, 66, f"직선: 측정 데이터 오차 평균 {ERR_TR['직선']} m", fontsize=16, color=GOLD, fontweight="bold")
save(fig, "of_line.png")

# 4. 새 데이터에서 구불구불 곡선
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, new=True)
ax.plot(vs, M["구불구불 곡선"](vs), color=RED, lw=3)
ax.annotate(f"실제 3 m → 예측 {WIGGLE_AT_15} m", (15, WIGGLE_AT_15), (24, 50), fontsize=16, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.plot([15, 15], [3, WIGGLE_AT_15], color=RED, lw=2, ls=":")
ax.legend(frameon=False, loc="upper left")
save(fig, "of_new.png")

# 5. 2차 곡선
fig, ax = plt.subplots(figsize=(8, 4.6)); base(ax, new=True)
ax.plot(vs, M["2차 곡선"](vs), color=GREEN, lw=3)
ax.text(30, 64, f"2차 곡선: 측정 {ERR_TR['2차 곡선']} m · 새 데이터 {ERR_TE['2차 곡선']} m", fontsize=16, color=GREEN, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
save(fig, "of_quad.png")

# 6. 세 모델 나란히
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
for ax, (k, title) in zip(axs, (("직선", "과소적합 (underfitting)"), ("2차 곡선", "알맞음"), ("구불구불 곡선", "과대적합 (overfitting)"))):
    base(ax, new=True); ax.plot(vs, M[k](vs), color=COL[k], lw=3)
    ax.set_title(f"{k}: {title}", fontsize=15, color=COL[k])
    if ax is not axs[0]: ax.set_ylabel("")
save(fig, "of_three.png")

# 7. 5겹 교차 검증 그림
fig, ax = plt.subplots(figsize=(10, 4.2))
F = folds()
for r, te in enumerate(F):
    for i in range(10):
        hid = i in te
        ax.add_patch(plt.Rectangle((i, 4 - r), 0.92, 0.8, color=RED if hid else BLUE, alpha=0.85 if hid else 0.35))
        ax.text(i + 0.46, 4 - r + 0.4, f"{SPEED[i]}", ha="center", va="center", fontsize=12, color="white" if hid else "black")
    ax.text(-0.25, 4 - r + 0.4, f"{r + 1}번째", ha="right", va="center", fontsize=14)
    ax.text(10.15, 4 - r + 0.4, f"→ 오차 {r + 1}", ha="left", va="center", fontsize=14, color=RED)
ax.set_xlim(-1.6, 12); ax.set_ylim(-0.2, 5); ax.axis("off")
ax.text(4.6, -0.15, "파랑 = 학습에 쓰는 8개, 빨강 = 숨겨 둔 2개 (속도 km/h)", ha="center", fontsize=13, color=GRAY)
save(fig, "of_folds.png")

# 8. 교차 검증 결과
fig, ax = plt.subplots(figsize=(8, 4.4))
ks = list(DEGS)
b = ax.bar(ks, [ERR_CV[k] for k in ks], color=[COL[k] for k in ks], width=0.55)
ax.set_yscale("log"); ax.set_ylabel("교차 검증 오차 (m, 로그 눈금)"); ax.set_ylim(1, 400)
for r, k in zip(b, ks):
    ax.text(r.get_x() + r.get_width() / 2, ERR_CV[k] * 1.15, f"{ERR_CV[k]} m", ha="center", fontsize=17, fontweight="bold", color=COL[k])
save(fig, "of_cv.png")

# 9. 차수 1~9 전부
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ds = list(SWEEP)
for j, (lab, c, m) in enumerate((("측정 데이터 오차", BLUE, "o-"), ("새 데이터 오차", RED, "^-"), ("교차 검증 오차", GOLD, "s-"))):
    ax.plot(ds, [max(SWEEP[d][j], 0.05) for d in ds], m, color=c, lw=2.5, label=lab)
ax.set_yscale("log"); ax.set_xticks(ds); ax.set_xlabel("다항식 차수"); ax.set_ylabel("평균 절대 오차 (m, 로그 눈금)")
ax.annotate("교차 검증 최소: 3차 2.5 m", (3, 2.5), (4.2, 0.4), fontsize=14, color=GOLD, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GOLD))
ax.legend(frameon=False, loc="upper left")
save(fig, "of_sweep.png")
print("ok")
