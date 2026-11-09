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


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def points(ax, new=True):
    ax.scatter(SPEED, DIST, s=70, color=BLUE, zorder=3, label="측정 데이터")
    if new:
        ax.scatter(SPEED_NEW, DIST_NEW, s=90, marker="^", color=RED, zorder=3, label="새 데이터")
    ax.set_xlabel("속도 (km/h)"); ax.set_ylabel("제동거리 (m)"); ax.set_ylim(-25, 80); ax.set_xlim(3, 107)
    ax.axhline(0, color=GRAY, lw=0.8)


# 1. 3주차 마지막 그림 (같은 숫자)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.axvspan(9.5, 20.5, color=BLUE, alpha=0.06)
ax.plot(DEGS, [BEYOND[d] for d in DEGS], "^-", color=RED, lw=2.5)
ax.axvline(9, color=GRAY, lw=1, ls="--")
ax.text(9.15, 9.6, "9차: 계수 10개 = 점 10개", fontsize=13, color=GRAY)
for d, xy in [(3, (2.2, 1.0)), (9, (5.6, 8.3)), (16, (16.6, 2.6))]:
    ax.annotate(f"{d}차 {BEYOND[d]} m", (d, BEYOND[d]), xy, fontsize=14, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks(range(1, 21, 2)); ax.set_xlim(0.5, 20.5); ax.set_ylim(0, 10.5)
ax.set_xlabel("다항식 차수"); ax.set_ylabel("새 데이터 오차 (m)")
save(fig, "dd_beyond.png")

# 2. 9차 곡선 vs 16차 최소 노름 곡선
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, d, c in [(axs[0], 9, RED), (axs[1], 16, BLUE)]:
    points(ax); ax.plot(vs, design(vs, d) @ minnorm(SPEED, DIST, d), color=c, lw=3)
    ax.set_title(f"{d}차: 측정 오차 0 m · 새 데이터 {BEYOND[d]} m · 계수 크기 {NORM[d]}", fontsize=14, color=c)
axs[0].legend(frameon=False, loc="upper left", fontsize=12)
save(fig, "dd_two_curves.png")

# 3. 계수 크기 vs 차수
fig, ax = plt.subplots(figsize=(8.5, 4.2))
nr = [float(np.linalg.norm(minnorm(SPEED, DIST, d))) for d in DEGS]
ax.plot(DEGS, nr, "o-", color=GOLD, lw=2.5)
ax.axvline(9, color=GRAY, lw=1, ls="--")
ax.annotate(f"9차 {NORM[9]}", (9, NORM[9]), (11.5, 118), fontsize=15, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate(f"16차 {NORM[16]}", (16, NORM[16]), (15, 75), fontsize=15, color=BLUE, fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))
ax.set_xticks(range(1, 21, 2)); ax.set_xlabel("다항식 차수"); ax.set_ylabel("계수 크기 ||w||"); ax.set_ylim(0, 145)
save(fig, "dd_norm.png")

# 4. 15차: 모든 점을 지나는 곡선이 무수히 많다
d = SHOW_DEG
X = design(SPEED, d); c0 = minnorm(SPEED, DIST, d)
_, _, Vt = np.linalg.svd(X); null = Vt[len(SPEED):]          # X @ v = 0 인 방향들
fig, ax = plt.subplots(figsize=(8.5, 4.4))
points(ax, new=False)
for k, t in enumerate([25, -25, 30]):
    v = null[k] / np.linalg.norm(null[k])
    ax.plot(vs, design(vs, d) @ (c0 + t * v), color=GRAY, lw=1.6, alpha=0.8, label="다른 해 (모두 점 10개를 지난다)" if k == 0 else None)
ax.plot(vs, design(vs, d) @ c0, color=BLUE, lw=3.2, label=f"계수가 가장 작은 해 (||w|| = {np.linalg.norm(c0):.1f})")
ax.legend(frameon=False, loc="upper left", fontsize=12)
save(fig, "dd_many.png")

# 5. 잡음이 없으면 피크도 없다
fig, ax = plt.subplots(figsize=(8.5, 4.2))
law = truth(SPEED_NEW)
for yy, c, lab in [(DIST, RED, "측정값 (잡음 있음) 으로 학습"), (truth(SPEED), GREEN, "물리 법칙 값 (잡음 없음) 으로 학습")]:
    ax.plot(DEGS, [mae(minnorm(SPEED, yy, d), d, SPEED_NEW, law) for d in DEGS], "o-", color=c, lw=2.5, label=lab)
ax.axvline(9, color=GRAY, lw=1, ls="--")
ax.annotate(f"9차 {NOISY[9]} m", (9, NOISY[9]), (11.3, 5.6), fontsize=14, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate(f"9차 {NOISEFREE[9]} m", (9, NOISEFREE[9]), (10.3, 1.0), fontsize=14, color=GREEN, fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN))
ax.set_xticks(range(1, 21, 2)); ax.set_xlabel("다항식 차수"); ax.set_ylabel("물리 법칙과의 차이 (m)")
ax.legend(frameon=False, loc="upper right", fontsize=12)
save(fig, "dd_noisefree.png")

# ---------- 손글씨 숫자 ----------
Xtr, ytr, Xte, yte = load_digits()

# 6. 숫자 예시
fig, axs = plt.subplots(2, 10, figsize=(12, 2.9))
for k, ax in enumerate(axs.flat):
    ax.imshow(Xtr[k].reshape(8, 8), cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(str(ytr[k]), fontsize=14)
save(fig, "dd_digits.png")

# 7. 무작위 특징 모델 그림
fig, ax = plt.subplots(figsize=(9, 3.0)); ax.axis("off")
L = {"in": (0.08, np.linspace(0.12, 0.88, 6)), "hid": (0.5, np.linspace(0.04, 0.96, 10)), "out": (0.92, np.linspace(0.25, 0.75, 4))}
for (x1, ys1), (x2, ys2), c, lw in [(L["in"], L["hid"], GRAY, 0.5), (L["hid"], L["out"], BLUE, 1.2)]:
    for a in ys1:
        for b in ys2:
            ax.plot([x1, x2], [a, b], color=c, lw=lw, alpha=0.6)
for key, col in [("in", "white"), ("hid", "#eeeeee"), ("out", "#dfe3fb")]:
    x, ys = L[key]
    ax.scatter([x] * len(ys), ys, s=260, color=col, edgecolor="black", zorder=3)
ax.text(0.08, -0.13, "픽셀 64칸", ha="center", fontsize=19)
ax.text(0.5, -0.13, "특징 p개 (ReLU)", ha="center", fontsize=19)
ax.text(0.92, -0.13, "숫자 10칸", ha="center", fontsize=19)
ax.text(0.29, 1.03, "무작위로 정하고 고정", ha="center", fontsize=19, color=GRAY, fontweight="bold")
ax.text(0.71, 1.03, "w 만 학습 (최소제곱)", ha="center", fontsize=19, color=BLUE, fontweight="bold")
ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.2, 1.12)
save(fig, "dd_model.png")

S = sweep((Xtr, ytr, Xte, yte))
SN = sweep((Xtr, ytr, Xte, yte), noisy=True)
SR = sweep((Xtr, ytr, Xte, yte), lam=LAM)


def curve_axes(ax, ylab="시험 오류율 (%)"):
    ax.set_xscale("log"); ax.axvline(N, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("특징 수 p (로그 눈금)"); ax.set_ylabel(ylab)
    ax.set_xticks([10, 50, 200, 1000, 4000]); ax.set_xticklabels(["10", "50", "200", "1000", "4000"])


# 8. 이중 하강 곡선
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(PS, [S[p][0] for p in PS], "o-", color=BLUE, lw=2.5)
curve_axes(ax); ax.set_ylim(0, 100)
ax.text(N * 1.06, 93, f"p = 학습 이미지 {N}장", fontsize=13, color=GRAY)
for p, xy, c in [(100, (14, 62), BLUE), (200, (420, 80), RED), (4000, (1300, 33), BLUE)]:
    ax.annotate(f"p = {p}: {CLEAN[p]}%", (p, CLEAN[p]), xy, fontsize=14, color=c, fontweight="bold", arrowprops=dict(arrowstyle="->", color=c))
save(fig, "dd_curve.png")

# 9. 계수 크기 vs p
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(PS, [S[p][1] for p in PS], "o-", color=GOLD, lw=2.5)
curve_axes(ax, "계수 크기 ||w|| (로그 눈금)"); ax.set_yscale("log")
ax.annotate(f"p = 200: {CLEAN_NORM[200]}", (200, S[200][1]), (500, 120), fontsize=14, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate(f"p = 4000: {CLEAN_NORM[4000]}", (4000, S[4000][1]), (900, 6), fontsize=14, color=BLUE, fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))
save(fig, "dd_norm_p.png")

# 10. 라벨 잡음 / 릿지
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (T, c, lab) in zip(axs, [(SN, RED, "라벨 20% 를 틀리게"), (SR, GREEN, f"릿지 λ = {LAM}")]):
    ax.plot(PS, [S[p][0] for p in PS], "o-", color=BLUE, lw=2, alpha=0.5, label="원래 (λ = 0)")
    ax.plot(PS, [T[p][0] for p in PS], "o-", color=c, lw=2.5, label=lab)
    curve_axes(ax); ax.set_ylim(0, 100); ax.legend(frameon=False, loc="upper right", fontsize=12)
axs[1].set_ylabel("")
save(fig, "dd_noise_ridge.png")

# 11. 데이터 수 바꾸기 (p = 200)
BN = sweep_n((Xtr, ytr, Xte, yte))
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(NS, [BN[n] for n in NS], "s-", color=BLUE, lw=2.5)
ax.set_xscale("log"); ax.axvline(200, color=GRAY, lw=1, ls="--"); ax.set_ylim(0, 100)
ax.set_xticks([50, 100, 200, 400, 1000, 2000]); ax.set_xticklabels(["50", "100", "200", "400", "1000", "2000"])
ax.set_xlabel("학습 이미지 수 n (로그 눈금, 특징 p = 200 고정)"); ax.set_ylabel("시험 오류율 (%)")
ax.annotate(f"n = 100: {BY_N[100]}%", (100, BY_N[100]), (55, 62), fontsize=14, color=BLUE, fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate(f"n = 200: {BY_N[200]}%", (200, BY_N[200]), (400, 80), fontsize=14, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
save(fig, "dd_by_n.png")
print("ok")
