"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 동전 실험
fig, ax = plt.subplots(figsize=(8.5, 3.6))
r = np.random.default_rng(2)
ax.scatter(np.array(LEAK) * 100, 1 + r.uniform(-0.15, 0.15, len(LEAK)), color=RED, s=45, label="누수: 전체 100명으로 숫자 고르기")
ax.scatter(np.array(OK) * 100, 0 + r.uniform(-0.15, 0.15, len(OK)), color=BLUE, s=45, label="정상: 학습 80명 안에서만 고르기")
ax.axvline(50, color=GRAY, ls="--"); ax.text(50.5, 1.45, "찍기 50%", color="#777777", fontsize=13)
ax.set_yticks([0, 1]); ax.set_yticklabels(["정상", "누수"]); ax.set_xlabel("5겹 교차 검증 정확도 (%) — 50번 반복"); ax.set_xlim(25, 100); ax.set_ylim(-0.5, 1.7)
save(fig, "tv_coin.png")

# 2. 정보 흐름
fig, ax = plt.subplots(figsize=(11, 3.4))
boxes = [("train (학습)", "가중치 $w$ 를 배운다", BLUE, 0.0, 6.0), ("val (검증)", "$\\lambda$ · 모델을 고른다", GOLD, 6.6, 2.4), ("test (시험)", "마지막에 딱 한 번", RED, 9.6, 2.4)]
for name, sub, c, x0, w in boxes:
    ax.add_patch(FancyBboxPatch((x0, 0.2), w, 1.6, boxstyle="round,pad=0.05", color=c, alpha=0.2))
    ax.text(x0 + w / 2, 1.25, name, ha="center", fontsize=18, fontweight="bold", color=c)
    ax.text(x0 + w / 2, 0.65, sub, ha="center", fontsize=14)
ax.annotate("", (6.5, 2.2), (3.0, 2.2), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", (9.5, 2.2), (7.8, 2.2), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(10.8, 2.25, "×  어디로도 거꾸로 흐르지 않는다", ha="center", fontsize=13, color=RED)
ax.set_xlim(-0.2, 12.2); ax.set_ylim(0, 2.6); ax.axis("off")
save(fig, "tv_flow.png")

# 3. 테스트로 고르면
ms = [1, 2, 5, 10, 20, 50, 100, 500]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogx(ms, [e_max(m) for m in ms], "o-", color=BLUE, lw=2.5)
ax.axhline(1.0, color=RED, ls="--"); ax.text(1.1, 1.08, "잡음 한 단위 (σ = 1)", color=RED, fontsize=13)
ax.annotate("5개 중 최고: +1.16", (5, 1.16), (6, 0.45), fontsize=14, arrowprops=dict(arrowstyle="->"))
ax.annotate("50개 중 최고: +2.25", (50, 2.25), (8, 2.6), fontsize=14, arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("test 로 비교한 후보 수 (로그 눈금)"); ax.set_ylabel("부풀려진 점수 (σ 단위)"); ax.set_ylim(-0.2, 3.3)
save(fig, "tv_emax.png")

# 4. 환자 단위 누수
X, y, ra, rb, te_rand, te_pat = patient_rmse()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
mask = np.zeros(len(X), bool); mask[te_rand] = True
for ax, m, t in ((axs[0], mask, f"사진 단위로 무작위 분할: RMSE {ra:.2f}"), (axs[1], te_pat, f"환자 단위로 분할: RMSE {rb:.2f}")):
    ax.scatter(X[~m, 0], X[~m, 1], s=14, color=BLUE, alpha=0.7, label="학습")
    ax.scatter(X[m, 0], X[m, 1], s=14, color=RED, alpha=0.9, label="시험")
    ax.set_title(t, fontsize=14); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
axs[0].legend(frameon=False, fontsize=12, loc="lower right")
save(fig, "tv_patient.png")

# 5. 시간 순서
t, yy, a, b, te_r, pa, pb = time_rmse()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
mask = np.zeros(365, bool); mask[te_r] = True
axs[0].scatter(t[~mask], yy[~mask], s=5, color=BLUE); axs[0].scatter(t[mask], yy[mask], s=9, color=RED); axs[0].set_title(f"날짜를 섞어 분할: RMSE {a:.2f}", fontsize=14)
cut = int(365 * 0.8)
axs[1].scatter(t[:cut], yy[:cut], s=5, color=BLUE); axs[1].scatter(t[cut:], yy[cut:], s=9, color=RED); axs[1].plot(t[cut:], pb, color="black", lw=2)
axs[1].set_title(f"시간 순서로 분할: RMSE {b:.1f}", fontsize=14)
for ax in axs: ax.set_xlabel("날짜 (일)")
axs[0].set_ylabel("값 (매일 +1 씩 오른다)")
save(fig, "tv_time.png")
print("ok")
