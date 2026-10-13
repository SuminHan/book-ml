"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py."""
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
x, y = study_pass(); xs = np.linspace(0, 10.5, 300)
jit = np.random.default_rng(0).uniform(-0.03, 0.03, len(y))


def base(ax):
    ax.scatter(x, y + jit, s=55, c=np.where(y == 1, BLUE, RED), zorder=3)
    ax.set_xlabel("공부 시간 (시간)"); ax.set_yticks([0, 0.5, 1], ["불합격 0", "0.5", "합격 1"]); ax.set_xlim(0, 10.5)


# 1. 직선으로 예측하면
fig, ax = plt.subplots(figsize=(8.5, 4.6)); base(ax)
k = np.polyfit(x, y, 1); ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3)
ax.axhspan(1, 1.6, color=RED, alpha=0.08); ax.axhspan(-0.6, 0, color=RED, alpha=0.08)
ax.text(8.2, 1.3, "확률 > 1 ?", color=RED, fontsize=16, fontweight="bold"); ax.text(0.3, -0.35, "확률 < 0 ?", color=RED, fontsize=16, fontweight="bold")
ax.set_ylim(-0.55, 1.55)
fig.tight_layout(); fig.savefig(O + "logit_line.png"); plt.close(fig)

# 2. 시그모이드
zs = np.linspace(-8, 8, 300)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(zs, sigmoid(zs), color=BLUE, lw=3.5); ax.axhline(0.5, color=GRAY, ls=":"); ax.axvline(0, color=GRAY, ls=":")
for zz in (-2, 0, 2):
    ax.scatter([zz], [sigmoid(zz)], color=RED, s=70, zorder=3); ax.text(zz + 0.3, sigmoid(zz) - 0.08, f"σ({zz}) = {sigmoid(zz):.2f}", fontsize=14, color=RED)
ax.set_xlabel("$z$"); ax.set_ylabel("$\\sigma(z)$"); ax.set_ylim(-0.05, 1.05)
fig.tight_layout(); fig.savefig(O + "logit_sigmoid.png"); plt.close(fig)

# 3. 학습 결과: 확률 곡선과 결정 경계
fig, ax = plt.subplots(figsize=(8.5, 4.6)); base(ax)
ax.plot(xs, sigmoid(W0 + W1 * xs), color=BLUE, lw=3)
ax.axvline(BOUNDARY, color=GRAY, ls="--", lw=2); ax.text(BOUNDARY + 0.15, 0.08, f"경계 {BOUNDARY}시간", fontsize=14, color=GRAY)
ax.scatter([7], [P7], s=200, color=GOLD, edgecolor="black", zorder=4)
ax.annotate(f"7시간 → 합격 확률 {P7:.0%}", (7, P7), (6.6, 0.55), fontsize=15, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.set_ylim(-0.08, 1.08)
fig.tight_layout(); fig.savefig(O + "logit_fit.png"); plt.close(fig)

# 4. 특징 두 개: 결정 경계는 직선
h, s, y2 = study_sleep(); w2 = fit(np.c_[np.ones_like(h), h, s], y2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(h, s, c=np.where(y2 == 1, BLUE, RED), s=55)
hh = np.linspace(0.5, 10, 50); ax.plot(hh, -(w2[0] + w2[1] * hh) / w2[2], color=GOLD, lw=3)
ax.text(1.0, 8.6, f"$\\sigma(z)=0.5 \\Leftrightarrow z = {w2[0]:.1f} + {w2[1]:.2f}\\,h + {w2[2]:.2f}\\,s = 0$", fontsize=14, color=GOLD, bbox=dict(fc="white", ec="none", alpha=0.9))
ax.set_xlabel("공부 시간 $h$"); ax.set_ylabel("수면 시간 $s$"); ax.set_ylim(3.8, 9.2)
fig.tight_layout(); fig.savefig(O + "logit_2d.png"); plt.close(fig)

# 5. 임계값을 바꾸면: 정밀도 · 재현율
p = sigmoid(W0 + W1 * x); ts = np.linspace(0.05, 0.95, 91)
prec = [((p >= t) & (y == 1)).sum() / max((p >= t).sum(), 1) for t in ts]
rec = [((p >= t) & (y == 1)).sum() / (y == 1).sum() for t in ts]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(ts, prec, color=BLUE, lw=3, label="정밀도 (합격이라 한 학생 중 진짜 합격)")
ax.plot(ts, rec, color=RED, lw=3, label="재현율 (진짜 합격 중 찾아낸 비율)")
ax.axvline(0.5, color=GRAY, ls=":"); ax.set_xlabel("임계값: 확률이 이 값 이상이면 '합격'"); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=13, loc="lower left")
fig.tight_layout(); fig.savefig(O + "logit_threshold.png"); plt.close(fig)
print("ok")
