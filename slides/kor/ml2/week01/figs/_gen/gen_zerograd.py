"""p62g1: zero_grad() 사용 vs 생략(그래디언트 축적) 시 손실 곡선 비교 -- 모식 시뮬레이션"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
import numpy as np

# 1차원 볼록 손실 J(w) = (w - w*)^2 에서 경사하강법 시뮬레이션.
# 정상: 매 스텝 grad = dJ/dw 만 사용해 업데이트.
# 결함: zero_grad 생략 -> 이전 스텝들의 grad가 누적되어 유효 학습률이 매 스텝 커짐.
w_star = 3.0
n_steps = 22

def run_normal(lr=0.15, n=n_steps):
    w = 0.0
    losses = []
    for t in range(n):
        grad = 2 * (w - w_star)
        w = w - lr * grad
        losses.append((w - w_star) ** 2)
    return np.array(losses)

def run_no_zero_grad(lr=2.1, n=n_steps, cap=1e8):
    # zero_grad() 누락 -> .grad 버퍼가 매 스텝 이전 그래디언트를 계속 더해감
    w = 0.0
    losses = []
    accum = 0.0
    for t in range(n):
        grad = 2 * (w - w_star)
        accum += grad
        w = w - lr * accum
        loss = (w - w_star) ** 2
        losses.append(loss)
        if loss > cap:
            break
    return np.array(losses)

loss_ok = run_normal()
loss_bad = run_no_zero_grad()

fig, ax = plt.subplots(figsize=(9.2, 5.4))
steps_ok = np.arange(1, len(loss_ok) + 1)
steps_bad = np.arange(1, len(loss_bad) + 1)

ax.plot(steps_ok, np.clip(loss_ok, 1e-4, None), color=KSA_BLUE, lw=2.4, marker="o", ms=4,
        label="zero_grad() 사용 — 정상 수렴")
ax.plot(steps_bad, np.clip(loss_bad, 1e-4, None), color=ACCENT_RED, lw=2.4, marker="o", ms=4,
        label="zero_grad() 생략 — 그래디언트 축적")
ax.scatter([steps_bad[-1]], [np.clip(loss_bad[-1], 1e-4, 5e7)],
           marker="x", s=200, color=ACCENT_RED, linewidths=3.2, zorder=5)
ax.annotate("손실이 폭주 → NaN", xy=(steps_bad[-1], 5e7),
            xytext=(steps_bad[-1] - 11, 3e5),
            fontsize=11.5, color=ACCENT_RED,
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5))

ax.set_yscale("log")
ax.set_ylim(1e-4, 1e9)
ax.set_xlim(0, n_steps + 1)
ax.set_xlabel("학습 스텝(에포크)", fontsize=12)
ax.set_ylabel("손실 $J$ (로그 스케일)", fontsize=12)
ax.set_title("zero_grad() 생략 → 그래디언트 축적 → 학습 발산 (모식 시뮬레이션)",
              fontsize=12.5, color=KSA_BLUE, pad=12)
ax.legend(fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", which="major", color=GREY, alpha=0.25)

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "zerograd_divergence.png"))
