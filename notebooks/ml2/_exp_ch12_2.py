"""One-shot verification for section 12.2 (preference-based reward model).

Trains a 2-feature Bradley-Terry reward model on 300 preference pairs only,
prints the recovered weights + 3 held-out validation pairs, and saves the
three matplotlib figures used in kor/src/ml2/chapter12/2.md.
Run once; adopt the printed numbers into the section text.
"""
import math
import random

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

IMG = "/home/smhan/book-ml/kor/src/images"
kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
if kr: plt.rcParams["font.sans-serif"] = [kr[0]]
plt.rcParams["axes.unicode_minus"] = False


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))


# ---------------- "true" hidden reward: 2*smooth - 1*rough ---------------
def r_star(traj):
    return 2.0 * traj[0] - 1.0 * traj[1]


rng = random.Random(42)

# 300 preference pairs: two random trajs per pair, human labels the better one
# by comparing hidden r_star (the learner NEVER sees r_star, only the label).
pairs = []
while len(pairs) < 300:
    a = (rng.uniform(0, 5), rng.uniform(0, 5))
    b = (rng.uniform(0, 5), rng.uniform(0, 5))
    ra, rb = r_star(a), r_star(b)
    if ra == rb:
        continue
    pairs.append((a, b) if ra > rb else (b, a))   # (win, lose)

print(f"생성된 선호 쌍: {len(pairs)}개")

# ---------------- learn r_phi(t) = w1*t[0] + w2*t[1] + b -----------------
# NOTE: b is NOT learned here. Bradley-Terry depends only on the difference
# r_w - r_l, in which any shared bias cancels exactly, so b is unidentifiable
# (it would just drift). The original section code matches this: only w1,w2 move.
w1, w2, b = 0.0, 0.0, 0.0
lr = 0.01
history = []
for epoch in range(200):
    total = 0.0
    for win, lose in pairs:
        rw = w1 * win[0] + w2 * win[1] + b
        rl = w1 * lose[0] + w2 * lose[1] + b
        p = sigmoid(rw - rl)
        g = p - 1.0
        total += -math.log(max(p, 1e-12))
        w1 -= lr * g * (win[0] - lose[0])
        w2 -= lr * g * (win[1] - lose[1])
        # b is deliberately not updated (cancels in r_w - r_l)
    history.append(total / len(pairs))

print(f"학습된 가중치: w1(부드러움)={w1:.3f}, w2(덜컹거림)={w2:.3f} (b=0, 학습되지 않음)")
print(f"진짜 가중치:   w1=2, w2=-1")
print(f"학습된 비율 w1/w2 = {w1 / w2:.2f}   (실제 비율 = -2)")

# held-out pairs (never shown in training) — binary "which side" + margin
val = [((4.0, 1.0), (1.0, 4.0)),
       ((3.0, 3.0), (1.0, 1.0)),
       ((5.0, 0.0), (0.0, 5.0))]
for (s, r1), (s2, r2) in val:
    win = (max(r1, r2), min(r1, r2))
    lose = (min(r1, r2), max(r1, r2))
    rw = w1 * win[0] + w2 * win[1]
    rl = w1 * lose[0] + w2 * lose[1]
    pred = "앞쪽" if rw > rl else "뒤쪽"
    print(f"({s},{r1}) vs ({s2},{r2}): 실제 선호=앞쪽(높은 쪽), 모델 예측={pred}, 차이={rw - rl:.2f}")

# ---------------- figure 1: recovered reward direction -------------------
import numpy as np
fig, ax = plt.subplots(figsize=(5.4, 5.4))
xs = [t for t in range(-5, 7)]
X, Y = np.meshgrid(xs, xs)
Z = w1 * X + w2 * Y
cs = ax.contour(X, Y, Z, levels=14, colors="#adb5bd", linewidths=1.0)
ax.clabel(cs, fontsize=7, fmt="%.0f")
# true direction (2, -1) and learned direction (w1, w2)
ax.arrow(0, 0, 2, -1, head_width=0.45, head_length=0.35, fc="#1971c2", ec="#1971c2",
         lw=2.2, zorder=5)
ax.text(2.2, -1.5, "진짜\n$r^*$=(2,-1)", color="#1971c2", fontsize=10)
ax.arrow(0, 0, w1 * 0.55, w2 * 0.55, head_width=0.45, head_length=0.4,
         fc="#e8590c", ec="#e8590c", lw=2.2, zorder=5)
ax.text(w1 * 0.55 + 0.25, w2 * 0.55 + 0.5, f"학습된\n(${w1:.2f},{w2:.2f}$)",
        color="#e8590c", fontsize=10)
ax.scatter([4, 1, 5], [1, 4, 0], s=60, color="#51cf66", zorder=4)
ax.scatter([1, 4, 0], [4, 1, 5], s=60, color="#ff8787", zorder=4)
ax.set_xlabel("부드러움")
ax.set_ylabel("덜컹거림")
ax.set_title("2개 특징 궤적공간에서 복원된 보상 방향")
fig.savefig(IMG + "/ch12_2_reward_direction.svg", bbox_inches="tight")
plt.close(fig)

# ---------------- figure 2: training loss curve --------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(range(1, 201), history, color="#1971c2", lw=1.6)
ax.set_xlabel("epoch")
ax.set_ylabel("평균 Bradley-Terry 손실  −log σ(r_w − r_l)")
ax.set_title("선호 쌍 300개만으로의 학습 곡선 (lr=0.01)")
ax.grid(alpha=0.3)
fig.savefig(IMG + "/ch12_2_reward_loss_curve.svg", bbox_inches="tight")
plt.close(fig)

print("SVG 저장 완료: ch12_2_reward_direction, ch12_2_reward_loss_curve")
