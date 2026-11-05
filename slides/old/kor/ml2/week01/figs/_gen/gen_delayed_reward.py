"""p26g1: 지도학습의 즉각 신호 vs 강화학습의 지연 보상 타임라인"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)

steps = np.arange(1, 11)

# --- top: 지도학습, 매 샘플 라벨 있음 ---
ax = axes[0]
ax.scatter(steps, np.ones_like(steps), s=140, color=KSA_BLUE, zorder=3)
for s in steps:
    ax.annotate("O", (s, 1), ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)
ax.set_ylim(0.5, 1.5)
ax.set_yticks([])
ax.set_title("지도학습: 매 샘플마다 정답이 즉시 붙음", fontsize=13, loc="left", color=KSA_BLUE)
for s in steps:
    ax.annotate("", xy=(s, 1.3), xytext=(s, 1.05),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=1))
ax.text(10.6, 1.0, "각 샘플 = 정답 1개", fontsize=10, va="center", color=GREY)
ax.set_xlim(0.3, 14.5)

# --- bottom: 강화학습, 보상은 끝에만 ---
ax = axes[1]
rewards = np.zeros_like(steps, dtype=float)
rewards[-1] = 1.0  # 에피소드 끝에만 승/패 신호
ax.axhline(0, color=GREY, lw=1)
markerline, stemlines, baseline = ax.stem(steps, rewards)
plt.setp(stemlines, color=ACCENT_RED, linewidth=2)
plt.setp(markerline, color=ACCENT_RED, markersize=11)
plt.setp(baseline, visible=False)
for s in steps[:-1]:
    ax.scatter([s], [0], s=90, facecolor="white", edgecolor=KSA_BLUE, lw=1.8, zorder=3)
ax.annotate("승/패 신호\n(에피소드 끝)", xy=(10, 1.0), xytext=(7.6, 1.35),
            fontsize=10.5, color=ACCENT_RED, ha="left",
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.5))
ax.annotate("이 수(예: 3번째 행동)가\n결과에 책임이 있나?", xy=(3, 0), xytext=(1.2, 1.35),
            fontsize=10, color=KSA_BLUE, ha="left",
            arrowprops=dict(arrowstyle="->", color=KSA_BLUE, lw=1.3))
ax.set_ylim(-0.4, 1.7)
ax.set_yticks([])
ax.set_title("강화학습: 보상은 에피소드가 끝나야 한 번 옴 (신용 할당 문제)",
              fontsize=13, loc="left", color=ACCENT_RED)
ax.set_xlim(0.3, 14.5)
ax.set_xticks(steps)
ax.set_xlabel("시간 스텝 / 행동 번호", fontsize=11)

plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "delayed_reward_timeline.png"))
