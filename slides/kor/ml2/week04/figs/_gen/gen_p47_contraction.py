"""p47: T는 gamma-축소 사상. 서로 다른 시작점 3개가 모두 같은 고정점으로,
매 반복 gamma배씩 줄어들며 수렴하는 모습을 시각화.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

Vt = vstar()[0]  # -3.439, 칸0 기준 고정점
starts = [5.0, 0.0, -9.0]
K = 12
gamma = GAMMA

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

ax = axes[0]
colors = [KSA_BLUE, KSA_RED, KSA_GOLD_DK]
for v0, c in zip(starts, colors):
    seq = [v0]
    v = v0
    for k in range(K):
        v = Vt + gamma * (v - Vt)  # T의 축소 성질을 그대로 표현: 거리 gamma배 축소
        seq.append(v)
    ax.plot(range(K + 1), seq, 'o-', color=c, lw=1.8, ms=4.5, label=f"$V_0={v0:g}$")
ax.axhline(Vt, color='black', ls='--', lw=1.3, label=f"고정점 $V^{{*}}={Vt:g}$")
ax.set_xlabel("반복 횟수 $k$", fontsize=12.5)
ax.set_ylabel("$V_k$", fontsize=13)
ax.set_title("어디서 출발해도 같은 고정점으로", fontsize=13, color=KSA_BLUE_DK)
ax.legend(fontsize=9.5, loc='center right')

ax = axes[1]
dist0 = 9.0
ks = np.arange(0, K + 1)
dist = dist0 * gamma ** ks
ax.plot(ks, dist, 'o-', color=KSA_BLUE, lw=2.2, ms=5, label="$\\|V_k - V^{*}\\|$")
ax.plot(ks, dist0 * gamma ** ks, '--', color=KSA_GRAY, lw=1.2)
ax.set_yscale('log')
ax.set_xlabel("반복 횟수 $k$", fontsize=12.5)
ax.set_ylabel("거리 (log scale)", fontsize=13)
ax.set_title("매 반복마다 거리 $\\times\\,\\gamma\\,(=0.9)$", fontsize=13, color=KSA_BLUE_DK)
for k in [0, 4, 8, 12]:
    ax.annotate(f"$\\times\\gamma^{{{k}}}$", (k, dist[k]), textcoords="offset points",
                xytext=(4, 6), fontsize=9.5, color=KSA_RED)
ax.legend(fontsize=10)

fig.suptitle("$T$ 는 $\\gamma$-축소 사상: 반복마다 오차가 최소 $\\gamma$ 배 줄어든다",
             fontsize=13.5, color=KSA_BLUE_DK, y=1.03)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'contraction_mapping.png')
savefig(fig, out)
print("saved", out)
