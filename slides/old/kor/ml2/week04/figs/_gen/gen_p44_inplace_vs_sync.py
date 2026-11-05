"""p44: in-place vs synchronous -- 왜 여기서 일치하는가?
왼쪽 패널: sweep 순서 0->4 (정보흐름과 반대) -- in-place == synchronous.
오른쪽 패널: sweep 순서 4->0 (정보흐름과 같음) -- in-place가 더 빠르게 수렴.
칸0의 V_k(0) 궤적을 두 방식으로 비교 (5칸 GridWorld, 오른쪽 정책, gamma=0.9).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

Vt = vstar()
V0_TRUE = Vt[0]
K = 6


def synchronous(order=None):
    """order 인자는 무시 -- synchronous는 항상 이전 sweep 값만 사용."""
    V = np.zeros(N)
    traj = [V[0]]
    for k in range(1, K + 1):
        newV = np.zeros(N)
        for s in range(N - 1):
            newV[s] = -1 + GAMMA * V[s + 1]
        V = newV
        traj.append(V[0])
    return traj


def inplace(order):
    V = np.zeros(N)
    traj = [V[0]]
    for k in range(1, K + 1):
        for s in order:
            if s == N - 1:
                continue
            V[s] = -1 + GAMMA * V[s + 1]
        traj.append(V[0])
    return traj


sync_traj = synchronous()
inplace_asc = inplace(range(0, N))          # 0->4, 정보흐름과 반대
inplace_desc = inplace(range(N - 1, -1, -1))  # 4->0, 정보흐름과 같음

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
ks = np.arange(K + 1)

ax = axes[0]
ax.plot(ks, sync_traj, 'o-', color=KSA_BLUE, lw=2.4, ms=8, label='synchronous', zorder=2)
ax.plot(ks, inplace_asc, 'x', color=KSA_RED, ms=11, mew=2.4, label='in-place (0→4 순서)', zorder=3)
ax.axhline(V0_TRUE, color=KSA_GRAY, ls='--', lw=1, label='$V^{*}(0)=-3.439$')
ax.set_title("sweep 순서 0→4\n(정보흐름과 반대) → 완전히 일치", fontsize=12.5, color=KSA_BLUE_DK)
ax.set_xlabel("sweep $k$", fontsize=12.5)
ax.set_ylabel("$V_k(0)$", fontsize=13)
ax.legend(fontsize=10, loc='lower right')

ax = axes[1]
ax.plot(ks, sync_traj, 'o-', color=KSA_BLUE, lw=2.4, ms=8, label='synchronous')
ax.plot(ks, inplace_desc, 's-', color=KSA_RED, lw=2.2, ms=8, label='in-place (4→0 순서)')
ax.axhline(V0_TRUE, color=KSA_GRAY, ls='--', lw=1, label='$V^{*}(0)=-3.439$')
ax.set_title("sweep 순서 4→0\n(정보흐름과 같음) → in-place가 더 빠름", fontsize=12.5, color=KSA_BLUE_DK)
ax.set_xlabel("sweep $k$", fontsize=12.5)
ax.legend(fontsize=10, loc='lower right')

for ax in axes:
    ax.set_xticks(ks)

fig.suptitle("in-place vs synchronous: 수렴 속도만 다르고 고정점은 항상 같다",
             fontsize=13.5, color=KSA_BLUE_DK, y=1.03)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'inplace_vs_sync.png')
savefig(fig, out)
print("saved", out)
print("sync:", [round(v, 3) for v in sync_traj])
print("inplace asc:", [round(v, 3) for v in inplace_asc])
print("inplace desc:", [round(v, 3) for v in inplace_desc])
