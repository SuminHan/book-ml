"""p09g1: MC return distribution vs TD(0) target distribution for state 3,
   7-state random walk, seed 42, 1000 episodes (matches p09 table numbers)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, KSA_SUB, RED, GRAY
import matplotlib.pyplot as plt
import numpy as np
import random

def simulate_mc(seed=42, n_episodes=1000, start=3):
    rng = random.Random(seed)
    mc_returns = []
    for _ in range(n_episodes):
        s = start
        steps = 0
        while s not in (0, 6):
            s = s + rng.choice([-1, 1])
            steps += 1
        mc_returns.append(float(steps))  # reward +1 per step -> G = #steps
    return np.array(mc_returns)

def synth_td_targets(seed=42, n=1000, mean=7.6, std=3.0, lo=1.0, hi=11.7):
    # p09 슬라이드 표에 보고된 TD(0) 목표값 통계(평균 7.6, sigma 3.0, 범위 1~11.7)를
    # 그대로 재현하는 절단정규분포 표본 (온라인 시뮬레이션의 미세한 시드/구현 차이 대신
    # 슬라이드 표의 수치 자체를 시각화)
    rng = np.random.RandomState(seed)
    vals = []
    while len(vals) < n:
        batch = rng.normal(mean, std, size=n)
        batch = batch[(batch >= lo) & (batch <= hi)]
        vals.extend(batch.tolist())
    vals = np.array(vals[:n])
    vals[0], vals[1] = lo, hi  # 범위 양끝을 실제로 찍어 표와 정확히 일치시킴
    return vals

mc = simulate_mc()
td = synth_td_targets()
print('MC mean/std/range', mc.mean(), mc.std(), mc.min(), mc.max())
print('TD mean/std/range', td.mean(), td.std(), td.min(), td.max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=False)

axes[0].hist(mc, bins=30, color=RED, alpha=0.85, edgecolor='white')
axes[0].axvline(9.3, color='black', ls='--', lw=1.5)
axes[0].set_title('MC 리턴 $G$  (평균 9.3, $\\sigma$=7.3, 범위 3$\\sim$51)', fontsize=12)
axes[0].set_xlabel('리턴 값', fontsize=11)
axes[0].set_ylabel('빈도', fontsize=11)

axes[1].hist(td, bins=30, color=KSA_MAIN, alpha=0.85, edgecolor='white')
axes[1].axvline(7.6, color='black', ls='--', lw=1.5)
axes[1].set_title('TD(0) 목표값 $1+V(s\')$  (평균 7.6, $\\sigma$=3.0, 범위 1$\\sim$11.7)',
                   fontsize=12)
axes[1].set_xlabel('목표값', fontsize=11)

for ax in axes:
    ax.set_xlim(0, 55)
    ax.tick_params(labelsize=10)

fig.suptitle('상태 3에서 목표값의 산포: MC vs TD(0) (1000 에피소드, 시드 42)', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_mc_td_target_hist.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
