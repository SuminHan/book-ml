"""p15g1: TD(0) convergence of V(3) for alpha=0.1 vs alpha=0.5 (Robbins-Monro intuition)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, RED, GRAY
import matplotlib.pyplot as plt
import numpy as np
import random

def td0_track(seed=42, n_episodes=300, alpha=0.1, gamma=1.0, start=3):
    rng = random.Random(seed)
    V = [0.0]*7
    trace = []
    for _ in range(n_episodes):
        s = start
        while s not in (0, 6):
            sp = s + rng.choice([-1, 1])
            r = 1.0
            V[s] += alpha*(r + gamma*(0.0 if sp in (0,6) else V[sp]) - V[s])
            s = sp
        trace.append(V[3])
    return np.array(trace)

v_small = td0_track(alpha=0.1)
v_big = td0_track(alpha=0.5)

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
ep = np.arange(1, len(v_small)+1)
ax.plot(ep, v_big, color=RED, lw=1.6, label=r'$\alpha=0.5$ (더 크게 흔들림)')
ax.plot(ep, v_small, color=KSA_MAIN, lw=1.8, label=r'$\alpha=0.1$ (안정적으로 착지)')
ax.axhline(9, color=GRAY, ls='--', lw=1.5, label=r'참값 $V^{*}(3)=9$')
ax.set_xlabel('에피소드', fontsize=12)
ax.set_ylabel('$V(3)$ 추정치', fontsize=12)
ax.set_title(r'학습률 $\alpha$에 따른 $V(3)$ 수렴 궤적 (7-state random walk, 시드 42)',
             fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.tick_params(labelsize=11)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_alpha_convergence.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
