"""p53g1: SARSA vs Q-learning learning curves on CliffWalking (matches p51 numbers:
   학습 중 평균 리턴 SARSA approx -31, Q-learning approx -53)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, RED, GRAY
import matplotlib.pyplot as plt
import numpy as np
import random

NROWS, NCOLS = 4, 12
START, GOAL = (3, 0), (3, 11)
N_STATES, N_ACTIONS = NROWS*NCOLS, 4  # 0 up,1 right,2 down,3 left
DR = {0: -1, 1: 0, 2: 1, 3: 0}
DC = {0: 0, 1: 1, 2: 0, 3: -1}

def s2rc(s):
    return divmod(s, NCOLS)

def rc2s(r, c):
    return r*NCOLS + c

def step(s, a):
    r, c = s2rc(s)
    nr, nc = r+DR[a], c+DC[a]
    nr = min(max(nr, 0), NROWS-1)
    nc = min(max(nc, 0), NCOLS-1)
    if nr == 3 and 1 <= nc <= 10:
        return rc2s(*START), -100.0, False
    ns = rc2s(nr, nc)
    done = (nr, nc) == GOAL
    return ns, -1.0, done

def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    row = Q[s]
    return max(range(N_ACTIONS), key=lambda a: row[a])

def train(method, seed=42, n_episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    rng = random.Random(seed)
    Q = [[0.0]*N_ACTIONS for _ in range(N_STATES)]
    returns = []
    start_s = rc2s(*START)
    for ep in range(n_episodes):
        s = start_s
        a = epsilon_greedy(Q, s, epsilon, rng)
        G = 0.0
        for _ in range(500):
            ns, r, done = step(s, a)
            G += r
            na = epsilon_greedy(Q, ns, epsilon, rng)
            if method == 'sarsa':
                target = r + gamma*Q[ns][na]
            else:  # q-learning
                target = r + gamma*max(Q[ns])
            Q[s][a] += alpha*(target - Q[s][a])
            s, a = ns, na
            if done:
                break
        returns.append(G)
    return np.array(returns)

sarsa_r = train('sarsa')
ql_r = train('qlearning')
print('SARSA last100 mean', sarsa_r[-100:].mean())
print('QL last100 mean', ql_r[-100:].mean())

def moving_avg(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='valid')

fig, ax = plt.subplots(figsize=(9.5, 5), dpi=200)
ax.plot(np.arange(len(sarsa_r)), sarsa_r, color=KSA_MAIN, alpha=0.18, lw=0.8)
ax.plot(np.arange(len(ql_r)), ql_r, color=RED, alpha=0.18, lw=0.8)
w = 20
ax.plot(np.arange(w-1, len(sarsa_r)), moving_avg(sarsa_r, w), color=KSA_MAIN, lw=2.2,
        label=f'SARSA (마지막 100ep 평균 {sarsa_r[-100:].mean():.1f})')
ax.plot(np.arange(w-1, len(ql_r)), moving_avg(ql_r, w), color=RED, lw=2.2,
        label=f'Q-learning (마지막 100ep 평균 {ql_r[-100:].mean():.1f})')
ax.set_ylim(-160, 5)
ax.set_xlabel('에피소드', fontsize=12)
ax.set_ylabel('에피소드 리턴 (20-에피소드 이동평균)', fontsize=12)
ax.set_title(r'학습 중(탐험 포함, $\varepsilon=0.1$) 리턴: SARSA vs Q-learning', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.tick_params(labelsize=11)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_sarsa_ql_learning_curves.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
