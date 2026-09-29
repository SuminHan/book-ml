"""p33: 수정된 정책 반복의 k-sweep 트레이드오프.
'왼쪽'(최악) 정책에서 시작, k sweep만 평가 후 개선을 반복.
k가 작을수록(=가치반복에 가까울수록) 이 예제에서는 훨씬 빨리 최적에 도달;
k=∞(=완전한 정책반복)는 첫 평가만으로 ~133 sweep을 낭비.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

GAMMA = 0.9
NS = 5  # states 0..4, 4=goal


def step_value(V, s, a):
    # a=0: left (wall at 0, bounce to self), a=1: right (goal at 4)
    if a == 0:
        s2 = max(s - 1, 0)
    else:
        s2 = s + 1
    if s2 == NS - 1:
        return -1 + GAMMA * 0.0
    return -1 + GAMMA * V[s2]


def evaluate_k_sweeps(policy, V, k):
    total = 0
    for _ in range(k):
        newV = V.copy()
        for s in range(NS - 1):
            newV[s] = step_value(V, s, policy[s])
        V = newV
        total += 1
    return V, total


def greedy(V):
    pol = np.zeros(NS - 1, dtype=int)
    for s in range(NS - 1):
        qs = [step_value(V, s, a) for a in (0, 1)]
        pol[s] = int(np.argmax(qs))
    return pol


def modified_policy_iteration(k, max_outer=500, tol=1e-6, eval_cap=2000):
    policy = np.zeros(NS - 1, dtype=int)  # 왼쪽(0) 정책 = 최악의 출발
    V = np.zeros(NS)
    total_sweeps = 0
    for outer in range(max_outer):
        if k == 'full':
            # 완전 평가: 변화가 tol 밑으로 떨어질 때까지
            for _ in range(eval_cap):
                newV = V.copy()
                for s in range(NS - 1):
                    newV[s] = step_value(V, s, policy[s])
                total_sweeps += 1
                if np.max(np.abs(newV - V)) < tol:
                    V = newV
                    break
                V = newV
        else:
            V, n = evaluate_k_sweeps(policy, V, k)
            total_sweeps += n
        new_policy = greedy(V)
        if np.array_equal(new_policy, policy) and outer > 0:
            return total_sweeps, outer + 1
        policy = new_policy
    return total_sweeps, max_outer


ks = [1, 2, 3, 5, 10, 'full']
labels = [f"$k={k}$" for k in ks[:-1]] + ["$k=\\infty$\n(완전 정책반복)"]
totals, outers = [], []
for k in ks:
    t, o = modified_policy_iteration(k)
    totals.append(t)
    outers.append(o)
    print(k, 'total_sweeps=', t, 'outer=', o)

fig, ax = plt.subplots(figsize=(8.2, 4.6))
colors = [KSA_BLUE] * (len(ks) - 1) + [KSA_RED]
bars = ax.bar(range(len(ks)), totals, color=colors, width=0.6)
for i, t in enumerate(totals):
    ax.text(i, t * 1.05 if t > 0 else 0.3, f"{t}", ha='center', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(range(len(ks)))
ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylabel("최적 정책까지 필요한 총 sweep 수 (log)", fontsize=12)
ax.set_title("'왼쪽'(최악)에서 출발 --- $k$ 가 작을수록 이 예제에서는 훨씬 빨리 수렴\n"
             "($k=\\infty$ 는 나쁜 정책을 끝까지 평가하느라 sweep을 낭비)",
             fontsize=12.5, color=KSA_BLUE_DK)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'modified_pi_tradeoff.png')
savefig(fig, out)
print("saved", out)
