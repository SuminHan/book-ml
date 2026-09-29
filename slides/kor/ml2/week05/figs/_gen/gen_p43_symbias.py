"""p43: 5칸 회랑에서 그냥 평균낸 값이 b의 대칭성에 따라 어떻게 수렴하는지.
대칭 b(50-50): 평균 -> V^b(2) = 0. 비대칭 b(70% 오른쪽): 평균 -> V^b(2) != 0, V^pi(2)=0.9 와도 다름.
"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
RED = '#C0392B'
SAND = '#D6CBB1'

GAMMA = 0.9


def simulate(p_right, n_episodes, rng, start=2, n_max=200):
    """5칸 회랑(0~4, 0/4 종료, gamma=0.9): b의 행동확률 p_right로 리턴 시뮬레이션."""
    means = []
    total, count = 0.0, 0
    checkpoints = np.unique(np.round(np.logspace(0, np.log10(n_episodes), 60)).astype(int))
    cp_set = set(checkpoints.tolist())
    for ep in range(1, n_episodes + 1):
        s = start
        rewards = []
        for _ in range(n_max):
            a_right = rng.random() < p_right
            if a_right:
                s2 = s + 1
            else:
                s2 = s - 1
            if s2 == 4:
                rewards.append(1.0); s = s2; break
            elif s2 == 0:
                rewards.append(-1.0); s = s2; break
            else:
                rewards.append(0.0); s = s2
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
        total += G
        count += 1
        if ep in cp_set:
            means.append((ep, total / count))
    return means


rng1 = np.random.default_rng(7)
rng2 = np.random.default_rng(7)
sym = simulate(0.5, 20000, rng1)
asym = simulate(0.7, 20000, rng2)

fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=200)
xs1 = [m[0] for m in sym]; ys1 = [m[1] for m in sym]
xs2 = [m[0] for m in asym]; ys2 = [m[1] for m in asym]

ax.plot(xs1, ys1, color=NAVY, lw=2.4, marker='o', markersize=3.5,
        label=r"대칭 $b$ (50–50) → 평균 $\to V^b(2) \approx 0$")
ax.plot(xs2, ys2, color=RED, lw=2.4, marker='s', markersize=3.5,
        label=r"비대칭 $b$ (오른쪽 70%) → 평균 $\to V^b(2) \approx$ " + f"{ys2[-1]:.2f}")
ax.axhline(0.9, color='gray', ls='--', lw=1.8, label=r"목표 $V^\pi(2) = 0.9$ (둘 다 도달 못함)")
ax.axhline(0, color='#bbbbbb', ls=':', lw=1.2)

ax.set_xscale('log')
ax.set_xlabel("에피소드 수 (로그 스케일)", fontsize=12.5)
ax.set_ylabel("그냥 평균낸 리턴 (편향 추정치)", fontsize=12.5)
ax.set_title("대칭 행동 정책은 우연히 0, 비대칭이면 다른 값으로 --- 둘 다 $V^\\pi$ 는 아니다",
              fontsize=12.5, color=NAVY)
ax.legend(fontsize=10, loc='center right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(-0.5, 1.05)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_3_sym_bias.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out, "sym_final=", ys1[-1], "asym_final=", ys2[-1])
