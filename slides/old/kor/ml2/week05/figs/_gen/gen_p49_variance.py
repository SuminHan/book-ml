"""p49: 분산이 큰 학습곡선(off-policy IS, rho*G0 in {0,3.6} 5칸 회랑) vs
분산이 작은 학습곡선(on-policy MC, G0 in [-1,1] 근처) 비교.
둘 다 실제 값(0.9, 대략 정규화)으로 수렴하지만 흔들림의 크기가 다름.
"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
RED = '#C0392B'

rng = np.random.default_rng(3)
N = 3000

# 저분산: on-policy MC 예측류. 평균 0.9, 표준편차 0.19 (p15의 (0,0) 첫방문 분산과 비슷한 스케일)
low_samples = rng.normal(0.9, 0.19, N)
low_running = np.cumsum(low_samples) / np.arange(1, N + 1)

# 고분산: off-policy IS. p44~46 그대로: rho*G0 = 3.6 (확률 0.25) 또는 0 (확률 0.75), 평균 0.9
p_hit = 0.25
hi_vals = np.where(rng.random(N) < p_hit, 3.6, 0.0)
hi_running = np.cumsum(hi_vals) / np.arange(1, N + 1)

fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
x = np.arange(1, N + 1)
ax.plot(x, hi_running, color=RED, lw=1.6,
        label=r"고분산 (off-policy, $\rho G_0 \in \{0, 3.6\}$)")
ax.plot(x, low_running, color=NAVY, lw=2.0,
        label=r"저분산 (on-policy MC, $G_0 \sim \mathcal{N}(0.9, 0.19^2)$)")
ax.axhline(0.9, color='gray', ls='--', lw=1.6, label="진짜 값 0.9 (둘 다 여기로 수렴)")

ax.set_xscale('log')
ax.set_xlim(1, N)
ax.set_xlabel("누적 에피소드 수 (로그 스케일)", fontsize=12.5)
ax.set_ylabel("누적 평균 (학습 곡선)", fontsize=12.5)
ax.set_title("같은 목표로 수렴해도 --- 고분산 곡선은 훨씬 오래 흔들린다",
              fontsize=13, color=NAVY)
ax.legend(fontsize=10.5, loc='upper right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(-0.3, 2.6)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_3_variance_curves.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
