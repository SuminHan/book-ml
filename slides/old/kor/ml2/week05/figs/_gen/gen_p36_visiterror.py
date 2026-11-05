"""p36: p32 표(학습 Q vs 정확 Q*, 방문 횟수)의 실제 숫자로
방문 횟수 vs |Q - Q*| 오차 산점도. sigma/sqrt(N) 참고곡선 덧그림."""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
RED = '#C0392B'
SAND = '#D6CBB1'

# p32 표에서 직접: (상태, 행동, 방문횟수, |학습Q - 정확Q*|)
rows = [
    ("(20,3) 스틱", 2678, abs(0.633 - 0.656)),
    ("(20,3) 히트", 168, abs(-0.839 - (-0.828))),
    ("(20,10) 스틱", 3143, abs(0.520 - 0.540)),
    ("(20,10) 히트", 169, abs(-0.811 - (-0.832))),
    ("(12,10) 스틱", 234, abs(-0.539 - (-0.532))),
    ("(12,10) 히트", 3770, abs(-0.253 - (-0.206))),
    ("(12,1) 스틱", 243, abs(-0.366 - (-0.380))),
    ("(12,1) 히트", 3806, abs(-0.131 - (-0.122))),
    ("(15,10) 스틱", 342, abs(-0.550 - (-0.532))),
    ("(15,10) 히트", 3873, abs(-0.404 - (-0.388))),
    ("(15,2) 스틱", 394, abs(-0.391 - (-0.363))),
    ("(15,2) 히트", 3768, abs(-0.356 - (-0.321))),
]

fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=200)
visits = np.array([r[1] for r in rows])
errs = np.array([r[2] for r in rows])
low = visits < 500
ax.scatter(visits[low], errs[low], s=110, color=RED, edgecolor='black', linewidth=0.8,
           label="방문 < 500회 (스틱, 12~20 경계)", zorder=3)
ax.scatter(visits[~low], errs[~low], s=110, color=NAVY, edgecolor='black', linewidth=0.8,
           label="방문 > 3000회 (히트, 주된 경로)", zorder=3)

offsets = {
    "(12,10) 히트": (-70, 14),
    "(15,2) 히트": (-10, 12),
    "(20,3) 스틱": (8, -14),
    "(20,10) 스틱": (8, -14),
    "(15,10) 히트": (8, -16),
    "(12,1) 히트": (8, 10),
}
for name, v, e in rows:
    dx, dy = offsets.get(name, (7, 6))
    ax.annotate(name, (v, e), textcoords="offset points", xytext=(dx, dy), fontsize=8.6,
                color='#333333')

# 참고 곡선: c/sqrt(N), c는 방문 적은 점들 평균에 맞춤
c = np.mean(errs[low] * np.sqrt(visits[low]))
xs = np.linspace(100, 4200, 200)
ax.plot(xs, c / np.sqrt(xs), color='gray', lw=1.8, ls='--', zorder=1,
        label=r"참고: $\sigma/\sqrt{N}$ 형태 (표준오차)")

ax.axvspan(0, 500, color=SAND, alpha=0.4, zorder=0)
ax.text(480, ax.get_ylim()[1]*0.95 if False else 0.052, "방문 부족\n위험 구간", fontsize=10,
        color=RED, ha='right', va='top')

ax.set_xlabel("방문 횟수 (상태·행동 쌍, 30만 에피소드)", fontsize=12)
ax.set_ylabel(r"학습 $Q$ 와 $Q^{*}$ 의 차이 $|\Delta Q|$", fontsize=12.5)
ax.set_title("방문이 적은 (상태,행동)일수록 Q 추정 오차가 크다 (p32 표 실제 값)",
              fontsize=12.5, color=NAVY)
ax.legend(fontsize=9.5, loc='upper left', bbox_to_anchor=(0.30, 1.0), frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0, 4200)
ax.set_ylim(0, 0.052)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_2_visit_error.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
