import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

rng = np.random.default_rng(42)
n = 40
# target: height mean 170 sd 8, weight mean 65 sd 6, correlated
z1 = rng.standard_normal(n)
z2 = 0.75 * z1 + np.sqrt(1 - 0.75**2) * rng.standard_normal(n)
h = 170 + 8 * z1
wt = 65 + 6 * z2
raw = np.column_stack([h, wt])

cov_raw = np.cov(raw.T)
w_raw, v_raw = np.linalg.eigh(cov_raw)
order = np.argsort(w_raw)[::-1]
w_raw, v_raw = w_raw[order], v_raw[:, order]
pc1_raw = v_raw[:, 0]
var_ratio_raw = w_raw[0] / w_raw.sum()

scaled = (raw - raw.mean(0)) / raw.std(0)
cov_s = np.cov(scaled.T)
w_s, v_s = np.linalg.eigh(cov_s)
order = np.argsort(w_s)[::-1]
w_s, v_s = w_s[order], v_s[:, order]
pc1_s = v_s[:, 0]
var_ratio_s = w_s[0] / w_s.sum()

corr = np.corrcoef(raw.T)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

# panel a: raw scatter + PC1 arrow
ax = axes[0]
ax.scatter(raw[:, 0], raw[:, 1], color=BLUE, alpha=0.6, s=30)
c = raw.mean(0)
scale = 18
ax.annotate('', xy=(c[0] + pc1_raw[0]*scale, c[1] + pc1_raw[1]*scale),
            xytext=(c[0] - pc1_raw[0]*scale, c[1] - pc1_raw[1]*scale),
            arrowprops=dict(arrowstyle='-', color=RED, lw=3))
ax.set_title(f'(a) 표준화 없이\nPC1$\\approx$({pc1_raw[0]:.2f}, {pc1_raw[1]:.2f}), 설명분산 {var_ratio_raw*100:.1f}%',
             fontsize=12)
ax.set_xlabel('키 (cm)', fontsize=11)
ax.set_ylabel('몸무게 (kg)', fontsize=11)
ax.set_aspect('auto')

# panel b: scaled scatter + PC1 arrow
ax = axes[1]
ax.scatter(scaled[:, 0], scaled[:, 1], color=BLUE, alpha=0.6, s=30)
scale2 = 2.2
ax.annotate('', xy=(pc1_s[0]*scale2, pc1_s[1]*scale2),
            xytext=(-pc1_s[0]*scale2, -pc1_s[1]*scale2),
            arrowprops=dict(arrowstyle='-', color=RED, lw=3))
ax.set_title(f'(b) 표준화 후 (StandardScaler)\nPC1$\\approx$({pc1_s[0]:.2f}, {pc1_s[1]:.2f}), 설명분산 {var_ratio_s*100:.1f}%',
             fontsize=12)
ax.set_xlabel('키 (표준화)', fontsize=11)
ax.set_ylabel('몸무게 (표준화)', fontsize=11)
ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)

# panel c: covariance vs correlation matrix heatmap
ax = axes[2]
mat = np.array([[cov_raw[0,0], cov_raw[0,1]], [corr[0,0]*1, corr[0,1]]])
labels = [['공분산\n키-키\n'+f'{cov_raw[0,0]:.1f}', '공분산\n키-몸무게\n'+f'{cov_raw[0,1]:.1f}'],
          ['상관계수\n키-키\n'+f'{corr[0,0]:.2f}', '상관계수\n키-몸무게\n'+f'{corr[0,1]:.2f}']]
disp = np.array([[cov_raw[0,0], cov_raw[0,1]], [corr[0,0], corr[0,1]]])
im = ax.imshow([[1,0.3],[1,0.3]], cmap='Blues', vmin=0, vmax=1.3)
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i][j], ha='center', va='center', fontsize=11,
                color='black')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('(c) 공분산 행렬 vs 상관행렬\n(단위 있음 vs 단위 없음)', fontsize=12)

fig.suptitle('키·몸무게 40명: 표준화 전후 PC1 방향과 공분산/상관행렬', fontsize=14, color=BLUE, y=1.03)
plt.tight_layout()
plt.savefig('/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week14/slides/kor/ml1/week14/figs/ch14_scale_corr.png',
            bbox_inches='tight', facecolor='white')
print("PC1 raw", pc1_raw, var_ratio_raw)
print("PC1 scaled", pc1_s, var_ratio_s)
print("corr", corr)
