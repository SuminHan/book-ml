import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
SAND = "#D6CBB1"
RED = "#C0392B"

# anchor facts stated in the deck text (p30/p31/p32): min lambda*=15.8 (MSE 9.03),
# fold std 1.66 -> 1SE upper bound 10.69, 1SE-rule pick lambda~=79.4 (MSE 10.59)
x_min, y_min = np.log(15.8), 9.03
x_b, y_b = np.log(79.4), 10.59
se_line = 10.69

a_r = (y_b - y_min) / (x_b - x_min) ** 2
a_l = 1.4 * a_r  # steeper left arm (overfitting side rises faster)

x = np.linspace(np.log(1.0), np.log(300.0), 400)
y = np.where(x >= x_min, y_min + a_r * (x - x_min) ** 2, y_min + a_l * (x - x_min) ** 2)

# left intersection with the 1SE line (for shading the plateau)
x_left = x_min - np.sqrt((se_line - y_min) / a_l)
lam_left = np.exp(x_left)

fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
ax.plot(np.exp(x), y, color=NAVY, lw=2.8, label="5-fold 평균 검증 MSE")

mask = (x >= x_left) & (x <= x_b)
ax.fill_between(np.exp(x[mask]), y[mask], se_line, color=SAND, alpha=0.55,
                 label="plateau (최소점과 통계적으로 구별 안 됨)")

ax.axhline(se_line, color="#555555", ls="--", lw=1.6)
ax.text(1.15, se_line + 0.10, f"min + 1 SE = {se_line}", fontsize=12, color="#333333")

ax.scatter([15.8], [9.03], s=90, color=NAVY, zorder=5)
ax.annotate("최소점\n$\\lambda^{*}\\approx15.8$  (MSE 9.03)", xy=(15.8, 9.03),
            xytext=(2.2, 8.35), fontsize=12.5, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))

ax.scatter([79.4], [10.59], s=110, color=RED, zorder=5, marker="D")
ax.annotate("1SE 규칙 선택\n$\\lambda\\approx79.4$ (5배 더 단순)", xy=(79.4, 10.59),
            xytext=(95, 9.15), fontsize=12.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

ax.text(0.02, 0.03,
        "참고: 시드 10개 반복 시 $\\lambda^{*}$는 약 5~20 사이에서 흩어짐\n"
        "(같은 최소점도 데이터가 조금 바뀌면 자리가 바뀌는 ``추정치'')",
        transform=ax.transAxes, fontsize=10.5, color="#555555", va="bottom")

ax.set_xscale("log")
ax.set_xlabel("$\\lambda$ (로그 스케일)", fontsize=13)
ax.set_ylabel("5-fold 평균 검증 MSE", fontsize=13)
ax.set_title("1SE 규칙: 평탄한 대지에서 가장 단순한 모델 고르기", fontsize=14, color=NAVY, pad=12)
ax.tick_params(labelsize=11)
ax.legend(fontsize=10.5, loc="upper center")
ax.set_xlim(1, 300)
ax.set_ylim(8.0, max(y.max(), 12.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week06/slides/kor/ml1/week06/figs/ch06_2_onese_rule.png",
            facecolor="white")
print("left intersection lambda approx:", lam_left)
