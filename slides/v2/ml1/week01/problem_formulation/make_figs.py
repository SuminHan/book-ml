"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (assets/formulation.json · assets/aml.json)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"


def bars(ax, labels, vals, colors, fmt, ylabel, ymax=None):
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    top = ymax or max(vals) * 1.18
    for i, v in enumerate(vals):
        ax.text(i, v + top * 0.02, fmt(v), ha="center", fontsize=17, fontweight="bold", color=colors[i])
    ax.set_xticks(range(len(vals)), labels, fontsize=13.5); ax.set_ylim(0, top); ax.set_ylabel(ylabel)


# 1. 와인: 같은 데이터, 두 문제
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
bars(a1, ["기준선\n가장 많은 품종", "로지스틱회귀"], [WINE_ACC["base"], WINE_ACC["model"]], [GRAY, BLUE],
     lambda v: f"{v}%", "시험 정확도 (%) ↑", 115)
a1.set_title("문제 A: 13성분 → 품종 (분류)", fontsize=15)
bars(a2, ["기준선\n학습 평균", "선형회귀"], [WINE_MAE["base"], WINE_MAE["model"]], [GRAY, GREEN],
     lambda v: f"{v:.2f}", "시험 MAE (도) ↓", 0.85)
a2.set_title("문제 B: 12성분 → 알코올 도수 (회귀)", fontsize=15)
fig.tight_layout(); fig.savefig(O + "pf_wine.png"); plt.close(fig)

# 2. 항공 승객: 기준선 vs 모델
fig, ax = plt.subplots(figsize=(10, 4.3))
labs = ["학습 기간 평균", "지난달 그대로", "작년 같은 달", "kNN 모델"]
vals = [AIR_MAE["mean"], AIR_MAE["last"], AIR_MAE["season"], AIR_MAE["knn"]]
bars(ax, labs, vals, [GRAY, BLUE, BLUE, RED], lambda v: f"{v}", "시험 MAE (천 명/달) ↓", 240)
ax.axhline(AIR_MAE["last"], color=BLUE, ls=":", lw=1.5)
fig.tight_layout(); fig.savefig(O + "pf_air.png"); plt.close(fig)

# 3. AlphaFold: 두 단백질의 예측 구조 (Cα 사슬, pLDDT 색) + 잔기별 pLDDT
fig = plt.figure(figsize=(13, 4.6))
cmap = matplotlib.colors.ListedColormap(["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"])     # AlphaFold DB 색 (<50, 50-70, 70-90, >90)
norm = matplotlib.colors.BoundaryNorm([0, 50, 70, 90, 100], 4)
for k, uid in enumerate(("P69905", "P37840")):
    a = AF[uid]; xyz = np.array(a["xyz"]); pl = np.array(a["plddt"]); xyz = xyz - xyz.mean(0)
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.plot(*xyz.T, color="#bbbbbb", lw=1)
    ax.scatter(*xyz.T, c=pl, cmap=cmap, norm=norm, s=16, depthshade=False)
    ax.text2D(0.5, 1.0, f"{AF_KO[uid]}\n아미노산 {len(a['seq'])}개, 평균 pLDDT {a['mean_plddt']:.0f}", transform=ax.transAxes, ha="center", va="top", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 1))
ax = fig.add_subplot(1, 3, 3)
for uid, c in (("P69905", BLUE), ("P37840", "#FF7D45")):
    ax.plot(np.arange(1, len(AF[uid]["plddt"]) + 1), AF[uid]["plddt"], color=c, lw=2, label=AF_KO[uid])
ax.axhline(70, color=GRAY, ls=":"); ax.text(2, 72, "70: 믿을 만함", color=GRAY, fontsize=11)
ax.set_xlabel("아미노산 위치"); ax.set_ylabel("pLDDT (예측 신뢰도)"); ax.set_ylim(0, 102); ax.legend(frameon=False, fontsize=11, loc="lower left")
fig.tight_layout(); fig.savefig(O + "pf_af.png"); plt.close(fig)

# 4. AML (assets/aml.json 이 있을 때만)
if HAVE_AML:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars(ax, ["전부 '정상'", "규칙 하나"], [AML["acc_zero"] * 100, AML["acc_rule"] * 100], [GRAY, BLUE],
         lambda v: f"{v:.2f}%", "정확도 (%)", 115)
    fig.tight_layout(); fig.savefig(O + "pf_aml_acc.png"); plt.close(fig)
print("figs done", "(AML 포함)" if HAVE_AML else "(AML 없음)")
