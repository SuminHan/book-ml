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

# 3. 넷플릭스 프라이즈: 숨긴 시험 (왼쪽) + RMSE (오른쪽)
from matplotlib.patches import FancyBboxPatch
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
def box(ax, x, y, w, h, c, t, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="none")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=tc, fontsize=15, fontweight="bold")
box(a1, 0, 2.2, 6.0, 1.2, BLUE, "공개: 별점 1억 개\n(학습)")
box(a1, 6.2, 2.2, 3.8, 1.2, GRAY, "숨김: 사용자마다\n가장 최근 300만 개")
box(a1, 6.2, 0.6, 1.8, 1.2, GOLD, "quiz\n공개 점수판")
box(a1, 8.2, 0.6, 1.8, 1.2, RED, "test\n우승 판정")
a1.annotate("", (8.1, 1.85), (8.1, 2.15), arrowprops=dict(arrowstyle="->", color="black"))
a1.annotate("", (10, 3.6), (0, 3.6), arrowprops=dict(arrowstyle="->", color=GRAY)); a1.text(0, 3.7, "시간 (1998 → 2005)", fontsize=13, color=GRAY)
a1.set_xlim(-0.2, 10.2); a1.set_ylim(0.3, 4.1); a1.axis("off")
vals = [("Cinematch\n(기준선)", NETFLIX["cinematch_test"], GRAY), ("목표\n(10% 개선)", NETFLIX["target_test"], GOLD), ("2009\n우승 팀", NETFLIX["winner_test"], BLUE)]
a2.bar(range(3), [v for _, v, _ in vals], color=[c for _, _, c in vals], width=0.6)
for k, (_, v, c) in enumerate(vals): a2.text(k, v + 0.004, f"{v:.4f}", ha="center", fontsize=14, fontweight="bold", color=c)
a2.set_xticks(range(3), [t for t, _, _ in vals], fontsize=13); a2.set_ylim(0.8, 0.98); a2.set_ylabel("숨긴 test 의 RMSE (작을수록 좋다)")
fig.tight_layout(); fig.savefig(O + "pf_netflix.png"); plt.close(fig)

# 4. AML (assets/aml.json 이 있을 때만)
if HAVE_AML:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars(ax, ["전부 '정상'", "규칙 하나"], [AML["acc_zero"] * 100, AML["acc_rule"] * 100], [GRAY, BLUE],
         lambda v: f"{v:.2f}%", "정확도 (%)", 115)
    fig.tight_layout(); fig.savefig(O + "pf_aml_acc.png"); plt.close(fig)
print("figs done", "(AML 포함)" if HAVE_AML else "(AML 없음)")
