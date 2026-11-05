"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (assets/hr.json · assets/baseball.json · assets/formulation.json)."""
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


# 1. 타구: 같은 데이터, 두 문제 (2024-06 인플레이 타구, 시험 30%)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
bars(a1, ["기준선\n늘 '홈런 아님'", "로지스틱회귀"], [BAT["acc_base"] * 100, BAT["acc_model"] * 100], [GRAY, BLUE],
     lambda v: f"{v:.1f}%", "시험 정확도 (%) ↑", 118)
a1.set_title("문제 A: 속도 · 발사각 → 홈런인가 (분류)", fontsize=15)
bars(a2, ["기준선\n학습 평균", "선형회귀"], [BAT["mae_base"], BAT["mae_model"]], [GRAY, GREEN],
     lambda v: f"{v:.1f} m", "시험 MAE (m) ↓", 47)
a2.set_title("문제 B: 속도 · 발사각 → 비거리 (회귀)", fontsize=15)
fig.tight_layout(); fig.savefig(O + "pf_batted.png"); plt.close(fig)

# 2. 다음 시즌 홈런 수: 기준선 vs 모델 (2023 → 2024, 207명)
fig, ax = plt.subplots(figsize=(10, 4.3))
labs = ["학습 평균", "작년 그대로", "kNN 모델", "선형회귀\n(작년 × 0.61 + 6.0)"]
vals = [NEXT["mae_mean"], NEXT["mae_last"], NEXT["mae_knn"], NEXT["mae_lin"]]
bars(ax, labs, vals, [GRAY, BLUE, RED, GREEN], lambda v: f"{v:.2f}", "시험 MAE (홈런 개수) ↓", 9.2)
ax.axhline(NEXT["mae_last"], color=BLUE, ls=":", lw=1.5)
fig.tight_layout(); fig.savefig(O + "pf_nexthr.png"); plt.close(fig)

# 5. 1쪽 질문: 모바일 야구 게임 화면 (컴투스프로야구2021, 사용자 제공 캡처) 위에 '치기 전 AI 예측' 자막 한 줄
# 원본 photos/com2us_baseball_2021.png (1631x924) 에서 에뮬레이터 창 테두리 (위 40px · 오른쪽 사이드바) 만 잘라낸다
def homerun_scene(shot, path):
    from matplotlib.patches import FancyBboxPatch as FB
    from PIL import Image
    im = np.asarray(Image.open(shot).convert("RGB").crop((0, 40, 1572, 924)))
    h, w = im.shape[:2]
    fig = plt.figure(figsize=(8.0, 8.0 * h / w))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    ax.imshow(im, zorder=0)
    # 아래 캐스터 자막 자리를 'AI 예측' 자막으로 덮는다
    ax.add_patch(FB((330, h - 62), 880, 48, boxstyle="round,pad=4,rounding_size=8", fc="#0b0f1a", ec="#3ad7ff", lw=2, alpha=0.96, zorder=5))
    ax.text(352, h - 38, "[AI 예측]", color="#3ad7ff", fontsize=12, fontweight="bold", va="center", zorder=6)
    ax.text(500, h - 38, "공을 치기 전에 예측합니다.  이번 타석은 홈런 아님", color="white", fontsize=12, fontweight="bold", va="center", zorder=6)
    fig.savefig(path, dpi=200); plt.close(fig)


homerun_scene("photos/com2us_baseball_2021.png", O + "pf_homerun.png")
# 6. 1쪽 답: 타석 100개 중 홈런 3개 (2024 MLB 2.99%) 를 격자로, AI 는 전부 '아님'
fig, ax = plt.subplots(figsize=(5.2, 5.2))
hr_cells = round(HR["mlb_hr"] / HR["mlb_pa"] * 100)          # 2.99% -> 3 칸
for k in range(100):
    r, c = divmod(k, 10)
    is_hr = k in (23, 56, 88)[:hr_cells]
    # AI 는 전부 '아님' → 홈런 아닌 칸은 맞음 (초록 ✓), 홈런 칸은 놓침 (빨강 ✗)
    ax.add_patch(plt.Rectangle((c, 9 - r), 0.86, 0.86, fc=(RED if is_hr else "#cfeae6"), ec="none"))
    ax.text(c + 0.43, 9 - r + 0.43, "✗" if is_hr else "✓", ha="center", va="center", fontsize=12,
            color=("white" if is_hr else GREEN), fontweight="bold", family="DejaVu Sans")
ax.set_xlim(-0.1, 10); ax.set_ylim(-0.1, 10); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(O + "pf_waffle.png"); plt.close(fig)


print("figs done")
