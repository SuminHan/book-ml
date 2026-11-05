"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (assets/formulation.json · assets/aml.json · assets/hr.json)."""
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

# 5. 1쪽 질문: 야구 게임 화면 (타자 뒤 시점) + AI 홈런 예측 패널. 2024 MLB 정규시즌 (assets/hr.json)
def homerun_scene(n_pa, path):
    from matplotlib.patches import Polygon, Ellipse, Circle, FancyBboxPatch as FB, Rectangle
    W, H = 160, 100
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    P = lambda pts, **k: ax.add_patch(Polygon(pts, closed=True, **k))
    # 밤하늘 (위로 갈수록 어둡게) + 조명탑
    sky = np.linspace(0, 1, 256)[:, None] * np.ones((1, 2))
    ax.imshow(sky, extent=[0, W, 62, H], origin="lower", aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("sky", ["#2b4a7a", "#0a1430"]), zorder=0)
    for lx in (18, 60, 100, 142):
        ax.add_patch(Rectangle((lx - 5, 92), 10, 4, fc="#e9eef7", ec="#9fb0c8", lw=0.6, zorder=1))
        ax.add_patch(Circle((lx, 94), 9, fc="#fff8d0", alpha=0.10, lw=0, zorder=1))
    # 관중석 (점 무늬)
    P([(0, 64), (W, 64), (W, 84), (0, 84)], fc="#1d2a44", lw=0, zorder=1)
    rng = np.random.default_rng(7)
    xs, ys = rng.uniform(0, W, 1400), rng.uniform(65, 83.5, 1400)
    ax.scatter(xs, ys, s=2.2, c=rng.choice(["#c9ced8", "#8a93a6", "#d9534f", "#f0d36b", "#5b7fbf"], 1400), lw=0, zorder=2)
    # 전광판 (가운데)
    ax.add_patch(FB((66, 66), 28, 20, boxstyle="round,pad=0.4", fc="#141414", ec="#c7b27a", lw=1.4, zorder=3))
    ax.text(80, 82.5, "2024 MLB", color="#f0d36b", fontsize=9, ha="center", fontweight="bold", zorder=4)
    for k, (t, v) in enumerate([("타석", f"{n_pa:,}"), ("홈런", "?")]):
        ax.text(70, 77.5 - k * 5, t, color="#e9e9e9", fontsize=9, zorder=4)
        ax.text(90, 77.5 - k * 5, v, color="#ffb84d", fontsize=9, ha="right", fontweight="bold", zorder=4)
    # 외야 펜스 + 광고판 (글자 없이 색만)
    P([(0, 58), (W, 58), (W, 64), (0, 64)], fc="#1f5a3a", lw=0, zorder=3)
    for k, c in enumerate(["#2e6fb5", "#d9d9d9", "#c0392b", "#2e6fb5", "#f0d36b", "#d9d9d9", "#2e6fb5", "#c0392b"]):
        ax.add_patch(Rectangle((2 + k * 20, 59), 16, 3.6, fc=c, lw=0, alpha=0.85, zorder=4))
    # 잔디 (원근감 있는 줄무늬: 멀수록 얇게)
    edges = [0, 9, 17, 24, 30.5, 36.5, 42, 47, 51.5, 55, 58]
    for k in range(len(edges) - 1):
        P([(0, edges[k]), (W, edges[k]), (W, edges[k + 1]), (0, edges[k + 1])],
          fc=("#3f8f3a" if k % 2 else "#357d31"), lw=0, zorder=3)
    # 내야 흙 + 내야 잔디 (원근 다이아몬드)
    home, first, second, third = (80, 4), (124, 26), (80, 41), (36, 26)
    t = np.linspace(np.pi * 0.08, np.pi * 0.92, 40)
    arc = [(80 + 62 * np.cos(a), 12 + 34 * np.sin(a)) for a in t]
    P([(30, 0), (130, 0)] + arc[::1][::-1][::-1], fc="#b9875a", lw=0, zorder=4)
    P([(80, 9), (115, 26), (80, 37), (45, 26)], fc="#3f8f3a", lw=0, zorder=5)
    ax.add_patch(Ellipse((80, 25), 15, 4.2, fc="#c49468", lw=0, zorder=6))          # 마운드
    for b in (first, second, third):
        P([(b[0] - 1.6, b[1]), (b[0], b[1] + 0.9), (b[0] + 1.6, b[1]), (b[0], b[1] - 0.9)], fc="white", lw=0, zorder=6)
    ax.plot([home[0], first[0] + 18], [home[1], first[1] + 9], color="white", lw=1.2, zorder=6)
    ax.plot([home[0], third[0] - 18], [home[1], third[1] + 9], color="white", lw=1.2, zorder=6)
    # 투수 (작게)
    px, py = 80, 27
    ax.add_patch(Rectangle((px - 1.3, py), 2.6, 4.5, fc="#eef1f6", lw=0, zorder=7))
    ax.plot([px - 0.8, px - 1.6], [py, py - 2.4], color="#eef1f6", lw=2.2, zorder=7)
    ax.plot([px + 0.8, px + 1.8], [py, py - 2.4], color="#eef1f6", lw=2.2, zorder=7)
    ax.plot([px + 1.3, px + 3.2], [py + 3.8, py + 5.6], color="#eef1f6", lw=2, zorder=7)
    ax.add_patch(Circle((px, py + 5.6), 1.05, fc="#d9a77e", lw=0, zorder=7))
    ax.add_patch(Ellipse((px, py + 6.3), 2.3, 0.9, fc="#1f3b73", lw=0, zorder=8))
    # 홈 플레이트 + 타석 선
    P([(73, 0), (87, 0), (87, 2.2), (80, 4.2), (73, 2.2)], fc="white", ec="#cccccc", lw=0.8, zorder=8)
    ax.add_patch(Rectangle((112, 0), 34, 13, fill=False, ec="white", lw=1.2, alpha=0.8, zorder=8))
    # 타자 (오른쪽 앞, 뒤에서 본 모습)
    U, T = "#d8dbe2", "#b5332e"
    P([(118, 0), (126, 0), (128, 22), (121, 22)], fc=U, ec="#9aa0ab", lw=0.8, zorder=9)         # 왼다리
    P([(132, 0), (140, 0), (137, 22), (130, 22)], fc=U, ec="#9aa0ab", lw=0.8, zorder=9)         # 오른다리
    ax.add_patch(Rectangle((118, 0), 8, 2.2, fc="#111", lw=0, zorder=10)); ax.add_patch(Rectangle((132, 0), 8, 2.2, fc="#111", lw=0, zorder=10))
    P([(119, 21), (139, 21), (142, 48), (117, 48)], fc=U, ec="#9aa0ab", lw=0.8, zorder=9)       # 몸통
    ax.plot([129.5, 129.5], [21, 48], color=T, lw=1.6, zorder=10)                                 # 등 가운데 줄
    ax.add_patch(Rectangle((119, 20), 20, 2.2, fc="#222", lw=0, zorder=10))                     # 벨트
    # 두 팔: 양 어깨에서 오른쪽 어깨 위 손으로 모인다 (뒤에서 본 타격 준비 자세)
    hx, hy = 141, 60
    P([(118, 47), (123, 45), (hx + 1, hy - 2), (hx - 3, hy + 1)], fc=U, ec="#9aa0ab", lw=0.8, zorder=9)
    P([(137, 46), (142, 45), (hx + 2.5, hy - 1), (hx - 1.5, hy)], fc=U, ec="#9aa0ab", lw=0.8, zorder=9)
    ax.plot([hx, 121], [hy + 1, 84], color="#7a4a22", lw=7, solid_capstyle="round", zorder=10)     # 배트
    ax.plot([126.5, 121], [77.5, 84], color="#2b1a0e", lw=7.5, solid_capstyle="round", zorder=10)
    ax.add_patch(Circle((hx, hy), 3, fc="#c6262e", lw=0, zorder=11))                            # 장갑 낀 손
    ax.add_patch(Circle((129.5, 55), 6, fc="#7a5a40", lw=0, zorder=10))                         # 머리
    ax.add_patch(Ellipse((129.5, 57.5), 13.5, 9.5, fc="#14284f", ec="#0a1530", lw=1, zorder=12))  # 헬멧
    ax.add_patch(Ellipse((135.5, 54.5), 6, 2.2, fc="#14284f", lw=0, zorder=12))                  # 챙
    # HUD: 스코어 (왼쪽 위)
    ax.add_patch(FB((3, 84), 40, 12, boxstyle="round,pad=0.6", fc="#0e1d3d", ec="#7fa7e6", lw=1.2, alpha=0.95, zorder=20))
    ax.text(6, 92.3, "AI 홈런 예측", color="white", fontsize=11, fontweight="bold", zorder=21)
    ax.text(6, 86.2, "B ● ○ ○   S ● ○   O ○ ○", color="#ffd166", fontsize=9, zorder=21)
    # HUD: AI 예측 패널 (왼쪽, 스코어 아래)
    ax.add_patch(FB((3, 52), 40, 26, boxstyle="round,pad=0.6", fc="#08121f", ec="#3ad7ff", lw=1.6, alpha=0.93, zorder=20))
    ax.text(6, 74, "이번 타석은?", color="#3ad7ff", fontsize=10, fontweight="bold", zorder=21)
    ax.text(23, 64.5, "홈런 아님", color="#5be38a", fontsize=19, fontweight="bold", ha="center", va="center", zorder=21)
    ax.text(6, 55.3, f"타석 #{n_pa:,}", color="#cfe9f5", fontsize=9, zorder=21)
    fig.savefig(path, dpi=160); plt.close(fig)


homerun_scene(HR["mlb_pa"], O + "pf_homerun.png")
print("figs done", "(AML 포함)" if HAVE_AML else "(AML 없음)")
