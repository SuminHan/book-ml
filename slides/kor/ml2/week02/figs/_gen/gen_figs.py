"""
ml2-week02 (멀티암 밴딧) 추가 개념도 생성 스크립트.
KSA 색상: 주색 #2E3192, 보조 #D6CBB1, 강조 #C0392B
흰 배경, dpi=200, 한글 폰트 Noto Sans CJK.
출력: ../ch02_g_*.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
import os

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
NAVYDK = "#20226A"
GOLD = "#D6CBB1"
GOLDDK = "#A9976B"
RED = "#C0392B"
GRAY = "#5B5B5B"

OUTDIR = os.path.join(os.path.dirname(__file__), "..")

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ───────────────────────────────────────────────────────────────
# 1. p05: 밴딧 문제의 정적 설정 (agent <-> k-arm bandit)
# ───────────────────────────────────────────────────────────────
def fig_bandit_setup():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Agent box
    agent = FancyBboxPatch((0.4, 2.1), 2.0, 1.8, boxstyle="round,pad=0.08,rounding_size=0.15",
                            linewidth=2.2, edgecolor=NAVY, facecolor="white")
    ax.add_patch(agent)
    ax.text(1.4, 3.0, "에이전트", ha="center", va="center", fontsize=17, color=NAVY, fontweight="bold")

    # Bandit box
    bandit = FancyBboxPatch((6.4, 0.5), 3.2, 5.0, boxstyle="round,pad=0.08,rounding_size=0.15",
                             linewidth=2.2, edgecolor=NAVYDK, facecolor=GOLD, alpha=0.35)
    ax.add_patch(bandit)
    ax.text(8.0, 5.75, "밴딧 ($k=3$ 팔)", ha="center", va="center", fontsize=15, color=NAVYDK, fontweight="bold")

    arms = [("A", 1.0, 1.1), ("B", 1.5, 3.0), ("C", 2.0, 4.9)]
    for name, q, ycen in arms:
        h = 0.55 + 0.35 * q
        rect = Rectangle((7.1, ycen - h/2), 1.8, h, facecolor=NAVY, edgecolor=NAVYDK, alpha=0.85)
        ax.add_patch(rect)
        ax.text(8.0, ycen, f"팔 {name}\n$q^*(\\!\\cdot\\!)=${'?'}", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

    # Arrows
    ax.annotate("", xy=(6.3, 3.6), xytext=(2.5, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    ax.text(4.3, 3.85, "행동 $a_t$ 선택", ha="center", fontsize=13, color=RED)

    ax.annotate("", xy=(2.5, 2.4), xytext=(6.3, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4))
    ax.text(4.3, 1.75, "보상 $R_t \\sim \\mathcal{N}(q^*(a_t), 1)$", ha="center", fontsize=12.5, color=NAVY)

    ax.text(1.4, 1.0, "목표: $\\sum_t R_t$ 최대화", ha="center", fontsize=12, color=GRAY)

    savefig(fig, "ch02_g_bandit_setup.png")


# ───────────────────────────────────────────────────────────────
# 2. p08: 탐욕적 선택의 함정 (구조적으로 갇힘)
# ───────────────────────────────────────────────────────────────
def fig_greedy_trap():
    rng = np.random.default_rng(11)
    steps = np.arange(1, 16)
    qA_true, qB_true = 2.0, 1.0

    QA = np.full(steps.shape, np.nan)
    QB = np.full(steps.shape, np.nan)
    # 팔 A: 1회만 시도(첫 보상 0.1로 불운), 이후 영원히 외면
    QA[:] = 0.1
    # 팔 B: 계속 시도, 온라인 평균이 진짜 평균(1.0) 쪽으로 수렴
    obs = rng.normal(qB_true, 1.0, size=len(steps))
    running = np.cumsum(obs) / steps
    QB = running
    QB[0] = 1.4  # 첫 관찰(문서 예시와 일치)
    for i in range(1, len(QB)):
        QB[i] = QB[i-1] + (obs[i] - QB[i-1]) / (i + 1)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(steps, QB, "-o", color=NAVY, lw=2.2, ms=5, label="$Q_t(B)$ (계속 시도)")
    ax.plot([1], [0.1], "o", color=RED, ms=10, zorder=5)
    ax.plot(steps, QA, "--", color=RED, lw=2.2, label="$Q_t(A)$ (1회 후 영원히 외면)")

    ax.axhline(qA_true, color=RED, lw=1.3, ls=":", alpha=0.8)
    ax.text(0.7, qA_true + 0.08, "$q^*(A)=2.0$ (진짜 최선)", color=RED, fontsize=11, va="bottom")
    ax.axhline(qB_true, color=NAVY, lw=1.3, ls=":", alpha=0.8)
    ax.text(0.7, qB_true + 0.08, "$q^*(B)=1.0$", color=NAVY, fontsize=11, va="bottom")

    ax.annotate("구조적으로 갇힘:\n다시 시도할 확률 0",
                xy=(1, 0.1), xytext=(6.3, 0.55),
                fontsize=12, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))

    ax.set_xlabel("시도 횟수 $t$", fontsize=13)
    ax.set_ylabel("추정치 $Q_t(a)$", fontsize=13)
    ax.set_title("탐욕적 선택의 함정: 첫 운이 영원한 확신이 된다", fontsize=14, color=NAVYDK, fontweight="bold")
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(-0.15, 2.35)
    ax.legend(loc="upper right", fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "ch02_g_greedy_trap.png")


# ───────────────────────────────────────────────────────────────
# 3. p27: 낙관적 초기화 궤적 (Q0=5.0)
# ───────────────────────────────────────────────────────────────
def fig_optimistic_traj():
    steps = [0, 1, 2, 3, 4, 5]
    QA = [5.0, 0.64, 0.64, 0.64, 0.64, 0.64]
    QB = [5.0, 5.0, 0.94, 0.94, 0.94, 0.94]
    QC = [5.0, 5.0, 5.0, 1.33, 1.90, 1.72]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(steps, QA, "-o", color=GOLDDK, lw=2.2, ms=6, label="$Q_t(A)$")
    ax.plot(steps, QB, "-s", color=NAVY, lw=2.2, ms=6, label="$Q_t(B)$")
    ax.plot(steps, QC, "-^", color=RED, lw=2.2, ms=6, label="$Q_t(C)$")

    for q, c, lbl in [(1.0, GOLDDK, "$q^*(A)$"), (1.5, NAVY, "$q^*(B)$"), (2.0, RED, "$q^*(C)$")]:
        ax.axhline(q, color=c, lw=1.0, ls=":", alpha=0.6)

    ax.axhline(5.0, color=GRAY, lw=1.2, ls="--", alpha=0.7)
    ax.text(0.05, 5.15, "초기값 $Q_0=5.0$ (낙관적)", fontsize=11, color=GRAY)

    ax.annotate("탐험 확률 0인데도\nA→B→C 순서로 한 번씩 당겨짐",
                xy=(3, 1.33), xytext=(2.6, 3.4),
                fontsize=11.5, color=NAVYDK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVYDK, lw=1.6))

    ax.set_xlabel("스텝", fontsize=13)
    ax.set_ylabel("추정치 $Q_t(a)$", fontsize=13)
    ax.set_title("낙관적 초기화: 실망하며 다음 팔로 자동 이동", fontsize=14, color=NAVYDK, fontweight="bold")
    ax.set_xticks(steps)
    ax.legend(loc="center right", fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "ch02_g_optimistic_traj.png")


# ───────────────────────────────────────────────────────────────
# 4. p40: 팔 개수(k)에 따른 탐색 비용 (책의 실측치 그대로)
# ───────────────────────────────────────────────────────────────
def fig_k_scaling():
    ks = ["k=3", "k=10"]
    optimistic = [4, 9]        # N_A+N_B (=3+1) , 나머지 9개 각 1회
    ucb = [107, 214]           # N_A+N_B(=35+72) , 나머지 팔 합계

    x = np.arange(len(ks))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar(x - w/2, optimistic, width=w, color=GOLDDK, label="낙관적 초기화 (탐색 비용)")
    b2 = ax.bar(x + w/2, ucb, width=w, color=NAVY, label="UCB (탐색 비용)")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{int(b.get_height())}",
                    ha="center", fontsize=12, color=GRAY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ks, fontsize=13)
    ax.set_ylabel("최선이 아닌 팔에 쓰인 시도 횟수 (2000스텝 중)", fontsize=11.5)
    ax.set_title("팔이 늘어나면: 낙관적 초기화는 $k{-}1$, UCB는 서브선형", fontsize=13.5, color=NAVYDK, fontweight="bold")
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 260)
    savefig(fig, "ch02_g_k_scaling.png")


# ───────────────────────────────────────────────────────────────
# 5. p41: 비정상 환경 (t=1500에서 q* 뒤바뀜)
# ───────────────────────────────────────────────────────────────
def fig_nonstationary():
    t1 = np.arange(0, 1500)
    t2 = np.arange(1500, 2000)
    qA1, qB1, qC1 = 1.0, 1.5, 2.0
    qA2, qB2, qC2 = 0.5, 2.0, 1.0

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(t1, np.full_like(t1, qA1, dtype=float), color=GOLDDK, lw=2.4)
    ax.plot(t2, np.full_like(t2, qA2, dtype=float), color=GOLDDK, lw=2.4, label="$q^*(A)$")
    ax.plot(t1, np.full_like(t1, qB1, dtype=float), color=NAVY, lw=2.4)
    ax.plot(t2, np.full_like(t2, qB2, dtype=float), color=NAVY, lw=2.4, label="$q^*(B)$")
    ax.plot(t1, np.full_like(t1, qC1, dtype=float), color=RED, lw=2.4)
    ax.plot(t2, np.full_like(t2, qC2, dtype=float), color=RED, lw=2.4, label="$q^*(C)$")

    for q1, q2, c in [(qA1, qA2, GOLDDK), (qB1, qB2, NAVY), (qC1, qC2, RED)]:
        ax.plot([1500, 1500], [q1, q2], color=c, lw=1.4, ls=":")

    ax.axvline(1500, color=GRAY, lw=1.2, ls="--", alpha=0.7)
    ax.text(1510, 2.15, "환경 전환\n($t=1500$)", fontsize=11, color=GRAY)

    ax.annotate("C가 최선 → B가 최선으로",
                xy=(1750, 1.9), xytext=(300, 1.85),
                fontsize=12, color=NAVYDK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVYDK, lw=1.6))

    ax.set_xlabel("스텝 $t$", fontsize=13)
    ax.set_ylabel("진짜 평균 보상 $q^*(a)$", fontsize=13)
    ax.set_title("비정상 환경: 최선 팔이 도중에 바뀐다", fontsize=14, color=NAVYDK, fontweight="bold")
    ax.set_ylim(0.2, 2.3)
    ax.legend(loc="lower left", fontsize=11, frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "ch02_g_nonstationary.png")


# ───────────────────────────────────────────────────────────────
# 6. p55: 2-상태 MDP 다이어그램
# ───────────────────────────────────────────────────────────────
def fig_two_state_mdp():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    c0 = Circle((2.6, 3.0), 1.15, facecolor=NAVY, edgecolor=NAVYDK, lw=2.2, alpha=0.9, zorder=3)
    c1 = Circle((7.4, 3.0), 1.15, facecolor=RED, edgecolor="#7B241C", lw=2.2, alpha=0.9, zorder=3)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.text(2.6, 3.15, "$s_0$", ha="center", va="center", fontsize=22, color="white", fontweight="bold")
    ax.text(2.6, 2.55, "$V(s_0){=}28.0$", ha="center", va="center", fontsize=11.5, color="white")
    ax.text(7.4, 3.15, "$s_1$", ha="center", va="center", fontsize=22, color="white", fontweight="bold")
    ax.text(7.4, 2.55, "$V(s_1){=}30.0$", ha="center", va="center", fontsize=11.5, color="white")

    # s0 -> s1 전이 (실선)
    ax.annotate("", xy=(6.25, 3.0), xytext=(3.75, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=NAVYDK, lw=2.6))
    ax.text(5.0, 3.35, "보상 $1.0$, $P{=}1$", ha="center", fontsize=12, color=NAVYDK, fontweight="bold")

    # s1 자기 전이 (self-loop, 원 위쪽에 작은 고리)
    loop = FancyArrowPatch((6.95, 4.02), (7.85, 4.02), connectionstyle="arc3,rad=-2.0",
                            arrowstyle="-|>", mutation_scale=20, color="#7B241C", lw=2.4)
    ax.add_patch(loop)
    ax.text(7.4, 5.35, "보상 $3.0$, $P{=}1$\n(자기 전이)", ha="center", fontsize=11.5, color="#7B241C", fontweight="bold")

    ax.text(5.0, 0.7, "$\\gamma=0.9$   $V(s_1)=3.0+0.9\\,V(s_1) \\Rightarrow V(s_1)=30$", ha="center",
            fontsize=12.5, color=GRAY)
    ax.text(5.0, 0.05, "$V(s_0)=1.0+0.9 \\times 30 = 28$", ha="center", fontsize=12.5, color=GRAY)

    ax.set_title("2-상태 MDP: 상태의 가치가 서로를 참조한다", fontsize=14, color=NAVYDK, fontweight="bold", y=1.02)
    savefig(fig, "ch02_g_two_state_mdp.png")


# ───────────────────────────────────────────────────────────────
# 7. p61: 1/N 갱신 vs 고정 alpha 갱신 (비정상 환경 추적)
# ───────────────────────────────────────────────────────────────
def fig_alpha_tracking():
    rng = np.random.default_rng(7)
    T = 2000
    switch = 1000
    true_mean = np.where(np.arange(T) < switch, 1.0, 3.0)
    obs = rng.normal(true_mean, 1.0)

    # 1/N 갱신 (모든 관측의 단순 온라인 평균)
    Q_avg = np.zeros(T)
    running_sum = 0.0
    for t in range(T):
        running_sum += obs[t]
        Q_avg[t] = running_sum / (t + 1)

    # 고정 alpha 갱신
    alpha = 0.05
    Q_alpha = np.zeros(T)
    q = 0.0
    for t in range(T):
        q = q + alpha * (obs[t] - q)
        Q_alpha[t] = q

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(true_mean, color=GRAY, lw=1.6, ls="--", label="진짜 평균 $q^*(B)$")
    ax.plot(Q_avg, color=NAVY, lw=2.0, label="$1/N_t(a)$ 갱신 (온라인 평균)")
    ax.plot(Q_alpha, color=RED, lw=2.0, label="고정 $\\alpha=0.05$ 갱신")
    ax.axvline(switch, color=GRAY, lw=1.0, ls=":", alpha=0.8)
    ax.text(switch + 15, 0.3, "전환\n($t{=}1000$)", fontsize=10.5, color=GRAY)

    ax.annotate("$1/N$: 옛 기록에 눌려\n천천히 따라옴", xy=(1550, Q_avg[1550]),
                xytext=(1250, 0.55), fontsize=11, color=NAVYDK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVYDK, lw=1.5))
    ax.annotate("고정 $\\alpha$: 빠르게\n새 평균을 따라감", xy=(1180, Q_alpha[1180]),
                xytext=(1350, 1.55), fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_xlabel("스텝 $t$", fontsize=13)
    ax.set_ylabel("팔 B의 추정치 $Q_t(B)$", fontsize=13)
    ax.set_title("비정상 환경: 고정 $\\alpha$가 변화에 더 빠르게 적응", fontsize=13.5, color=NAVYDK, fontweight="bold")
    ax.set_ylim(-0.1, 3.9)
    ax.legend(loc="upper left", fontsize=10.5, frameon=True, facecolor="white", edgecolor="none")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "ch02_g_alpha_tracking.png")


if __name__ == "__main__":
    fig_bandit_setup()
    fig_greedy_trap()
    fig_optimistic_traj()
    fig_k_scaling()
    fig_nonstationary()
    fig_two_state_mdp()
    fig_alpha_tracking()
