"""
Week15 (VAE/GAN/Diffusion) enrichment diagrams.
Run with the figenv python. Outputs PNGs into ../ (figs/).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#D6CBB1"
GOLDDK = "#A9976B"
RED = "#C0392B"
GRAY = "#5B5B5B"

OUT = "../"

def savefig(fig, name):
    fig.savefig(OUT + name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------
# 1. p06g1: 일반 오토인코더 vs VAE 잠재 공간
# ---------------------------------------------------------------
def fig_ae_vs_vae():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

    # 왼쪽: 일반 AE -- 흩어진 점들, 빈 공간 많음, 클러스터 간 간격
    ax = axes[0]
    clusters = [(-3, 2.5), (3, 3), (-2.5, -3), (2.8, -2.2), (0, 0.2)]
    for cx, cy in clusters:
        pts = rng.normal(0, 0.35, size=(40, 2)) + np.array([cx, cy])
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=BLUE, alpha=0.75)
    # 빈 공간에 무작위 쿼리 포인트(물음표) 표시
    empties = np.array([[0.8, -0.8], [-1.2, 1.0], [1.6, 1.4], [-0.6, -1.6]])
    ax.scatter(empties[:, 0], empties[:, 1], s=140, marker="x", color=RED, linewidths=2.5)
    for ex, ey in empties:
        ax.text(ex + 0.15, ey + 0.15, "?", color=RED, fontsize=15, fontweight="bold")
    ax.set_title("일반 오토인코더: 흩어진 점 + 빈 공간", fontsize=12.5)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRAY)

    # 오른쪽: VAE -- 표준정규분포에 맞춰 매끄럽게, 빈틈없이 채워짐
    ax = axes[1]
    pts = rng.normal(0, 1.0, size=(700, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=BLUE, alpha=0.35)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, alpha in [(1, 0.9), (2, 0.55), (3, 0.3)]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=GOLDDK, lw=1.6, alpha=alpha)
    query = np.array([[0.8, -0.8], [-1.2, 1.0], [1.6, 1.4], [-0.6, -1.6]])
    ax.scatter(query[:, 0], query[:, 1], s=120, marker="*", color=RED)
    ax.set_title(r"VAE: $\mathcal{N}(0,1)$로 매끄럽게 채워짐", fontsize=12.5)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRAY)

    fig.suptitle("잠재 공간 $z$: 오토인코더 vs VAE", fontsize=14, color=BLUE, y=1.02)
    savefig(fig, "gen_ae_vs_vae_latent.png")


# ---------------------------------------------------------------
# 2. p14g1: x=1 vs x=5의 우도 비교 (p(x) = N(x; 0, 2))
# ---------------------------------------------------------------
def fig_likelihood_x1_x5():
    from scipy.stats import norm
    x = np.linspace(-8, 8, 400)
    var = 2.0
    y = norm.pdf(x, loc=0, scale=np.sqrt(var))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(x, y, color=BLUE, lw=2.5, label=r"$p(x) = \mathcal{N}(x;0,2)$")
    ax.fill_between(x, y, color=BLUE, alpha=0.08)

    for xv, label, color in [(1, "x=1", GOLDDK), (5, "x=5", RED)]:
        yv = norm.pdf(xv, loc=0, scale=np.sqrt(var))
        ax.vlines(xv, 0, yv, color=color, lw=2.2, linestyle="--")
        ax.scatter([xv], [yv], s=70, color=color, zorder=5)
        logp = norm.logpdf(xv, loc=0, scale=np.sqrt(var))
        ax.annotate(f"{label}\n" + r"$\log p(x)\approx$" + f"{logp:.2f}",
                    xy=(xv, yv), xytext=(xv + (0.6 if xv < 4 else -3.0), yv + 0.05),
                    fontsize=12, color=color, fontweight="bold")

    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$p(x)$", fontsize=13)
    ax.set_title("두 봉우리 데이터의 우도: $x=1$ vs $x=5$", fontsize=14, color=BLUE)
    ax.tick_params(labelsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    savefig(fig, "gen_likelihood_x1_x5.png")


# ---------------------------------------------------------------
# 3. p19g1: 인코더가 배운 두 좁은 정규분포 (원점 대칭)
# ---------------------------------------------------------------
def fig_encoder_two_bumps():
    rng = np.random.default_rng(1)
    mu1 = np.array([0.5, -0.88])
    mu2 = np.array([-0.77, 1.42])
    std = np.sqrt(np.exp(-1.7))  # log sigma^2 ~ -1.7
    pts1 = rng.normal(0, std, size=(300, 2)) + mu1
    pts2 = rng.normal(0, std, size=(300, 2)) + mu2

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(pts1[:, 0], pts1[:, 1], s=12, color=BLUE, alpha=0.6, label=r"왼쪽 봉우리($x\approx1$)")
    ax.scatter(pts2[:, 0], pts2[:, 1], s=12, color=RED, alpha=0.6, label=r"오른쪽 봉우리($x\approx9$)")
    ax.scatter([mu1[0]], [mu1[1]], s=90, color=BLUE, edgecolor="white", zorder=5, marker="X")
    ax.scatter([mu2[0]], [mu2[1]], s=90, color=RED, edgecolor="white", zorder=5, marker="X")
    ax.scatter([0], [0], s=140, color=GOLDDK, marker="*", zorder=6, label="원점 (사전분포 중심)")

    # 표준정규 사전분포 등고선(참고)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, lw=1.2, ls=":", alpha=0.7)

    ax.axhline(0, color=GRAY, lw=0.6)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel(r"$z_1$", fontsize=13)
    ax.set_ylabel(r"$z_2$", fontsize=13)
    ax.set_title("인코더가 두 봉우리에 배운 $q_\\phi(z|x)$", fontsize=14, color=BLUE)
    ax.legend(fontsize=10.5, loc="lower left", framealpha=0.9)
    ax.set_aspect("equal")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    savefig(fig, "gen_encoder_two_bumps.png")


# ---------------------------------------------------------------
# 4. p24g1: beta-VAE 트레이드오프 (p22 표 데이터 그대로)
# ---------------------------------------------------------------
def fig_beta_vae_tradeoff():
    beta = np.array([0, 0.1, 0.5, 1.0, 2.0])
    recon = np.array([0.016, 0.068, 0.232, 0.378, 0.546])
    kl = np.array([19.1, 5.61, 1.70, 1.22, 0.90])

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax2 = ax1.twinx()

    l1, = ax1.plot(beta, recon, "o-", color=BLUE, lw=2.4, ms=8, label="복원 손실")
    l2, = ax2.plot(beta, kl, "s-", color=RED, lw=2.4, ms=8, label="KL")

    ax1.set_xlabel(r"$\beta$", fontsize=13)
    ax1.set_ylabel("복원 손실", fontsize=12.5, color=BLUE)
    ax2.set_ylabel("KL 발산", fontsize=12.5, color=RED)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_xticks(beta)
    ax1.set_title(r"$\beta$-VAE: 복원 손실 vs KL 발산의 줄다리기", fontsize=14, color=BLUE)
    ax1.legend(handles=[l1, l2], fontsize=11, loc="center right")
    for s in ["top"]:
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    savefig(fig, "gen_beta_vae_tradeoff.png")


# ---------------------------------------------------------------
# 5. p32g1: 판별자 정확도(D*(0))가 0.5로 수렴 (p37 표 데이터 그대로)
# ---------------------------------------------------------------
def fig_discriminator_converge():
    steps = np.array([0, 10, 20, 30, 40])
    d_star = np.array([0.667, 0.625, 0.572, 0.527, 0.507])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(steps, d_star, "o-", color=BLUE, lw=2.6, ms=9, label=r"$D^*(0)$ (판별자 정확도)")
    ax.axhline(0.5, color=RED, lw=2.0, ls="--", label=r"내쉬 균형 $D=0.5$")
    for xv, yv in zip(steps, d_star):
        ax.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(xv, yv + 0.012),
                    fontsize=10.5, ha="center", color=BLUE)
    ax.set_xlabel("교대 최적화 스텝", fontsize=13)
    ax.set_ylabel(r"$D^*(0)$", fontsize=13)
    ax.set_ylim(0.45, 0.72)
    ax.set_title("판별자 정확도가 균형점 50%로 수렴", fontsize=14, color=BLUE)
    ax.legend(fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    savefig(fig, "gen_discriminator_converge.png")


# ---------------------------------------------------------------
# 6. p42g1: 모드 붕괴 -- 3개 모드 데이터 vs GAN이 1개 모드에 집중
# ---------------------------------------------------------------
def fig_mode_collapse():
    rng = np.random.default_rng(2)
    x = np.linspace(-10, 10, 500)

    def gauss(x, mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    data_dist = (gauss(x, -6, 0.8) + gauss(x, 0, 0.8) + gauss(x, 6, 0.8)) / 3

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    ax = axes[0]
    ax.plot(x, data_dist, color=GRAY, lw=2.2, label="실제 데이터 분포 (3개 모드)")
    samples_ok = np.concatenate([
        rng.normal(-6, 0.8, 300), rng.normal(0, 0.8, 300), rng.normal(6, 0.8, 300)
    ])
    ax.hist(samples_ok, bins=40, density=True, color=BLUE, alpha=0.55, label="정상 학습된 생성자")
    ax.set_title("다양성 유지: 세 모드 모두 재현", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_xlabel("$x$", fontsize=12)

    ax = axes[1]
    ax.plot(x, data_dist, color=GRAY, lw=2.2, label="실제 데이터 분포 (3개 모드)")
    samples_collapse = rng.normal(0, 0.8, 900)
    ax.hist(samples_collapse, bins=40, density=True, color=RED, alpha=0.6, label="모드 붕괴: 1개 모드에 집중")
    ax.set_title("모드 붕괴: 다양성 상실", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_xlabel("$x$", fontsize=12)

    fig.suptitle("모드 붕괴(mode collapse): $D$를 속이는 데는 충분하지만", fontsize=14, color=BLUE, y=1.03)
    for ax in axes:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    savefig(fig, "gen_mode_collapse.png")


# ---------------------------------------------------------------
# 7. p45g1: Diffusion 정방향/역방향 -- 분포가 점점 노이즈로, 다시 데이터로
# ---------------------------------------------------------------
def fig_diffusion_forward_reverse():
    rng = np.random.default_rng(3)
    x = np.linspace(-6, 6, 400)

    def gauss(x, mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    data_dist = 0.5 * gauss(x, -2, 0.5) + 0.5 * gauss(x, 2, 0.5)

    ts = [0, 1, 2, 3]  # t=0(데이터), 점점 노이즈로
    labels = [r"$x_0$ (데이터)", r"$x_{T/3}$", r"$x_{2T/3}$", r"$x_T$ (순수 노이즈)"]
    colors = [BLUE, "#6B6FBF", GOLDDK, RED]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, t in enumerate(ts):
        blend = t / (len(ts) - 1)
        # 데이터 분포와 표준정규 사이를 (알파 스케줄처럼) 섞어 노이즈가 누적되는 모양을 표현
        noise_dist = gauss(x, 0, 1.0)
        mix = (1 - blend) * data_dist + blend * noise_dist
        # 봉우리를 뭉개서(분산 증가) 실제 forward process 느낌을 더함
        smoothed = np.convolve(mix, gauss(np.linspace(-3, 3, 60), 0, 0.15 + blend * 0.9), mode="same")
        smoothed /= np.trapezoid(smoothed, x)
        ax.plot(x, smoothed + i * 0.02, color=colors[i], lw=2.4, label=labels[i])

    ax.annotate("", xy=(0.72, 0.92), xytext=(0.28, 0.92), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))
    ax.text(0.5, 0.95, "정방향 과정(노이즈 추가, 고정)", ha="center", fontsize=11.5,
            color=GRAY, transform=ax.transAxes)
    ax.annotate("", xy=(0.28, 0.06), xytext=(0.72, 0.06), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(0.5, 0.01, "역방향 과정(신경망이 학습, $\\epsilon_\\theta$)", ha="center", fontsize=11.5,
            color=BLUE, transform=ax.transAxes)

    ax.set_xlabel("$x$", fontsize=13)
    ax.set_yticks([])
    ax.set_title("Diffusion: 데이터 분포 $\\leftrightarrow$ 표준정규 노이즈", fontsize=14, color=BLUE)
    ax.legend(fontsize=10.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    savefig(fig, "gen_diffusion_forward_reverse.png")


if __name__ == "__main__":
    fig_ae_vs_vae()
    fig_likelihood_x1_x5()
    fig_encoder_two_bumps()
    fig_beta_vae_tradeoff()
    fig_discriminator_converge()
    fig_mode_collapse()
    fig_diffusion_forward_reverse()
