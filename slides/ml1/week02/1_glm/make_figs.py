"""GLM 덱 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (assert 로 확인한 값) 와 같은 함수로 계산한다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import data as D

plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d5c"
O = "figs/"
os.makedirs(O, exist_ok=True)
IRIS_NAMES = ["setosa", "versicolor", "virginica"]
IRIS_COL = [GOLD, GREEN, BLUE]


def types_fig(study, chess, goals, iris, seed=0):
    """맞힐 값 y 의 종류 네 가지: 실수 · 0/1 · 개수 · 여러 클래스."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    x, y = study; ax[0].scatter(x, y, s=34, color=GRAY)
    ax[0].set_xlabel("공부 시간"); ax[0].set_title("점수: 아무 실수", fontsize=16, fontweight="bold")
    x, y = chess; i = rng.choice(len(x), 300, replace=False)
    ax[1].scatter(x[i], y[i] + rng.uniform(-0.06, 0.06, 300), s=18, c=np.where(y[i] == 1, BLUE, RED), alpha=0.35, edgecolors="none")
    ax[1].set_yticks([0, 1], ["흑 승 0", "백 승 1"]); ax[1].set_xlabel("레이팅 차이"); ax[1].set_title("체스 승패: 0 / 1", fontsize=16, fontweight="bold")
    x, y = goals; ax[2].scatter(x, y + rng.uniform(-0.25, 0.25, len(y)), s=8, color=GREEN, alpha=0.3, edgecolors="none")
    ax[2].set_xlabel("Elo 차이 (상대보다)"); ax[2].set_title("월드컵 골 수: 0, 1, 2, …", fontsize=16, fontweight="bold"); ax[2].set_yticks(range(0, 9, 2))
    x, y = iris
    for k in range(3):
        m = y == k; ax[3].scatter(x[m], k + rng.uniform(-0.12, 0.12, m.sum()), s=18, color=IRIS_COL[k], alpha=0.6, edgecolors="none")
    ax[3].set_yticks([0, 1, 2], IRIS_NAMES); ax[3].set_xlabel("꽃잎 길이 (cm)"); ax[3].set_title("붓꽃 종: 셋 중 하나", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(O + "glm_types.png"); plt.close(fig)


def goal_means(x, y, step=100, min_n=15):
    """Elo 차이 step 구간마다 실제 평균 골 (줄이 min_n 개 이상인 구간만)."""
    out = []
    for lo in np.arange(-600, 600, step):
        m = (x >= lo) & (x < lo + step)
        if m.sum() >= min_n: out.append((lo + step / 2, y[m].mean(), m.sum()))
    return np.array(out)


def goals_axes(ax, x, y):
    """검은 점 = Elo 차이 100 구간마다 실제 평균 골 (점 크기 = 경기 수)."""
    g = goal_means(x, y)
    ax.scatter(g[:, 0], g[:, 1], s=g[:, 2] * 0.5 + 20, color="k", zorder=4, label="실제 평균 골 (점 크기 = 경기 수)")
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("Elo 차이 (내 팀 − 상대)   ← 약팀  ·  강팀 →"); ax.set_ylabel("한 경기에 넣은 골"); ax.set_xlim(-600, 600)


def goals_line_fig(x, y, k, at=-450):
    """개수에 직선: 약팀 골을 너무 낮게, 끝에서는 음수."""
    xs = np.linspace(-600, 600, 200); z0 = -k[1] / k[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(-0.6, 0, color=RED, alpha=0.08)
    goals_axes(ax, x, y); ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3, label="직선")
    m = (x >= at - 50) & (x < at + 50); real, line = y[m].mean(), np.polyval(k, at)
    ax.annotate("", (at, line), (at, real), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2))
    ax.text(at - 140, real + 0.45, f"약팀: 실제 {real:.2f}골\n직선은 {line:.2f}골", fontsize=14, color=BLUE, fontweight="bold", va="center")
    ax.annotate(f"{-z0:.0f}점 넘게 뒤지면\n직선은 골 < 0", (-560, np.polyval(k, -560)), (-420, -0.5), fontsize=14, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_ylim(-0.6, 3.4); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_goals_line.png"); plt.close(fig)


def goals_pois_fig(x, y, k, wp, marks=(-300, 0, 300)):
    """포아송 회귀 e^z: 언제나 양수, Elo 100 마다 같은 배수."""
    xs = np.linspace(-600, 600, 200); pm = lambda v: np.exp(wp[0] + wp[1] * v)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    goals_axes(ax, x, y)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=2, ls="--", label="직선")
    ax.plot(xs, pm(xs), color=BLUE, lw=3.5, label="포아송 회귀 $e^{z}$")
    for v in marks:
        ax.plot(v, pm(v), "o", color=BLUE, ms=10, zorder=5)
        ax.annotate(f"{v:+d}: {pm(v):.2f}골" if v else f"0: {pm(v):.2f}골", (v, pm(v)), (-30, 26), textcoords="offset points", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(80, 0.25, f"Elo 100 앞설 때마다 ×{np.exp(100 * wp[1]):.2f}", fontsize=14, color=BLUE, fontweight="bold")
    ax.set_ylim(-0.6, 4.6); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_goals_pois.png"); plt.close(fig)


def claims_axes(ax, x, y):
    """검은 점 = 등급 10점 구간마다 실제 평균 청구 (점 크기 = 계약 수, 로그)."""
    g = D.claim_means(x, y)
    ax.scatter(g[:, 0], g[:, 1], s=25 * np.log10(g[:, 2]) + 10, color="k", zorder=4, label="실제 평균 청구 (점 크기 = 계약 수)")
    ax.set_xlabel("보너스-말러스 등급 x   (50 = 오래 무사고 · 100 = 처음 · 높을수록 사고 이력)")
    ax.set_ylabel("1년 사고 청구 횟수"); ax.set_xlim(45, 155); ax.set_ylim(0, 1.0)


def claims_line_fig(x, y, k, at=145):
    """개수에 직선: 등급이 높은 쪽에서 실제 평균은 위로 휘는데 직선은 못 따라간다."""
    xs = np.linspace(45, 155, 200)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    claims_axes(ax, x, y); ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=3, label="직선")
    m = (x >= at - 5) & (x < at + 5); real, line = y[m].mean(), np.polyval(k, at)
    ax.annotate("", (at, line), (at, real), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(at - 4, real + 0.08, f"등급 {at - 5}~{at + 4}: 실제 {real:.2f}번 · 직선 {line:.2f}번", fontsize=14, color=RED, fontweight="bold", ha="right")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_claims_line.png"); plt.close(fig)


def claims_pois_fig(x, y, k, wp, marks=(50, 100, 150)):
    """포아송 회귀 e^z: 언제나 양수, 등급 10점마다 같은 배수."""
    xs = np.linspace(45, 155, 200); pm = lambda v: np.exp(wp[0] + wp[1] * v)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    claims_axes(ax, x, y)
    ax.plot(xs, np.polyval(k, xs), color=GOLD, lw=2, ls="--", label="직선")
    ax.plot(xs, pm(xs), color=BLUE, lw=3.5, label="포아송 회귀 $e^{z}$")
    for v in marks:
        ax.plot(v, pm(v), "o", color=BLUE, ms=10, zorder=5)
        ax.annotate(f"{v}: {pm(v):.2f}번", (v, pm(v)), (-70, 14) if v == 150 else ((10, -26) if v == 100 else (6, 14)), textcoords="offset points", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(62, 0.55, f"등급 10점 오를 때마다 ×{np.exp(10 * wp[1]):.2f}", fontsize=14, color=BLUE, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_claims_pois.png"); plt.close(fig)

def softmax_fig(x, y, W, v=4.9, seed=1):
    """꽃잎 길이 하나로 세 종의 확률 (소프트맥스 회귀)."""
    rng = np.random.default_rng(seed); xs = np.linspace(1, 7, 400); P = D.softmax_p(W, xs); pv = D.softmax_p(W, v)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k in range(3):
        ax.plot(xs, P[:, k], color=IRIS_COL[k], lw=3.5, label=IRIS_NAMES[k])
        m = y == k; ax.scatter(x[m], -0.07 - 0.04 * k + rng.uniform(-0.012, 0.012, m.sum()), s=14, color=IRIS_COL[k], alpha=0.6, edgecolors="none")
    ax.axvline(v, color=GRAY, ls=":")
    ax.annotate(f"{v} cm\nversicolor {pv[1]:.0%}\nvirginica {pv[2]:.0%}", (v, 0.5), (5.25, 0.08), fontsize=14, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("확률"); ax.set_ylim(-0.17, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_softmax.png"); plt.close(fig)


def dists_fig(study, chess, goals, iris, line=(38.58, 5.25), chess_at=200, n_near=500, even=50, iris_at=4.9):
    """2쪽 데이터 네 개에 분포 맞추기 (glm_types 와 같은 순서). 막대 = 실제 데이터, 검은 선/점 = 같은 평균의 분포."""
    from math import lgamma, exp, log
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.9))
    # 점수: 직선에서 벗어난 정도 ~ 정규
    x, y = study; r = y - (line[0] + line[1] * x); sd = r.std()
    a = ax[0]; a.hist(r, bins=np.arange(-12, 13, 3), density=True, color=GRAY, alpha=0.5, label=f"{len(r)}명")
    u = np.linspace(-13, 13, 300); a.plot(u, np.exp(-u ** 2 / (2 * sd * sd)) / (sd * np.sqrt(2 * np.pi)), color="k", lw=2.5, label="정규")
    a.set_xlabel("점수 − 직선 예측"); a.set_title("점수 → 정규", fontsize=17, fontweight="bold"); a.legend(frameon=False, fontsize=11)
    # 체스: +200 점 근처 n_near 판 ~ 베르누이
    x, y = chess; i = np.argsort(np.abs(x - chess_at), kind="stable")[:n_near]; ph = y[i].mean()
    a = ax[1]; a.bar([0, 1], [1 - ph, ph], color=[RED, BLUE], width=0.5)
    for k, v in enumerate([1 - ph, ph]): a.text(k, v + 0.02, f"{int(round(v * n_near))}판", ha="center", fontsize=13)
    a.set_xticks([0, 1], ["흑 승 0", "백 승 1"]); a.set_ylim(0, 1)
    a.set_xlabel(f"+{chess_at}점 근처 {n_near}판"); a.set_title("체스 → 베르누이", fontsize=17, fontweight="bold")
    # 보험: 등급 50 (오래 무사고) 계약의 1년 청구 횟수 ~ 포아송 (같은 평균)
    x, y = goals; v = y[x == 50].astype(int); lam = v.mean(); ks = np.arange(0, 4)
    cnt = np.bincount(v, minlength=len(ks))[:len(ks)]; pois = np.array([exp(k * log(lam) - lam - lgamma(k + 1)) for k in ks])
    a = ax[2]; a.bar(ks, cnt / len(v), color=GREEN, alpha=0.6, width=0.7, label=f"{len(v):,}계약")
    a.plot(ks, pois, "o-", color="k", lw=2, ms=6, label="포아송")
    a.set_xticks(ks); a.set_xlabel("1년 청구 횟수 (등급 50)"); a.set_title("보험 → 포아송", fontsize=17, fontweight="bold")
    a.legend(frameon=False, fontsize=11, loc="upper right")
    # 붓꽃: 꽃잎 길이 근처 ~ 범주형
    x, y = iris; m = np.abs(x - iris_at) <= 0.1 + 1e-9; n = np.bincount(y[m], minlength=3)
    a = ax[3]; a.bar(range(3), n / n.sum(), color=IRIS_COL, width=0.5)
    for k, v in enumerate(n): a.text(k, v / n.sum() + 0.02, f"{v}송이", ha="center", fontsize=13)
    a.set_xticks(range(3), IRIS_NAMES); a.set_ylim(0, 1)
    a.set_xlabel(f"꽃잎 {iris_at - 0.1:.1f}~{iris_at + 0.1:.1f} cm ({n.sum()}송이)"); a.set_title("붓꽃 → 범주형", fontsize=17, fontweight="bold")
    for a in ax: a.set_yticks([])
    fig.tight_layout(); fig.savefig(O + "glm_dists.png"); plt.close(fig)
    return sd, ph, round(lam, 2), n

def pois_pmf_fig(lams=((50, 0.059), (100, 0.216), (150, 0.793)), kmax=4):
    """포아송 분포 모양: 평균 λ 하나로 정해진다 (보험 모델의 등급 50 · 100 · 150 평균)."""
    from math import lgamma, exp, log
    ks = np.arange(0, kmax + 1); cols = [GREEN, GRAY, RED]; wd = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, ((d, lam), c) in enumerate(zip(lams, cols)):
        pm = [exp(k * log(lam) - lam - lgamma(k + 1)) for k in ks]
        ax.bar(ks + (i - 1) * wd, pm, width=wd, color=c, alpha=0.85, label=f"등급 {d}: λ = {lam:.2f}번")
    ax.set_xticks(ks); ax.set_xlabel("1년 사고 청구 횟수 y"); ax.set_ylabel("확률"); ax.legend(frameon=False, fontsize=13)
    fig.tight_layout(); fig.savefig(O + "glm_pois_pmf.png"); plt.close(fig)

def pois_examples_fig(sets, kmax=6):
    """개수 데이터 여러 개: 막대 = 실제 비율, 검은 점 = 같은 평균의 포아송. sets = [(제목, x 라벨, 개수 배열, 평균, 단위)]."""
    from math import lgamma, exp, log
    fig, ax = plt.subplots(1, len(sets), figsize=(13, 3.9))
    for a, (title, xl, cnt, lam, unit) in zip(ax, sets):
        cnt = np.asarray(cnt); ks = np.arange(len(cnt)); n = cnt.sum()
        pm = np.array([exp(k * log(lam) - lam - lgamma(k + 1)) for k in ks])
        a.bar(ks, cnt / n, color=GREEN, alpha=0.6, width=0.7, label=f"실제 {n:,}{unit}")
        a.plot(ks, pm, "o-", color="k", lw=2, ms=6, label=f"포아송 λ = {lam:.2f}")
        a.set_xticks(ks); a.set_xlabel(xl); a.set_title(title, fontsize=17, fontweight="bold"); a.set_yticks([])
        a.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout(); fig.savefig(O + "glm_pois_examples.png"); plt.close(fig)

def pois_origin_fig(poker=D.POKER, kmax=6):
    """포아송의 발명: 포커에서 드문 패를 많은 판에서 센다. 판 수 x 확률 = 1.27 로 같게 → 판은 많고 패는 드물수록 포아송."""
    from math import comb, exp, factorial
    T = comb(52, 5); ks = np.arange(kmax + 1)
    fig, ax = plt.subplots(1, len(poker), figsize=(13, 3.7), sharey=True)
    for a, (name, (c, n)) in zip(ax, poker.items()):
        p = c / T; lam = n * p; po = [lam ** k * exp(-lam) / factorial(k) for k in ks]
        bi = [comb(n, k) * p ** k * (1 - p) ** (n - k) if k <= n else 0 for k in ks]
        a.bar(ks, bi, color=BLUE, alpha=0.55, width=0.7, label=f"실제 ({n}판, 이항)")
        a.plot(ks, po, "o-", color="k", lw=2, ms=6, label=f"포아송 λ = {lam:.2f}")
        pct = f"{100 * p:.1f}%" if p > 0.01 else f"{100 * p:.3f}%"
        a.set_title(f"{name} ({pct}) · {n}판", fontsize=16, fontweight="bold")
        a.set_xticks(ks); a.set_xlabel(f"{n}판 중 {name} 횟수"); a.set_yticks([]); a.legend(frameon=False, fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_pois_origin.png"); plt.close(fig)


def scatter_cells_fig(nx=12, ny=8, lam=2, seed=3, kmax=8):
    """점을 흩뿌리고 칸마다 세기: 제멋대로 (포아송) · 고르게 (좁음) · 뭉침 (넓음). 아래 = 칸별 개수 막대 vs 같은 평균의 포아송."""
    from math import exp, factorial
    rng = np.random.default_rng(seed); N = nx * ny * lam
    rand = np.c_[rng.uniform(0, nx, N), rng.uniform(0, ny, N)]
    gx, gy = np.meshgrid(np.arange(nx * 2) / 2 + 0.25, np.arange(ny) + 0.5)
    reg = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-0.12, 0.12, (N, 2))
    cen = np.c_[rng.uniform(0.8, nx - 0.8, 8), rng.uniform(0.8, ny - 0.8, 8)]
    clu = np.clip(np.repeat(cen, N // 8, 0) + rng.normal(0, 0.45, (N, 2)), 0.01, [nx - 0.01, ny - 0.01])
    sets = [("제멋대로 (서로 상관없음)", rand, "포아송", "#2a9d5c"), ("고르게 (서로 밀어냄)", reg, "포아송보다 좁다", RED), ("뭉침 (서로 끌어당김)", clu, "포아송보다 넓다", RED)]
    ks = np.arange(kmax + 1); po = [lam ** k * exp(-lam) / factorial(k) for k in ks]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6), gridspec_kw={"height_ratios": [1.05, 1]})
    out = []
    for j, (title, P, verdict, col) in enumerate(sets):
        a = ax[0, j]; a.scatter(P[:, 0], P[:, 1], s=7, color="k")
        for v in range(nx + 1): a.axvline(v, color="#cccccc", lw=0.6)
        for v in range(ny + 1): a.axhline(v, color="#cccccc", lw=0.6)
        a.add_patch(plt.Rectangle((2, 5), 1, 1, fill=False, ec="#2a9d5c", lw=2.5))
        a.set_xlim(0, nx); a.set_ylim(0, ny); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values(): s_.set_visible(True)
        a.set_title(title, fontsize=16, fontweight="bold")
        c = np.histogram2d(P[:, 0], P[:, 1], bins=[nx, ny], range=[[0, nx], [0, ny]])[0].ravel().astype(int)
        b = ax[1, j]; cnt = np.bincount(np.minimum(c, kmax), minlength=kmax + 1) / len(c)
        b.bar(ks, cnt, color=GRAY, alpha=0.6, width=0.7, label="칸별 점 개수")
        b.plot(ks, po, "o-", color="k", lw=2, ms=5, label=f"포아송 λ = {lam}")
        b.set_xticks(ks, [str(k) for k in ks[:-1]] + [f"{kmax}+"]); b.set_yticks([]); b.set_xlabel(f"한 칸의 점 수 (평균 {c.mean():.0f}, 분산 {c.var():.1f})")
        b.set_title(f"→ {verdict}", fontsize=16, fontweight="bold", color=col); b.legend(frameon=False, fontsize=11)
        out.append((round(c.mean(), 2), round(c.var(), 2)))
    fig.tight_layout(); fig.savefig(O + "glm_pois_scatter.png"); plt.close(fig)
    return out

def _merc(lon, lat, z):
    """경위도 → z 단계 타일 픽셀 좌표 (웹 메르카토르)."""
    lat = np.radians(lat); n = 256 * 2 ** z
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.log(np.tan(lat) + 1 / np.cos(lat)) / np.pi) / 2 * n


def basemap(ax, lon0, lon1, lat0, lat1, z=13, style="osm"):
    """OpenStreetMap 타일을 받아 (data/tiles 에 저장) 흑백 · 어둡게 바꿔 붙인다. 좌표는 z 단계 픽셀."""
    import io, urllib.request
    from PIL import Image
    x0, y1 = _merc(lon0, lat0, z); x1, y0 = _merc(lon1, lat1, z)
    tx = range(int(x0 // 256), int(x1 // 256) + 1); ty = range(int(y0 // 256), int(y1 // 256) + 1)
    cache = os.path.join("data", "tiles", style, str(z)); os.makedirs(cache, exist_ok=True)
    big = Image.new("RGB", (256 * len(tx), 256 * len(ty)))
    for i, x in enumerate(tx):
        for j, y in enumerate(ty):
            p = os.path.join(cache, f"{x}_{y}.png")
            if not os.path.exists(p):
                req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers={"User-Agent": "ksa-ml-course-slides/1.0 (educational, one-off figure)"})
                open(p, "wb").write(urllib.request.urlopen(req, timeout=30).read())
            big.paste(Image.open(p).convert("RGB"), (256 * i, 256 * j))
    from PIL import ImageOps
    g = ImageOps.grayscale(big); g = ImageOps.invert(g).point(lambda v: int(25 + v * 0.42)); big = g.convert("RGB")
    ax.imshow(big, extent=(tx[0] * 256, (tx[-1] + 1) * 256, (ty[-1] + 1) * 256, ty[0] * 256), zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def v1_map_fig(e, n, lon, lat, box=D.V1_BOX, cell=D.V1_CELL, z=13):
    """V-1 실제 낙하 지점 (런던 시의회 구역 887발, Shaw 2025) 을 런던 지도 (OpenStreetMap · CARTO) 위에.
    노란 네모 = 남쪽 12 x 9 km, 0.5 km 칸으로 센다. 영국 격자 (m) → 경위도는 887점으로 맞춘 1차식 (이 넓이에서 오차 수 m)."""
    H, k = D.v1_cells(e, n, box, cell); x0, x1, y0, y1 = box
    A = np.c_[e, n, np.ones_like(e)]; cl = np.linalg.lstsq(A, lon, rcond=None)[0]; ca = np.linalg.lstsq(A, lat, rcond=None)[0]
    to_px = lambda E, N: _merc(np.c_[E, N, np.ones_like(E)] @ cl, np.c_[E, N, np.ones_like(E)] @ ca, z)
    fig, ax = plt.subplots(figsize=(8.6, 6.6)); fig.patch.set_facecolor("#262626")
    basemap(ax, lon.min() - 0.01, lon.max() + 0.01, lat.min() - 0.006, lat.max() + 0.006, z)
    for v in np.arange(x0, x1 + 1, cell):
        px, py = to_px(np.array([v, v], float), np.array([y0, y1], float)); ax.plot(px, py, color="#f2c94c", lw=0.35, alpha=0.55, zorder=2)
    for v in np.arange(y0, y1 + 1, cell):
        px, py = to_px(np.array([x0, x1], float), np.array([v, v], float)); ax.plot(px, py, color="#f2c94c", lw=0.35, alpha=0.55, zorder=2)
    bx, by = to_px(np.array([x0, x1, x1, x0, x0], float), np.array([y0, y0, y1, y1, y0], float)); ax.plot(bx, by, color="#f2c94c", lw=2.5, zorder=5)
    px, py = _merc(lon, lat, z)
    ax.scatter(px, py, s=14, color="#ff3b30", edgecolors="#ffd0cc", linewidths=0.4, zorder=4)
    ii = np.argwhere(H == 2); c = ii[np.argmin(((ii - np.array(H.shape) * [0.55, 0.6]) ** 2).sum(1))]
    cx, cy = x0 + c[0] * cell, y0 + c[1] * cell
    qx, qy = to_px(np.array([cx, cx + cell, cx + cell, cx, cx], float), np.array([cy, cy, cy + cell, cy + cell, cy], float))
    ax.plot(qx, qy, color="#7CFC9A", lw=2.5, zorder=6)
    ax.annotate("이 칸: 2발", (qx[2], qy[2]), (qx[2] + 120, qy[2] - 160), color="#7CFC9A", fontsize=16, fontweight="bold", zorder=7,
                arrowprops=dict(arrowstyle="->", color="#7CFC9A", lw=2))
    ax.text(bx[0] + 8, by[0] - 12, f"0.5 km 칸 {H.size}개 · {k}발", color="#f2c94c", fontsize=14, fontweight="bold", zorder=7)
    ax.text(0.995, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#bbbbbb", zorder=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    fig.tight_layout(pad=0.2); fig.savefig(O + "glm_v1_map.png", facecolor=fig.get_facecolor()); plt.close(fig)
    return len(e), k


def v1_sim_fig(e, n, reps=10000):
    """실제 칸별 개수 vs 포아송 vs 제멋대로 1만 번 (95% 범위). 오른쪽 = 분산/평균 의 시뮬레이션 분포."""
    from math import exp, factorial
    H, k = D.v1_cells(e, n); cells = H.size; tab = np.bincount(np.minimum(H.ravel(), 5), minlength=6)[:5]
    S = D.v1_sim(k, cells, reps)[:, :5]; lam = k / cells; ks = np.arange(5)
    lo, hi = np.percentile(S, 2.5, 0), np.percentile(S, 97.5, 0)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    a.bar(ks, tab, color="#e8322f", alpha=0.75, width=0.6, label="실제 (칸 수)")
    a.errorbar(ks + 0.33, (lo + hi) / 2, yerr=(hi - lo) / 2, fmt="none", ecolor=GRAY, elinewidth=6, alpha=0.7, label="제멋대로 1만 번: 95% 범위")
    a.plot(ks, [cells * lam ** j * exp(-lam) / factorial(j) for j in ks], "o", color="k", ms=7, label=f"포아송 λ = {lam:.2f}")
    for j in ks: a.text(j - 0.05, tab[j] + 5, str(tab[j]), ha="center", fontsize=13, color="#b01c1a", fontweight="bold")
    a.set_xticks(ks); a.set_xlabel("한 칸에 떨어진 폭탄 수"); a.set_ylabel("칸 수"); a.legend(frameon=False, fontsize=12)
    rng = np.random.default_rng(1); R = []
    for _ in range(reps):
        h = np.bincount(rng.integers(0, cells, k), minlength=cells); R.append(h.var() / h.mean())
    R = np.array(R); r = H.var() / H.mean()
    b = ax[1]; b.hist(R, bins=40, color=GRAY, alpha=0.6); b.axvline(r, color="#e8322f", lw=3)
    b.text(r + 0.01, b.get_ylim()[1] * 0.85, f"실제 {r:.2f}", color="#b01c1a", fontsize=14, fontweight="bold")
    b.set_xlabel("분산 ÷ 평균 (제멋대로면 1 근처)"); b.set_yticks([]); b.set_title("제멋대로 1만 번의 분산 ÷ 평균", fontsize=14)
    fig.tight_layout(); fig.savefig(O + "glm_v1_sim.png"); plt.close(fig)
    return tab, lo, hi, round(r, 3), np.percentile(R, [2.5, 97.5]).round(3)

if __name__ == "__main__":
    st, ch, ir = D.study(), D.chess(), D.iris(); cx, cy = D.claims()
    print("dists", dists_fig(st, ch, (cx, cy), ir))
    k = np.polyfit(cx, cy, 1); claims_line_fig(cx, cy, k)
    wp = D.fit_glm(cx, cy, np.exp, lr=0.5, steps=20000, scale=10, shift=50, w0=np.log(cy.mean())); claims_pois_fig(cx, cy, k, wp)
    W = D.fit_softmax(*ir); softmax_fig(*ir, W); pois_pmf_fig()
    pois_examples_fig([("런던: 구역에 떨어진 폭탄", "0.25 km² 구역의 V-1 수", D.V1_BOMBS, D.V1_MEAN, "구역"),
                       ("기병대: 말에 차여 숨진 병사", "한 부대 · 한 해의 사망자", D.HORSE_KICKS, D.HORSE_MEAN, "부대-년"),
                       ("자동차 보험: 1년 사고 청구", "한 운전자의 1년 청구 횟수", D.CLAIMS_COUNTS, D.CLAIMS_MEAN, "계약")])
    ve, vn = D.v1(); vl, va = D.v1_lonlat(); print("v1 map", v1_map_fig(ve, vn, vl, va)); print("v1 sim", v1_sim_fig(ve, vn))
    pois_origin_fig(); print("scatter", scatter_cells_fig())
    print("figs ok", wp.round(5))
