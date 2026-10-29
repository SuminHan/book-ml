"""이 덱(v2_cs)만 쓰는 그림. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림은 원본 덱 slides/v2/ml1/week01/course_intro/figs/ 를 그대로 쓴다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR"})
SRC = "../../../v2/ml1/week01/course_intro/figs/"
WINTERS = ((1974, 1980, "1차 AI 겨울\n1974~80"), (1987, 1993, "2차 AI 겨울\n1987~93"))   # p.10 연표와 같은 기간


def winters_fixed(src, out, winters):
    """퍼셉트론 인기 그래프의 회색 띠(신경망 겨울)를 지우고, 연표와 같은 AI 겨울 기간을 파란 띠로 다시 칠한다."""
    im = np.asarray(Image.open(src).convert("RGB")).astype(float)
    s = im.shape[1] / 2000                                  # 원본 2160px, 아래 좌표는 2000px 기준
    def px(yr): return (333 + (yr - 1940) * 19.525) * s     # 눈금: 1940 → 333, 2020 → 1895
    top, bot = int(82 * s), int(805 * s)
    plot = im[top:bot]
    gray = (plot.min(axis=2) > 222) & (plot.max(axis=2) < 250) & (np.ptp(plot, axis=2) < 6)
    plot[gray] = 255                                       # 옛 회색 띠 지우기
    for x0, x1 in ((955, 1135), (1445, 1630)):              # 옛 'A.I. Winter' 글자 지우기
        im[int(135 * s):int(185 * s), int(x0 * s):int(x1 * s)] = 255
    blue = np.array([207, 227, 245], float)
    for a, b, _ in winters:                                 # 흰 바탕만 파랗게 (곡선·글자는 그대로)
        sub = im[top:bot, int(px(a)):int(px(b))]
        white = sub.min(axis=2) > 235
        sub[white] = 0.3 * sub[white] + 0.7 * blue
    h, w = im.shape[:2]
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(im.astype(np.uint8))
    for a, b, lab in winters:
        ax.text((px(a) + px(b)) / 2, 100 * s, lab, ha="center", va="top", fontsize=13,
                color="#3a7bb5", linespacing=1.15)
    fig.savefig(out, dpi=200); plt.close(fig)


winters_fixed(SRC + "ai_winters_perceptron.jpg", "figs/ai_winters_fixed.png", WINTERS)


def xor_lines(out, X, T):
    """1969 XOR: 직선 하나로는 0 과 1 을 못 가르고, 직선 두 개(OR · NAND)면 가운데 띠로 가른다 (7주차 xor_backprop 그림과 같은 모양)."""
    BLUE, RED, GREEN, GOLD = "#2E3192", "#d62728", "#2a9d8f", "#A9976B"
    xs = np.linspace(-0.4, 1.4, 2)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax in axs:
        for (a, b), v in zip(X, T):
            ax.scatter(a, b, s=380, color=RED if v else BLUE, marker="o" if v else "s", zorder=3, edgecolor="black")
            ax.text(a + 0.11, b + 0.11, str(v), fontsize=17, fontweight="bold")
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_aspect("equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.tick_params(labelsize=13)
        ax.set_xlabel("$A$", fontsize=15); ax.set_ylabel("$B$", fontsize=15)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[0].plot(xs, 0.5 - xs, color=GREEN, lw=3)
    axs[0].scatter(1, 1, s=1100, facecolor="none", edgecolor=RED, lw=2.5, zorder=4)   # 직선 하나면 늘 하나는 틀린다
    axs[0].set_title("직선 하나: 못 가른다", fontsize=16, color=RED)
    axs[1].plot(xs, 0.5 - xs, color=GREEN, lw=3); axs[1].plot(xs, 1.5 - xs, color=GOLD, lw=3)
    axs[1].fill_between(xs, 0.5 - xs, 1.5 - xs, color=RED, alpha=0.1)
    axs[1].set_title("직선 두 개: 가른다", fontsize=16, color=GREEN)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


xor_lines("figs/xor_lines.png", [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0])


def portrait(src, out, top=0.0, size=480):
    """인물 사진을 정사각형으로 자른다 (원본 덱 intro_people 과 같은 자르기). top = 위아래 여백 중 위쪽 비율."""
    im = Image.open(src).convert("RGB")
    w, h = im.size; s = min(w, h); y0 = int((h - s) * top)
    im.crop(((w - s) // 2, y0, (w - s) // 2 + s, y0 + s)).resize((size, size)).save(out, quality=92)


def strip_banner(src, out, frac):
    """사진 위쪽 제목 띠(frac 비율)를 잘라 낸다."""
    im = Image.open(src).convert("RGB")
    im.crop((0, int(im.size[1] * frac), im.size[0], im.size[1])).save(out, quality=92)


PHOTOS = "../../../v2/ml1/week01/course_intro/photos/"
portrait(PHOTOS + "lecun.jpg", "figs/lecun_sq.jpg", 0.1)
portrait(PHOTOS + "feifei.jpg", "figs/feifei_sq.jpg", 0.05)
strip_banner(SRC + "alexnet_authors.jpg", "figs/alexnet_authors_crop.jpg", 0.07)


def roadmap(out, weeks, colors):
    """이번 학기 16주 (2027-1 학생 공지용 강의계획서 5. 주차별 계획). 8주씩 두 줄, 칸 색 = 분야."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(15, 3.4)); ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0.6, 4.2)
    for i, (wk, name, kind) in enumerate(weeks):
        fc, ec = colors[kind]
        x, y, w, h = 0.62 + (i % 8) * 1.12, 3.4 - (i // 8) * 1.5, 1.06, 1.3
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=2))
        ax.text(x, y, f"{wk}주\n{name}", ha="center", va="center", fontsize=15, fontweight="bold", linespacing=1.25)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


ROADMAP16 = [
    (1, "기계학습 소개\n· kNN", "ml"), (2, "회귀 · 분류", "ml"), (3, "SVM · 커널", "ml"), (4, "트리 · GBDT", "ml"),
    (5, "나이브 베이즈\n· GDA", "ml"), (6, "k-means\n· EM · PCA", "ml"), (7, "신경망 ·\n역전파", "dl"), (8, "프로젝트 1\n+ 중간", "proj"),
    (9, "CNN", "dl"), (10, "표현 학습\n(임베딩)", "dl"), (11, "RNN", "dl"), (12, "어텐션 ·\n트랜스포머", "dl"),
    (13, "그래프", "dl"), (14, "VAE · GAN ·\nDiffusion", "dl"), (15, "LLM", "llm"), (16, "프로젝트 2\n+ 기말", "proj"),
]
roadmap("figs/intro_roadmap.png", ROADMAP16, {"ml": ("#eef0fa", "#2E3192"), "dl": ("#f6f1e4", "#A9976B"),
                                               "proj": ("#fbeeee", "#d62728"), "llm": ("#e8f4f2", "#2a9d8f")})


def scaling_law(out, laws):
    """Kaplan et al. (2020) 그림 1 을 논문의 근사식으로 다시 그린다: 계산 · 데이터 · 매개변수를 늘리면 시험 손실이 거듭제곱으로 준다 (로그-로그에서 직선)."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, (xlabel, x0, x1, xc, a, unit) in zip(axs, laws):
        x = np.logspace(np.log10(x0), np.log10(x1), 100)
        ax.plot(x, (x / xc) ** (-a), color="#2E3192", lw=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(f"{xlabel} ({unit})", fontsize=15)
        ax.text(0.95, 0.9, f"$L \\propto x^{{-{a}}}$", transform=ax.transAxes, ha="right", fontsize=16)
        ax.set_yticks([2, 3, 4, 5, 6]); ax.set_yticklabels(["2", "3", "4", "5", "6"]); ax.minorticks_off()
        dec = int(round(np.log10(x1 / x0))); step = 1 if dec <= 4 else 2
        ax.set_xticks(np.logspace(np.log10(x0), np.log10(x1), dec // step + 1))
        ax.tick_params(labelsize=12); ax.grid(alpha=0.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[0].set_ylabel("Test loss L", fontsize=15)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def users_100m(out, rows):
    """사용자 1억 명까지 걸린 달 수 (UBS 추정, Reuters 2023-02-02). ChatGPT 만 빨강, 나머지는 회색, 막대마다 숫자."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [r[0] for r in rows][::-1]; months = [r[1] for r in rows][::-1]
    cols = ["#d62728" if n == "ChatGPT" else "#9a9aa6" for n in names]
    ax.barh(names, months, color=cols, height=0.55)
    for i, (n, m) in enumerate(zip(names, months)):
        ax.text(m + 0.6, i, f"{m}개월", va="center", fontsize=17, fontweight="bold",
                color="#d62728" if n == "ChatGPT" else "#333333")
    ax.set_xlim(0, max(months) * 1.18); ax.set_xticks([]); ax.tick_params(axis="y", labelsize=17, length=0)
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


# Kaplan et al. 2020, Fig. 1: L = (C/2.3e8)^-0.050, L = (D/5.4e13)^-0.095, L = (N/8.8e13)^-0.076
scaling_law("figs/scaling_law.png", [("Compute", 1e-9, 1e1, 2.3e8, 0.050, "PF-days"),
                                     ("Dataset size", 1e7, 1e10, 5.4e13, 0.095, "tokens"),
                                     ("Parameters", 1e5, 1e9, 8.8e13, 0.076, "count")])
users_100m("figs/users_100m.png", [("ChatGPT", 2), ("TikTok", 9), ("Instagram", 30), ("Spotify", 54)])


def emergence(out, tasks):
    """Wei et al. (2022) 그림 2 에서 네 과제를 골라, 그래프를 눈으로 읽은 근사값으로 다시 그린다.
    점선 = 찍어서 맞히는 수준. 한동안 그 근처에 머물다 어느 크기를 넘으면 갑자기 뛴다."""
    fig, axs = plt.subplots(1, len(tasks), figsize=(13, 3.3))
    for ax, (title, lx, acc, rand, ymax) in zip(axs, tasks):
        x = 10.0 ** np.array(lx)
        ax.axhline(rand, color="#d6337f", ls="--", lw=2)
        ax.plot(x, acc, color="#2E3192", lw=2.5, marker="o", ms=7)
        ax.annotate(f"{acc[-1]:g}%", (x[-1], acc[-1]), textcoords="offset points", xytext=(-6, 8),
                    ha="right", fontsize=14, fontweight="bold", color="#d62728")
        ax.set_xscale("log"); ax.set_xlim(1e18, 1e25); ax.set_ylim(-2, ymax)
        ax.set_xticks([1e18, 1e20, 1e22, 1e24]); ax.minorticks_off(); ax.tick_params(labelsize=12)
        ax.set_title(title, fontsize=15); ax.set_xlabel("Training FLOPs", fontsize=13)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axs[0].set_ylabel("Accuracy (%)", fontsize=14)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


# (제목, log10 학습 FLOPs, 정답률 %, 찍기 수준 %, y 위쪽) — Wei et al. 2022 Fig. 2 근사 (모델: GPT-3 · LaMDA · LaMDA · Chinchilla)
emergence("figs/emergence.png", [
    ("모듈러 산술", [20.2, 20.6, 21.0, 21.3, 21.6, 22.0, 22.2, 23.5], [0, 0.3, 0.5, 0.5, 0.5, 1.5, 8.7, 33], 0, 50),
    ("발음 기호로 바꾸기", [18.4, 19.5, 20.0, 20.3, 20.7, 21.0, 21.4, 22.3, 23.1, 23.7], [0.2, 0.2, 0.3, 0.3, 0.4, 0.6, 1, 1, 15, 22.5], 0, 50),
    ("섞인 철자 바로잡기", [18.4, 19.5, 19.9, 20.1, 20.6, 21.0, 21.3, 21.6, 22.0, 22.4, 23.1, 23.7], [1, 0, 0, 0, 0.4, 1, 1.2, 1.5, 2.3, 2, 10.5, 12.3], 0, 50),
    ("여러 과목 객관식 (MMLU)", [20.9, 21.4, 22.1, 23.8], [25.5, 27.5, 29.5, 60], 25, 70),
])


def exams(out, rows):
    """GPT-3.5 → GPT-4 시험 백분위 (응시자 중 몇 % 보다 잘했나). OpenAI GPT-4 Technical Report (2023) Table 1, 범위는 아래쪽 값."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (name, a, b) in enumerate(rows[::-1]):
        ax.annotate("", (b, i), (a, i), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5, mutation_scale=18,
                                                         shrinkA=9, shrinkB=9))
        ax.scatter([a], [i], s=160, color="#9a9aa6", zorder=3); ax.scatter([b], [i], s=160, color="#d62728", zorder=3)
        ax.text(a, i + 0.28, f"{a}", ha="center", fontsize=14, color="#555555")
        ax.text(b, i + 0.28, f"{b}", ha="center", fontsize=14, fontweight="bold", color="#d62728")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=16)
    ax.set_xlim(0, 105); ax.set_ylim(-0.5, len(rows) - 0.3); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("백분위 (응시자 중 몇 % 보다 잘했나)", fontsize=14); ax.grid(axis="x", alpha=0.3)
    ax.scatter([], [], s=120, color="#9a9aa6", label="GPT-3.5 (2022)"); ax.scatter([], [], s=120, color="#d62728", label="GPT-4 (2023)")
    ax.legend(loc="lower left", fontsize=13, frameon=False, ncol=2, bbox_to_anchor=(0, 1.0))
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


exams("figs/exams.png", [("미국 변호사 시험", 10, 90), ("로스쿨 입학시험 LSAT", 40, 88), ("SAT 수학", 70, 89),
                         ("GRE 언어", 63, 99), ("생물 올림피아드 준결선", 31, 99)])


def course_flow(out, steps, n_blue):
    """한 학기 흐름 (원본 덱 intro_course_flow 와 같은 모양). 리뷰를 먼저 쓰고, 그다음 내 논문이 평가를 받는다."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    BLUE, RED, GRAY = "#2E3192", "#d62728", "#888888"
    fig, ax = plt.subplots(figsize=(14, 3.3)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 3.3)
    for i, s in enumerate(steps):
        x, blue = 1.2 + i * 2.32, i < n_blue
        ax.add_patch(FancyBboxPatch((x - 0.975, 1.65 - 0.675), 1.95, 1.35, boxstyle="round,pad=0.02",
                                    fc="#eef0fa" if blue else "#fbeeee", ec=BLUE if blue else RED, lw=2))
        ax.text(x, 1.65, s, ha="center", va="center", fontsize=15, fontweight="bold")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.0, 1.65), (x + 1.32, 1.65), arrowstyle="-|>", mutation_scale=22, color=GRAY, lw=2))
    ax.text(3.5, 0.25, "AI를 도구로 파고든다", ha="center", fontsize=14, color=BLUE)
    ax.text(10.5, 0.25, "학회처럼 심사를 주고받는다", ha="center", fontsize=14, color=RED)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


course_flow("figs/intro_course_flow.png", ["데이터셋 찾기", "실험", "논문 쓰기", "KSA Review\n제출", "좋은 리뷰\n쓰기", "동료 평가\n받기"], 3)
