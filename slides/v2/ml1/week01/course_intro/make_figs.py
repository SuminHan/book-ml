"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/intro_*.png 를 다시 만든다)
사진 원본은 photos/ (출처는 tex 마지막 '이미지 출처' 쪽). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps, ImageEnhance
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"


def box(ax, x, y, w, h, text, fc, ec, fs=16, color="black", bold=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=GRAY, lw=2.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def square(path, box_=None, top=0.0):
    im = Image.open(path).convert("RGB")
    if box_:
        im = im.crop(box_)
    w, h = im.size; s = min(w, h)
    y0 = int((h - s) * top)
    return im.crop(((w - s) // 2, y0, (w - s) // 2 + s, y0 + s)).resize((360, 360))


# 1. 과학과 공학
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2)
for y, (a, b, name, col) in zip((3.0, 1.1), (("돈", "지식", "과학", BLUE), ("지식", "돈", "공학", RED))):
    box(ax, 2.0, y, 1.8, 1.1, a, "#f4f4f8", GRAY, fs=24)
    box(ax, 6.6, y, 1.8, 1.1, b, "#f4f4f8", GRAY, fs=24)
    arrow(ax, (3.0, y), (5.6, y), color=col, lw=3.5)
    ax.text(4.3, y + 0.35, "만든다", ha="center", fontsize=15, color=col)
    ax.text(9.0, y, name, ha="center", va="center", fontsize=30, fontweight="bold", color=col)
fig.savefig(O + "intro_sci_eng.png", bbox_inches="tight"); plt.close(fig)

# 2. 연표: 아이디어는 오래됐고, 터진 건 공학이 갖춰진 뒤
def X(yr): return (yr - 1940) * 0.5 if yr < 2000 else 30 + (yr - 2000) * 2.2   # 2000년 이후를 넓게
fig, ax = plt.subplots(figsize=(14, 4.8)); ax.axis("off")
ax.set_xlim(-3, 92); ax.set_ylim(-3.4, 3.4)
ax.plot([X(1940), X(2027)], [0, 0], color="#bbbbbb", lw=3, zorder=0)
ax.plot([X(2000)] * 2, [-0.25, 0.25], color="#999999", lw=1.5)
col = {"idea": BLUE, "engine": GOLD, "boom": RED}
ups = {1943: 1, 1958: -1, 1986: 1, 1998: -1, 2005: 1, 2007: -1, 2009: 2.0, 2012: -2.0, 2016: 1, 2022: -1, 2024: 2.0}
for yr, name, kind in TIMELINE:
    up = ups[yr]
    ax.plot([X(yr), X(yr)], [0, up * 0.75], color=col[kind], lw=2)
    ax.scatter([X(yr)], [0], s=140, color=col[kind], zorder=3)
    ax.text(X(yr), up * 0.85, f"{yr}\n{name}" if up > 0 else f"{name}\n{yr}", ha="center",
            va="bottom" if up > 0 else "top", fontsize=12.5, color=col[kind], fontweight="bold" if kind == "boom" else "normal")
for i, (k, lab) in enumerate((("idea", "아이디어 (과학)"), ("engine", "계산·데이터 (공학)"), ("boom", "터진 순간"))):
    ax.scatter([0 + i * 19], [-3.2], s=110, color=col[k]); ax.text(1.2 + i * 19, -3.2, lab, va="center", fontsize=13)
fig.savefig(O + "intro_timeline.png", bbox_inches="tight"); plt.close(fig)

# 3. 사람들
people = [("lecun.jpg", None, 0.1, "얀 르쿤", "CNN\n2018 튜링상"),
          ("hinton.jpg", None, 0.0, "제프리 힌턴", "역전파\n2024 노벨 물리학상"),
          ("bengio.jpg", None, 0.05, "요슈아 벤지오", "딥러닝\n2018 튜링상"),
          ("sutskever_rao.jpg", (40, 0, 330, 300), 0.0, "일리야 서츠케버", "AlexNet\nOpenAI 공동창업"),
          ("ng.jpg", None, 0.05, "앤드루 응", "Google Brain · Coursera\nBaidu 수석 과학자"),
          ("feifei.jpg", None, 0.05, "페이페이 리", "ImageNet\nStanford HAI"),
          ("hassabis.jpg", None, 0.1, "데미스 허사비스", "AlphaGo · AlphaFold\n2024 노벨 화학상")]
fig, axs = plt.subplots(1, 7, figsize=(17, 3.9))
for ax, (f, b, top, name, what) in zip(axs, people):
    ax.imshow(square("photos/" + f, b, top)); ax.axis("off")
    ax.set_title(name, fontsize=17, fontweight="bold", color=BLUE, pad=6)
    ax.text(0.5, -0.07, what, transform=ax.transAxes, ha="center", va="top", fontsize=13, color="#333", linespacing=1.3)
fig.subplots_adjust(wspace=0.08)
fig.savefig(O + "intro_people.png", bbox_inches="tight"); plt.close(fig)

# 4. 연구실에서 회사로
rows = [("토론토 (힌턴)", "DNNresearch → Google (2013)"),
        ("NYU (르쿤)", "Facebook AI Research (2013)"),
        ("Stanford (응)", "Google Brain (2011) · Coursera (2012) · Baidu (2014)"),
        ("토론토 → Google (서츠케버)", "OpenAI 공동창업 (2015)"),
        ("UCL (허사비스)", "DeepMind (2010) → Google (2014)"),
        ("Stanford (페이페이 리)", "Google Cloud (2017) · World Labs (2024)")]
fig, ax = plt.subplots(figsize=(13.5, 4.6)); ax.axis("off"); ax.set_xlim(0, 13.5); ax.set_ylim(-0.6, len(rows))
ax.text(2.6, len(rows) - 0.1, "연구실", ha="center", fontsize=17, fontweight="bold", color=BLUE)
ax.text(9.3, len(rows) - 0.1, "회사 · 창업", ha="center", fontsize=17, fontweight="bold", color=RED)
for i, (a, b) in enumerate(rows):
    y = len(rows) - 1 - i - 0.25
    box(ax, 2.6, y, 4.4, 0.66, a, "#eef0fa", BLUE, fs=14)
    box(ax, 9.3, y, 7.0, 0.66, b, "#fbeeee", RED, fs=14)
    arrow(ax, (4.85, y), (5.75, y), color=GRAY, lw=2)
fig.savefig(O + "intro_valley.png", bbox_inches="tight"); plt.close(fig)

# 5. 이 수업의 흐름
steps = ["데이터셋 찾기", "실험", "논문 쓰기", "KSA Review\n제출", "동료 평가\n받기", "좋은 리뷰\n쓰기"]
fig, ax = plt.subplots(figsize=(14, 3.3)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 3.3)
for i, s in enumerate(steps):
    x = 1.2 + i * 2.32
    box(ax, x, 1.65, 1.95, 1.35, s, "#eef0fa" if i < 3 else "#fbeeee", BLUE if i < 3 else RED, fs=15)
    if i < len(steps) - 1:
        arrow(ax, (x + 1.0, 1.65), (x + 1.32, 1.65), color=GRAY, lw=2)
ax.text(3.5, 0.25, "AI를 도구로 파고든다", ha="center", fontsize=14, color=BLUE)
ax.text(10.5, 0.25, "학회처럼 심사를 주고받는다", ha="center", fontsize=14, color=RED)
fig.savefig(O + "intro_course_flow.png", bbox_inches="tight"); plt.close(fig)

# 6a. 정확도 90%의 함정: 100명 (10 x 10), 환자 10명, AI는 '모두 건강'
rng6 = np.random.default_rng(7)
sick_idx = set(rng6.choice(N_PEOPLE, N_SICK, replace=False).tolist())
fig, ax = plt.subplots(figsize=(6.4, 6.4)); ax.axis("off"); ax.set_aspect("equal")
for i in range(N_PEOPLE):
    ax.scatter(i % 10, 9 - i // 10, s=420, color=RED if i in sick_idx else "#5b8bd9")
ax.text(4.5, -1.3, "AI 판정: 모두 건강", ha="center", fontsize=19, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eeeeee", ec="none"))
ax.set_xlim(-0.8, 9.8); ax.set_ylim(-2.1, 9.8)
fig.savefig(O + "intro_acc_trap.png", bbox_inches="tight"); plt.close(fig)

# 6b. 재현율과 정밀도: AI가 고른 12명 vs 진짜 환자 10명
from matplotlib.patches import Circle
fig, ax = plt.subplots(figsize=(9, 6.2)); ax.axis("off"); ax.set_aspect("equal")
ax.set_xlim(-0.2, 9.2); ax.set_ylim(-0.4, 6.4)
ax.add_patch(Circle((3.3, 3.2), 2.55, fc="#f2c94c22", ec="#e0a800", lw=3.5))
ax.add_patch(Circle((5.7, 3.2), 2.55, fc="#d6272822", ec=RED, lw=3.5))
ax.text(2.4, 6.05, f"AI가 고른 {TP + FP}명", ha="center", fontsize=19, fontweight="bold", color="#c98f00")
ax.text(6.6, 6.05, f"진짜 환자 {TP + FN}명", ha="center", fontsize=19, fontweight="bold", color=RED)
for k in range(FP):                      # 헛짚음: 건강한데 AI가 고름
    ax.scatter(1.65 + (k % 2) * 0.7, 2.75 + (k // 2) * 0.9, s=380, color="#5b8bd9")
for k in range(TP):                      # 맞힘
    ax.scatter(4.15 + (k % 2) * 0.7, 1.45 + (k // 2) * 0.82, s=380, color=RED)
for k in range(FN):                      # 놓침
    ax.scatter(7.0, 2.75 + k * 0.9, s=380, color=RED)
ax.text(2.0, 1.55, f"헛짚음 {FP}", ha="center", fontsize=15, color="#5b8bd9", fontweight="bold")
ax.text(4.5, 4.95, f"맞힘 {TP}", ha="center", fontsize=15, color=RED, fontweight="bold")
ax.text(7.0, 1.55, f"놓침 {FN}", ha="center", fontsize=15, color=RED, fontweight="bold")
fig.savefig(O + "intro_recall.png", bbox_inches="tight"); plt.close(fig)

# 7. 데이터가 부족하면: 증강
cat = square("photos/cat_snow.jpg")
aug = [(cat, "원본"), (ImageOps.mirror(cat), "좌우 뒤집기"),
       (cat.crop((40, 30, 300, 290)).resize((360, 360)), "잘라 확대"),
       (cat.rotate(15, fillcolor=(235, 238, 242)), "회전"),
       (ImageEnhance.Brightness(cat).enhance(0.55), "어둡게"),
       (ImageEnhance.Color(cat).enhance(0.0), "흑백")]
fig, axs = plt.subplots(1, 6, figsize=(16, 3.3))
for ax, (im, lab) in zip(axs, aug):
    ax.imshow(im); ax.axis("off"); ax.set_title(lab, fontsize=15, color=BLUE if lab == "원본" else "black", fontweight="bold")
fig.savefig(O + "intro_augment.png", bbox_inches="tight"); plt.close(fig)

# 8. 딥러닝 이전: EM 으로 두 무리 찾기, 로그우도는 매번 오른다
rng = np.random.default_rng(EM_SEED)
xs = np.concatenate([rng.normal(m, s, n) for m, s, n in EM_GROUPS])
mu, sd, pi = np.array([-0.5, 0.5]), np.array([2.0, 2.0]), np.array([0.5, 0.5])
def pdf(x, m, s): return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
lls, snaps = [], {}
for it in range(EM_ITERS + 1):
    dens = pi * pdf(xs[:, None], mu, sd)
    lls.append(np.log(dens.sum(1)).sum())
    if it in (0, 2, EM_ITERS): snaps[it] = (mu.copy(), sd.copy(), pi.copy())
    r = dens / dens.sum(1, keepdims=True)
    nk = r.sum(0); pi = nk / len(xs); mu = (r * xs[:, None]).sum(0) / nk
    sd = np.sqrt((r * (xs[:, None] - mu) ** 2).sum(0) / nk)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
grid = np.linspace(-5.5, 7, 300)
a1.hist(xs, bins=30, density=True, color="#dddddd")
for it, c_, ls in ((0, GRAY, ":"), (2, GOLD, "--"), (EM_ITERS, RED, "-")):
    m_, s_, p_ = snaps[it]
    a1.plot(grid, (p_ * pdf(grid[:, None], m_, s_)).sum(1), color=c_, ls=ls, lw=2.5, label=f"{it}번째")
a1.legend(frameon=False, fontsize=13); a1.set_yticks([]); a1.set_title("라벨 없이 두 무리 찾기", fontsize=16)
a2.plot(range(EM_ITERS + 1), lls, "o-", color=BLUE, lw=2.5)
a2.set_xlabel("반복 횟수"); a2.set_ylabel("로그우도"); a2.set_title("로그우도는 매번 오른다", fontsize=16)
fig.tight_layout(); fig.savefig(O + "intro_em.png"); plt.close(fig)

# 9. 바텀업 vs 탑다운
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.0));
for a in (a1, a2): a.axis("off"); a.set_xlim(0, 7); a.set_ylim(0, 5.2)
stairs = ["선형대수", "확률", "미적분", "머신러닝", "딥러닝", "Diffusion 논문"]
for i, s in enumerate(stairs):
    box(a1, 1.1 + i * 0.95, 0.4 + i * 0.72, 1.6, 0.55, s, "#f4f4f8" if i < 5 else "#fbeeee", GRAY if i < 5 else RED, fs=12.5)
a1.text(3.5, 5.0, "바텀업: 기초부터 차례로 (몇 년)", ha="center", fontsize=17, fontweight="bold", color=GRAY)
box(a2, 3.5, 4.2, 3.0, 0.7, "Diffusion 논문", "#fbeeee", RED, fs=16)
qs = [("노이즈를 왜 넣지?", 1.3), ("손실이 왜 MSE?", 3.5), ("U-Net이 뭐지?", 5.7)]
for q, x in qs:
    box(a2, x, 2.6, 2.0, 0.65, q, "#eef0fa", BLUE, fs=12.5)
    arrow(a2, (3.5, 3.82), (x, 2.95), color=BLUE, lw=1.8)
for q, x, par in (("가우시안 분포", 1.1, 1.3), ("회귀와 MSE", 3.5, 3.5), ("CNN", 5.7, 5.7)):
    box(a2, x, 1.15, 1.7, 0.6, q, "#f4f4f8", GRAY, fs=12.5)
    arrow(a2, (par, 2.25), (x, 1.47), color=GRAY, lw=1.6)
a2.text(3.5, 0.25, "모를 때마다 AI에게 묻고, 필요한 만큼만 내려간다", ha="center", fontsize=14, color=BLUE)
a2.text(3.5, 5.0, "탑다운: 읽고 싶은 논문부터 (며칠)", ha="center", fontsize=17, fontweight="bold", color=RED)
fig.savefig(O + "intro_topdown.png", bbox_inches="tight"); plt.close(fig)

# 10. 이번 학기 17주
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(-0.3, 4.3)
for i, (wk, name) in enumerate(ROADMAP):
    row = 0 if i < 8 else (1 if i < 16 else 2)
    col_i = i if i < 8 else (i - 8 if i < 16 else 0)
    proj = wk in (8, 16)
    fc = "#fbeeee" if proj else ("#eef0fa" if row == 0 else ("#f6f1e4" if row == 1 else "#e8f4f2"))
    ec = RED if proj else (BLUE if row == 0 else (GOLD if row == 1 else GREEN))
    box(ax, 0.62 + col_i * 1.12, 3.5 - row * 1.45, 1.02, 1.1, f"{wk}주\n{name}", fc, ec, fs=11)
fig.savefig(O + "intro_roadmap.png", bbox_inches="tight"); plt.close(fig)
# 11. 2024 노벨상: 물리학상 힌턴, 화학상 허사비스
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, (f, top, name, prize, c_) in zip(axs, (("hinton.jpg", 0.0, "제프리 힌턴", "2024 노벨 물리학상", BLUE),
                                              ("hassabis.jpg", 0.1, "데미스 허사비스", "2024 노벨 화학상", RED))):
    ax.imshow(square("photos/" + f, None, top)); ax.axis("off")
    ax.set_title(prize, fontsize=19, fontweight="bold", color=c_, pad=8)
    ax.text(0.5, -0.06, name, transform=ax.transAxes, ha="center", va="top", fontsize=17)
fig.savefig(O + "intro_nobel.png", bbox_inches="tight"); plt.close(fig)
print("ok")
