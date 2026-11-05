"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py 와 같은 폴더의 것을 쓴다."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
os.chdir(os.path.dirname(os.path.abspath(__file__)))          # 덱 폴더에서 실행 -> figs/ 에 저장
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"

# 1. 메일 목업
fig, ax = plt.subplots(figsize=(8, 4.2)); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84, boxstyle="round,pad=0.02", fc="#f7f7fb", ec=BLUE, lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); r_ = fig.canvas.get_renderer(); x_ = 0.1
for seg, col, gap in (("제목: [안내]", "black", 0.012), ("무료", RED, 0.012), ("쿠폰", "black", 0.012), ("당첨", RED, 0.0), ("을 축하합니다", "black", 0)):
    t_ = ax.text(x_, 0.8, seg, fontsize=17, fontweight="bold", color=col)
    x_ = t_.get_window_extent(r_).transformed(ax.transData.inverted()).x1 + gap
ax.text(0.1, 0.62, "지금 확인하시면 쿠폰을 즉시 지급해 드립니다.", fontsize=15, color="#444")
ax.text(0.5, 0.25, "스팸일까, 정상일까?", fontsize=24, fontweight="bold", color=RED, ha="center")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(O + "nb_mail.png", bbox_inches="tight"); plt.close(fig)

# 2. 1000통 트리 막대
fig, ax = plt.subplots(figsize=(9, 4.4))
s, h = 1000 * SPAM_PRIOR, 1000 * (1 - SPAM_PRIOR); sf, hf = s * P_FREE["spam"], h * P_FREE["ham"]
ax.barh([1, 0], [s, h], color=["#f4a3a3", "#a9c5e8"], height=0.55)
ax.barh([1, 0], [sf, hf], color=[RED, BLUE], height=0.55)
ax.text(sf / 2, 1, f"'무료' {sf:.0f}", color="white", ha="center", va="center", fontsize=16, fontweight="bold")
ax.text(max(hf, 60) + 10, 0, f"'무료' {hf:.0f}", color=BLUE, va="center", fontsize=16, fontweight="bold")
ax.text(s + 10, 1, f"스팸 {s:.0f}통", va="center", fontsize=15); ax.text(h + 10, 0, f"정상 {h:.0f}통", va="center", fontsize=15)
ax.set_yticks([]); ax.set_xlim(0, 820); ax.set_xlabel("메일 수 (1000통 중)")
ax.set_title(f"'무료'가 든 {sf+hf:.0f}통 중 스팸 {sf:.0f}통  →  {sf/(sf+hf):.1%}", fontsize=18, color=RED)
fig.tight_layout(); fig.savefig(O + "nb_tree.png"); plt.close(fig)

# 3. 증거가 쌓이면: 단어 보기 전 / '무료' / '당첨' / 둘 다 (슬라이드의 1000통 표와 같은 숫자)
p1 = posterior(SPAM_PRIOR, P_FREE["spam"], P_FREE["ham"])
pw = posterior(SPAM_PRIOR, P_WIN["spam"], P_WIN["ham"])
p2 = posterior(p1, P_WIN["spam"], P_WIN["ham"])
vals = [SPAM_PRIOR, p1, pw, p2]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
bars = ax.bar(["단어 보기 전\n(메일함 스팸 비율)", "'무료' 확인", "'당첨' 확인", "'무료' + '당첨'\n확인"], vals, color=[GRAY, GOLD, GOLD, RED], width=0.55)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.03, f"{v:.1%}", ha="center", fontsize=18, fontweight="bold")
ax.set_ylim(0, 1.12); ax.set_ylabel("스팸일 확률")
fig.tight_layout(); fig.savefig(O + "nb_update.png"); plt.close(fig)

# 4. 스팸이 1%뿐인 메일함: 1000통 격자
fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.axis("off")
cols, rows = 50, 20
ns = int(1000 * LOW_PRIOR); nsf = int(round(ns * P_FREE["spam"])); nhf = int(round((1000 - ns) * P_FREE["ham"]))
rng = np.random.default_rng(2); hamf = set(rng.choice(np.arange(ns, 1000), nhf, replace=False))
spf = set(np.random.default_rng(12).choice(ns, nsf, replace=False))      # 스팸 영역 안에서도 흩어지게
for i in range(1000):
    r, c = divmod(i, cols)
    col = (RED if i in spf else "#f4a3a3") if i < ns else (BLUE if i in hamf else "#cfd8e3")
    ax.add_patch(Rectangle((c, rows - r), 0.82, 0.82, color=col))
ax.set_xlim(-0.5, cols + 0.5); ax.set_ylim(0, rows + 1.5)
pl = posterior(LOW_PRIOR, P_FREE["spam"], P_FREE["ham"])
ax.text(0, rows + 1.2, f"■ '무료' 든 스팸 {nsf}통", color=RED, fontsize=15, fontweight="bold")
ax.text(15, rows + 1.2, f"■ '무료' 든 정상 약 {nhf}통", color=BLUE, fontsize=15, fontweight="bold")
ax.text(34, rows + 1.2, f"'무료' 메일 중 스팸 ≈ {pl:.0%}", color="#333", fontsize=15, fontweight="bold")
fig.savefig(O + "nb_grid.png", bbox_inches="tight"); plt.close(fig)

# 5. 사전확률 곡선 (스팸 비율 -> '무료' 메일이 스팸일 확률). 라벨은 곡선 왼쪽 위 빈 곳에 (x축과 겹치지 않게)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
pr = np.logspace(-3, -0.05, 200); ax.semilogx(pr, [posterior(p, P_FREE["spam"], P_FREE["ham"]) for p in pr], color=BLUE, lw=3)
for (p0, lab), ty in (((LOW_PRIOR, "스팸 1% 메일함"), 0.50), ((SPAM_PRIOR, "스팸 30% 메일함"), 0.93)):
    q = posterior(p0, P_FREE["spam"], P_FREE["ham"]); ax.scatter([p0], [q], color=RED, s=90, zorder=3)
    ax.annotate(f"{lab} → {q:.0%}", (p0, q), (0.0012, ty), fontsize=17, fontweight="bold", color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("메일함의 스팸 비율 (사전확률)"); ax.set_ylabel("'무료' 메일이 스팸일 확률"); ax.set_ylim(0, 1.02)
ax.set_xticks([1e-3, 1e-2, 1e-1, 0.5]); ax.set_xticklabels(["0.1%", "1%", "10%", "50%"])
fig.tight_layout(); fig.savefig(O + "nb_prior_curve.png"); plt.close(fig)

# 6. 말뭉치 단어 확률 (베르누이: 단어가 든 메일 수 + 1) / (메일 수 + 2). 수식 라벨은 영어
Ms, Mh = len(SPAM), len(HAM)
ns_ = lambda w: sum(w in m.split() for m in SPAM); nh_ = lambda w: sum(w in m.split() for m in HAM)
words = ["무료", "당첨", "쿠폰", "회의", "자료", "내일"]
ps = [(ns_(w) + 1) / (Ms + 2) for w in words]; ph = [(nh_(w) + 1) / (Mh + 2) for w in words]
fig, ax = plt.subplots(figsize=(9, 4.2)); xi = np.arange(len(words))
ax.bar(xi - 0.2, ps, 0.4, color=RED, label="P(word present | spam)"); ax.bar(xi + 0.2, ph, 0.4, color=BLUE, label="P(word present | ham)")
ax.set_xticks(xi); ax.set_xticklabels(words, fontsize=16); ax.legend(frameon=False); ax.set_ylabel("probability (smoothed)")
fig.tight_layout(); fig.savefig(O + "nb_words.png"); plt.close(fig)


# 7. 우리 반 20명: 조건부 확률
from matplotlib.patches import Circle, Wedge
def people(ax, highlight=None):
    """highlight: 'soccer' | 'glasses' | None"""
    for i, (sc, gl) in enumerate(zip(CLASS_SOCCER, CLASS_GLASSES)):
        if i < 8: x, y = (i % 4) * 1.2, 1.4 - (i // 4) * 1.6
        else: j = i - 8; x, y = 5.6 + (j % 6) * 1.2, 1.4 - (j // 6) * 1.6
        on = highlight is None or (highlight == "soccer" and sc) or (highlight == "glasses" and gl)
        a = 1 if on else 0.15; col = GREEN if sc else "#8d99ae"
        ax.add_patch(Wedge((x, y - 0.55), 0.48, 0, 180, color=col, alpha=a))
        ax.add_patch(Circle((x, y), 0.3, color="#f1c9a5", alpha=a))
        if gl:
            for dx in (-0.12, 0.12): ax.add_patch(Circle((x + dx, y + 0.03), 0.09, fill=False, lw=2.2, color="#222", alpha=a))
            ax.plot([x - 0.03, x + 0.03], [y + 0.05, y + 0.05], color="#222", lw=2, alpha=a)
    ax.set_xlim(-0.8, 12.4); ax.set_ylim(-1.1, 2.1); ax.set_aspect("equal"); ax.axis("off")
    ax.text(1.8, 2.05, "축구부 8명", ha="center", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(8.6, 2.05, "축구부 아닌 12명", ha="center", fontsize=14, color="#555", fontweight="bold")
for name, hl, txt in (("nb_class.png", None, "20명 · 안경 쓴 사람 9명"), ("nb_class_soccer.png", "soccer", "축구부 8명 중 안경 3명  →  3/8 = 37.5%"),
                      ("nb_class_glasses.png", "glasses", "안경 9명 중 축구부 3명  →  3/9 = 33.3%")):
    fig, ax = plt.subplots(figsize=(10, 3.4)); people(ax, hl)
    ax.set_title(txt, fontsize=17, color=RED if hl else "#333", pad=18)
    fig.savefig(O + name, bbox_inches="tight"); plt.close(fig)

print("figs ok", f"p1={p1:.4f} pw={pw:.4f} p2={p2:.4f} low={posterior(LOW_PRIOR, P_FREE['spam'], P_FREE['ham']):.4f}")
