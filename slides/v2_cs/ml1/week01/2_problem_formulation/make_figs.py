"""이 덱(v2_cs)만 쓰는 그림. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림은 원본 덱 slides/v2/ml1/week01/problem_formulation/figs/ 를 그대로 쓴다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.datasets import load_wine
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
CLASS_COLORS = [BLUE, RED, GREEN]


def icon(ax, name, x, y, zoom):
    """figs/emoji/<name>.png (Apple Color Emoji 를 PNG 로 뽑은 것) 를 데이터 좌표 (x, y) 에 붙인다."""
    ab = AnnotationBbox(OffsetImage(plt.imread(f"figs/emoji/{name}.png"), zoom=zoom), (x, y), frameon=False)
    ax.add_artist(ab)


def bars(ax, labels, vals, colors, fmt, ylabel, ymax):
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.02, fmt(v), ha="center", fontsize=17, fontweight="bold", color=colors[i])
    ax.set_xticks(range(len(vals)), labels, fontsize=13.5); ax.set_ylim(0, ymax); ax.set_ylabel(ylabel)


def wine_two_problems(acc, mae, out):
    """같은 와인 178병, 두 문제. 이 덱에서는 문제 A = 알코올 도수 (회귀), 문제 B = 품종 (분류).
    acc/mae = {"base": ..., "model": ...}: 값은 원본 덱 data.py 의 WINE_ACC / WINE_MAE 와 같다."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    bars(a1, ["기준선\n학습 평균", "선형회귀"], [mae["base"], mae["model"]], [GRAY, GREEN],
         lambda v: f"{v:.2f}", "시험 MAE (도) ↓", 0.85)
    a1.set_title("문제 A: 12성분 → 알코올 도수 (회귀)", fontsize=15)
    bars(a2, ["기준선\n가장 많은 품종", "로지스틱회귀"], [acc["base"], acc["model"]], [GRAY, BLUE],
         lambda v: f"{v}%", "시험 정확도 (%) ↑", 115)
    a2.set_title("문제 B: 13성분 → 품종 (분류)", fontsize=15)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def sup_unsup(X, y, out):
    """와인 178병을 두 성분(flavanoids, color intensity)으로 그린다. 왼쪽은 품종(y)을 색으로, 오른쪽은 y 없이 회색."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True, sharey=True)
    for k, c in enumerate(CLASS_COLORS):
        a1.scatter(X[y == k, 0], X[y == k, 1], s=26, color=c, alpha=0.85, label=f"class {k}")
    a2.scatter(X[:, 0], X[:, 1], s=26, color=GRAY, alpha=0.8)
    for ax, t in ((a1, "지도학습: 정답 $y$ (품종) 이 있다"), (a2, "비지도학습: 입력 $x$ 만 있다")):
        ax.set_title(t, fontsize=15, fontweight="bold"); ax.set_xlabel("flavanoids"); ax.set_ylabel("color intensity")
    a1.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def recipe(out):
    """학습의 뼈대: 데이터 -> 모델 -> 손실 -> 최적화, 최적화가 모델의 파라미터를 갱신하는 고리."""
    fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.8)
    boxes = [("① 데이터", "$(x,\\,y)$ 쌍", BLUE, "data"), ("② 모델", "$\\hat{y}=f_w(x)$", GOLD, "model"),
             ("③ 손실함수", "$L(w)$: 얼마나 틀렸나", RED, "loss"), ("④ 최적화", "$w$ 를 고친다", GREEN, "opt")]
    bw, gap = 2.25, 0.5
    for i, (t, s, c, ic) in enumerate(boxes):
        x0 = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, 1.2), bw, 1.5, boxstyle="round,pad=0.03", fc=c, ec="none"))
        ax.text(x0 + bw / 2, 2.15, t, ha="center", va="center", color="white", fontsize=17, fontweight="bold")
        ax.text(x0 + bw / 2, 1.65, s, ha="center", va="center", color="white", fontsize=13)
        icon(ax, ic, x0 + bw / 2, 3.2, 0.3)
        if i < 3:
            ax.annotate("", (x0 + bw + gap - 0.05, 1.95), (x0 + bw + 0.05, 1.95), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    x_mod, x_opt = 1 * (bw + gap) + bw / 2, 3 * (bw + gap) + bw / 2     # 최적화 -> 모델 되돌아가는 고리
    ax.plot([x_opt, x_opt, x_mod], [1.15, 0.8, 0.8], color=GRAY, lw=2)
    ax.annotate("", (x_mod, 1.15), (x_mod, 0.8), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax.text((x_mod + x_opt) / 2, 0.4, "손실이 줄어들 때까지 반복", ha="center", fontsize=14, color=GRAY)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def grad_descent(w0, lr, steps, out):
    """L(w) = (w - 3)^2 + 1 위에서 경사하강법 w <- w - lr * L'(w) 를 steps 번. 점마다 번호를 붙인다."""
    L = lambda w: (w - 3) ** 2 + 1
    dL = lambda w: 2 * (w - 3)
    ws = [w0]
    for _ in range(steps): ws.append(ws[-1] - lr * dL(ws[-1]))
    g = np.linspace(-2.2, 8.2, 300)
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.plot(g, L(g), color=BLUE, lw=3)
    ax.plot(ws, [L(w) for w in ws], "o--", color=RED, lw=1.5, ms=9)
    for k, w in enumerate(ws):
        ax.text(w, L(w) + 2.2, f"{k}", ha="center", fontsize=13, color=RED, fontweight="bold")
    ax.axvline(3, color=GRAY, lw=1, ls=":"); ax.text(3.45, 9.2, "최솟값", color=GRAY, fontsize=13)
    ax.set_xlabel("파라미터 $w$"); ax.set_ylabel("손실 $L(w)$"); ax.set_ylim(-1, 40); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return ws


if __name__ == "__main__":
    d = load_wine()
    cols = [d.feature_names.index("flavanoids"), d.feature_names.index("color_intensity")]
    sup_unsup(d.data[:, cols], d.target, "figs/pf_sup_unsup.png")
    recipe("figs/pf_recipe.png")
    wine_two_problems({"base": 39, "model": 100}, {"base": 0.66, "model": 0.45}, "figs/pf_wine.png")
    print([round(w, 2) for w in grad_descent(-1.0, 0.3, 4, "figs/pf_gd.png")])
