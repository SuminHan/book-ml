"""'트랜스포머 = 그래프 신경망' 쪽 그림 (나머지 그림은 원본 덱 slides/v2/ml1/week12/transformer/figs/).
실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py
친구 값은 쇼츠 lessons/ml1/transformer_gnn/src/data.py 와 같고, 어텐션 무게는 어텐션 덱 손 계산 (예시 벡터) 에서 나온 값."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#999999"

FRIENDS = {"A": 1, "B": 4, "C": 2, "D": 5}            # '나' 의 친구 4명과 각자의 값
OTHER = ("E", 3)                                      # 친구의 친구 (나와는 직접 안 이어짐)
WORDS = ["나는", "밤새", "게임을", "해서", "피곤하다"]
VEC = np.array([[0, 0, 2], [3, 0, 0], [2, 2, 0], [0, 1, 0], [1, 0, 0]], dtype=float)   # 어텐션 덱 손 계산과 같은 예시 벡터
_e = VEC @ VEC[4] / np.sqrt(3); ATT = list(np.exp(_e) / np.exp(_e).sum())            # '피곤하다' 가 각 단어를 보는 무게 (0.08 · 0.45 · 0.25 · 0.08 · 0.14)
assert sum(FRIENDS.values()) / len(FRIENDS) == 3 and abs(sum(ATT) - 1) < 1e-9


def save(fig, name):
    fig.tight_layout(); fig.savefig("figs/" + name, transparent=False, facecolor="white"); plt.close(fig)


def node(ax, xy, text, fc="white", ec=BLUE, r=0.42, fs=15, tc="black"):
    ax.add_patch(plt.Circle(xy, r, fc=fc, ec=ec, lw=2.5, zorder=3)); ax.text(*xy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


# 1. GNN 한 층: 친구 4명의 값을 평균 내 '나' 를 고친다
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(-3.0, 3.9); ax.set_ylim(-2.4, 2.4)
pos = {"A": (-2.2, 1.4), "B": (2.2, 1.4), "C": (-2.2, -1.4), "D": (2.2, -1.4)}
for k, (x, y) in pos.items():
    ax.annotate("", (0.45 * x / np.hypot(x, y), 0.45 * y / np.hypot(x, y)), (x * 0.8, y * 0.8),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=GOLD)); ax.text(x * 0.5, y * 0.5 + 0.25, "¼", fontsize=15, color="#7a6a3f", ha="center")
    node(ax, (x, y), f"{k}\n{FRIENDS[k]}", fs=14)
ex = (3.3, 0.0); ax.plot([pos["B"][0], ex[0]], [pos["B"][1], ex[1]], color=GRAY, lw=1.5, zorder=1)
node(ax, ex, f"{OTHER[0]}\n{OTHER[1]}", ec=GRAY, fs=13, r=0.36, tc=GRAY)
node(ax, (0, 0), "나\n→ 3", fc="#eef0fb", r=0.55, fs=15)
ax.text(0, -2.25, "(1 + 4 + 2 + 5) ÷ 4 = 3", ha="center", fontsize=15, color=BLUE)
save(fig, "tg_gnn.png")

# 2. 문장 = 모든 단어가 이어진 그래프, '피곤하다' 로 오는 선의 굵기 = 어텐션 무게
fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.3, 2.5)
ang = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
P = [(2.0 * np.cos(a), 2.0 * np.sin(a)) for a in ang]
for i in range(5):
    for j in range(i + 1, 5):
        ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color="#dddddd", lw=1.2, zorder=1)
f = WORDS.index("피곤하다")
for j, w in enumerate(ATT):
    if j == f: continue
    ax.plot([P[j][0], P[f][0]], [P[j][1], P[f][1]], color=RED, lw=1 + 22 * w, alpha=0.75, zorder=2, solid_capstyle="round")
    mx, my = (P[j][0] * 0.55 + P[f][0] * 0.45), (P[j][1] * 0.55 + P[f][1] * 0.45)
    ax.text(mx, my, f"{w:.2f}", fontsize=13, color=RED, ha="center", va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=1))
for k, (x, y) in enumerate(P):
    node(ax, (x, y), WORDS[k], fc="#fbeaea" if k == f else "white", ec=RED if k == f else BLUE, r=0.5, fs=13)
ax.text(P[f][0] + 0.15, P[f][1] - 0.85, f"자기 자신 {ATT[f]:.2f}", fontsize=12, color=RED, ha="center")
save(fig, "tg_sentence.png")

# 3. 앞만 보기 마스크: 같은 문장, 아래 삼각형만 (GPT)
n = len(WORDS); M = np.tril(np.ones((n, n)))
fig, ax = plt.subplots(figsize=(5.0, 4.4))
ax.imshow(M, cmap="Blues", vmin=0, vmax=4)
for i in range(n):
    for j in range(n):
        ax.text(j, i, "O" if M[i, j] else "×", ha="center", va="center", fontsize=15, color=BLUE if M[i, j] else GRAY)
ax.set_xticks(range(n), WORDS, fontsize=13); ax.set_yticks(range(n), WORDS, fontsize=13)
ax.set_xlabel("보는 단어 (키)"); ax.set_ylabel("지금 단어 (쿼리)"); ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "tg_mask.png")
print("ok", [round(v, 3) for v in ATT])
