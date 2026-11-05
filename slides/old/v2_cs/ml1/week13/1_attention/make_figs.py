"""이 덱에만 있는 그림 원본: '나는 밤새 게임을 해서 피곤하다' 손 계산 (트랜스포머 덱 그래프 그림과 같은 숫자).
실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  — 나머지 그림은 원본 덱 slides/v2/ml1/week12/attention/figs/.
단어 벡터는 손 계산용 예시 (Q = K = V = 이 벡터, d = 3)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GRAY = "#2E3192", "#d62728", "#999999"
WORDS = ["나는", "밤새", "게임을", "해서", "피곤하다"]
V = np.array([[0, 0, 2], [3, 0, 0], [2, 2, 0], [0, 1, 0], [1, 0, 0]], dtype=float)


def attention(q, K, Vals):
    e = K @ q / np.sqrt(len(q)); a = np.exp(e) / np.exp(e).sum()
    return e, a, a @ Vals


e, a, z = attention(V[4], V, V)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b = ax.bar(WORDS, a, color=[GRAY, RED, BLUE, GRAY, GRAY])
for r, v in zip(b, a):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=14)
ax.set_ylabel("'피곤하다' 가 보는 비율"); ax.set_ylim(0, 0.55)
fig.tight_layout(); fig.savefig("figs/at_tired.png"); plt.close(fig)
print("scores", e.round(3), "weights", a.round(3), "z", z.round(2))
e2, a2, _ = attention(np.array([0, 1, 0.]), V, V); print("quiz q=(0,1,0) weights", a2.round(2))
