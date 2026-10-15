"""연결 예측 커버 그림 (assets/link_prediction_drawn.png). 얼굴은 Twemoji (CC BY 4.0), data/twemoji/ 에 받아 둔다.
   ~/Documents/manim_shorts/_env/v/bin/python assets/draw_link_prediction.py"""
import os, urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
plt.rcParams.update({"font.family": "Noto Sans CJK KR"})
pos = {"Adam": (0.12, 0.62), "Sophia": (0.58, 0.92), "Maya": (0.6, 0.55), "Maria": (0.35, 0.28), "David": (0.85, 0.15)}
emo = {"Adam": "1f468-1f3fe", "Sophia": "1f469-1f3fb", "Maya": "1f469-1f3fd-200d-1f9b0", "Maria": "1f469-1f3ff-200d-1f9b1", "David": "1f468-1f3fc-200d-1f9b0"}
ring = {"Adam": "#2a9d8f", "Sophia": "#f4b400", "Maya": "#6fa8dc", "Maria": "#9b72cf", "David": "#e76f51"}
E = [("Adam", "Maria"), ("Sophia", "Maria"), ("Sophia", "Maya"), ("Maya", "Maria"), ("Maria", "David"), ("Maya", "David")]
NEW = [("Adam", "Sophia"), ("Sophia", "David")]
os.makedirs("data/twemoji", exist_ok=True)
for code in emo.values():
    p = f"data/twemoji/{code}.png"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{code}.png", p)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, title, new in [(axs[0], "시점 t: 지금의 친구 관계", []), (axs[1], "시점 t′ (t′ > t): 새로 생긴 연결 (점선)", NEW)]:
    for a, b in E: ax.plot(*zip(pos[a], pos[b]), color="#333333", lw=2.5, zorder=1)
    for a, b in new: ax.plot(*zip(pos[a], pos[b]), color="#d62728", lw=2.8, ls="--", zorder=1)
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=1900, color=ring[n], edgecolor="white", linewidth=3, zorder=2)
        img = np.asarray(Image.open(f"data/twemoji/{emo[n]}.png").convert("RGBA"))
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.5), (x, y), frameon=False, zorder=3))
        ax.text(x, y + 0.125 if n != "David" else y - 0.16, n, ha="center", fontsize=12, fontweight="bold", zorder=3)
    ax.set_title(title, fontsize=13); ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.08, 1.1); ax.axis("off")
fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.92], color="#999999", ls="--", lw=1.5))
fig.tight_layout(); fig.savefig("assets/link_prediction_drawn.png", dpi=130, facecolor="white")
print("ok")
