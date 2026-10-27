"""이 덱(v2_cs)만 쓰는 그림. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림은 원본 덱 slides/v2/ml1/week01/course_intro/figs/ 를 그대로 쓴다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR"})
SRC = "../../../../v2/ml1/week01/course_intro/figs/"
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
