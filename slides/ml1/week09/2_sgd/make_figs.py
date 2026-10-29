"""이 덱(v2_cs)만 쓰는 그림. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
나머지 그림은 원본 덱 slides/v2/ml1/week02/sgd/figs/ 를 그대로 쓴다."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 15})
BLUE, RED = "#2E3192", "#d62728"
AE = os.path.expanduser("~/book-ml/lessons/ml1/sgd/src/assets/")   # SGD 쇼츠 실험 sgd_ae_exp.py 결과 (오토인코더, 시험 숫자 0·1·4·7)


def ae_strip(src, steps, out):
    """같은 6만 장 1에폭: 전체 배치 1걸음 vs 32장씩 걸음마다 다시 그린 숫자. 맨 오른쪽은 원본."""
    cols = [(src + "ae/full_1.png", "6만 장 보고\n1걸음", RED)]
    cols += [(src + f"ae/b32_{s}.png", f"32장씩\n{s:,}걸음", BLUE) for s in steps]
    cols += [(src + "ae/orig.png", "원본", "black")]
    fig, axs = plt.subplots(1, len(cols), figsize=(2.0 * len(cols), 2.75))
    for ax, (path, lab, col) in zip(axs, cols):
        ax.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(2.5 if col != "black" else 1)
        ax.set_title(lab, fontsize=14, color=col, fontweight="bold" if col == RED else None)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.74, bottom=0.03, wspace=0.08)
    fig.savefig(out, dpi=160); plt.close(fig)


# p.1 오늘의 질문: 다시 그리기(오토인코더)에서도 걸음 수가 차이를 만든다
ae_strip(AE, (0, 12, 50, 200, 1875), "figs/sgd_ae_steps.png")
