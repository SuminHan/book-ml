import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
TAN = '#D6CBB1'
GREEN = '#1E8449'
RED = '#C0392B'

fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)

# center word box
def box(x, y, w, h, text, fc, ec='black', fontsize=13, textcolor='black'):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=fc, ec=ec, lw=1.5)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=textcolor)

# center word
box(1.2, 3.5, 1.8, 0.9, '중심 단어\n"고양이는"', TAN)

# true context (label 1)
box(5.0, 5.6, 2.0, 0.8, '진짜 문맥어\n"귀엽다"  (label=1)', '#C8E6C9', textcolor=GREEN, fontsize=12)
ax.annotate('', xy=(4.0, 5.6), xytext=(2.1, 3.75), arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2))

neg_words = ['"자동차"', '"정책"', '"금융"', '"날씨"']
ys = [4.55, 3.5, 2.45, 1.4]
for w, y in zip(neg_words, ys):
    box(5.0, y, 2.0, 0.7, f'가짜 단어\n{w}  (label=0)', '#F5CBA9', textcolor=RED, fontsize=12)
    ax.annotate('', xy=(4.0, y), xytext=(2.1, 3.4), arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.3, alpha=0.8))

# binary classifier box
box(8.7, 3.5, 2.6, 4.8, '이진 분류기\n$\\sigma(\\;v_c \\cdot v_w\\;)$\n\n1개는 정답(1)\n$k_{neg}$=4개는\n가짜(0)', 'white', ec=BLUE, fontsize=13, textcolor=BLUE)
for y in [5.6] + ys:
    ax.annotate('', xy=(7.4, 3.5), xytext=(6.0, y), arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.6))

ax.set_xlim(0, 10.2)
ax.set_ylim(0.6, 6.4)
ax.axis('off')
ax.set_title('네거티브 샘플링: $1+k_{neg}$개의 이진 분류로 softmax를 피한다', fontsize=15, color=BLUE, pad=14)

plt.tight_layout()
plt.savefig('/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week14/slides/kor/ml1/week14/figs/ch14_negsampling.png',
            bbox_inches='tight', facecolor='white')
print("done")
