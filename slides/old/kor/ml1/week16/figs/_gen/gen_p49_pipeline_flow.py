import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.6, 3.6))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 3.6)
ax.axis('off')

boxes = [
    (0.4, '모델을\n정의'),
    (3.1, '틀린 정도를\n손실함수로\n수치화'),
    (5.8, '손실을 줄이는\n방향으로\n파라미터 조정'),
    (8.5, '평가\n(val/test)'),
]
w, h, y0 = 2.0, 2.0, 0.9
for x, label in boxes:
    box = FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0.08,rounding_size=0.12',
                          linewidth=1.6, edgecolor=KSA_MAIN, facecolor=KSA_LIGHT, zorder=3)
    ax.add_patch(box)
    ax.text(x + w/2, y0 + h/2, label, ha='center', va='center', fontsize=12.5, zorder=4)

for i in range(3):
    x0 = boxes[i][0] + w
    x1 = boxes[i+1][0]
    arr = FancyArrowPatch((x0, y0 + h/2), (x1, y0 + h/2), arrowstyle='-|>',
                           mutation_scale=22, linewidth=1.8, color=KSA_MAIN, zorder=2)
    ax.add_patch(arr)

# feedback loop: 평가 -> 모델 정의 (과적합/누수면 다시)
arr2 = FancyArrowPatch((9.5, y0), (1.4, 0.15), connectionstyle='arc3,rad=-0.25',
                        arrowstyle='-|>', mutation_scale=20, linewidth=1.6,
                        color=KSA_RED, linestyle='--', zorder=2)
ax.add_patch(arr2)
ax.text(5.3, -0.15, '과적합·누수 발견 시 되돌아감 (붉은 점선)', fontsize=11, color=KSA_RED, ha='center')

ax.set_title('거의 모든 장에서 반복되는 4단계 구조', fontsize=14.5, pad=6)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_3_pipeline_4stage.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
