import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

years = ['2010\nNEC-UIUC', '2011\nXRCE', '2012\nAlexNet', '2013\nClarifai', '2014\nGoogLeNet', '2015\nResNet']
err = [28.2, 25.8, 16.4, 11.7, 6.7, 3.6]
colors = [KSA_GRAY]*2 + [KSA_RED] + [KSA_SUB]*2 + [KSA_MAIN]

fig, ax = plt.subplots(figsize=(9.2, 4.6))
x = np.arange(len(years))
ax.bar(x, err, color=colors, edgecolor='black', linewidth=0.8, zorder=3)
for xi, v in zip(x, err):
    ax.text(xi, v + 0.5, f'{v}%', ha='center', va='bottom', fontsize=13)

ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=12)
ax.set_ylabel('ImageNet top-5 오차율 (%)', fontsize=13)
ax.set_title('AlexNet(2012): 손으로 만든 특징 시대의 종료', fontsize=14, pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_ylim(0, 32)

ax.annotate('약 10%p 하락', xy=(2, 16.4), xytext=(2.9, 22),
            fontsize=13, color=KSA_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.6))

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_1_imagenet_error_history.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
