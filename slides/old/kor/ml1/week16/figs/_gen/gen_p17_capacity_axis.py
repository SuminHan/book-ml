import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

# Left: GBDT, x = n_estimators (capacity), real numbers from p17 table
ax = axes[0]
n_est = [20, 50, 200]
train = [0.988, 1.000, 1.000]
val = [0.944, 0.961, 0.983]
ax.plot(n_est, train, 'o-', color=KSA_MAIN, linewidth=2.2, markersize=8, label='train')
ax.plot(n_est, val, 's-', color=KSA_RED, linewidth=2.2, markersize=8, label='val')
ax.set_xlabel('n\\_estimators (트리 수 = 모델 크기)', fontsize=11.5)
ax.set_ylabel('정확도', fontsize=12)
ax.set_title('GBDT: x축 = 용량(트리 수)', fontsize=13)
ax.set_ylim(0.90, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

# Right: NN, x = epoch (fixed capacity, illustrative example curve)
ax = axes[1]
epoch = np.arange(1, 41)
train_nn = 1 - 0.75*np.exp(-epoch/6)
val_nn = 1 - 0.75*np.exp(-epoch/6) - np.where(epoch > 20, (epoch-20)*0.0012, 0)
ax.plot(epoch, train_nn, '-', color=KSA_MAIN, linewidth=2.2, label='train')
ax.plot(epoch, val_nn, '-', color=KSA_RED, linewidth=2.2, label='val')
ax.axvline(20, color='gray', ls=':', lw=1.2)
ax.text(20.5, 0.35, 'val 포화\n(예시)', fontsize=10.5, color='gray')
ax.set_xlabel('epoch (학습량, 구조는 고정)', fontsize=11.5)
ax.set_ylabel('정확도', fontsize=12)
ax.set_title('신경망: x축 = 학습량(에포크)', fontsize=13)
ax.set_ylim(0.2, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

fig.suptitle('같은 "train/val 곡선"인데 x축의 의미가 다르다', fontsize=13.5, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_1_capacity_vs_epoch_axis.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
