"""p18g1: 파라미터 공유 - RNN은 시퀀스 길이와 무관, MLP 윈도우는 비례해서 증가."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np

V, H = 26, 8
window = np.arange(1, 21)
mlp_params = (window * V) * H + H  # 첫 은닉층만 계산 (교재 예시와 동일한 단순화)
rnn_params = np.full_like(window, H*V + H*H + V*H + H + V, dtype=float)  # 514

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(window, mlp_params, color=KSA_RED, lw=2.4, marker='o', markersize=4, label='MLP (윈도우 펼침)')
ax.plot(window, rnn_params, color=KSA_MAIN, lw=2.6, label='RNN (파라미터 공유) = 514 (일정)')

ax.scatter([5], [5*V*H+H], color=KSA_RED, zorder=5, s=55)
ax.annotate('윈도우=5\n1,048개', xy=(5, 5*V*H+H), xytext=(2.2, 2200),
            fontsize=10.5, color=KSA_RED,
            arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.2))
ax.annotate('514개 (V=26, H=8)\n— 문장이 몇 단어든 동일', xy=(18, 514), xytext=(11.5, 1050),
            fontsize=10.5, color=KSA_MAIN,
            arrowprops=dict(arrowstyle='->', color=KSA_MAIN, lw=1.2))

ax.set_xlabel('필요한 문맥 길이 (단어 수)', fontsize=12)
ax.set_ylabel('파라미터 개수', fontsize=12)
ax.set_title('파라미터 공유: RNN은 시퀀스 길이에 무관, MLP는 비례 증가', fontsize=13, color=KSA_MAIN)
ax.legend(loc='upper left', fontsize=11, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0.5, 20.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_1_param_sharing.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
