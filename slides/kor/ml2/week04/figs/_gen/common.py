"""공통 스타일 설정 - KSA ml2-week04 개념 도식용."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

KSA_BLUE = '#2E3192'
KSA_BLUE_DK = '#20226A'
KSA_GOLD = '#D6CBB1'
KSA_GOLD_DK = '#A9976B'
KSA_RED = '#C0392B'
KSA_GRAY = '#5B5B5B'

GAMMA = 0.9
N = 5  # states 0..4, state 4 = goal(terminal)


def vstar():
    """오른쪽 정책의 참값 (5칸 GridWorld, reward -1, gamma=0.9)."""
    V = [0.0] * N
    for s in range(N - 2, -1, -1):
        V[s] = -1 + GAMMA * V[s + 1]
    return V  # [-3.439, -2.710, -1.900, -1.000, 0.0]


def savefig(fig, path, dpi=200):
    fig.patch.set_facecolor('white')
    fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
