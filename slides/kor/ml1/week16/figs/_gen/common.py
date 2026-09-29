import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.facecolor'] = 'white'

KSA_MAIN = '#2E3192'
KSA_SUB = '#D6CBB1'
KSA_RED = '#C0392B'
KSA_GRAY = '#8A8A8A'
KSA_LIGHT = '#EDEAE2'
