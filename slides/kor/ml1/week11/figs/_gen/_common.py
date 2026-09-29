"""공통 스타일 설정: KSA 색, 한글 폰트, 흰 배경, dpi 200."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 13

KSA_MAIN = '#2E3192'
KSA_SUB = '#D6CBB1'
KSA_RED = '#C0392B'
KSA_GRAY = '#7f8c8d'
DPI = 200
