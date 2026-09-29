"""Common matplotlib setup for ml2-week06 figures (KSA colors, Korean font)."""
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_MAIN = '#2E3192'
KSA_SUB = '#D6CBB1'
RED = '#C0392B'
GRAY = '#888888'
