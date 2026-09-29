"""공통 설정: 한글 폰트, KSA 색, 저장 헬퍼."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = "#2E3192"
KSA_TAN = "#D6CBB1"
ACCENT_RED = "#C0392B"
GREY = "#888888"

def save(fig, path, dpi=200):
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    print("saved:", path)
