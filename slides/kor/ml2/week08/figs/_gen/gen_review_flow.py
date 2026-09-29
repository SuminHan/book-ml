"""ch08_2_rl_review_flow.png -- 좋은 리뷰를 쓰는 과정 흐름도 (p37)"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'
GREEN = '#DCEAD8'
PINK = '#F5D5D0'

fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(1.3, 13.2)
ax.axis('off')
ax.set_aspect('equal')

def box(cx, cy, w, h, text, fc=TAN, ec=NAVY, tc='black', fs=13.5, lw=2.2):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle="round,pad=0.08,rounding_size=0.18",
                        fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=tc,
             zorder=3, linespacing=1.35)

def diamond(cx, cy, w, h, text, fc='white', ec=RED, fs=12.5):
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    poly = plt.Polygon(pts, closed=True, fc=fc, ec=ec, lw=2.4, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color='black',
             zorder=3, linespacing=1.3)

def varrow(x, y1, y2, color=NAVY, lw=2.2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))

def diag_arrow(x1, y1, x2, y2, text, tx, ty, color=NAVY, lw=2.0, fs=12):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw))
    ax.text(tx, ty, text, ha='center', va='center', fontsize=fs, color=color,
             fontweight='bold', zorder=4)

# Boxes (top to bottom)
box(5, 12.4, 8.8, 1.15, "발표 + 보고서로 주장 파악", fc=NAVY, tc='white', fs=14.5)
box(5, 10.7, 8.8, 1.3, "체크리스트 4항목 적용\n(환경·알고리즘·정당화·정직성)")
box(5, 9.0, 8.8, 1.1, "각 항목에 질문 초안 작성")
diamond(5, 6.7, 9.2, 2.4, "개념(장·절) 근거 +\n답이 새 계산·실험인가?")
box(2.0, 4.3, 3.7, 1.2, "채택 (3점 질문)", fc=GREEN, fs=13.5)
box(8.0, 4.3, 3.7, 1.2, "감상평으로 하락", fc=PINK, fs=13.5)
box(5, 2.1, 8.8, 1.1, "3단계(1/2/3점)로 자기 채점")

varrow(5, 11.82, 11.35)
varrow(5, 10.05, 9.55)
varrow(5, 8.45, 7.9)

diag_arrow(3.6, 6.15, 2.15, 4.9, "예", 3.15, 5.75, color=NAVY)
diag_arrow(6.4, 6.15, 7.85, 4.9, "아니오", 7.35, 5.75, color=NAVY)

diag_arrow(2.6, 3.7, 4.35, 2.65, "", 0, 0)
diag_arrow(7.4, 3.7, 5.65, 2.65, "", 0, 0)

# loop back: 감상평으로 하락 -> 다시 쓰기 -> 질문 초안 작성 box
ax.annotate('', xy=(9.55, 9.0), xytext=(9.55, 4.3),
            arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.0,
                             connectionstyle='arc3,rad=-0.4'))
ax.text(9.82, 6.65, "다시 쓰기", rotation=90, ha='center', va='center',
         fontsize=11.5, color=RED, fontweight='bold')

plt.tight_layout(pad=0.3)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_2_rl_review_flow.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
