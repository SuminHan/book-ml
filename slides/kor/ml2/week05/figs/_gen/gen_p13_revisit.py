"""p13g1: p13 표의 에피소드(상태 (0,0)를 세 번 방문, 미끄러짐 두 번)를
시간순 타임라인으로 표현. 그리드 위 화살표는 뒤엉키므로 가로 시퀀스로 대체.
"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
SAND = '#D6CBB1'
RED = '#C0392B'
GRAY = '#888888'

# 스텝: (t, 상태, 방문순번or None)
steps = [
    (0, "(0,0)", "1st"),
    (1, "(0,1)", None),
    (2, "(0,0)", "2nd"),
    (3, "(0,1)", None),
    (4, "(0,0)", "3rd"),
    (5, "(0,1)", None),
    (6, "(0,2)", None),
    (7, "(1,2)", None),
    (8, "터미널", None),
]

fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=200)
n = len(steps)
xs = list(range(n))
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(-1.3, 1.9)
ax.axis('off')

box_w, box_h = 0.82, 0.9
for x, (t, state, visit) in zip(xs, steps):
    is_term = state == "터미널"
    is_home = state == "(0,0)"
    face = SAND if is_term else (RED if is_home else 'white')
    edge = NAVY
    alpha = 0.35 if is_home else 1.0
    rect = mpatches.FancyBboxPatch((x - box_w / 2, -box_h / 2), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.8, edgecolor=edge,
                                    facecolor=face, alpha=1.0 if not is_home else 0.18)
    if is_home:
        rect.set_edgecolor(RED)
        rect.set_linewidth(2.4)
    ax.add_patch(rect)
    txtcolor = NAVY if not is_term else NAVY
    ax.text(x, 0.05, state, ha='center', va='center', fontsize=12.5,
            color=RED if is_home else NAVY, fontweight='bold' if (is_home or is_term) else 'normal')
    ax.text(x, -0.85, f"$t={t}$", ha='center', va='center', fontsize=10, color=GRAY)
    if visit is not None:
        ax.text(x, 0.85, visit, ha='center', va='bottom', fontsize=12,
                color=RED, fontweight='bold')
    if x < n - 1:
        ax.annotate("", xy=(x + 1 - box_w / 2 - 0.03, 0), xytext=(x + box_w / 2 + 0.03, 0),
                    arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.6))

ax.text((n - 1) / 2, 1.55,
        "같은 에피소드 안에서 (0,0)을 세 번 방문 (미끄러짐 두 번, 빨간 칸)",
        ha='center', va='bottom', fontsize=14, color=NAVY, fontweight='bold')
ax.text((n - 1) / 2, -1.25,
        "첫방문 MC: 오직 1st 방문의 리턴($-0.139$)만 카운트 — 2nd·3rd 방문(빨간 글자)은 버림",
        ha='center', va='top', fontsize=11.5, color=RED)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_1_revisit_path.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
