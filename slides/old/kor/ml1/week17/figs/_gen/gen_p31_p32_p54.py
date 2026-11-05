import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = "#2E3192"
KSA_TAN = "#D6CBB1"
RED = "#C0392B"
GRAY = "#8a8a8a"
LIGHT = "#EDEBE3"

OUT = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week17/slides/kor/ml1/week17/figs"


def rbox(ax, x, y, w, h, text, fc, ec, tc="black", fs=13, bold=True, lw=1.8):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
    return r


def arrow(ax, xy_from, xy_to, color="black", lw=1.8, style="->"):
    a = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=16,
                         color=color, linewidth=lw)
    ax.add_patch(a)


# ---------------------------------------------------------------
# D6 (p31g1): three-tier pyramid -- pretrain -> SFT -> prompting, + RAG/agent branch
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)

# pyramid layers (bottom = widest = pretraining)
layers = [
    (0.5, 0.3, 8.5, 1.1, "사전학습 (Pretraining)\n인터넷 규모 텍스트 --- 폭넓은 능력을 압축", "#c9cbe8", KSA_BLUE),
    (1.3, 1.55, 6.9, 1.05, "SFT (지시 따르기 형식)\n사람이 쓴 (질문,답변)으로 다듬기", "#8890c9", KSA_BLUE),
    (2.3, 2.75, 4.9, 1.0, "프롬프팅 (입력 레벨)\n가중치 불변, 지금 이 요청만", KSA_BLUE, "#1b1e63"),
]
for (x, y, w, h, text, fc, ec) in layers:
    tc = "white" if fc == KSA_BLUE else "#1b1e63"
    rbox(ax, x, y, w, h, text, fc, ec, tc=tc, fs=12.5)

ax.text(5, 4.15, "학습 단계 (가중치를 고침)", fontsize=12, color=KSA_BLUE, fontweight="bold", ha="center")

# side branch: RAG / agent, inference time
rbox(ax, 7.6, 2.75, 2.0, 1.0, "RAG /\n에이전트\n(추론 시점)", LIGHT, RED, tc=RED, fs=11.5)
arrow(ax, (7.6, 3.25), (7.2, 3.25), color=RED)
ax.text(9.6, 3.9, "학습 없이,\n컨텍스트로 보완", fontsize=10.5, color=RED, ha="center")

ax.annotate("", xy=(9.6, 3.6), xytext=(9.6, 4.25),
            arrowprops=dict(arrowstyle="-", color=RED, lw=0))

ax.set_xlim(0, 10.2)
ax.set_ylim(0, 4.6)
ax.axis("off")
ax.set_title("프롬프트가 \"행동을 바꾼다\": 세 겹의 맨 위층일 뿐", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/pretrain_sft_prompt_layers.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# D7 (p32g1): prompt structure diagram -- system / few-shot / user -> LLM -> output
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=200)

blocks = [
    (1.0, 5.6, 6.0, 0.9, "System message\n\"너는 ... 전문가다. 항상 JSON으로 답한다.\"", KSA_BLUE, "white"),
    (1.0, 4.5, 6.0, 0.9, "Few-shot 예시 (선택)\n입력 $\\to$ 출력 쌍 1$\\sim$4개", "#8890c9", "white"),
    (1.0, 3.4, 6.0, 0.9, "User message\n이번 질문/요청 자체", KSA_TAN, "#1b1e63"),
]
for (x, y, w, h, text, fc, tc) in blocks:
    rbox(ax, x, y, w, h, text, fc, "#1b1e63", tc=tc, fs=12)

# bracket down to single sequence
ax.annotate("", xy=(4.0, 2.65), xytext=(4.0, 3.4),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.text(4.35, 3.0, "모두 하나의\n토큰 시퀀스로 이어짐", fontsize=10.5, ha="left", va="center")

rbox(ax, 1.6, 1.6, 4.8, 1.0, "LLM\n다음 토큰 예측 (조건부 확률)", "#EDEBE3", RED, tc="#1b1e63", fs=12.5)
ax.annotate("", xy=(4.0, 1.6), xytext=(4.0, 2.65),
            arrowprops=dict(arrowstyle="-", color="black", lw=0))

ax.annotate("", xy=(4.0, 0.8), xytext=(4.0, 1.6),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
rbox(ax, 2.3, 0.0, 3.4, 0.75, "출력", "white", "black", tc="black", fs=12)

ax.set_xlim(0, 8.2)
ax.set_ylim(-0.2, 6.7)
ax.axis("off")
ax.set_title("프롬프트의 구조: 결국 하나의 입력 시퀀스", fontsize=15, fontweight="bold", pad=6)
fig.tight_layout()
fig.savefig(f"{OUT}/prompt_structure.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# D8 (p54g1): RAG vs Agent flow diagram
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), dpi=200)

# RAG flow
ax = axes[0]
steps = ["질문", "관련 문서\n검색", "프롬프트에\n문서 삽입", "LLM", "답변"]
n = len(steps)
xs = [0.6 + i * 1.9 for i in range(n)]
for x, s in zip(xs, steps):
    fc = KSA_BLUE if s == "LLM" else ("#EDEBE3" if s in ("질문", "답변") else KSA_TAN)
    tc = "white" if s == "LLM" else "#1b1e63"
    rbox(ax, x, 0.15, 1.5, 0.7, s, fc, "#1b1e63", tc=tc, fs=10.5)
for i in range(n - 1):
    arrow(ax, (xs[i] + 1.5, 0.5), (xs[i+1], 0.5))
ax.set_xlim(0, xs[-1] + 2.0)
ax.set_ylim(-0.1, 1.1)
ax.axis("off")
ax.set_title("RAG: 검색해서 컨텍스트에 넣기", fontsize=13.5, fontweight="bold", loc="left")

# Agent flow (loop)
ax = axes[1]
steps2 = ["질문", "LLM 판단", "도구 호출\n(검색/계산/코드)", "결과를\n컨텍스트에 추가", "LLM", "답변"]
n2 = len(steps2)
xs2 = [0.5 + i * 1.6 for i in range(n2)]
for x, s in zip(xs2, steps2):
    fc = KSA_BLUE if s in ("LLM 판단", "LLM") else ("#EDEBE3" if s in ("질문", "답변") else RED)
    tc = "white" if fc in (KSA_BLUE, RED) else "#1b1e63"
    rbox(ax, x, 0.15, 1.35, 0.75, s, fc, "#1b1e63", tc=tc, fs=9.5)
for i in range(n2 - 1):
    arrow(ax, (xs2[i] + 1.35, 0.52), (xs2[i+1], 0.52))
# loop-back arrow from "결과를 컨텍스트에 추가" back to "LLM 판단" (agent may call tool again),
# arced well above the boxes so it never crosses their text
loop_y = 1.55
ax.plot([xs2[3] + 0.67, xs2[3] + 0.67], [0.9, loop_y], color=RED, lw=1.6)
ax.plot([xs2[3] + 0.67, xs2[1] + 0.67], [loop_y, loop_y], color=RED, lw=1.6)
ax.annotate("", xy=(xs2[1] + 0.67, 0.9), xytext=(xs2[1] + 0.67, loop_y),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax.text((xs2[1] + xs2[3]) / 2 + 0.67, loop_y + 0.18, "필요하면 도구를 다시 호출 (반복)",
        fontsize=10, color=RED, ha="center")

ax.set_xlim(0, xs2[-1] + 1.8)
ax.set_ylim(-0.1, 2.1)
ax.axis("off")
ax.set_title("에이전트: 스스로 도구를 호출하고 반복", fontsize=13.5, fontweight="bold", loc="left")

fig.suptitle("에이전트와 RAG: \"모델이 필요할 때 바깥에 접근\"", fontsize=15.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/rag_agent_flow.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

print("done: pretrain_sft_prompt_layers.png, prompt_structure.png, rag_agent_flow.png")
