"""13.3 그림 생성 (단 한 번 실행): (1) Reacher 과제 상시도 (2) log밀도/σ 클램프.
숫자 검증용이 아니라 정적 다이어그램.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
if kr: plt.rcParams["font.sans-serif"] = [kr[0]]
plt.rcParams["axes.unicode_minus"] = False
IMG = "/home/smhan/book-ml/kor/src/images"

# ---------- 그림 1: Reacher-v5 상시도 ----------
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ax.add_patch(plt.Rectangle((-0.3, -0.3), 0.6, 0.6, facecolor="#f4f4f4",
                            edgecolor="#bbbbbb", lw=1.2, zorder=1))
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(0.2 * np.cos(th), 0.2 * np.sin(th), color="#d62728", ls="--", lw=1.6,
        zorder=2, label="목표 범위 (반경 0.2 원)")
t1, t2 = np.deg2rad(70), np.deg2rad(-40)
elbow = 0.1 * np.array([np.cos(t1), np.sin(t1)])
tip = elbow + 0.11 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
target = np.array([0.16, 0.08])
ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], color="#1f5fa8",
        lw=7, solid_capstyle="round", zorder=3)
ax.plot(0, 0, "o", ms=11, color="#333333", zorder=5)
ax.plot(elbow[0], elbow[1], "o", ms=9, color="#1f5fa8", mec="k", mew=0.8, zorder=5)
ax.plot(tip[0], tip[1], "o", ms=10, color="#2ca02c", mec="k", mew=0.9, zorder=5)
ax.plot(target[0], target[1], "*", ms=17, color="#d62728", mec="k", mew=0.6, zorder=6)
ax.annotate("", xy=tip, xytext=target,
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.8), zorder=4)
ax.text(-0.285, 0.135, "어깨 (고정)", fontsize=10, ha="left")
ax.text(elbow[0] - 0.015, elbow[1] + 0.045, "팔꿈치\n(0.1 m)", fontsize=9.5, ha="right")
ax.text(tip[0] + 0.012, tip[1] + 0.02, "fingertip", fontsize=10)
ax.text(target[0] - 0.005, target[1] - 0.075, "목표 (매 에피소드 무작위)",
        fontsize=10, ha="center", color="#a02020")
ax.text(0.005, 0.235, "오차 벡터 = fingertip − target  (관찰 obs[8:10])",
        fontsize=10, ha="left", color="#444444")
ax.text(-0.29, -0.29, "팔 길이 0.1 + 0.11 = 쭉 펴면 0.21 m\n"
        "→ 목표(≤0.2)는 거의 모두 도달 가능 (13.1의 도달 가능 영역)",
        fontsize=9.5, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe8", ec="#d8c46a", lw=1))
ax.text(0.30, 0.275, "아레나 ±0.3", fontsize=8.5, ha="right", color="#888888")
ax.set_xlim(-0.34, 0.34)
ax.set_ylim(-0.34, 0.34)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Reacher-v5 (상시도) — 50스텝(1초) 안에 fingertip을 목표에 붙인다", fontsize=11)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
plt.tight_layout()
p1 = os.path.join(IMG, "ch13_3_reacher_task.svg")
plt.savefig(p1, bbox_inches="tight")
print("saved", p1, os.path.getsize(p1), "bytes")

# ---------- 그림 2: (a) log밀도와 σ (b) log_std 클램프 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4))
a = np.linspace(-3, 3, 600)
for sigma, ls in zip([1.0, 0.61, 0.25, 0.10], ["-", "--", "-.", ":"]):
    ld = -0.5 * np.log(2 * np.pi * sigma**2) - a**2 / (2 * sigma**2)
    ax1.plot(a, ld, ls, lw=1.7, label=f"σ = {sigma}")
ax1.axhline(0, color="#444444", lw=1.0, ls=":")
ax1.text(2.5, 0.35, "log 밀도 = 0\n(밀도 1)", fontsize=8.5, ha="right")
ax1.set_title("(a) 가우시안(μ=0)의 log 밀도 — σ가 작을수록 정점이 0 위(양수)",
              fontsize=10)
ax1.set_xlabel("행동 a")
ax1.set_ylabel("log 밀도")
ax1.legend(fontsize=8.5)
ax1.grid(alpha=0.3)
xs = np.linspace(-3, 0.7, 400)
ax2.plot(xs, np.exp(xs), color="#1f5fa8", lw=2.0)
ax2.axvline(0, color="#d62728", ls="--", lw=1.6)
ax2.fill_between(xs, 0, np.exp(xs), where=(xs > 0), color="#f4cccc", alpha=0.8)
ax2.annotate("클램프 상한: log_std ≤ 0\n(σ ≤ 1 — 이 절의 실습)",
             (0.02, 1.05), xytext=(0.22, 2.35), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#d62728"))
ax2.plot(-0.5, np.exp(-0.5), "o", color="#2ca02c", ms=7)
ax2.annotate("초기 σ = e^−0.5 ≈ 0.61", (-0.5, 0.61), xytext=(-2.7, 0.42),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="#2ca02c"))
ax2.set_title("(b) σ = exp(log_std) — 학습 가능한 단일 파라미터", fontsize=10)
ax2.set_xlabel("log_std")
ax2.set_ylabel("σ")
ax2.set_xlim(-3, 0.7)
ax2.set_ylim(0, 3.2)
ax2.grid(alpha=0.3)
plt.tight_layout()
p2 = os.path.join(IMG, "ch13_3_logprob_sigma.svg")
plt.savefig(p2, bbox_inches="tight")
print("saved", p2, os.path.getsize(p2), "bytes")
