"""p46: 은닉 유닛(선형 경계) 개수가 늘수록 2D 입력 공간이 더 잘게 쪼개짐"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import save, KSA_BLUE, KSA_TAN, ACCENT_RED, GREY
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)

def make_units(n):
    # 각 유닛: w1*x + w2*y + b = 0 (원점 근처를 지나는 무작위 방향의 직선)
    angles = np.linspace(0, np.pi, n, endpoint=False) + rng.uniform(-0.2, 0.2, n)
    w1 = np.cos(angles)
    w2 = np.sin(angles)
    b = rng.uniform(-0.55, 0.55, n)
    return w1, w2, b

def region_count(gx, gy, w1, w2, b):
    # 각 픽셀에서 활성 패턴(부호 벡터)을 세어 서로 다른 조합 수 리턴(색칠용)
    code = np.zeros(gx.shape, dtype=np.int64)
    for i in range(len(w1)):
        s = (w1[i] * gx + w2[i] * gy + b[i]) > 0
        code = code * 2 + s.astype(np.int64)
    return code

fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
xs = np.linspace(-1.4, 1.4, 500)
ys = np.linspace(-1.4, 1.4, 500)
gx, gy = np.meshgrid(xs, ys)

for ax, n, title in zip(axes, [4, 12], ["은닉 유닛 4개 → 선형 경계 4개", "은닉 유닛 12개 → 선형 경계 12개"]):
    w1, w2, b = make_units(n)
    code = region_count(gx, gy, w1, w2, b)
    ax.imshow(code, extent=[-1.4, 1.4, -1.4, 1.4], origin="lower",
              cmap="Pastel1", alpha=0.9, aspect="equal")
    for i in range(n):
        if abs(w2[i]) > 1e-6:
            yy = (-w1[i] * xs - b[i]) / w2[i]
            ax.plot(xs, yy, color=KSA_BLUE, lw=1.6)
        else:
            xx0 = -b[i] / w1[i]
            ax.axvline(xx0, color=KSA_BLUE, lw=1.6)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12.5, color=KSA_BLUE)
    n_regions = len(np.unique(code))
    ax.text(0, -1.65, f"조각난 영역 수 ≈ {n_regions}", ha="center", fontsize=11, color=ACCENT_RED)

fig.suptitle("유닛(경계선)이 많아질수록 입력 공간이 기하급수적으로 잘게 쪼개진다",
             fontsize=13.5, color="black", y=1.02)
plt.tight_layout()
save(fig, os.path.join(os.path.dirname(__file__), "..", "relu_layer_partition.png"))
