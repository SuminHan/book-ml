"""GLM 덱 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (assert 로 확인한 값) 와 같은 함수로 계산한다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import data as D

plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d5c"
O = "figs/"
os.makedirs(O, exist_ok=True)
IRIS_NAMES = ["setosa", "versicolor", "virginica"]
IRIS_COL = [GOLD, GREEN, BLUE]


def types_fig(study, chess, bike, iris, seed=0):
    """맞힐 값 y 의 종류 네 가지: 실수 · 0/1 · 개수 · 여러 클래스."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    x, y = study; ax[0].scatter(x, y, s=34, color=GRAY)
    ax[0].set_xlabel("공부 시간"); ax[0].set_title("점수: 아무 실수", fontsize=16, fontweight="bold")
    x, y = chess; i = rng.choice(len(x), 300, replace=False)
    ax[1].scatter(x[i], y[i] + rng.uniform(-0.06, 0.06, 300), s=18, c=np.where(y[i] == 1, BLUE, RED), alpha=0.35, edgecolors="none")
    ax[1].set_yticks([0, 1], ["흑 승 0", "백 승 1"]); ax[1].set_xlabel("레이팅 차이"); ax[1].set_title("체스 승패: 0 / 1", fontsize=16, fontweight="bold")
    t, c = bike; ax[2].scatter(t, c, s=16, color=GREEN, alpha=0.5, edgecolors="none")
    ax[2].set_xlabel("기온 (°C)"); ax[2].set_title("따릉이 대여: 0, 1, 2, …", fontsize=16, fontweight="bold"); ax[2].set_ylim(0, None)
    x, y = iris
    for k in range(3):
        m = y == k; ax[3].scatter(x[m], k + rng.uniform(-0.12, 0.12, m.sum()), s=18, color=IRIS_COL[k], alpha=0.6, edgecolors="none")
    ax[3].set_yticks([0, 1, 2], IRIS_NAMES); ax[3].set_xlabel("꽃잎 길이 (cm)"); ax[3].set_title("붓꽃 종: 셋 중 하나", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(O + "glm_types.png"); plt.close(fig)


def bike_axes(ax, t, c):
    ax.scatter(t, c, s=20, color=GREEN, alpha=0.45, edgecolors="none", label=f"아침 6시 대여 수 ({len(t)}일)")
    ax.set_xlabel("기온 (°C)"); ax.set_ylabel("대여 수"); ax.set_xlim(-20, 33)


def bike_line_fig(t, c, k):
    """개수에 직선: 추우면 음수."""
    ts = np.linspace(-20, 33, 200); z0 = -k[1] / k[0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axhspan(-150, 0, color=RED, alpha=0.08)
    bike_axes(ax, t, c); ax.plot(ts, np.polyval(k, ts), color=GOLD, lw=3, label="직선")
    ax.annotate(f"{z0:.1f}°C 아래면\n대여 수 < 0 ?", (-17, np.polyval(k, -17)), (-12.5, -120), fontsize=15, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_ylim(-150, 850); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_bike_line.png"); plt.close(fig)


def bike_pois_fig(t, c, k, wp):
    """포아송 회귀 e^z: 언제나 양수, 1도마다 같은 배수."""
    ts = np.linspace(-20, 33, 200); pm = lambda v: np.exp(wp[0] + wp[1] * v)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bike_axes(ax, t, c)
    ax.plot(ts, np.polyval(k, ts), color=GOLD, lw=2, ls="--", label="직선")
    ax.plot(ts, pm(ts), color=BLUE, lw=3.5, label="포아송 회귀 $e^{z}$")
    for v in (0, 20):
        ax.plot(v, pm(v), "o", color=BLUE, ms=9); ax.annotate(f"{v}°C: {pm(v):.0f}대", (v, pm(v)), (-75, 22), textcoords="offset points", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(14, 40, f"1°C 오를 때마다 ×{np.exp(wp[1]):.3f}", fontsize=15, color=BLUE, fontweight="bold")
    ax.set_ylim(-150, 850); ax.legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_bike_pois.png"); plt.close(fig)


def softmax_fig(x, y, W, v=4.9, seed=1):
    """꽃잎 길이 하나로 세 종의 확률 (소프트맥스 회귀)."""
    rng = np.random.default_rng(seed); xs = np.linspace(1, 7, 400); P = D.softmax_p(W, xs); pv = D.softmax_p(W, v)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k in range(3):
        ax.plot(xs, P[:, k], color=IRIS_COL[k], lw=3.5, label=IRIS_NAMES[k])
        m = y == k; ax.scatter(x[m], -0.07 - 0.04 * k + rng.uniform(-0.012, 0.012, m.sum()), s=14, color=IRIS_COL[k], alpha=0.6, edgecolors="none")
    ax.axvline(v, color=GRAY, ls=":")
    ax.annotate(f"{v} cm\nversicolor {pv[1]:.0%}\nvirginica {pv[2]:.0%}", (v, 0.5), (5.25, 0.08), fontsize=14, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("확률"); ax.set_ylim(-0.17, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=12)
    fig.tight_layout(); fig.savefig(O + "glm_softmax.png"); plt.close(fig)


if __name__ == "__main__":
    st, ch, (t, c), ir = D.study(), D.chess(), D.bike(), D.iris()
    types_fig(st, ch, (t, c), ir)
    k = np.polyfit(t, c, 1); bike_line_fig(t, c, k)
    wp = D.fit_glm(t, c, np.exp, lr=0.002, steps=20000, scale=10, shift=10, w0=np.log(c.mean())); bike_pois_fig(t, c, k, wp)
    W = D.fit_softmax(*ir); softmax_fig(*ir, W)
    print("figs ok", wp.round(4))
