"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py. ① 더티 데이터 그림 (dd_*): 반 10명 키 예시 + scikit-learn 내장 iris.
② 정규화 그림 (fs_*): assets/scale.json · assets/modal.json, 소리 · 비디오 원본은 lab/data/ (exp_modal.py 가 쓰는 파일)."""
import os, sys, shutil, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"

# ======== ① 더티 데이터


def bars(ax, rows, mean_v, colors=None, ylim=(0, 270), mean_label=None):
    names = [n for n, _ in rows]; vals = [v for _, v in rows]
    cs = colors or [BLUE] * len(rows)
    ax.bar(range(len(rows)), vals, color=cs, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v}", ha="center", fontsize=13, color=cs[i] if cs[i] != BLUE else "black")
    ax.set_xticks(range(len(rows)), names, fontsize=13); ax.set_ylim(*ylim); ax.set_ylabel("키 (cm)")
    ax.axhline(mean_v, color=RED, ls="--", lw=2.5)
    ax.set_xlim(-0.6, len(rows) + 1.3)
    ax.text(len(rows) + 1.25, mean_v, mean_label or f"평균\n{mean_v:g}cm", ha="right", va="center", color=RED, fontsize=16,
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))


# 1. 질문: 평균 131.2
fig, ax = plt.subplots(figsize=(10, 4.4))
bars(ax, CLASS, round(MEAN_DIRTY, 1), [GOLD if v == 0 else BLUE for _, v in CLASS], ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_hook.png"); plt.close(fig)

# 2. 결측치 0 을 빼면 164
fig, ax = plt.subplots(figsize=(10, 4.4))
bars(ax, REAL, MEAN_CLEAN, ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_clean.png"); plt.close(fig)

# 3. 중복: 민수 ×3 → 166
rows = REAL + [(DUP, DUP_V)] * 2
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bars(ax, rows, MEAN_DUP, [RED if n == DUP else BLUE for n, _ in rows], ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_dup.png"); plt.close(fig)

# 4. 이상치: 오타 250 vs 농구 선수 205
rows = REAL + [(TYPO, TYPO_V), (ATH, ATH_V)]
cs = [RED if n == TYPO else (GOLD if n == ATH else BLUE) for n, _ in rows]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bars(ax, rows, round(mean([v for _, v in rows]), 1), cs, ylim=(0, 290))
ax.text(len(rows) - 2, TYPO_V + 18, "오타?", ha="center", color=RED, fontsize=15, fontweight="bold")
ax.text(len(rows) - 1, ATH_V + 18, "농구 선수", ha="center", color=GOLD, fontsize=15, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "dd_outlier.png"); plt.close(fig)

# 5. 그림으로 찾기: 상자 그림 (세로) — 왼쪽: 읽는 법, 오른쪽: 우리 반 키
vals = np.array([v for _, v in REAL] + [TYPO_V, ATH_V])
q1, med, q3 = np.percentile(vals, [25, 50, 75]); iqr = q3 - q1; fence = q3 + 1.5 * iqr
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
# 읽는 법 (모양만 보여 주는 그림)
from matplotlib.patches import Rectangle
a1.add_patch(Rectangle((0.3, 3), 0.8, 3, fc="#eef0fa", ec="black", lw=1.6))
a1.plot([0.3, 1.1], [4.3, 4.3], color=RED, lw=3)
a1.plot([0.7, 0.7], [6, 7.6], color="black", lw=1.4); a1.plot([0.55, 0.85], [7.6, 7.6], color="black", lw=1.4)
a1.plot([0.7, 0.7], [3, 1.6], color="black", lw=1.4); a1.plot([0.55, 0.85], [1.6, 1.6], color="black", lw=1.4)
a1.scatter([0.7], [9.3], s=110, color=RED)
for y, t, c in ((9.3, "이상치 후보: 수염보다 바깥", RED), (7.6, "수염 끝: 상자에서 상자 길이 × 1.5 안의 가장 먼 값", GRAY),
                (6, "75% (Q3): 이보다 큰 값이 1/4", "black"), (4.3, "중앙값 (50%): 한가운데", RED),
                (3, "25% (Q1): 이보다 작은 값이 1/4", "black"), (1.6, "수염 끝", GRAY)):
    a1.text(1.3, y, t, va="center", fontsize=13.5, color=c)
a1.annotate("", (0.2, 6), (0.2, 3), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
a1.text(0.12, 4.5, "상자 = 가운데 절반", rotation=90, va="center", ha="right", fontsize=13, color=BLUE)
a1.set_xlim(-0.3, 6.2); a1.set_ylim(0.8, 10.2); a1.axis("off"); a1.set_title("상자 그림 읽는 법", fontsize=16, fontweight="bold")
# 우리 반 키
a2.boxplot(vals, positions=[0], widths=0.45, medianprops=dict(color=RED, lw=2.5),
           flierprops=dict(marker="o", markerfacecolor=RED, markersize=11, markeredgecolor=RED))
a2.axhline(fence, color=GRAY, ls=":", lw=1.5); a2.text(0.32, fence + 2, f"Q3 + 1.5 × 상자 길이 = {fence:g}", color=GRAY, fontsize=12)
for v in (TYPO_V, ATH_V):
    a2.text(0.32, v, f"{v} cm", color=RED, fontsize=14, va="center", fontweight="bold")
a2.text(0.32, med, f"중앙값 {med:g}", color=RED, fontsize=12.5, va="center")
a2.set_xlim(-0.6, 1.6); a2.set_xticks([]); a2.set_ylabel("키 (cm)"); a2.spines["bottom"].set_visible(False)
a2.set_title("우리 반 키 (10명)", fontsize=16, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "dd_box.png"); plt.close(fig)

# 6. 모델도 속는다: iris 꽃잎 길이 → 너비, '비었음'을 0으로 적은 꽃 6송이
from sklearn.datasets import load_iris
X = load_iris().data; pl, pw = X[:, 2], X[:, 3]
rng = np.random.default_rng(1)
bad = rng.choice(np.where(pl > 4.5)[0], 6, replace=False)
pw_dirty = pw.copy(); pw_dirty[bad] = 0
xs = np.linspace(0.8, 7.2, 50)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.scatter(pl, pw_dirty, s=22, color=BLUE, alpha=0.6)
ax.scatter(pl[bad], pw_dirty[bad], s=90, color=RED, zorder=3, label="'비었음'을 0으로 적은 6송이")
for y_, c_, lab in ((pw, GREEN, "깨끗한 데이터의 직선"), (pw_dirty, RED, "0 이 섞인 데이터의 직선")):
    k = np.polyfit(pl, y_, 1); ax.plot(xs, np.polyval(k, xs), color=c_, lw=2.8, label=f"{lab} (기울기 {k[0]:.2f})")
ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("꽃잎 너비 (cm)"); ax.legend(frameon=False, fontsize=12.5, loc="upper left")
fig.tight_layout(); fig.savefig(O + "dd_iris.png"); plt.close(fig)
print("slopes", [round(np.polyfit(pl, y_, 1)[0], 3) for y_ in (pw, pw_dirty)])


# ======== ② 정규화와 표준화
shutil.copy("../../../../../lessons/ml1/feature_scaling/src/assets/neuron.png", O + "fs_neuron.png")


def acc_bars(ax, labels, vals, colors, title=None):
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v}%", ha="center", fontsize=20, fontweight="bold", color=colors[i])
    ax.set_xticks(range(len(vals)), labels, fontsize=15); ax.set_ylim(0, 112); ax.set_yticks([0, 50, 100])
    ax.set_ylabel("시험 정확도 (%)")
    if title: ax.set_title(title, fontsize=15)


# 1. 질문: 와인 그대로 72% vs 98%
fig, ax = plt.subplots(figsize=(8, 4.2))
acc_bars(ax, ["그대로", "Min-Max", "Z-score"], [ACC["raw"], ACC["minmax"], ACC["z"]], [RED, BLUE, BLUE])
fig.tight_layout(); fig.savefig(O + "fs_hook.png"); plt.close(fig)

# 2. 성분 13가지의 범위 (로그 축)
lo = np.array([J["ranges"][n][0] for n in NAMES]); hi = np.array([J["ranges"][n][1] for n in NAMES])
order = np.argsort(hi)
short = {"od280/od315_of_diluted_wines": "od280/od315", "alcalinity_of_ash": "alcalinity of ash",
         "nonflavanoid_phenols": "nonflavanoid phenols", "color_intensity": "color intensity",
         "total_phenols": "total phenols", "malic_acid": "malic acid"}
fig, ax = plt.subplots(figsize=(10, 5.2))
for r, i in enumerate(order):
    c = RED if NAMES[i] == "proline" else (GOLD if NAMES[i] == "hue" else BLUE)
    ax.plot([lo[i], hi[i]], [r, r], color=c, lw=7, solid_capstyle="round")
    ax.text(hi[i] * 1.12, r, f"{lo[i]:g} ~ {hi[i]:,g}", va="center", fontsize=11.5, color=c)
ax.set_yticks(range(13), [short.get(NAMES[i], NAMES[i]) for i in order], fontsize=12)
ax.set_xscale("log"); ax.set_xlim(0.1, 8000); ax.set_xlabel("성분 값 (로그 눈금: 한 칸 = 10배)")
fig.tight_layout(); fig.savefig(O + "fs_ranges.png"); plt.close(fig)


# 3·4. tanh 곡선 위의 와인 12병: 그대로 vs 표준화
def tanh_fig(xs, xlim, name, note):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    t = np.linspace(*xlim, 600); ax.plot(t, np.tanh(t), color=GRAY, lw=2.5)
    ax.scatter(xs, np.tanh(xs), s=110, color=RED if xlim[1] > 10 else BLUE, zorder=3)
    ax.axhline(0, color="#ccc", lw=1); ax.set_ylim(-1.25, 1.25); ax.set_xlim(*xlim)
    ax.set_xlabel("w·x + b (뉴런에 들어가는 합)"); ax.set_ylabel("출력 tanh")
    ax.text(0.97, 0.12, note, transform=ax.transAxes, fontsize=14, va="bottom", ha="right", bbox=dict(fc="white", ec="none"))
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


tanh_fig(MANY_RAW, (-20, 215), "fs_tanh_raw.png", f"와인 12병의 합: {MANY_RAW.min():.0f} ~ {MANY_RAW.max():.0f}\n출력은 모두 1.00, 기울기 0")
tanh_fig(MANY_Z, (-3, 3), "fs_tanh_z.png", f"표준화한 12병의 합: {MANY_Z.min():.2f} ~ {MANY_Z.max():.2f}\n곡선의 가파른 곳에 고루 퍼진다")

# 5. 프롤린 178병: 그대로 / Min-Max / Z-score (모양은 같고 눈금만 바뀐다)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, v, t, c in ((axs[0], PRO, "그대로 (mg/L)", RED), (axs[1], (PRO - PRO.min()) / (PRO.max() - PRO.min()), "Min-Max", BLUE),
                    (axs[2], (PRO - PRO.mean()) / PRO.std(), "Z-score", GREEN)):
    ax.hist(v, bins=20, color=c, alpha=0.85); ax.set_title(t, fontsize=15); ax.set_yticks([])
    ax.axvline(v[0], color="black", lw=2, ls="--")
    ax.text(v[0], ax.get_ylim()[1] * 0.62, f" 1번 와인\n {v[0]:.2f}" if t != "그대로 (mg/L)" else f" 1번 와인\n {v[0]:,.0f}", fontsize=12)
axs[1].set_xticks([0, 0.5, 1]); axs[2].set_xticks([-2, -1, 0, 1, 2, 3])
fig.tight_layout(); fig.savefig(O + "fs_hist3.png"); plt.close(fig)

# 6. 학습 손실 곡선
fig, ax = plt.subplots(figsize=(8.5, 4.2))
steps = np.arange(len(CURVE["raw"])) * 10
for k, c, lab in (("raw", RED, f"그대로 ({ACC['raw']}%)"), ("minmax", BLUE, f"Min-Max ({ACC['minmax']}%)"), ("z", GREEN, f"Z-score ({ACC['z']}%)")):
    ax.plot(steps, CURVE[k], color=c, lw=3, label=lab)
ax.set_xlabel("학습 횟수"); ax.set_ylabel("학습 손실 (교차 엔트로피)"); ax.legend(frameon=False, title="괄호 = 시험 정확도")
fig.tight_layout(); fig.savefig(O + "fs_curve.png"); plt.close(fig)

# 7. 흑백 이미지: MNIST 숫자 하나 (픽셀 값) + 정확도
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
a1.imshow(DIGIT, cmap="gray", vmin=0, vmax=255)
a1.set_xticks([]); a1.set_yticks([]); a1.set_title("28 × 28 픽셀, 값 0 ~ 255 (uint8)", fontsize=14)
acc_bars(a2, ["그대로\n0~255", "÷ 255\n0~1", "Z-score\n(0.1307, 0.3081)"],
         [MN_ACC["raw"], MN_ACC["div255"], MN_ACC["z"]], [RED, BLUE, GREEN], title="MNIST, 같은 신경망 · 같은 학습")
fig.tight_layout(); fig.savefig(O + "fs_mnist.png"); plt.close(fig)

# 8. 컬러 이미지: CIFAR-10 한 장 + R · G · B 채널 + 채널별 평균
fig, axs = plt.subplots(1, 5, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
axs[0].imshow(CF_ONE); axs[0].set_title("32 × 32 × 3", fontsize=13)
for k, (nm, cm) in enumerate((("R", "Reds_r"), ("G", "Greens_r"), ("B", "Blues_r"))):
    axs[k + 1].imshow(CF_ONE[..., k], cmap="gray", vmin=0, vmax=255); axs[k + 1].set_title(f"{nm} 채널", fontsize=13)
for a in axs[:4]: a.set_xticks([]); a.set_yticks([])
cs = ["#d62728", "#2ca02c", "#1f77b4"]
axs[4].bar(range(3), CF_MEAN, yerr=CF_STD, color=cs, capsize=6, width=0.6)
for i, (m_, s_) in enumerate(zip(CF_MEAN, CF_STD)):
    axs[4].text(i, 0.08, f"{m_:.3f}\n±{s_:.3f}", ha="center", color="white", fontsize=11, fontweight="bold")
axs[4].set_xticks(range(3), ["R", "G", "B"]); axs[4].set_ylim(0, 0.8); axs[4].set_title("1만 장의 채널별 평균 ± 표준편차", fontsize=12)
fig.tight_layout(); fig.savefig(O + "fs_cifar.png"); plt.close(fig)

# 9. 시계열: 항공 승객
fig, ax = plt.subplots(figsize=(11, 4.6))
months = 1949 + np.arange(len(AIR_P)) / 12; te_m = months[AIR_NTR:]
ax.axhspan(AIR["lo"], AIR["hi"], color=BLUE, alpha=0.08)
ax.text(1955.6, AIR["lo"] + 12, f"학습 기간의 범위 {AIR['lo']:.0f} ~ {AIR['hi']:.0f}", color=BLUE, fontsize=13)
ax.plot(months[:AIR_NTR + 1], AIR_P[:AIR_NTR + 1], color=BLUE, lw=2.2, label="학습 (1949~1958)")
ax.plot(te_m, AIR_P[AIR_NTR:], color="black", lw=2.2, label="실제 (1959~1960)")
ax.plot(te_m, AIR["pred_global"], color=RED, lw=2.4, ls="--", label=f"전체 기간 하나로 표준화: 오차 {AIR_MAE['global']}천 명")
ax.plot(te_m, AIR["pred_window"], color=GREEN, lw=2.4, ls="--", label=f"12달 창마다 표준화: 오차 {AIR_MAE['window']}천 명")
ax.axvline(1959, color=GRAY, ls=":"); ax.set_xlabel("연도"); ax.set_ylabel("국제선 승객 (천 명/달)")
ax.legend(frameon=False, fontsize=12.5, loc="upper left")
fig.tight_layout(); fig.savefig(O + "fs_air.png"); plt.close(fig)

# 10. 소리: 같은 '0' 두 사람, 그대로 (÷ 32768) vs 피크 정규화
jk = np.array(SND["jackson"]) / 32768; th = np.array(SND["theo"]) / 32768
fig, axs = plt.subplots(2, 2, figsize=(12, 4.6), sharey=True)
for col, (w, nm) in enumerate(((jk, "jackson"), (th, "theo"))):
    t = np.arange(len(w)) / SND["sr"]
    axs[0, col].plot(t, w, color=BLUE, lw=0.6); axs[0, col].set_title(f"{nm}: ÷ 32768, 가장 큰 값 {PEAK[nm]:.2f}", fontsize=13)
    axs[1, col].plot(t, w / np.abs(w).max(), color=GREEN, lw=0.6); axs[1, col].set_title(f"{nm}: 가장 큰 값으로 나누기 → 1.00", fontsize=13)
for a in axs.ravel(): a.set_ylim(-1.05, 1.05); a.set_yticks([-1, 0, 1])
for a in axs[1]: a.set_xlabel("시간 (초)")
fig.tight_layout(); fig.savefig(O + "fs_sound.png"); plt.close(fig)

# 11. 소리: 스펙트로그램 그대로 vs dB
from scipy.signal import spectrogram
f, t, S = spectrogram(jk, fs=SND["sr"], nperseg=256, noverlap=192)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.0))
a1.pcolormesh(t, f, S, cmap="magma", shading="auto"); a1.set_title(f"에너지 그대로: {S.min():.0e} ~ {S.max():.0e}", fontsize=13)
db = 10 * np.log10(S + 1e-10)
a2.pcolormesh(t, f, db, cmap="magma", shading="auto", vmin=db.max() - 80); a2.set_title("10·log10 (dB): 큰 값과 작은 값이 같이 보인다", fontsize=13)
for a in (a1, a2): a.set_xlabel("시간 (초)"); a.set_ylabel("주파수 (Hz)")
fig.tight_layout(); fig.savefig(O + "fs_spec.png"); plt.close(fig)
print("spec range", S.min(), S.max())

# 12. 비디오: 말 프레임 5장 — 원본 / 점점 어두워지는 영상 / 프레임마다 표준화 / 영상 전체로 표준화
from PIL import Image, ImageSequence
v = np.stack([np.array(fr.convert("RGB")) for fr in ImageSequence.Iterator(Image.open("lab/data/horse.gif"))]) / 255
idx = [0, 3, 6, 9, 12]; fade = np.linspace(1, 0.25, len(v))[:, None, None, None]
dark = v * fade
per_frame = (dark - dark.mean((1, 2, 3), keepdims=True)) / dark.std((1, 2, 3), keepdims=True)
per_clip = (dark - dark.mean()) / dark.std()
rows = [(dark, "점점 어두워지는 영상", (0, 1)), (per_frame, "프레임마다 표준화", (-2.5, 2)), (per_clip, "영상 전체로 표준화", (-2.5, 2))]
fig, axs = plt.subplots(3, 5, figsize=(12, 5.4))
for r, (arr, title, (a, b)) in enumerate(rows):
    for c, i in enumerate(idx):
        axs[r, c].imshow(arr[i, ..., 0], cmap="gray", vmin=a, vmax=b); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        if r == 0: axs[r, c].set_title(f"프레임 {i}", fontsize=12)
    axs[r, 0].set_ylabel(title, fontsize=12, rotation=0, ha="right", va="center", labelpad=10)
fig.tight_layout(); fig.savefig(O + "fs_video.png"); plt.close(fig)
print("figs done")
