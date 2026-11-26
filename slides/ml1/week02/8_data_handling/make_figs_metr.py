"""더티 데이터 그림 (METR-LA). 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs_metr.py
숫자는 exp_metr.py 가 만든 assets/metr.json, 데이터는 lab/data/metr_test.parquet"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_metr import load_speed, PATH, fill_methods

os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"
SENSOR = 125                      # a sensor with short gaps, a long outage and rush-hour jams
WEEK = ("2012-06-11", "2012-06-17")
METHODS = ["sensor mean", "ffill", "interpolate", "time-of-day mean"]
LABEL = {"sensor mean": "센서 평균\nfillna(S.mean())", "ffill": "직전 값\nffill()", "interpolate": "선으로 잇기\ninterpolate()",
         "time-of-day mean": "같은 시각 평균\n(시각별 평균)"}


def week_plot(raw, sensor, week, name):
    """one sensor for a week: zeros (sensor down) in red"""
    s = raw.loc[week[0]:week[1], sensor]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(s.index, s.values, color=BLUE, lw=1.2)
    z = s[s == 0]
    ax.scatter(z.index, z.values, color=RED, s=10, zorder=3, label="속도 0 (센서 고장)")
    ax.set_ylabel("속도 (mph)"); ax.set_ylim(-3, 75); ax.legend(loc="lower left", frameon=False)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def missing_map(raw, name):
    """sensors x time: where the zeros are"""
    z = (raw == 0).values.T
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.imshow(z, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f3f3f8", RED]), interpolation="nearest")
    days = pd.date_range(raw.index[0].normalize() + pd.Timedelta("1D"), raw.index[-1], freq="7D")
    ax.set_xticks([raw.index.get_indexer([d], method="nearest")[0] for d in days]); ax.set_xticklabels([d.strftime("%m/%d") for d in days])
    ax.set_ylabel("센서 207개"); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def hist_plot(raw, name):
    """distribution of all speeds: spike at 0 (blank) and wall at 70 (cap)"""
    v = raw.values.ravel()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(v, bins=np.arange(0, 72, 1), color=BLUE, alpha=0.85)
    ax.hist(v[v == 0], bins=np.arange(0, 72, 1), color=RED)
    ax.set_xlabel("속도 (mph)"); ax.set_ylabel("개수"); ax.set_yscale("log"); ax.set_ylim(10, 2e6)
    ax.annotate("0: 12.1%\n(센서 고장)", (0.6, (v == 0).sum()), (8, (v == 0).sum() / 3), color=RED, fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("70 에서 끝\n(측정 상한)", (69.5, (v == 70).sum()), (40, 3e5), color=GOLD, fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOLD))
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def box_by_hour(raw, sensor, name):
    """one sensor, speed by hour of day (zeros removed): rush-hour points outside the whiskers are real jams"""
    s = raw[sensor].replace(0, np.nan).dropna()
    data = [s[s.index.hour == h].values for h in range(24)]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.boxplot(data, positions=range(24), widths=0.6, flierprops=dict(marker="o", ms=3, mfc=RED, mec="none", alpha=0.5),
               medianprops=dict(color=RED), boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    ax.set_xlabel("시각 (시)"); ax.set_ylabel("속도 (mph)"); ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([str(h) for h in range(0, 24, 3)])
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def fill_example(raw, sensor, start, hours, name):
    """hide `hours` of real data in one sensor and fill it four ways"""
    S = raw.replace(0, np.nan)
    t0 = S.index.get_loc(pd.Timestamp(start)); n = hours * 12
    truth = S[sensor].iloc[t0:t0 + n].copy()
    H = S.copy(); H.iloc[t0:t0 + n, H.columns.get_loc(sensor)] = np.nan
    F = fill_methods(H)
    win = slice(t0 - n, t0 + 2 * n)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, m, c in zip(axs, METHODS, [GRAY, GOLD, GREEN, BLUE]):
        ax.plot(S.index[win], S[sensor].iloc[win], color="black", lw=1.2)
        ax.plot(truth.index, F[m][sensor].iloc[t0:t0 + n], color=c, lw=3)
        mae = np.abs(F[m][sensor].iloc[t0:t0 + n].values - truth.values).mean()
        ax.axvspan(truth.index[0], truth.index[-1], color=RED, alpha=0.07)
        ax.set_title(f"{LABEL[m].splitlines()[0]}: 오차 {mae:.1f}", fontsize=14, color=c)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H시")); ax.tick_params(axis="x", labelsize=11)
    axs[0].set_ylabel("속도 (mph)")
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def fill_bars(J, name):
    """MAE of the four fills for 5-min, 1-hour, 12-hour holes"""
    keys = [("fill_5min", "5분 구멍"), ("fill_1h", "1시간 구멍"), ("fill_12h", "12시간 구멍")]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    w = 0.2
    for k, (m, c) in enumerate(zip(METHODS, [GRAY, GOLD, GREEN, BLUE])):
        vals = [J[key][m] for key, _ in keys]
        b = ax.bar(np.arange(3) + (k - 1.5) * w, vals, w, color=c, label=LABEL[m].replace("\n", " "))
        ax.bar_label(b, fmt="%.1f", fontsize=11)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel("오차 (mph)")
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper left"); ax.set_ylim(0, 11)
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def model_bars(J, name):
    """15-min-ahead linear regression MAE"""
    m = J["model"]
    items = [("지금 속도 그대로\n(기준선)", m["baseline (now)"], GRAY), ("0 을 속도로 믿고\n학습", m["zeros as speed"], RED),
             ("0 을 빈칸으로 바꾸고\n학습", m["drop blank rows"], BLUE)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    b = ax.barh(range(3), [v for _, v, _ in items], color=[c for *_, c in items])
    ax.bar_label(b, fmt="%.2f mph", fontsize=14, padding=4)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for l, *_ in items]); ax.invert_yaxis(); ax.set_xlim(3.0, 3.45)
    ax.set_xlabel("15분 뒤 속도 오차 (MAE, 작을수록 좋다)")
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


if __name__ == "__main__":
    raw = load_speed(PATH)
    J = json.load(open("assets/metr.json"))
    week_plot(raw, SENSOR, WEEK, "mt_week.png")
    missing_map(raw, "mt_map.png")
    hist_plot(raw, "mt_hist.png")
    box_by_hour(raw, SENSOR, "mt_box.png")
    fill_example(raw, SENSOR, "2012-06-13 06:00", 2, "mt_fill_example.png")
    fill_bars(J, "mt_fill.png")
    model_bars(J, "mt_model.png")
    print("ok")
