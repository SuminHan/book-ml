"""METR-LA (LA 고속도로 센서 207개, 5분 간격 속도 mph) 로 더티 데이터 실험 → assets/metr.json
데이터: Hugging Face witgaw/METR-LA 의 test.parquet (2012-06-04 ~ 06-27, Li et al. 2018 DCRNN 의 시험 기간)
실행: ~/Documents/manim_shorts/_env/v/bin/python exp_metr.py   (lab/data/metr_test.parquet 가 없으면 내려받는다)"""
import os, json, urllib.request
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "lab/data/metr_test.parquet")
URL = "https://huggingface.co/datasets/witgaw/METR-LA/resolve/main/test.parquet"


def load_speed(path):
    """(time x sensor) speed table from the windowed parquet (current speed column)"""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_parquet(path, columns=["node_id", "t0_timestamp", "x_t+0_d0"])
    df["time"] = pd.to_datetime(df["t0_timestamp"])
    return df.pivot(index="time", columns="node_id", values="x_t+0_d0")


def gap_runs(mask):
    """lengths of consecutive True runs in each column of a boolean table"""
    out = []
    for c in mask.columns:
        v = mask[c].values.astype(int)
        d = np.diff(np.r_[0, v, 0])
        out += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    return np.array(out)


def fill_methods(S):
    """the four ways to fill blanks (S has NaN for missing)"""
    tod = S.groupby(S.index.time).transform("mean")           # same sensor, same time of day
    return {"sensor mean": S.fillna(S.mean()), "ffill": S.ffill(), "interpolate": S.interpolate(limit_direction="both"),
            "time-of-day mean": S.fillna(tod)}


def grade_fill(S, block, n, seed=0):
    """hide n blocks of `block` observed steps, fill, MAE on the hidden truth"""
    rng = np.random.default_rng(seed)
    obs = S.notna().values
    T, N = obs.shape
    hide = np.zeros_like(obs)
    tries = 0
    while hide.sum() < n * block and tries < n * 50:
        tries += 1
        t, c = rng.integers(1, T - block - 1), rng.integers(N)
        if obs[t - 1:t + block + 1, c].all() and not hide[t - 1:t + block + 1, c].any():
            hide[t:t + block, c] = True
    H = S.mask(hide)
    truth = S.values[hide]
    return {k: round(float(np.abs(v.values[hide] - truth).mean()), 2) for k, v in fill_methods(H).items()}


def windows(S, lag=12, ahead=3):
    """rows: past `lag` speeds of a sensor -> speed `ahead` steps later. returns X, y, time of target"""
    A = S.values
    X, y, t = [], [], []
    for c in range(A.shape[1]):
        a = A[:, c]
        idx = np.arange(lag, len(a) - ahead + 1)
        X.append(np.stack([a[i - lag:i] for i in idx])); y.append(a[idx + ahead - 1]); t.append(idx + ahead - 1)
    return np.concatenate(X), np.concatenate(y), np.concatenate(t)


def model_mae(S_raw):
    """15-min-ahead linear regression: zeros as numbers vs blanks handled. MAE on test rows with real target and real inputs"""
    S_nan = S_raw.replace(0, np.nan)
    cut = int(len(S_raw) * 0.7)
    Xr, yr, t = windows(S_raw)
    Xn, yn, _ = windows(S_nan)
    Xi, yi, _ = windows(S_nan.interpolate(limit=12))
    test_ok = (t >= cut) & np.isfinite(Xn).all(1) & np.isfinite(yn)       # same honest test rows for all
    out = {}
    tr = t < cut
    m = LinearRegression().fit(Xr[tr], yr[tr]); out["zeros as speed"] = np.abs(m.predict(Xr[test_ok]) - yn[test_ok]).mean()
    ok = tr & np.isfinite(Xn).all(1) & np.isfinite(yn)
    m = LinearRegression().fit(Xn[ok], yn[ok]); out["drop blank rows"] = np.abs(m.predict(Xn[test_ok]) - yn[test_ok]).mean()
    ok = tr & np.isfinite(Xi).all(1) & np.isfinite(yi)
    m = LinearRegression().fit(Xi[ok], yi[ok]); out["interpolate short gaps"] = np.abs(m.predict(Xn[test_ok]) - yn[test_ok]).mean()
    base = np.abs(Xn[test_ok, -1] - yn[test_ok]).mean()
    return {k: round(float(v), 2) for k, v in out.items()} | {"baseline (now)": round(float(base), 2), "test rows": int(test_ok.sum())}


if __name__ == "__main__":
    S_raw = load_speed(PATH)
    S = S_raw.replace(0, np.nan)
    zero = S_raw == 0
    allzero = zero.all(axis=1)
    runs = gap_runs(zero & ~allzero.values[:, None])
    J = {
        "shape": list(S_raw.shape), "start": str(S_raw.index.min()), "end": str(S_raw.index.max()),
        "zero_pct": round(float(zero.values.mean() * 100), 1),
        "mean_with_zero": round(float(S_raw.values.mean()), 1), "mean_without_zero": round(float(np.nanmean(S.values)), 1),
        "allzero_times": int(allzero.sum()), "dups": int(S_raw.reset_index().duplicated().sum()),
        "max": float(S_raw.values.max()), "at_max_pct": round(float((S_raw.values == 70).mean() * 100), 1),
        "low_nonzero": int(((S_raw > 0) & (S_raw < 5)).values.sum()),
        "runs": {"n": int(len(runs)), "one_step": int((runs == 1).sum()), "le_3": int((runs <= 3).sum()),
                 "ge_1h": int((runs >= 12).sum()), "ge_1day": int((runs >= 288).sum()), "max_steps": int(runs.max())},
        "worst_sensors": {str(k): round(float(v) * 100) for k, v in zero.mean().sort_values(ascending=False).head(3).items()},
        "fill_5min": grade_fill(S, 1, 3000), "fill_1h": grade_fill(S, 12, 300), "fill_12h": grade_fill(S, 144, 100),
        "model": model_mae(S_raw),
    }
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    json.dump(J, open(os.path.join(HERE, "assets/metr.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(J, ensure_ascii=False, indent=1))
