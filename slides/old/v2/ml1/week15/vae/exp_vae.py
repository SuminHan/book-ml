"""오토인코더 (AE) vs VAE, 잠재 공간 2칸 → assets/vae.json · assets/vae.npz
쇼츠 lessons/ml1/vae/src/train_vae.py 와 같은 구조 (sklearn digits 8×8 1,797장, 64→256→2 · 256→64, Adam 1e-3, 400바퀴, 배치 128).
AE = 같은 구조에서 구름 없이 점 하나 (μ 만), 손실 = 복원 (BCE) 만. VAE = 복원 + KL.
비교: 복원 오차, 그리고 잠재 공간의 '빈 곳' 에서 뽑아 되살린 그림이 숫자처럼 보이는가
(숫자 분류기 = sklearn 로지스틱 회귀 (원본으로 학습) 의 최대 확률 ≥ 0.9 인 비율)."""
import os, json, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
d = load_digits(); Xn = d.data / 16.0; y = d.target; X = torch.tensor(Xn, dtype=torch.float32)
clf = LogisticRegression(max_iter=2000).fit(Xn, y)

def train(vae):
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 4 if vae else 2))
    dec = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, 64), nn.Sigmoid())
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3)
    for ep in range(400):
        for idx in torch.randperm(len(X)).split(128):
            x = X[idx]; h = enc(x)
            if vae:
                mu, lv = h[:, :2], h[:, 2:]; z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum()
            else:
                z = h; kl = torch.tensor(0.0)
            rec = nn.functional.binary_cross_entropy(dec(z), x, reduction="sum")
            loss = (rec + kl) / len(x); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = enc(X); mu = h[:, :2]
        recon = dec(mu).numpy(); err = float(((recon - Xn) ** 2).sum(1).mean())
        m, s = mu.mean(0), mu.std(0)
        g = np.linspace(-2, 2, 9); grid = torch.tensor([[a, b] for b in g[::-1] for a in g], dtype=torch.float32) * s + m   # 평균 ± 2 표준편차 격자
        G = dec(grid).numpy()
        torch.manual_seed(1); zr = torch.randn(2000, 2) * s + m; R = dec(zr).numpy()
        torch.manual_seed(2); N01 = dec(torch.randn(2000, 2)).numpy()               # 표준정규분포 N(0, 1) 에서 뽑기
        from scipy.spatial import cKDTree
        tree = cKDTree(mu.numpy()); dd = tree.query(mu.numpy(), k=2)[0][:, 1]
        lo, hi = mu.numpy().min(0), mu.numpy().max(0); gg = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 60), np.linspace(lo[1], hi[1], 60)), -1).reshape(-1, 2)
        gap = float((tree.query(gg)[0] > 3 * np.median(dd)).mean())                # 지도 네모 안에서 데이터와 먼 빈 곳 비율
    conf = lambda A: float((clf.predict_proba(A).max(1) >= 0.9).mean())
    pred = clf.predict(N01); hist = np.bincount(pred, minlength=10) / len(pred)
    return dict(mu=mu.numpy(), recon=recon, err=err, grid=G, grid_z=grid.numpy(), conf_grid=conf(G), conf_rand=conf(R), conf_data=conf(Xn), rand=R[:16],
                lat_mean=m.numpy().tolist(), lat_std=s.numpy().tolist(), n01_hist=hist.tolist(), n01_kinds=int((hist >= 0.02).sum()), n01=N01[:16], gap=gap)

out = {k: train(k == "vae") for k in ("ae", "vae")}
summary = {k: {kk: v[kk] for kk in ("err", "conf_grid", "conf_rand", "conf_data", "lat_mean", "lat_std", "n01_hist", "n01_kinds", "gap")} for k, v in out.items()}
print(json.dumps(summary, indent=1))
json.dump(summary, open(os.path.join(HERE, "assets", "vae.json"), "w"), indent=1)
np.savez_compressed(os.path.join(HERE, "assets", "vae.npz"), y=y, x=Xn, **{f"{k}_{kk}": v[kk] for k, v in out.items() for kk in ("mu", "recon", "grid", "grid_z", "rand", "n01")})
