"""데이터 종류별 정규화 실험 → assets/modal.json (슬라이드 · 그림이 쓰는 숫자. 실습 노트북과 같은 데이터 · 같은 설정)
데이터는 lab/data/ (실습 노트북이 내려받는 곳, git 제외).
  흑백 이미지: MNIST (torchvision) 학습 1만 장 · 시험 1만 장, MLP 784-64-10 (tanh), SGD 0.1, 배치 100, 3에폭, 시드 0~2 평균
  컬러 이미지: CIFAR-10 시험 1만 장 (Hugging Face uoft-cs parquet), 채널별 평균 · 표준편차
  언어: UCI SMS Spam 5,574통, 단어 사전 크기 (그대로 공백으로 자르기 vs 소문자 + 단어만)
  시계열: 국제선 항공 승객 1949~1960 (Box & Jenkins), 앞 10년 학습 · 뒤 2년 시험, 12달 → 다음 달, kNN (k = 3)
  소리: Free Spoken Digit Dataset, 숫자 0 을 말한 6명 (8 kHz, int16)
  비디오: Muybridge, The Horse in Motion (1878) GIF, Wikimedia Commons (퍼블릭 도메인)"""
import io, json, os, zipfile
import numpy as np, pandas as pd, torch
from PIL import Image, ImageSequence
from scipy.io import wavfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "lab", "data")
J = {}

# ---- 흑백 이미지: MNIST
tr = datasets.MNIST(D, train=True, download=True); te = datasets.MNIST(D, train=False, download=True)
Xtr, ytr = tr.data[:10000].reshape(-1, 784).float(), tr.targets[:10000]
Xte, yte = te.data.reshape(-1, 784).float(), te.targets
full = tr.data.float() / 255
J["mnist"] = dict(dtype=str(tr.data.dtype), lo=int(tr.data.min()), hi=int(tr.data.max()), mean=float(full.mean()), std=float(full.std()))
SCALE = {"raw": lambda x: x, "div255": lambda x: x / 255, "z": lambda x: (x / 255 - 0.1307) / 0.3081}
acc, sat = {}, {}
for name, f in SCALE.items():
    a_list, s_list = [], []
    for s in range(3):
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(784, 64), torch.nn.Tanh(), torch.nn.Linear(64, 10))
        opt = torch.optim.SGD(net.parameters(), lr=0.1); A = f(Xtr)
        with torch.no_grad(): s_list.append(float((net[1](net[0](A)).abs() > 0.99).float().mean()))
        for ep in range(3):
            perm = torch.randperm(len(A))
            for i in range(0, len(A), 100):
                idx = perm[i:i + 100]; opt.zero_grad()
                torch.nn.functional.cross_entropy(net(A[idx]), ytr[idx]).backward(); opt.step()
        with torch.no_grad(): a_list.append(float((net(f(Xte)).argmax(1) == yte).float().mean()))
    acc[name], sat[name] = float(np.mean(a_list)), float(np.mean(s_list))
J["mnist"].update(acc=acc, sat=sat, digit=tr.data[0].numpy().tolist(), label=int(tr.targets[0]))

# ---- 컬러 이미지: CIFAR-10 시험 1만 장
df = pd.read_parquet(os.path.join(D, "c10_test.parquet"))
im = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
J["cifar"] = dict(shape=list(im.shape), mean=(im / 255).mean((0, 1, 2)).tolist(), std=(im / 255).std((0, 1, 2)).tolist(),
                  one=im[23].tolist(), one_label=int(df["label"][23]))

# ---- 언어: SMS 단어 사전
z = zipfile.ZipFile(os.path.join(D, "sms.zip"))
sms = pd.read_csv(z.open("SMSSpamCollection"), sep="\t", names=["label", "text"], quoting=3)
v_raw = CountVectorizer(lowercase=False, token_pattern=r"\S+").fit(sms.text)
v_norm = CountVectorizer().fit(sms.text)
L = sms.text.str.split().str.len()
J["sms"] = dict(n=len(sms), vocab_raw=len(v_raw.vocabulary_), vocab_norm=len(v_norm.vocabulary_), len_min=int(L.min()), len_max=int(L.max()))

# ---- 시계열: 항공 승객
p = pd.read_csv(os.path.join(D, "air.csv")).Passengers.values.astype(float)
W, NTR = 12, 120
X = np.stack([p[i:i + W] for i in range(len(p) - W)]); y = p[W:]
is_tr = np.arange(W, len(p)) < NTR
mu, sd, lo, hi = p[:NTR].mean(), p[:NTR].std(), p[:NTR].min(), p[:NTR].max()
m = KNeighborsRegressor(3).fit((X[is_tr] - mu) / sd, (y[is_tr] - mu) / sd)
pred_g = m.predict((X[~is_tr] - mu) / sd) * sd + mu
wm, ws = X.mean(1), X.std(1)
m = KNeighborsRegressor(3).fit((X[is_tr] - wm[is_tr, None]) / ws[is_tr, None], (y[is_tr] - wm[is_tr]) / ws[is_tr])
pred_w = m.predict((X[~is_tr] - wm[~is_tr, None]) / ws[~is_tr, None]) * ws[~is_tr] + wm[~is_tr]
J["air"] = dict(p=p.tolist(), ntr=NTR, mean=mu, std=sd, lo=lo, hi=hi, test_hi=float(p[NTR:].max()),
                pred_global=pred_g.tolist(), pred_window=pred_w.tolist(),
                mae_global=float(np.abs(pred_g - y[~is_tr]).mean()), mae_window=float(np.abs(pred_w - y[~is_tr]).mean()))

# ---- 소리: 숫자 0 을 말한 6명
SPK = ["jackson", "lucas", "george", "nicolas", "yweweler", "theo"]
waves = {}
for s in SPK:
    sr, a = wavfile.read(os.path.join(D, f"0_{s}_0.wav")); waves[s] = a
J["sound"] = dict(sr=sr, dtype=str(a.dtype), peak={s: float(np.abs(w).max() / 32768) for s, w in waves.items()},
                  jackson=waves["jackson"].tolist(), theo=waves["theo"].tolist())

# ---- 비디오: 머이브리지 말 GIF
g = Image.open(os.path.join(D, "horse.gif"))
v = np.stack([np.array(f.convert("RGB")) for f in ImageSequence.Iterator(g)])
J["video"] = dict(shape=list(v.shape), dtype=str(v.dtype), frame_mean=(v / 255).mean((1, 2, 3)).tolist(),
                  clip_mean=float((v / 255).mean()), clip_std=float((v / 255).std()))

json.dump(J, open(os.path.join(HERE, "assets", "modal.json"), "w"))
print({k: (J["mnist"][k] if k in J["mnist"] else None) for k in ("acc", "sat", "mean", "std")})
print("cifar", np.round(J["cifar"]["mean"], 3), np.round(J["cifar"]["std"], 3))
print("sms", J["sms"]); print("air mae", J["air"]["mae_global"], J["air"]["mae_window"], max(pred_g), max(pred_w))
print("peaks", J["sound"]["peak"]); print("video", J["video"]["shape"], J["video"]["clip_mean"], J["video"]["clip_std"])
