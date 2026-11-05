"""글자 언어 모델 GRU → assets/char_gru.json  (RNN · LSTM 은 ../lstm/exp_char.py 의 char_cmp.json 과 같은 조건)
tiny Shakespeare, Embedding 128 → nn.GRU (은닉 128) → Linear, 길이 조각 64 · 128, 배치 64, Adam 3e-3, 2,000걸음, 시드 0.
지표: 마지막 100걸음 평균 학습 손실 + 검증 손실 (뒤 10% 글자, 조각 200개) + 한 걸음 평균 시간 (CPU, 세 모델 같이 잰다)."""
import os, json, time, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "assets", "tinyshakespeare.txt")).read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); cut = int(len(data) * 0.9); tr, va = data[:cut], data[cut:]
CELL = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
class M(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.e = nn.Embedding(len(chars), 128); self.r = CELL[kind](128, 128, batch_first=True); self.o = nn.Linear(128, len(chars))
    def forward(self, x): return self.o(self.r(self.e(x))[0])
def get(d, seq, n, g):
    ix = torch.randint(0, len(d) - seq - 1, (n,), generator=g); return torch.stack([d[i:i + seq] for i in ix]), torch.stack([d[i + 1:i + seq + 1] for i in ix])
out = {}
for seq in (64, 128):
    torch.manual_seed(0); m = M("gru"); opt = torch.optim.Adam(m.parameters(), 3e-3); g = torch.Generator().manual_seed(0); L = []
    for step in range(2000):
        x, y = get(tr, seq, 64, g); loss = nn.functional.cross_entropy(m(x).reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
    x, y = get(va, seq, 200, torch.Generator().manual_seed(1))
    with torch.no_grad(): v = nn.functional.cross_entropy(m(x).reshape(-1, len(chars)), y.reshape(-1)).item()
    out[f"gru_{seq}"] = dict(train=sum(L[-100:]) / 100, val=v, curve=L); print("gru", seq, round(out[f'gru_{seq}']['train'], 3), round(v, 3), flush=True)
# 가중치 수 (순환 층만) · 한 걸음 시간 (조각 128, 50걸음 평균)
for kind in CELL:
    torch.manual_seed(0); m = M(kind); opt = torch.optim.Adam(m.parameters(), 3e-3); g = torch.Generator().manual_seed(0)
    out[f"params_{kind}"] = sum(p.numel() for p in m.r.parameters())
    t0 = time.perf_counter()
    for step in range(50):
        x, y = get(tr, 128, 64, g); loss = nn.functional.cross_entropy(m(x).reshape(-1, len(chars)), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    out[f"sec_{kind}"] = (time.perf_counter() - t0) / 50; print(kind, out[f"params_{kind}"], round(out[f"sec_{kind}"], 4), flush=True)
json.dump(out, open(os.path.join(HERE, "assets", "char_gru.json"), "w"))
