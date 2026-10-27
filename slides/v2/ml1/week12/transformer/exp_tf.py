"""작은 글자 트랜스포머 (GPT 꼴, 디코더만) vs RNN · LSTM → assets/tf.json
tiny Shakespeare, week11/lstm/exp_char.py 와 같은 조건: 조각 64, 배치 64, Adam 3e-3 (트랜스포머는 1e-3), 2,000걸음, 시드 0,
검증 = 뒤 10% 글에서 조각 200개 (시드 1). 트랜스포머: 폭 128, 4층, 헤드 4, 위치 임베딩 학습.
위치 정보를 뺀 모델도 같이 학습 (순서를 모르면?). 걸음당 시간도 잰다 (CPU)."""
import os, json, time, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "assets", "tinyshakespeare.txt")).read(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); cut = int(len(data) * 0.9); tr, va = data[:cut], data[cut:]
V, SEQ = len(chars), 64

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        a = self.ln1(x); x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, pos=True):
        super().__init__(); self.pos = pos
        self.tok = nn.Embedding(V, d); self.pe = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, V)
    def forward(self, x):
        T = x.shape[1]; h = self.tok(x) + (self.pe(torch.arange(T)) if self.pos else 0)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in self.blocks: h = b(h, mask)
        return self.out(self.ln(h))

class RNNLM(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.e = nn.Embedding(V, 128); self.r = (nn.RNN if kind == "rnn" else nn.LSTM)(128, 128, batch_first=True); self.o = nn.Linear(128, V)
    def forward(self, x): return self.o(self.r(self.e(x))[0])

def get(d, n, g):
    ix = torch.randint(0, len(d) - SEQ - 1, (n,), generator=g); return torch.stack([d[i:i + SEQ] for i in ix]), torch.stack([d[i + 1:i + SEQ + 1] for i in ix])

def run(name, make, lr):
    torch.manual_seed(0); m = make(); opt = torch.optim.Adam(m.parameters(), lr); g = torch.Generator().manual_seed(0); L = []
    t0 = time.time()
    for step in range(2000):
        x, y = get(tr, 64, g); loss = nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
    sec = (time.time() - t0) / 2000
    m.eval(); x, y = get(va, 200, torch.Generator().manual_seed(1))
    with torch.no_grad(): v = nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
    # 글 써 보기
    g2 = torch.Generator().manual_seed(0); s = torch.tensor([[stoi[c] for c in "ROMEO:"]])
    with torch.no_grad():
        for _ in range(200):
            p = torch.softmax(m(s[:, -SEQ:])[0, -1] / 0.8, 0); s = torch.cat([s, torch.multinomial(p, 1, generator=g2)[None]], 1)
    sample = "".join(chars[i] for i in s[0].tolist())
    n_par = sum(p.numel() for p in m.parameters())
    print(name, round(v, 3), round(sum(L[-100:]) / 100, 3), f"{sec * 1000:.0f} ms/step", n_par, flush=True)
    return dict(val=v, train=sum(L[-100:]) / 100, ms=sec * 1000, params=n_par, curve=L[::10], sample=sample)

if __name__ == "__main__":
    torch.set_num_threads(4)
    out = {}
    out["rnn"] = run("rnn", lambda: RNNLM("rnn"), 3e-3)
    out["lstm"] = run("lstm", lambda: RNNLM("lstm"), 3e-3)
    out["tf"] = run("tf", lambda: GPT(), 1e-3)
    out["tf_nopos"] = run("tf_nopos", lambda: GPT(pos=False), 1e-3)
    json.dump(out, open(os.path.join(HERE, "assets", "tf.json"), "w"))
