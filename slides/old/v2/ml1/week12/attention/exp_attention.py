"""어텐션 실험 → assets/attn.json
Seq2Seq (GRU) 에 내적 어텐션 (Luong 2015 식: 점수 = s_k · h_j, 맥락 = Σ α h_j, 출력 = W[s_k; 맥락]) 을 더한다.
숫자열 베끼기 길이 5 ~ 30 으로 학습 (week12/seq2seq 실험과 같은 설정, 은닉 32, 4,000걸음), 시험은 40 · 50 까지.
어텐션 없음 vs 있음, 입력 그대로 vs 거꾸로. 어텐션 지도 = 길이 12 숫자열 하나."""
import os, json, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))

class AttnS2S(nn.Module):
    def __init__(self, vin, vout, hid, attn=True):
        super().__init__(); self.attn = attn
        self.ei, self.eo = nn.Embedding(vin, hid), nn.Embedding(vout, hid)
        self.enc, self.dec = nn.GRU(hid, hid, batch_first=True), nn.GRU(hid, hid, batch_first=True)
        self.out = nn.Linear(2 * hid if attn else hid, vout)
    def step(self, H, mask, o):
        if not self.attn: return self.out(o), None
        score = o @ H.transpose(1, 2)                                  # (B, K, T)
        score = score.masked_fill(~mask[:, None, :], -1e9); a = score.softmax(-1)
        return self.out(torch.cat([o, a @ H], -1)), a
    def forward(self, x, y_in):
        H, h = self.enc(self.ei(x)); o, _ = self.dec(self.eo(y_in), h); return self.step(H, x != 0, o)
    @torch.no_grad()
    def greedy(self, x, n, bos=1):
        H, s = self.enc(self.ei(x)); t = torch.full((x.shape[0], 1), bos); outs, As = [], []
        for _ in range(n):
            o, s = self.dec(self.eo(t), s); lg, a = self.step(H, x != 0, o); t = lg.argmax(-1); outs.append(t); As.append(a)
        return torch.cat(outs, 1), (torch.cat(As, 1) if self.attn else None)

def train_copy(attn, rev, hid=32, steps=4000):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m = AttnS2S(12, 13, hid, attn); opt = torch.optim.Adam(m.parameters(), 3e-3); L = []
    for step in range(steps):
        n = int(torch.randint(5, 31, (1,), generator=g)); d = torch.randint(0, 10, (64, n), generator=g)
        x = (d.flip(1) if rev else d) + 1; y = d + 3
        yi = torch.cat([torch.ones(64, 1, dtype=torch.long), y], 1)[:, :-1]
        loss = nn.functional.cross_entropy(m(x, yi)[0].reshape(-1, 13), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); L.append(loss.item())
    return m, L

def eval_copy(m, rev, lengths=(5, 10, 15, 20, 25, 30, 40, 50)):
    res = {}
    for n in lengths:
        d = torch.randint(0, 10, (500, n), generator=torch.Generator().manual_seed(n))
        x = (d.flip(1) if rev else d) + 1; p, _ = m.greedy(x, n); p = p - 3
        res[n] = dict(seq=(p == d).all(1).float().mean().item(), tok=(p == d).float().mean().item())
    return res

def copy_map(m, rev, n=12):
    d = torch.randint(0, 10, (1, n), generator=torch.Generator().manual_seed(99)); x = (d.flip(1) if rev else d) + 1
    p, a = m.greedy(x, n); return dict(src=(x[0] - 1).tolist(), out=(p[0] - 3).tolist(), attn=a[0].tolist())

if __name__ == "__main__":
    out = {}
    for attn in (False, True):
        for rev in (False, True):
            m, L = train_copy(attn, rev); k = f"{'attn' if attn else 'plain'}_{'rev' if rev else 'fwd'}"
            out[k] = dict(acc=eval_copy(m, rev), curve=L[::20])
            if attn: out[k]["map"] = copy_map(m, rev)
            print(k, {n: round(v["tok"] * 100) for n, v in out[k]["acc"].items()}, {n: round(v["seq"] * 100) for n, v in out[k]["acc"].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "assets", "attn.json"), "w"))
