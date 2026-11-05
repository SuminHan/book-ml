"""Seq2Seq 장난감 실험 → assets/s2s.json
A. 영어 숫자 말 → 숫자 (0 ~ 999, 'nine hundred four' → '904'). 1,000개 중 800개 학습, 200개 시험. GRU 인코더-디코더 은닉 128, 3,000걸음.
B. 숫자열 그대로 베끼기 (길이 5 ~ 30), 은닉 32, 길이별 문장 전체 정답률. 입력 그대로 vs 거꾸로."""
import os, json, random, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def words(n):
    if n < 10: return [ONES[n]]
    if n < 20: return [TEENS[n - 10]]
    if n < 100: return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    return [ONES[n // 100], "hundred"] + (words(n % 100) if n % 100 else [])

class S2S(nn.Module):
    def __init__(self, vin, vout, hid):
        super().__init__()
        self.ei, self.eo = nn.Embedding(vin, hid), nn.Embedding(vout, hid)
        self.enc, self.dec = nn.GRU(hid, hid, batch_first=True), nn.GRU(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, vout)
    def forward(self, x, y_in):
        _, h = self.enc(self.ei(x)); o, _ = self.dec(self.eo(y_in), h); return self.out(o)
    @torch.no_grad()
    def greedy(self, x, n, bos):
        _, h = self.enc(self.ei(x)); t = torch.full((x.shape[0], 1), bos); outs = []
        for _ in range(n):
            o, h = self.dec(self.eo(t), h); t = self.out(o).argmax(-1); outs.append(t)
        return torch.cat(outs, 1)

def pad(seqs, v):
    L = max(map(len, seqs)); return torch.tensor([s + [v] * (L - len(s)) for s in seqs])

def run_numbers():
    torch.manual_seed(0); random.seed(0)
    vin = ["<pad>"] + sorted(set(ONES + TEENS + TENS[2:] + ["hundred"])); si = {w: i for i, w in enumerate(vin)}
    vout = ["<pad>", "<s>", "</s>"] + list("0123456789"); so = {c: i for i, c in enumerate(vout)}
    nums = list(range(1000)); random.shuffle(nums); tr, te = nums[:800], nums[800:]
    def enc(ns):
        x = pad([[si[w] for w in words(n)] for n in ns], 0)
        y = [[so[c] for c in str(n)] + [2] for n in ns]
        return x, pad([[1] + s[:-1] for s in y], 0), pad(y, 0)
    torch.manual_seed(0); random.seed(0)                        # 실습 노트북과 같은 순서
    m = S2S(len(vin), len(vout), 128); opt = torch.optim.Adam(m.parameters(), 3e-3); L = []
    for step in range(3000):
        x, yi, yo = enc(random.sample(tr, 64))
        loss = nn.functional.cross_entropy(m(x, yi).reshape(-1, len(vout)), yo.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
    def acc(ns):
        x, _, yo = enc(ns); p = m.greedy(x, 4, 1); ok = 0; outs = []
        for n, row in zip(ns, p.tolist()):
            s = "".join(vout[t] for t in row).split("</s>")[0]; outs.append(s); ok += s == str(n)
        return ok / len(ns), outs
    a_tr, _ = acc(tr); a_te, outs = acc(te)
    ex = [(" ".join(words(n)), o, str(n)) for n, o in zip(te, outs)]
    wrong = [e for e in ex if e[1] != e[2]]
    return dict(train_acc=a_tr, test_acc=a_te, examples=ex[:12], wrong=wrong[:10], curve=L[::10])

def run_copy(rev, lengths=(5, 10, 15, 20, 25, 30), hid=32, steps=4000):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m = S2S(12, 13, hid); opt = torch.optim.Adam(m.parameters(), 3e-3)  # in: 0 pad, 1..10 digits ; out: 0 pad 1 <s> 2 </s> 3..12 digits
    for step in range(steps):
        n = int(torch.randint(5, 31, (1,), generator=g)); d = torch.randint(0, 10, (64, n), generator=g)
        x = (d.flip(1) if rev else d) + 1; y = d + 3
        yi = torch.cat([torch.ones(64, 1, dtype=torch.long), y], 1)[:, :-1]
        loss = nn.functional.cross_entropy(m(x, yi).reshape(-1, 13), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    res = {}
    for n in lengths:
        d = torch.randint(0, 10, (500, n), generator=torch.Generator().manual_seed(n))
        x = (d.flip(1) if rev else d) + 1; p = m.greedy(x, n, 1) - 3
        res[n] = dict(seq=(p == d).all(1).float().mean().item(), tok=(p == d).float().mean().item())
    return res

if __name__ == "__main__":
    out = dict(numbers=run_numbers()); print("numbers", out["numbers"]["train_acc"], out["numbers"]["test_acc"], out["numbers"]["wrong"], flush=True)
    out["copy"] = run_copy(False); print("copy", out["copy"], flush=True)
    out["copy_rev"] = run_copy(True); print("copy_rev", out["copy_rev"], flush=True)
    json.dump(out, open(os.path.join(HERE, "assets", "s2s.json"), "w"), indent=1)
