"""글자 단위 RNN 언어 모델 → assets/charrnn.json
tiny Shakespeare (Karpathy char-rnn, 약 111만 글자), nn.RNN(은닉 128, tanh) + Linear, 길이 64 조각 · 배치 64, Adam 3e-3, 2,000걸음, 시드 0.
걸음 0 · 200 · 2,000 에서 'ROMEO:' 로 시작해 200글자 생성 (온도 0.8, 시드 0). 손실 = 다음 글자 교차 엔트로피 (nats)."""
import os, json, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "assets", "tinyshakespeare.txt")).read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
torch.manual_seed(0)
class CharRNN(nn.Module):
    def __init__(self, V, H=128):
        super().__init__(); self.emb = nn.Embedding(V, H); self.rnn = nn.RNN(H, H, batch_first=True); self.out = nn.Linear(H, V)
    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h); return self.out(o), h
def sample(m, start="ROMEO:", n=200, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.tensor([[stoi[c] for c in start]]); out = start; h = None
    with torch.no_grad():
        logits, h = m(x, h)
        for _ in range(n):
            p = torch.softmax(logits[0, -1] / temp, 0); c = torch.multinomial(p, 1, generator=g).item(); out += chars[c]
            logits, h = m(torch.tensor([[c]]), h)
    return out
m = CharRNN(len(chars)); opt = torch.optim.Adam(m.parameters(), 3e-3); g = torch.Generator().manual_seed(0)
losses, samples = [], {}
for step in range(2001):
    if step in (0, 200, 2000): samples[step] = sample(m)
    if step == 2000: break
    ix = torch.randint(0, len(data) - 65, (64,), generator=g); xb = torch.stack([data[i:i + 64] for i in ix]); yb = torch.stack([data[i + 1:i + 65] for i in ix])
    logits, _ = m(xb); loss = nn.functional.cross_entropy(logits.reshape(-1, len(chars)), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); losses.append(loss.item())
json.dump(dict(V=len(chars), n_chars=len(text), losses=losses, samples=samples), open(os.path.join(HERE, "assets", "charrnn.json"), "w"))
print(len(chars), len(text), losses[0], sum(losses[-50:]) / 50); print(samples[200]); print("----"); print(samples[2000])
