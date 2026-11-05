"""먼 기억 실험 → assets/memory.json
과제: 길이 T 수열. 첫 칸 = 0 또는 1 (기억할 비트), 나머지 칸 = 무작위 잡음 (표준정규). 마지막 칸에서 첫 비트를 맞힌다.
모델: nn.RNN / nn.LSTM (은닉 32) + Linear(32, 2), Adam 1e-2, 배치 128, 1,500걸음, 시드 0. T = 5, 20, 50, 100. 시험 = 새 수열 2,000개 정확도."""
import os, json, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
def batch(T, n, g):
    bit = torch.randint(0, 2, (n,), generator=g); x = torch.randn(n, T, 1, generator=g); x[:, 0, 0] = bit.float() * 2 - 1
    return x, bit
class Net(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.r = (nn.RNN if kind == "rnn" else nn.LSTM)(1, 32, batch_first=True); self.o = nn.Linear(32, 2)
    def forward(self, x): return self.o(self.r(x)[0][:, -1])
out = {}
for T in (5, 20, 50, 100):
    for kind in ("rnn", "lstm"):
        torch.manual_seed(0); m = Net(kind); opt = torch.optim.Adam(m.parameters(), 1e-2); g = torch.Generator().manual_seed(0)
        for step in range(1500):
            x, y = batch(T, 128, g); opt.zero_grad(); nn.functional.cross_entropy(m(x), y).backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        x, y = batch(T, 2000, torch.Generator().manual_seed(99))
        with torch.no_grad(): acc = (m(x).argmax(1) == y).float().mean().item()
        out[f"{kind}_{T}"] = acc; print(kind, T, round(acc, 3), flush=True)
json.dump(out, open(os.path.join(HERE, "assets", "memory.json"), "w"))
