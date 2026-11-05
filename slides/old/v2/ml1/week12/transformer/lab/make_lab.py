"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Tiny Shakespeare (Karpathy char-rnn, 처음 실행 때 내려받기). 슬라이드 실험 exp_tf.py 는 2,000걸음, 노트북은 1,000걸음."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (transformer) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 트랜스포머 실습

입력 = 단어 + 위치, 블록 = 멀티헤드 어텐션 → 피드포워드 (지름길 · LayerNorm), 블록을 쌓는다.

- **파트 1**: 위치 정보 (사인 · 코사인) · 앞만 보기 마스크
- **파트 2**: 트랜스포머 블록 직접 쌓기 → 가중치 수를 $12d^2$ 와 비교
- **파트 3**: 셰익스피어 작은 GPT 학습 · 글 써 보기

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 파트 3 은 몇 분 걸린다."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn"""),
("md", """## 파트 1. 위치 정보 · 마스크

$\\text{PE}(pos, 2i) = \\sin(pos / 10000^{2i/d})$, $\\text{PE}(pos, 2i+1) = \\cos(pos / 10000^{2i/d})$."""),
("code", """def positional_encoding(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / 10000 ** (2 * (i // 2) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))   #@ANS return None  # TODO: sin on even dims, cos on odd dims

PE = positional_encoding(64, 128)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
axs[0].imshow(PE[:, :64].T, cmap="RdBu", aspect="auto"); axs[0].set_xlabel("position"); axs[0].set_ylabel("dimension")
for k in [0, 20, 40]:
    axs[1].plot(PE[:, k], label=f"dim {k}")
axs[1].set_xlabel("position"); axs[1].legend(); plt.show()

# Nearby positions have similar codes, far ones differ
print("similarity pos 10 vs 11:", round(float(PE[10] @ PE[11]), 1), "  pos 10 vs 50:", round(float(PE[10] @ PE[50]), 1))"""),
("code", """def causal_mask(n):
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)   #@ANS return None  # TODO: True above the diagonal (future words)

tokens = ["today", "lunch", "what", "eat", "?"]
mask = causal_mask(5)
print(mask.int())
scores = torch.zeros(5, 5).masked_fill(mask, float("-inf"))
print("weights:"); print(scores.softmax(-1).numpy().round(2))
print("visible cells:", int((~mask).sum()))"""),
("md", """## 파트 2. 트랜스포머 블록

정규화 먼저 (GPT 방식): $x \\leftarrow x + \\text{Attn}(\\text{LN}(x))$, $\\quad x \\leftarrow x + \\text{FFN}(\\text{LN}(x))$."""),
("code", """class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        x = x + self.ff(self.ln2(x))                     #@ANS x = None  # TODO: feed-forward with a shortcut
        return x

def count_params(module):
    return sum(p.numel() for p in module.parameters())

for d in [128, 512]:
    n = count_params(Block(d, 4 if d == 128 else 8))
    print(f"d = {d}: block has {n:,} weights,  12 d^2 = {12 * d * d:,}")"""),
("md", """## 파트 3. 작은 GPT

폭 128, 4층, 머리 4. 길이 64 조각에서 매 칸의 다음 글자를 맞힌다 (11주차와 같은 데이터)."""),
("code", """url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/shakespeare.txt"):
    import urllib.request; urllib.request.urlretrieve(url, "data/shakespeare.txt")
text = open("data/shakespeare.txt").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
cut = int(len(data) * 0.9)
train_data, val_data = data[:cut], data[cut:]
print(len(text), "characters,", len(chars), "distinct")"""),
("code", """class GPT(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, seq=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)                  # learned position vectors (like GPT)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))      #@ANS h = None  # TODO: word vector + position vector
        mask = causal_mask(T)
        for block in self.blocks:
            h = block(h, mask)
        return self.out(self.ln(h))

def get_batch(data, seq=64, batch=64, gen=None):
    ix = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
    return x, y

def train_gpt(train_data, vocab, steps=1000):
    torch.manual_seed(0)
    model = GPT(vocab); opt = torch.optim.Adam(model.parameters(), 1e-3); gen = torch.Generator().manual_seed(0)
    losses = []
    for step in range(steps):
        x, y = get_batch(train_data, gen=gen)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step % 200 == 0:
            print(f"step {step}: loss {loss.item():.2f}")
    return model, losses

model, losses = train_gpt(train_data, len(chars))
print("weights:", f"{count_params(model):,}")
model.eval()
x, y = get_batch(val_data, batch=200, gen=torch.Generator().manual_seed(1))
with torch.no_grad():
    print("validation loss:", round(nn.functional.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1)).item(), 3))
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss"); plt.show()"""),
("code", """def generate(model, start, stoi, chars, n=300, temperature=0.8, seq=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([[stoi[c] for c in start]])
    with torch.no_grad():
        for _ in range(n):
            logits = model(x[:, -seq:])                  # only the last 64 characters fit
            p = torch.softmax(logits[0, -1] / temperature, dim=0)
            x = torch.cat([x, torch.multinomial(p, 1, generator=g)[None]], 1)
    return "".join(chars[i] for i in x[0].tolist())

print(generate(model, "ROMEO:", stoi, chars, 300))"""),
("md", """**확인 질문**
1. 파트 1 에서 위치 10 과 11 의 위치 벡터는 비슷하고 10 과 50 은 다르다. 이것이 왜 쓸모 있을까?
2. 파트 2 에서 실제 가중치 수가 $12d^2$ 보다 조금 많은 이유는?
3. 파트 3 의 `generate` 에서 `x[:, -seq:]` 로 마지막 64 글자만 넣는 이유는?
4. (도전) `GPT` 에서 위치 벡터를 빼고 다시 학습하면 검증 손실은? (슬라이드: 1.61 → 1.68, 2,000걸음)

데이터 출처: Tiny Shakespeare, Andrej Karpathy char-rnn (2015)."""),
]

if __name__ == "__main__":
    REL = os.path.relpath(HERE, os.path.expanduser("~/book-ml"))
    COLAB = f"https://colab.research.google.com/github/SuminHan/book-ml/blob/main/{REL}/{TOPIC}.ipynb"
    kind, first = CELLS[0]                                   # 첫 칸 제목 아래에 Colab 배지
    CELLS[0] = (kind, first.replace("\n", f"\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})\n", 1))
    lab.build(CELLS, f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb")
    sys.path.insert(0, os.path.join(HERE, "../../../../tools"))
    from expected import add_expected                         # 학생용 코드 칸 아래에 '예상 결과' (정답을 실제 실행한 출력)
    print(add_expected(f"{HERE}/{TOPIC}.ipynb", f"{HERE}/{TOPIC}_solution.ipynb"))
