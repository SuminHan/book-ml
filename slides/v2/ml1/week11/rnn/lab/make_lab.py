"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: Tiny Shakespeare (Karpathy char-rnn, 처음 실행 때 내려받기). 슬라이드 실험 exp_charrnn.py 는 2,000걸음, 노트북은 1,000걸음."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (rnn) 이 파일명 앞에 붙는다

CELLS = [
("md", """# RNN 순환 신경망 실습

단어 (글자) 를 하나씩 읽으며 메모를 고쳐 쓴다: $\\mathbf{h}_t = \\tanh(W_x \\mathbf{x}_t + W_h \\mathbf{h}_{t-1} + \\mathbf{b})$.

- **파트 1**: numpy 로 RNN 한 칸 → 손 계산, 두 문장의 메모
- **파트 2**: PyTorch `nn.RNN` 글자 단위 언어 모델 (셰익스피어)
- **파트 3**: 글 써 보기 · 온도

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 파트 2 는 몇 분 걸린다."""),
("code", """import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn"""),
("md", """## 파트 1. RNN 한 칸

슬라이드의 손 계산: 메모 한 칸, $w_x = 1$, $w_h = 0.5$, 입력 1, 0, 1 → 0.762, 0.363, 0.828."""),
("code", """def rnn_step(x, h, Wx, Wh, b):
    return np.tanh(Wx @ x + Wh @ h + b)                  #@ANS return None  # TODO: one RNN step

def run_rnn(xs, Wx, Wh, b):
    h = np.zeros(Wh.shape[0]); hs = []
    for x in xs:
        h = rnn_step(x, h, Wx, Wh, b); hs.append(h)
    return hs

hs = run_rnn([np.array([1.0]), np.array([0.0]), np.array([1.0])], np.array([[1.0]]), np.array([[0.5]]), np.zeros(1))
print([round(float(h[0]), 3) for h in hs])

# Same 4-cell RNN as the slides (random weights, seed 11): word order changes the final memory
rng = np.random.default_rng(11)
emb = {w: rng.normal(0, 1, 4) for w in ("dog", "bites", "man")}
Wx, Wh = rng.normal(0, 0.8, (4, 4)), rng.normal(0, 0.8, (4, 4))
for sentence in [["dog", "bites", "man"], ["man", "bites", "dog"]]:
    final = run_rnn([emb[w] for w in sentence], Wx, Wh, np.zeros(4))[-1]
    bag = sum(emb[w] for w in sentence)
    print(" ".join(sentence), " final memory:", final.round(2), "  sum of words:", bag.round(2))"""),
("md", """## 파트 2. 글자 단위 언어 모델

셰익스피어 희곡 약 111만 글자. 길이 64 조각에서 매 칸의 **다음 글자**를 맞힌다."""),
("code", """url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/shakespeare.txt"):
    import urllib.request; urllib.request.urlretrieve(url, "data/shakespeare.txt")
text = open("data/shakespeare.txt").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
print(len(text), "characters,", len(chars), "distinct;  ln(V) =", round(np.log(len(chars)), 2))
print(text[:200])"""),
("code", """class CharRNN(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h)
        return self.out(o), h

def get_batch(data, seq=64, batch=64, gen=None):
    ix = torch.randint(0, len(data) - seq - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + seq + 1] for i in ix])  #@ANS y = None  # TODO: the next character at every position
    return x, y

torch.manual_seed(0)
model = CharRNN(len(chars)); opt = torch.optim.Adam(model.parameters(), 3e-3); gen = torch.Generator().manual_seed(0)
losses = []
for step in range(1000):
    x, y = get_batch(data, gen=gen)
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step}: loss {loss.item():.2f}")
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss"); plt.show()"""),
("md", """## 파트 3. 글 써 보기

시작 글자를 넣고, 나온 확률대로 다음 글자를 뽑아 다시 넣는다. 온도 $T$: 확률을 $\\text{softmax}(z / T)$ 로. 작으면 무난, 크면 엉뚱."""),
("code", """def generate(model, start, n=300, temperature=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([[stoi[c] for c in start]]); out = start; h = None
    with torch.no_grad():
        logits, h = model(x, h)
        for _ in range(n):
            p = torch.softmax(logits[0, -1] / temperature, dim=0)   #@ANS p = None  # TODO: probabilities with temperature
            c = torch.multinomial(p, 1, generator=g).item()
            out += chars[c]
            logits, h = model(torch.tensor([[c]]), h)
    return out

for T in [0.5, 1.0, 1.5]:
    print(f"--- temperature {T} ---"); print(generate(model, "ROMEO:", 200, T)); print()"""),
("md", """**확인 질문**
1. 파트 1 에서 두 문장의 '단어 합' 은 같은데 마지막 메모는 다르다. 왜인가?
2. 파트 2 의 첫 손실이 ln 65 근처인 이유는?
3. 파트 3 에서 온도를 0.1 로 하면 어떤 글이 나오나? 왜 같은 말을 되풀이하기 쉬울까?
4. (도전) 은닉 크기를 128 → 256 으로, 또는 `nn.RNN` 을 `nn.LSTM` 으로 바꾸면 손실이 어떻게 달라지나? (다음 시간)

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
