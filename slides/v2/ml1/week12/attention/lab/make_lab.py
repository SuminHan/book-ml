"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: 직접 만든다 (손 계산 단어 벡터, 무작위 숫자열). 슬라이드 실험 exp_attention.py 와 같은 설정."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (attention) 이 파일명 앞에 붙는다

CELLS = [
("md", """# 어텐션 실습

질문 (Q) 을 모든 색인 (K) 과 비교해 비율을 만들고, 그 비율대로 내용 (V) 을 섞는다: $\\text{softmax}(QK^\\top / \\sqrt{d})\\,V$.

- **파트 1**: numpy 어텐션 → "그것은" 손 계산, $\\sqrt{d}$ 로 나누는 이유
- **파트 2**: Seq2Seq 에 어텐션 더하기 → 숫자열 베끼기 길이별 정답률
- **파트 3**: 어텐션 지도

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn"""),
("md", """## 파트 1. 어텐션 함수

슬라이드의 예: 동물은 도로를 건너지 않았다. **그것은** 너무 피곤했다. 단어 벡터 (3칸) 는 설명을 위해 직접 만든 예시, Q = K = V = 이 벡터."""),
("code", """def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)                        #@ANS scores = None  # TODO: scaled dot-product scores
    weights = softmax(scores)
    return weights @ V, weights                          #@ANS return None, weights  # TODO: mix the values

words = ["animal", "street", "cross", "it"]
E = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [2.8, 0.3, 0]], float)
out, w = attention(E, E, E)
print("weights of 'it':", dict(zip(words, w[3].round(3).tolist())))
print("new vector of 'it':", out[3].round(2))
plt.imshow(w, cmap="YlOrBr"); plt.xticks(range(4), words); plt.yticks(range(4), words)
plt.xlabel("key"); plt.ylabel("query"); plt.colorbar(); plt.show()"""),
("md", """$\\sqrt{d}$ 로 나누지 않으면? 칸마다 평균 0, 분산 1 인 무작위 벡터의 내적은 표준편차가 $\\sqrt{d}$. 칸이 많으면 점수가 커지고 softmax 가 한 단어에 몰린다."""),
("code", """rng = np.random.default_rng(0)
for d in [3, 64, 512]:
    q, K = rng.normal(size=d), rng.normal(size=(10, d))
    raw = K @ q
    print(f"d = {d:3d}: std of q.k = {raw.std():5.1f}   max weight without sqrt(d): {softmax(raw).max():.2f}"
          f"   with sqrt(d): {softmax(raw / np.sqrt(d)).max():.2f}")"""),
("md", """## 파트 2. Seq2Seq + 어텐션

지난 시간의 GRU 인코더-디코더에서, 디코더의 메모 $\\mathbf{s}_k$ 를 쿼리로, 인코더의 모든 칸 $\\mathbf{h}_1, \\dots, \\mathbf{h}_T$ 를 키 · 밸류로 쓴다. 맥락 벡터 $\\sum_j \\alpha_j \\mathbf{h}_j$ 를 $\\mathbf{s}_k$ 옆에 붙여 다음 글자를 고른다 (Luong 2015 의 내적 어텐션)."""),
("code", """class Seq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, hidden, use_attention):
        super().__init__()
        self.use_attention = use_attention
        self.src_emb, self.tgt_emb = nn.Embedding(n_src, hidden), nn.Embedding(n_tgt, hidden)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden if use_attention else hidden, n_tgt)

    def attend(self, H, s):
        if not self.use_attention:
            return self.out(s), None
        scores = s @ H.transpose(1, 2)                   # (batch, output steps, input steps)
        alpha = scores.softmax(-1)                       #@ANS alpha = None  # TODO: weights over the input positions
        context = alpha @ H                              #@ANS context = None  # TODO: weighted sum of encoder states
        return self.out(torch.cat([s, context], -1)), alpha

    def forward(self, x, y_in):
        H, h = self.encoder(self.src_emb(x))
        s, _ = self.decoder(self.tgt_emb(y_in), h)
        return self.attend(H, s)[0]

    @torch.no_grad()
    def greedy(self, x, n_steps, bos=1):
        H, h = self.encoder(self.src_emb(x))
        t = torch.full((x.shape[0], 1), bos); outs, alphas = [], []
        for _ in range(n_steps):
            s, h = self.decoder(self.tgt_emb(t), h)
            logits, alpha = self.attend(H, s)
            t = logits.argmax(-1); outs.append(t); alphas.append(alpha)
        return torch.cat(outs, 1), (torch.cat(alphas, 1) if self.use_attention else None)"""),
("code", """def train_copy(use_attention, hidden=32, steps=4000):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    model = Seq2Seq(12, 13, hidden, use_attention)       # input: 0 pad, 1..10 digits / output: 0 pad, 1 <s>, 2 </s>, 3..12 digits
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for step in range(steps):
        n = int(torch.randint(5, 31, (1,), generator=g))
        d = torch.randint(0, 10, (64, n), generator=g)
        x, y = d + 1, d + 3
        y_in = torch.cat([torch.ones(64, 1, dtype=torch.long), y], 1)[:, :-1]
        loss = nn.functional.cross_entropy(model(x, y_in).reshape(-1, 13), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model

def copy_accuracy(model, lengths):
    acc = []
    for n in lengths:
        d = torch.randint(0, 10, (500, n), generator=torch.Generator().manual_seed(n))
        pred, _ = model.greedy(d + 1, n)
        acc.append(((pred - 3) == d).float().mean().item() * 100)
    return acc

lengths = [5, 10, 15, 20, 25, 30, 40, 50]                 # trained on 5 ~ 30 only
models = {}
for use_attention in [False, True]:
    name = "attention" if use_attention else "plain"
    models[name] = train_copy(use_attention)
    acc = copy_accuracy(models[name], lengths)
    print(f"{name:9s}", [round(a) for a in acc])
    plt.plot(lengths, acc, "o-", label=name)
plt.axvspan(32, 52, color="0.93"); plt.axhline(10, ls=":", color="gray")
plt.xlabel("sequence length"); plt.ylabel("digits correct (%)"); plt.legend(); plt.show()"""),
("md", """## 파트 3. 어텐션 지도

길이 12 숫자열 하나를 베끼며, 출력 숫자 (행) 가 입력 숫자 (열) 를 보는 비율 $\\alpha$ 를 그린다."""),
("code", """def plot_attention_map(model, digits):
    x = torch.tensor([digits]) + 1
    pred, alpha = model.greedy(x, len(digits))
    plt.imshow(alpha[0].numpy(), cmap="YlOrBr", vmin=0, vmax=1)
    plt.xticks(range(len(digits)), digits); plt.yticks(range(len(digits)), (pred[0] - 3).tolist())
    plt.xlabel("input digit (key)"); plt.ylabel("output digit (query)"); plt.show()
    return alpha[0].argmax(-1).tolist()

print("most-attended input position per output:", plot_attention_map(models["attention"], [1, 9, 1, 0, 1, 8, 2, 8, 5, 0, 9, 5]))"""),
("md", """**확인 질문**
1. 파트 1 에서 "it" 벡터를 `[0.3, 2.8, 0]` 으로 바꾸면 비율은 어떻게 바뀌나?
2. 파트 1 의 $\\sqrt{d}$ 실험에서, 나누지 않을 때 최대 비율이 1 에 가까워지면 학습에 왜 나쁜가?
3. 파트 2 에서 학습 때 없던 길이 40 · 50 에서 두 모델은 어떻게 다른가? 왜 그럴까?
4. (도전) 입력을 거꾸로 넣어 (`d.flip(1) + 1`) 다시 학습하면 어텐션 지도는 어떤 모양이 될까?"""),
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
