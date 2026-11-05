"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
데이터: text8 (Matt Mahoney, 처음 실행 때 31MB 내려받기). 여기서는 data/text8 → ../assets/text8 링크 (git 제외).
슬라이드 실험 exp_w2v.py 는 1,700만 단어 전부 · 약 2분, 노트북은 앞 800만 단어 · 1바퀴."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (word2vec) 이 파일명 앞에 붙는다

CELLS = [
("md", """# word2vec 실습

함께 나오는 단어를 맞히는 연습 (Skip-gram + 네거티브 샘플링) 으로 단어 벡터를 배운다.

- **파트 1**: 창에서 (가운데, 주변) 쌍 만들기 · 손 계산 손실
- **파트 2**: PyTorch 로 Skip-gram 직접 학습 (text8 앞 800만 단어)
- **파트 3**: 가까운 단어 · 비유 풀기

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요. 파트 2 는 1 ~ 3분 걸린다."""),
("code", """import os, collections, zipfile, urllib.request
import numpy as np
import torch, torch.nn as nn"""),
("md", """## 파트 1. 쌍 만들기 · 손 계산"""),
("code", """def make_pairs(tokens, window=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j != i:
                pairs.append((center, tokens[j]))          #@ANS pass  # TODO: add the (center, context) pair
    return pairs

sentence = ["cute", "little", "cat", "sleeps", "inside"]
pairs = make_pairs(sentence)
print(len(pairs), "pairs:", pairs[:6], "...")

sigmoid = lambda z: 1 / (1 + np.exp(-z))
v_cat, u_sleeps, u_car = np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([0, -1, 1])
loss = -np.log(sigmoid(v_cat @ u_sleeps)) - np.log(sigmoid(-v_cat @ u_car))   #@ANS loss = None  # TODO: positive pair term + negative term
print("loss:", round(loss, 2))"""),
("md", """## 파트 2. Skip-gram 학습

5번 이상 나온 단어만, 흔한 단어는 솎아 내고 ($t = 10^{-4}$), 창은 앞뒤 최대 5칸, 네거티브 5개, 100칸."""),
("code", """os.makedirs("data", exist_ok=True)
if not os.path.exists("data/text8"):
    urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", "data/text8.zip")
    zipfile.ZipFile("data/text8.zip").extractall("data")
words = open("data/text8").read().split()[:8_000_000]
count = collections.Counter(words)
vocab = [w for w, c in count.most_common() if c >= 5]
index = {w: i for i, w in enumerate(vocab)}
freq = np.array([count[w] for w in vocab], float); freq /= freq.sum()
rng = np.random.default_rng(0)
ids = np.array([index[w] for w in words if w in index])
keep = rng.random(len(ids)) < np.minimum(1, np.sqrt(1e-4 / freq[ids]) + 1e-4 / freq[ids])
ids = ids[keep]
print(len(words), "words,", len(vocab), "vocab,", len(ids), "after subsampling")"""),
("code", """def skipgram_pairs(ids, window, rng):
    centers, contexts = [], []
    size = rng.integers(1, window + 1, len(ids))        # random window size per position
    for off in range(1, window + 1):
        pos = np.arange(len(ids) - off)
        pos = pos[size[pos] >= off]
        centers += [ids[pos], ids[pos + off]]; contexts += [ids[pos + off], ids[pos]]
    c, o = np.concatenate(centers), np.concatenate(contexts)
    p = rng.permutation(len(c))
    return torch.tensor(c[p]), torch.tensor(o[p])

def train_skipgram(centers, contexts, n_vocab, freq, dim=100, neg=5, batch=4096, lr=0.003):
    torch.manual_seed(0)
    v = nn.Embedding(n_vocab, dim, sparse=True); u = nn.Embedding(n_vocab, dim, sparse=True)
    nn.init.uniform_(v.weight, -0.5 / dim, 0.5 / dim); nn.init.zeros_(u.weight)
    opt = torch.optim.SparseAdam(list(v.parameters()) + list(u.parameters()), lr=lr)
    noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float)
    for step, s in enumerate(range(0, len(centers), batch)):
        c, o = centers[s:s + batch], contexts[s:s + batch]
        n = torch.multinomial(noise, len(c) * neg, replacement=True).view(len(c), neg)
        vc = v(c)
        pos = (vc * u(o)).sum(1)                          # dot product with the true context
        negs = torch.bmm(u(n), vc[:, :, None]).squeeze(2) # dot products with the negatives
        loss = -(nn.functional.logsigmoid(pos).mean() + nn.functional.logsigmoid(-negs).sum(1).mean())   #@ANS loss = None  # TODO: negative-sampling loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.2f}")
    return v.weight.detach().numpy()

centers, contexts = skipgram_pairs(ids, 5, rng)
print(len(centers), "pairs")
E = train_skipgram(centers, contexts, len(vocab), freq)
E = E / np.linalg.norm(E, axis=1, keepdims=True)"""),
("md", """## 파트 3. 가까운 단어 · 비유"""),
("code", """def nearest(word, E, vocab, index, k=5):
    sims = E @ E[index[word]]
    best = np.argsort(-sims)[1:k + 1]
    return [(vocab[j], round(float(sims[j]), 2)) for j in best]

def analogy(a, b, c, E, vocab, index, k=5):
    q = E[index[b]] - E[index[a]] + E[index[c]]        #@ANS q = None  # TODO: b - a + c
    sims = E @ q / np.linalg.norm(q)
    sims[[index[a], index[b], index[c]]] = -9            # do not answer with the question words
    return [(vocab[j], round(float(sims[j]), 2)) for j in np.argsort(-sims)[:k]]

for w in ["cat", "computer", "france", "king"]:
    print(f"{w:9s}", nearest(w, E, vocab, index))
for a, b, c in [("france", "paris", "germany"), ("man", "he", "woman"), ("man", "king", "woman"), ("big", "bigger", "small")]:
    print(f"{b} - {a} + {c} =", analogy(a, b, c, E, vocab, index))"""),
("md", """**확인 질문**
1. 파트 1 의 5단어 문장에서 창을 1칸으로 하면 쌍은 몇 개인가?
2. 파트 3 의 결과를 슬라이드 (1,700만 단어 전부) 와 비교하면? 데이터가 적으면 어떤 비유부터 틀릴까?
3. `analogy` 에서 질문 단어 (a, b, c) 를 빼지 않으면 어떤 답이 나올까?
4. (도전) `words` 를 1,700만 단어 전부로 늘리거나 `dim` 을 200 으로 바꿔 king − man + woman 에서 queen 의 순위를 비교해 보라.

데이터 출처: text8, Matt Mahoney (영어 위키백과 2006년 덤프 앞 1억 글자)."""),
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
