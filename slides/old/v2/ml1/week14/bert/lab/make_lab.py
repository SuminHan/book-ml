"""실습 노트북 원본: 이 파일을 고치고 실행하면 <주제>.ipynb(학생용) / <주제>_solution.ipynb(정답) 이 다시 만들어진다.
   ~/Documents/manim_shorts/_env/v/bin/python make_lab.py
코드 줄 끝의 '#@ANS 빈칸버전' = solution 노트북엔 앞부분, student 노트북엔 빈칸버전이 들어간다.
모델: Hugging Face bert-base-uncased (처음 실행 때 약 440MB 내려받기). 슬라이드 실험 exp_bert.py 와 같은 문장."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/book-ml/lessons/_tools"))
import lab

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.basename(os.path.dirname(HERE))          # 폴더 이름 (bert) 이 파일명 앞에 붙는다

CELLS = [
("md", """# BERT 빈칸 채우기 실습

트랜스포머 인코더가 빈칸의 앞뒤를 모두 보고 원래 단어를 맞힌다.

- **파트 1**: 15% 가리기 (80% [MASK] · 10% 무작위 · 10% 그대로)
- **파트 2**: 실제 BERT 로 빈칸 채우기
- **파트 3**: 문맥 벡터 ("bank") · 가중치 수

`None` 으로 비워 둔 `TODO` 줄을 채우고 셀을 차례로 실행하세요."""),
("code", """import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM"""),
("md", """## 파트 1. 가리기"""),
("code", """def mask_tokens(tokens, vocab, rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    out, targets = list(tokens), {}
    for i, t in enumerate(tokens):
        if rng.random() < rate:                          # choose 15% of the positions
            targets[i] = t
            r = rng.random()
            if r < 0.8:
                out[i] = "[MASK]"                        #@ANS out[i] = None  # TODO: 80%: replace with [MASK]
            elif r < 0.9:
                out[i] = rng.choice(vocab)               # 10%: a random word
            # else 10%: keep the original word
    return out, targets

sentence = "the cat sat on the mat and slept because it was a very long and tiring day for the little cat".split()
vocab = ["apple", "run", "blue", "music", "river", "seven"]
masked, targets = mask_tokens(sentence, vocab)
print(" ".join(masked)); print("to predict:", targets)

# On many sentences the split is close to 15% / 80-10-10
picked = masked_n = 0
for s in range(2000):
    m, tg = mask_tokens(sentence, vocab, seed=s)
    picked += len(tg); masked_n += sum(m[i] == "[MASK]" for i in tg)
print(f"picked {picked / (2000 * len(sentence)):.1%} of tokens, {masked_n / picked:.1%} of them became [MASK]")"""),
("md", """## 파트 2. 실제 BERT"""),
("code", """tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()

def fill_mask(sentence, tok, model, k=3):
    t = tok(sentence, return_tensors="pt")
    pos = (t["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    with torch.no_grad():
        probs = model(**t).logits[0, pos].softmax(-1)    #@ANS probs = None  # TODO: softmax over the vocabulary at the [MASK] position
    top = probs.topk(k)
    return [(tok.decode([int(i)]).strip(), round(float(p) * 100, 1)) for p, i in zip(top.values, top.indices)]

for s in ["I went to the [MASK] to buy some bread.", "I went to the [MASK] to see a doctor.",
          "The capital of France is [MASK].", "She is a [MASK] at the hospital.", "He is a [MASK] at the hospital."]:
    print(f"{s:42s}", fill_mask(s, tok, mlm))"""),
("md", """## 파트 3. 문맥 벡터 · 가중치 수"""),
("code", """def word_vector(sentence, word, tok, model):
    t = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        h = model.bert(**t).last_hidden_state[0]         # one 768-number vector per token
    k = tok.convert_ids_to_tokens(t["input_ids"][0]).index(word)
    v = h[k].numpy()
    return v / np.linalg.norm(v)

sents = ["I deposited money at the bank.", "The bank approved my loan.",
         "We sat on the bank of the river.", "The river bank was covered with grass."]
V = np.array([word_vector(s, "bank", tok, mlm) for s in sents])
print((V @ V.T).round(2))"""),
("code", """bert = mlm.bert
count = lambda m: sum(p.numel() for p in m.parameters())
print("word table :", f"{bert.embeddings.word_embeddings.weight.numel():,}")
print("one block  :", f"{count(bert.encoder.layer[0]):,}", "  12 d^2 =", f"{12 * 768 ** 2:,}")
print("total      :", f"{count(bert):,}")"""),
("md", """**확인 질문**
1. 파트 1 에서 고른 토큰을 전부 [MASK] 로 바꾸지 않는 이유는?
2. 파트 2 에 여러분의 문장을 넣어 보라. 앞만 보면 맞히기 어렵고 뒤를 봐야 맞힐 수 있는 문장을 만들어 보라.
3. 파트 3 에서 은행끼리 · 강둑끼리 유사도가 더 높은 이유는? word2vec 이라면?
4. (도전) "She is a [MASK] at the hospital." 의 편향을 줄이려면 어떤 방법이 있을까?

모델: J. Devlin 외 (2018) BERT, Hugging Face `bert-base-uncased`."""),
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
