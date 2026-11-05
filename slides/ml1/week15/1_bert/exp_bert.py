"""실제 BERT (bert-base-uncased, Hugging Face 캐시) 로 재기 → assets/bert.json
1) 빈칸 채우기 확률 (쇼츠 lessons/ml1/bert 와 같은 문장 + 더)
2) 같은 단어 'bank' 의 문맥별 벡터 (마지막 층) 코사인 유사도: 돈 은행 2문장 vs 강둑 2문장
3) 가중치 수 (부분별)."""
import os, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", local_files_only=True).eval()
out = {"fill": {}}
for s in ["I went to the [MASK] to buy some bread.", "I went to the [MASK] to see a doctor.", "The capital of France is [MASK].",
          "The capital of Korea is [MASK].", "She is a [MASK] at the hospital.", "He is a [MASK] at the hospital."]:
    t = tok(s, return_tensors="pt"); i = (t["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    with torch.no_grad(): p = mlm(**t).logits[0, i].softmax(-1)
    top = p.topk(5); out["fill"][s] = [(tok.decode([int(j)]).strip(), float(v)) for v, j in zip(top.values, top.indices)]
    print(s, [(w, round(v * 100, 1)) for w, v in out["fill"][s][:3]])
BANK = ["I deposited money at the bank.", "The bank approved my loan.", "We sat on the bank of the river.", "The river bank was covered with grass."]
vecs = []
for s in BANK:
    t = tok(s, return_tensors="pt")
    with torch.no_grad(): h = mlm.bert(**t).last_hidden_state[0]
    k = tok.convert_ids_to_tokens(t["input_ids"][0]).index("bank"); vecs.append(h[k].numpy())
V = np.array(vecs); V = V / np.linalg.norm(V, axis=1, keepdims=True); out["bank_sents"] = BANK; out["bank_sim"] = (V @ V.T).tolist()
print(np.round(V @ V.T, 2))
b = mlm.bert; n = lambda m: sum(p.numel() for p in m.parameters())
out["params"] = dict(total=n(b), word_emb=b.embeddings.word_embeddings.weight.numel(), pos_emb=b.embeddings.position_embeddings.weight.numel(),
                     layer=n(b.encoder.layer[0]), layers=len(b.encoder.layer), vocab=tok.vocab_size)
print(out["params"])
json.dump(out, open(os.path.join(HERE, "assets", "bert.json"), "w"), indent=1)
