"""word2vec (Skip-gram + 네거티브 샘플링) 을 PyTorch 로 직접 학습 → assets/w2v.npz
데이터: text8 (위키백과 영어 앞 1억 글자, 단어 약 1,700만 개, Matt Mahoney). assets/text8 (git 에 올리지 않음, .gitignore).
설정: 5번 이상 나온 단어, 흔한 단어 솎아 내기 (t = 1e-4), 창 앞뒤 최대 5칸, 네거티브 5개, 100차원, 1바퀴, SparseAdam 0.003, 배치 4,096.
저장: 단어 목록 · 벡터 (자주 나온 3만 단어만) · 손실 곡선."""
import os, time, collections, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); rng = np.random.default_rng(0)
words = open(os.path.join(HERE, "assets", "text8")).read().split()
cnt = collections.Counter(words); vocab = [w for w, c in cnt.most_common() if c >= 5]; idx = {w: i for i, w in enumerate(vocab)}
ids = np.array([idx[w] for w in words if w in idx]); freq = np.array([cnt[w] for w in vocab], float); f = freq / freq.sum()
keep = rng.random(len(ids)) < np.minimum(1, np.sqrt(1e-4 / f[ids]) + 1e-4 / f[ids]); ids = ids[keep]
print(len(words), "words,", len(vocab), "vocab,", len(ids), "after subsampling", flush=True)
noise = torch.tensor(f ** 0.75 / (f ** 0.75).sum(), dtype=torch.float)
V, D, W, NEG, B = len(vocab), 100, 5, 5, 4096
emb_in, emb_out = nn.Embedding(V, D, sparse=True), nn.Embedding(V, D, sparse=True)
nn.init.uniform_(emb_in.weight, -0.5 / D, 0.5 / D); nn.init.zeros_(emb_out.weight)
opt = torch.optim.SparseAdam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.003)
# pairs: dynamic window
cen, ctx = [], []
for off in range(1, W + 1):
    m = rng.integers(1, W + 1, len(ids)) >= off                 # window size per position >= offset
    a = np.arange(len(ids) - off)
    sel = a[m[:len(ids) - off]]
    cen += [ids[sel], ids[sel + off]]; ctx += [ids[sel + off], ids[sel]]
cen, ctx = np.concatenate(cen), np.concatenate(ctx); perm = rng.permutation(len(cen)); cen, ctx = torch.tensor(cen[perm]), torch.tensor(ctx[perm])
print(len(cen), "pairs", flush=True)
L = []; t0 = time.time()
for s in range(0, len(cen), B):
    c, o = cen[s:s + B], ctx[s:s + B]; neg = torch.multinomial(noise, len(c) * NEG, replacement=True).view(len(c), NEG)
    vc = emb_in(c); pos = (vc * emb_out(o)).sum(1); ng = torch.bmm(emb_out(neg), vc[:, :, None]).squeeze(2)
    loss = -(nn.functional.logsigmoid(pos).mean() + nn.functional.logsigmoid(-ng).sum(1).mean())
    opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
    if (s // B) % 2000 == 0: print(s // B, round(loss.item(), 3), round(time.time() - t0), flush=True)
E = emb_in.weight.detach().numpy()[:30000]
np.savez_compressed(os.path.join(HERE, "assets", "w2v.npz"), vocab=np.array(vocab[:30000]), E=E, loss=np.array(L[::50]), n_words=len(words), n_vocab=V, n_pairs=len(cen))
