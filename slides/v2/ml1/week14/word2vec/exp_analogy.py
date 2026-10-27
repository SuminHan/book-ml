"""학습한 벡터 (assets/w2v.npz, 자주 나온 3만 단어) 로 Google 비유 문제 (Mikolov 2013, 19,544문제) 풀기 → assets/analogy.json
a : b = c : ?  →  b − a + c 와 코사인 유사도가 가장 큰 단어 (a, b, c 제외). 네 단어가 모두 3만 단어 안에 있는 문제만.
+ 슬라이드 예시들의 1 ~ 5위, 이웃 단어."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "assets", "w2v.npz")); vocab = [str(w) for w in Z["vocab"]]; ix = {w: i for i, w in enumerate(vocab)}
E = Z["E"] / np.linalg.norm(Z["E"], axis=1, keepdims=True)

def analogy(a, b, c, k=5):
    q = E[ix[b]] - E[ix[a]] + E[ix[c]]; s = E @ q; s[[ix[a], ix[b], ix[c]]] = -9
    o = np.argsort(-s)[:k]; return [(vocab[j], round(float(s[j] / np.linalg.norm(q)), 2)) for j in o]

def neighbors(w, k=5):
    s = E @ E[ix[w]]; o = np.argsort(-s)[1:k + 1]; return [(vocab[j], round(float(s[j]), 2)) for j in o]

res = {}; sec = None
for line in open(os.path.join(HERE, "assets", "questions-words.txt")):
    if line.startswith(":"): sec = line[2:].strip(); res[sec] = [0, 0, 0]; continue
    a, b, c, d = line.lower().split(); res[sec][2] += 1
    if all(w in ix for w in (a, b, c, d)):
        res[sec][1] += 1; res[sec][0] += analogy(a, b, c, 1)[0][0] == d
sem = [k for k in res if not k.startswith("gram")]; syn = [k for k in res if k.startswith("gram")]
tot = lambda ks, i: sum(res[k][i] for k in ks)
out = dict(sections=res, sem=(tot(sem, 0), tot(sem, 1), tot(sem, 2)), syn=(tot(syn, 0), tot(syn, 1), tot(syn, 2)),
           examples={" ".join(t): analogy(*t) for t in [("man", "king", "woman"), ("france", "paris", "germany"), ("japan", "tokyo", "korea"),
                                                       ("man", "he", "woman"), ("big", "bigger", "small"), ("walking", "walked", "swimming")]},
           neighbors={w: neighbors(w) for w in ("cat", "dog", "computer", "apple", "korea", "france", "king", "queen")})
print(out["sem"], out["syn"]); print(json.dumps(out["examples"], ensure_ascii=False)[:800])
json.dump(out, open(os.path.join(HERE, "assets", "analogy.json"), "w"), indent=1)
