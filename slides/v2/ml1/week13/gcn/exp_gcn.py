"""카라테 클럽 GCN 층 수 실험 → assets/layers.json
쇼츠 lessons/ml1/gcn/src/train_gcn.py 와 같은 설정 (특징 = 단위행렬, 이름표 0번 · 33번, 은닉 16, Adam 0.01, weight decay 5e-4, 200번),
층 수만 바꾼다: 0층 (그래프 안 씀, Linear 만) · 1 · 2 · 3 · 4 · 6 · 8층. 시드 10개, 나머지 32명 중 정답 수.
'층 수 = 몇 걸음 떨어진 친구까지' 를 보이기 위해 0번 · 33번에서의 최단 거리별 사람 수도 저장."""
import os, json, numpy as np, networkx as nx, torch
HERE = os.path.dirname(os.path.abspath(__file__))
G = nx.karate_club_graph(); n = G.number_of_nodes()
A = nx.to_numpy_array(G, weight=None) + np.eye(n); d = A.sum(1)
Ah = torch.tensor(A / np.sqrt(np.outer(d, d)), dtype=torch.float32)
y = torch.tensor([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)]); X = torch.eye(n)
lab = torch.tensor([0, 33]); rest = [i for i in range(n) if i not in (0, 33)]

def run(layers, seed):
    torch.manual_seed(seed)
    dims = [n] + [16] * (layers - 1) + [2] if layers else [n, 2]
    Ws = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
    opt = torch.optim.Adam(Ws.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        h = X
        for k, W in enumerate(Ws):
            h = W(h) if layers == 0 else Ah @ W(h)
            if k < len(Ws) - 1: h = torch.relu(h)
        loss = torch.nn.functional.cross_entropy(h[lab], y[lab]); opt.zero_grad(); loss.backward(); opt.step()
    return int((h.argmax(1)[rest] == y[rest]).sum())

out = {}
for L in (0, 1, 2, 3, 4, 6, 8):
    r = [run(L, s) for s in range(10)]; out[L] = r; print(L, r, np.mean(r), flush=True)
dist0 = nx.single_source_shortest_path_length(G, 0); dist33 = nx.single_source_shortest_path_length(G, 33)
hops = {i: min(dist0[i], dist33[i]) for i in range(n)}
json.dump(dict(layers=out, hops=hops, n_edges=G.number_of_edges(), degree=[int(G.degree(i)) for i in range(n)]), open(os.path.join(HERE, "assets", "layers.json"), "w"))
