# word2vec

- 원본 쇼츠: [`lessons/ml1/word2vec/`](../../../../../lessons/ml1/word2vec/README.md)
- 구성: king − man + woman = ? · 원-핫의 한계 → 함께 나오는 단어 (Firth 1957) → Skip-gram · CBOW → 식 (네거티브 샘플링) → 손 계산 손실 1.63 → 실험 text8 1,700만 단어 직접 학습 (cat → dog, apple → macintosh) → 비유 paris − france + germany = berlin, king → queen 은 5위 → Google 비유 19,544문제 의미 23.3% · 문법 29.1% → 한계 (단어마다 벡터 하나 · 편향) → BERT + 문제풀이
- 상태: 13쪽. exp_w2v.py → assets/w2v.npz (약 2분), exp_analogy.py → assets/analogy.json. assets/text8 (100MB) · questions-words.txt 는 내려받기, text8 은 .gitignore
- 실습: 쌍 만들기 · 손 계산 · PyTorch Skip-gram (text8 앞 800만 단어, 약 1분) · 이웃 · 비유
