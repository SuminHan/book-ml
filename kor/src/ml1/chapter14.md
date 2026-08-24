# Chapter 14. 표현학습: PCA, word2vec, Node2Vec, PageRank (Representation Learning)

1901년, 통계학자 칼 피어슨(Karl Pearson)은 흥미로운 질문을 던졌다: 여러 변수를
측정한 데이터가 있을 때, 정보를 최대한 보존하면서 변수의 개수를 줄일 수 있는
"가장 좋은 직선(또는 평면)"은 무엇인가? 그의 답 — 데이터가 가장 많이 퍼져 있는
방향을 찾아라 — 이 지금 **주성분분석**(Principal Component Analysis, PCA)이라
불리는 방법의 시작이다.[^cs229] 120년도 더 전에 나온 이 아이디어가, 지금은 수백
차원짜리 이미지 임베딩을 사람이 눈으로 볼 수 있는 2차원으로 줄이는 데도 그대로
쓰인다.

넓히면 이 질문이 곧 **표현학습**(representation learning)이다: 관측 데이터
속에 숨어 있는 저차원 "요약 모습"을 찾아내는 일. 이 장은 데이터의 종류를
바꾸어가며 그 주제를 세 단계로 확장한다. 이미 벡터인 데이터를 회전·압축하는
선형 도구(PCA)에서 출발해, 벡터가 아니었던 **단어**를 학습으로 벡터로 만드는
word2vec[^word2vec]으로, 그리고 벡터가 아니었던 **그래프 노드**까지 벡터로
만드는 Node2Vec[^node2vec]과 PageRank[^pagerank]까지 이른다. 바뀌는 것은
데이터의
형태(행렬 → 텍스트 → 그래프)뿐이며, "함께 자주 나타나는 것들은 가까운 벡터로
모인다"는 핵심 원칙은 처음부터 끝까지 한 줄이다.

![두 개의 새로운 모델 아키텍처 — CBOW는 컨텍스트 단어로부터 현재 단어를, Skip-gram은 현재 단어로부터 주변 단어를 예측 (원 논문 Figure 1).](../images/ref_word2vec.png)

그런데 왜 이렇게까지 차원을 줄이고 "벡터 공간"을 만들고 싶은가? 두 가지 이유가
겹친다. 첫째, 저차원 표현은 **실용**적이다: 데이터가 사람이 눈으로 볼 수 있을
만큼 작아지고, 거리 기반 모델(kNN, k-means)이 차원의 저주 없이 동작하며,
저장·추론 비용도 크게 줄어든다. 둘째, 그것은 **이해**를 가능하게 한다:
"유사한 것이 가까운 곳으로 모이는" 하나의 벡터 공간을 만들면, 그 공간 위에서
군집화·유사 검색·추론을 모두 같은 방식으로 할 수 있다. 이 장의 세 알고리즘은
모두 이 "벡터 공간 만들기"를, 서로 다른 형태의 데이터에 대해, 하는 방법이다.

**이 장이 어디에서 오고 어디로 가나.** 이전 장(Chapter 13, LLM)에서 본
"다음 토큰 예측이 지식을 부산물로 남긴다"는 이야기의 바로 앞자리에 이 장이
있다. word2vec이 하는 "중심 단어 하나로 주변 단어를 맞히는" 작은 게임을 데이터
규모와 모델 크기만 키우면 LLM의 사전학습이 되기 때문이다 — 즉 이 장은 다음 장의
LLM이 **어떻게 동작하는지**를, 그 원형이 되는 "작은 게임"에서 먼저 이해하는 장이다.[^cs224n]
"학습 목표는 수단일 뿐, 그 과정에서 나오는 임베딩이 진짜 목표물"이라는 패턴도
여기서 처음 등장한다. 반대로 다음 장(Chapter 15, 잠재변수 생성모델)은 이 장을
뒤집어본다: 이 장이 관측 데이터 속에 숨은 저차원 구조를 **찾는** 쪽이라면, 15장은
그 구조(잠재 공간)가 데이터를 어떻게 **만들어내는가**를 학습하는 생성 모델
(EM/GMM → VAE[^vae] → GAN[^gan]·Diffusion[^ddpm])로 넘어간다. 이 장에서 만든 임베딩 공간은 곧
15장의 "잠재 공간"이 첫 모습을 드러내는 자리다.

![VAE의 방향적 그래프 모델 (원 논문 Figure 1) — 데이터 𝐱, 잠재 변수 𝐳, 파라미터 𝜃 노드와 생성 모델 p_𝜃(𝐳), p_𝜃(𝐱|𝐳) 및 인식 모델 q_𝜙(𝐳|𝐱)의 구조.](../images/ref_vae.png)

![Figure 1: GAN이 학습되면서 샘플 품질이 개선되는 과정을 MNIST로 보여준다 — 판별기 D와 생성기 G가 동시에 갱신되며 (a)에서 (d)로 갈수록 생성 샘플이 실제 데이터에 가까워진다.](../images/ref_gan.png)

**이 장을 마치면:**

- 공분산 행렬의 고유값 문제를 이용해 "분산을 최대화하라"는 문장이 곧 "가장 큰
  고유벡터를 찾아라"는 말과 같다는 것을 라그랑주 승수법으로 유도하고 설명할 수
  있다.
- bag-of-words·one-hot의 한계를 짚고, skip-gram과 네거티브 샘플링이 단어
  벡터를 어떻게 만드는지, "왕 − 남자 + 여자 ≈ 여왕"[^word2vec2013b]이 왜 성립하는지
  설명할 수 있다.
- 그래프 노드를 무작위 걷기 시퀀스로 바꿔 word2vec에 그대로 돌리는
  Node2Vec의 원리와 p·q 계수의 역할을 설명할 수 있다.
- PageRank가 무작위 걷기의 "장기 체류 확률"이라는 재귀식임을 알고,
  Node2Vec의 무작위 걷기와 같은 뿌리임을 연결해 설명할 수 있다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [14.1 PCA: 분산을 최대로 보존하는 축 찾기](chapter14/1.md) — 비지도학습의
  관점에서 차원축소의 필요성(차원의 저주·비용·시각화)을 짚고, 중심화 →
  공분산 행렬 → 고유값/고유벡터로 주성분을 찾는 절차를 다룬다. 4개 점의
  손계산과 붓꽃 실습에서 버린 성분의 고유값이 재구성 오차와 정확히 같음을
  확인한다.
- [14.2 word2vec: 단어를 벡터로](chapter14/2.md) — bag-of-words·LSI부터
  word2vec까지의 흐름을 지나 skip-gram의 구조와 네거티브 샘플링을 다루고,
  학습된 벡터의 기하학에서 "왕 − 남자 + 여자 ≈ 여왕" 같은 선형 관계를
  들여다본다.
- [14.3 Node2Vec과 PageRank](chapter14/3.md) — 무작위 걷기로 "문장"을 만들어
  노드 임베딩을 학습하는 Node2Vec과 p·q 계수를 다루고, 같은 무작위 걷기를
  "중요도"라는 다른 질문에 쓴 PageRank의 재귀식·댐핑 팩터를 짚는다. Zachary
  카라테 클럽에서 라벨 없이도 두 파벌과 허브 노드가 회복되는 모습을
  확인한다.

[^word2vec]: Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
[^node2vec]: Grover, A., Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks." KDD 2016. arXiv:1607.00653.
[^cs224n]: Stanford CS224N: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/
[^vae]: Kingma, D. P., Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
[^gan]: Goodfellow, I. J. et al. (2014). "Generative Adversarial Networks." arXiv:1406.2661.
[^ddpm]: Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." arXiv:2006.11239.
[^cs229]: 더 깊이 보려면: Stanford CS229: Machine Learning, Lecture Notes. https://cs229.stanford.edu/main_notes.pdf — 이 장의 차원 축소(주성분분석, PCA)와 비지도 학습은 CS229 강의 노트의 PCA·차원 축소 단원과 직접 겹친다.
[^word2vec2013b]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). "Distributed Representations of Words and Phrases and their Compositionality." NeurIPS 2013. arXiv:1310.4546 — 단어 유사도/유추(analogy) 분석("왕 − 남자 + 여자 ≈ 여왕")을 본격적으로 도입한 논문.
[^pagerank]: Brin, S., Page, L. (1998). "The Anatomy of a Large-Scale Hypertextual Web Search Engine." WWW7 (The 7th International World Wide Web Conference). https://web.stanford.edu/pub/papers/law/
