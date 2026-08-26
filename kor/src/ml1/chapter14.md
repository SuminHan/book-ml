# Chapter 14. 표현학습: PCA, word2vec, 임베딩의 활용 (Representation Learning)

1901년, 통계학자 칼 피어슨(Karl Pearson)은 흥미로운 질문을 던졌다: 여러 변수를
측정한 데이터가 있을 때, 정보를 최대한 보존하면서 변수의 개수를 줄일 수 있는
"가장 좋은 직선(또는 평면)"은 무엇인가? 그의 답 — 데이터가 가장 많이 퍼져 있는
방향을 찾아라 — 이 지금 **주성분분석**(Principal Component Analysis, PCA)이라
불리는 방법의 시작이다.[^cs229] 120년도 더 전에 나온 이 아이디어가, 지금은 수백
차원짜리 이미지 임베딩을 사람이 눈으로 볼 수 있는 2차원으로 줄이는 데도 그대로
쓰인다.

넓히면 이 질문이 곧 **표현학습**(representation learning)이다: 관측 데이터
속에 숨어 있는 저차원 "요약 모습"을 찾아내는 일. 이 장은 이미 벡터인 데이터를
회전·압축하는 선형 도구(PCA)에서 출발해, 벡터가 아니었던 **단어**를 학습으로
벡터로 만드는 word2vec[^word2vec]으로 확장한 뒤, 그렇게 만든 임베딩을 **실전
시스템에서 어떻게 쓰는지**(Transformer의 입력층, 검색·RAG)까지 이어간다.
바뀌는 것은 데이터의 형태와 쓰임새뿐이며, "함께 자주 나타나는 것들은 가까운
벡터로 모인다"는 핵심 원칙은 처음부터 끝까지 한 줄이다.

그런데 왜 이렇게까지 차원을 줄이고 "벡터 공간"을 만들고 싶은가? 두 가지 이유가
겹친다. 첫째, 저차원 표현은 **실용**적이다: 데이터가 사람이 눈으로 볼 수 있을
만큼 작아지고, 거리 기반 모델(kNN, k-means)이 차원의 저주 없이 동작하며,
저장·추론 비용도 크게 줄어든다. 둘째, 그것은 **이해**를 가능하게 한다:
"유사한 것이 가까운 곳으로 모이는" 하나의 벡터 공간을 만들면, 그 공간 위에서
군집화·유사 검색·추론을 모두 같은 방식으로 할 수 있다.

**이 장이 어디에서 오고 어디로 가나.** 13장(그래프 표현학습)에서 이미
"임베딩을 만드는 아이디어"를 그래프 위에서 한 번 봤다 — 이 장의 word2vec은
그 아이디어의 원형(텍스트 버전)이다. 순서가 뒤죽박죽인 것처럼 보일 수 있다:
13장의 Node2Vec가 이 장 14.2절의 skip-gram을 전제로 설명하기 때문이다. 그래서
13.2절은 skip-gram이 "중심 단어 하나로 주변 단어를 예측하는 작은 신경망"임을
그 자리에서 한 번 짚어 자기완결적으로 읽히도록 했다 — 14.2절에서는 그
"작은 신경망"의 실제 구조(입력·은닉·출력층의 크기와 loss)를 처음부터 다시
완전히 풀이한다. 12장에서 배운 Transformer의 입력층
자체가 학습된 임베딩 테이블이라는 것, 그리고 RAG처럼 임베딩 유사도로
문서를 검색하는 실전 응용이 이 장의 후반부(14.3)다. 다음 장(Chapter 15,
잠재변수 생성모델)은 이 장을 뒤집어본다: 이 장이 관측 데이터 속에 숨은
저차원 구조를 **찾는** 쪽이라면, 15장은 그 구조(잠재 공간)가 데이터를 어떻게
**만들어내는가**를 학습하는 생성 모델(VAE[^vae] → GAN[^gan]·Diffusion[^ddpm])로
넘어간다.

**이 장을 마치면:**

- 공분산 행렬의 고유값 문제를 이용해 "분산을 최대화하라"는 문장이 곧 "가장 큰
  고유벡터를 찾아라"는 말과 같다는 것을 라그랑주 승수법으로 유도하고 설명할 수
  있다.
- bag-of-words·one-hot의 한계를 짚고, skip-gram과 네거티브 샘플링이 단어
  벡터를 어떻게 만드는지, "왕 − 남자 + 여자 ≈ 여왕"[^word2vec2013b]이 왜 성립하는지
  설명할 수 있다.
- Transformer의 입력 임베딩층이 word2vec과 같은 아이디어를 end-to-end로
  학습한 것임을 설명하고, RAG가 "질문도 벡터, 문서도 벡터, 가까운 것을
  검색"이라는 임베딩 유사도 검색으로 동작하는 원리를 설명할 수 있다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [14.1 PCA: 분산을 최대로 보존하는 축 찾기](chapter14/1.md) — 비지도학습의
  관점에서 차원축소의 필요성(차원의 저주·비용·시각화)을 짚고, 중심화 →
  공분산 행렬 → 고유값/고유벡터로 주성분을 찾는 절차를 다룬다.
- [14.2 word2vec: 단어를 벡터로](chapter14/2.md) — bag-of-words·LSI부터
  word2vec까지의 흐름을 지나 skip-gram의 구조와 네거티브 샘플링을 다루고,
  학습된 벡터의 기하학에서 "왕 − 남자 + 여자 ≈ 여왕" 같은 선형 관계를
  들여다본다.
- [14.3 임베딩을 실전에 쓰기: Transformer 임베딩과 RAG](chapter14/3.md) —
  Transformer의 입력층이 학습된 임베딩 테이블이라는 것, 그리고 RAG가
  임베딩 유사도 검색으로 동작하는 원리를 다룬다. 코사인 유사도를 손으로
  계산해 작은 문서 집합에서 "가장 가까운 문서"를 찾아보는 미니 RAG 데모를
  실습한다.

[^word2vec]: Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
[^cs224n]: Stanford CS224N: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/
[^vae]: Kingma, D. P., Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
[^gan]: Goodfellow, I. J. et al. (2014). "Generative Adversarial Networks." arXiv:1406.2661.
[^ddpm]: Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." arXiv:2006.11239.
[^cs229]: 더 깊이 보려면: Stanford CS229: Machine Learning, Lecture Notes. https://cs229.stanford.edu/main_notes.pdf — 이 장의 차원 축소(주성분분석, PCA)와 비지도 학습은 CS229 강의 노트의 PCA·차원 축소 단원과 직접 겹친다.
[^word2vec2013b]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). "Distributed Representations of Words and Phrases and their Compositionality." NeurIPS 2013. arXiv:1310.4546 — 단어 유사도/유추(analogy) 분석("왕 − 남자 + 여자 ≈ 여왕")을 본격적으로 도입한 논문.
