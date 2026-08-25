# Chapter 12. 어텐션과 트랜스포머 (Attention & Transformer)

2017년, 구글 브레인의 연구자들은 논문 제목을 다소 도발적으로 지었다 —
**"Attention Is All You Need"**. 그때까지 시퀀스를 다루는 최고 성능의
모델들은 모두 RNN(또는 그 변형인 LSTM)을 기반으로 했는데, 이 논문은 순환
구조를 완전히 걷어내고 **어텐션**(attention) 메커니즘만으로 RNN보다 더
좋은 성능을 냈다. 이 구조가 **트랜스포머**(Transformer)이며, 지금 우리가
쓰는 거의 모든 대규모 언어모델(LLM)의 근간이다.[^transformer]

Chapter 11에서 우리는 RNN/LSTM이 순환 구조로 "기억을 이끈다"는 것을,
그리고 그 기억이 두 가지 비용을 치른다는 것을 보았다: 병렬화가 안 되는
순차 연산, 그리고 먼 과거의 정보가 흐려지는 그래디언트 소실. 어텐션은
그 두 가지 비용에 대한 답안이다. 흥미로운 것은 어텐션의 기원이 새로운
아키텍처가 아니라 RNN에 붙인 "패치"였다는 점이다 — 2014년 기계번역
연구(Bahdanau et al.)([^bahdanau])에서 "원문 전체를 은닉 상태 한 개에 압축하는
병목"을 발견하고, "생성의 각 단계마다 원문에 다시 돌아가 적절한 비율로
쓴다"는 아이디어를 도입한 것이다. 3년 뒤 "Attention Is All You Need"의
저자들이 한 것은 그 메커니즘 하나만 남기고 RNN 본체를 통째로 걷어내는
것이었다. 즉, 이 장은 Chapter 11 서사의 귀결이며, 그 구조의
조각들(Query·Key·Value, 스케일드 닷-프로덕트 어텐션, 멀티헤드,
positional encoding)이 어떤 직관 위에서 왜 그 형태인지 하나씩 분해한다.

제목의 "All"은 무엇이 전부였던가 — 이 장의 핵심 질문이기도 하다. 핵심은
이렇다: RNN이 "과거를 담은 한 벡터(요약)"만 다음 단계로 넘기던 것을,
트랜스포머는 **모든 단어가 한 번의 행렬 곱으로 서로를 직접 보는** 단일
메커니즘으로 순환 구조 전체를 대체했다. 그리고 이 장을 마치면 다음 장의
기계도 돌아가는 방식을 이미 알게 된다 — Chapter 13에서 다룰 LLM의
자기회귀 생성(토큰을 하나씩 만들어 나가는 과정)은 12.3절의 causal mask
위에 정확히 올라가기 때문이다.

### 학습 목표

이 챕터를 마치면 다음을 할 수 있다.

- 어텐션을 "모든 단어에 대한 주의(attention) 분포"로 설명하고, Query
  ("나는 무엇을 찾고 있는가"), Key("나는 무엇을 갖고 있는가"), Value
  ("실제로 전달할 내용")의 세 역할이 왜 같은 입력에서 서로 다른 학습
  가능한 변환(\\(W\_Q, W\_K, W\_V\\))을 거쳐야 하는지 설명한다.
- 스케일드 닷-프로덕트 어텐션 \\(\text{softmax}(QK^T / \sqrt{d\_k})\\,V\\)의
  4단계를 작은 예로 손으로 계산하고, 왜 \\(\sqrt{d\_k}\\)로 나누어야 하는지
  (무작위 내적의 표준편차가 정확히 \\(\sqrt{d\_k}\\)라) 수식으로 정당화한다.
- 멀티헤드 어텐션이 왜 "하나의 점수로 여러 관계를 타협해야 하는" 단일
  헤드보다 나은지 "관점의 수" 관점에서 설명하고, 파라미터 수가 단일
  헤드와 같도록 설계되는 이유(\\(d\_k = d\_{model} / h\\))를 확인한다.
- 어텐션 연산 자체는 단어 순서를 구분하지 못한다는 것(퍼뮤테이션
  등변성)을 확인하고, 사인/코사인 positional encoding이 그 결손을
  어떻게, 그리고 "더하는" 방식으로 보완하는지 설명한다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [12.1 어텐션 아이디어와 Query·Key·Value](chapter12/1.md) —
  RNN의 순차성과 "은닉 상태 하나로 원문 전체를 압축한다"는 병목의
  문제, 2014년 기계번역에서 태어난 어텐션의 역사, 그리고 같은 입력
  임베딩에서 Q·K·V가 왜 서로 다른 세 벡터로 나와야 하는지 2단어 예로
  손으로 확인한다.
- [12.2 Scaled Dot-Product Attention과 Self-Attention](chapter12/2.md) —
  4단계(\\(QK^T\\) → 스케일링 → softmax → V 가중합)를 손으로 추적하고,
  softmax가 점수 차이를 지수적으로 증폭하기 때문에 \\(\sqrt{d\_k}\\)
  스케일링이 필수인 이유를 무작위 내적의 분산으로 풀어본 뒤, "그것은"
  이 "동물"에 56%의 주의를 주는 대명사 지시 실험으로 self-attention을
  시각화한다.
- [12.3 Multi-Head Attention과 Positional Encoding](chapter12/3.md) —
  단일 헤드가 "관점"에서 부족한 이유(미러 예로 확인), 멀티헤드 공식과
  "파라미터는 그대로, 관점은 h배" 설계, 시계(clock)로 보는
  positional encoding의 회전 성질과 "왜 더하기인가", 그리고 causal
  mask와 전체 트랜스포머 블록 조립까지 RNN 대비로 마무리한다.

이 장의 주제(어텐션과 트랜스포머)를 더 깊이 다루는 자료: [^cs224n]

[^transformer]: Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
[^bahdanau]: Bahdanau, D., Cho, K., Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv:1409.0473.
[^cs224n]: Stanford CS224N: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/
