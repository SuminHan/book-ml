# Chapter 12. 어텐션과 트랜스포머 (Attention & Transformer)

2017년, 구글 브레인의 연구자들은 논문 제목을 다소 도발적으로 지었다 —
**"Attention Is All You Need"**. 그때까지 시퀀스를 다루는 최고 성능의
모델들은 모두 RNN(또는 그 변형인 LSTM)을 기반으로 했는데, 이 논문은 순환
구조를 완전히 걷어내고 **어텐션**(attention) 메커니즘만으로 RNN보다 더
좋은 성능을 냈다. 이 구조가 **트랜스포머**(Transformer)이며, 지금 우리가
쓰는 거의 모든 대규모 언어모델(LLM)의 근간이다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [12.1 어텐션 아이디어와 Query·Key·Value](chapter12/1.md)
- [12.2 Scaled Dot-Product Attention과 Self-Attention](chapter12/2.md)
- [12.3 Multi-Head Attention과 Positional Encoding](chapter12/3.md)
