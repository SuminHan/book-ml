# Chapter 11. 시퀀스 모델 (Sequence Models)

1990년, 인지과학자 제프리 엘먼(Jeffrey Elman)은 논문 "Finding Structure in
Time"에서 질문 하나를 던졌다: 신경망이 "지금 보고 있는 입력"뿐 아니라 "조금
전에 무엇을 봤는지"도 기억하게 만들 수 있을까? 그가 제안한 구조 — 은닉층의
출력을 다음 시점의 입력으로 다시 연결하는 순환(recurrent) 구조 — 는 지금
RNN(Recurrent Neural Network)이라 불리는 모델의 원형이다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [11.1 RNN: 은닉 상태와 순전파](chapter11/1.md)
- [11.2 BPTT와 numpy RNN 언어모델 실습](chapter11/2.md)
- [11.3 그래디언트 소실의 재발과 LSTM/GRU](chapter11/3.md)
