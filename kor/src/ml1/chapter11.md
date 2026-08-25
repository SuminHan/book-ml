# Chapter 11. 시퀀스 모델 (Sequence Models)

1990년, 인지과학자 제프리 엘먼(Jeffrey Elman)은 논문 "Finding Structure in
Time"[^elman1990]에서 질문 하나를 던졌다: 신경망이 "지금 보고 있는 입력"뿐 아니라 "조금
전에 무엇을 봤는지"도 기억하게 만들 수 있을까? 그가 제안한 구조 — 은닉층의
출력을 다음 시점의 입력으로 다시 연결하는 순환(recurrent) 구조 — 는 지금
RNN(Recurrent Neural Network)이라 불리는 모델의 원형이다. 이 "기억"을
펼쳐서 배우게 하면 다음 문자를 예측하는 언어모델이 되며(11.2), 펼쳐지는
과정에서 사라지는 그래디언트를 다루려면 게이트가 필요해지고(11.3), 결국
"순차적 압축 자체의 한계"를 극복하러 Chapter 12의 어텐션[^transformer]으로 이어진다.

![트랜스포머 전체 구조 -- 인코더(왼쪽)와 디코더(오른쪽)의 Multi-Head Attention + Feed Forward 스택 (원 논문 Figure 1).](../images/ref_transformer.png)

이 장은 Chapter 10의 "재사용" 위에서 자연스럽게 이어진다. CNN이 같은 필터를
**공간**(이미지 위 위치)에 걸쳐 재사용했다면, RNN은 같은 가중치를 **시간**
(시퀀스의 시점)에 걸쳐 재사용한다 — 둘 다 "입력이 크든 길든 파라미터 수는
일정하다"는 같은 결론에 닿는다. 그러나 그 재사용은 두 가지 대가를 동반한다.
시점을 하나씩 순서대로 계산해야 하고, 그래디언트가 시퀀스 길이에 걸쳐
반복 곱해진다. 이 두 가지가 이 장의 그래디언트 소실(11.3)[^pascanu2013]이자, Chapter 12가
"순환 구조를 걷어내고 과거를 직접 다시 본다"는 접근으로 전환하게 되는
직접적 계기다.

## 학습 목표

이 장을 마치면 다음을 할 수 있다.

- RNN의 순전파 수식(은닉 상태 갱신식)을 손으로 유도하고, 파라미터 수가
  시퀀스 길이에 무관한 이유를 설명하며, 같은 입력이 반복되면 은닉 상태가
  고정점으로 수렴하는 행동을 예측한다.
- BPTT의 재귀 구조("본 시점의 출력 오차 + 다음 시점에서 흘러온
  그래디언트"의 합)를 유도하고, numpy[^numpy]로 문자 단위 RNN 언어모델을 학습
  시키며, 왜 그래디언트를 공유 가중치에 `+=`로 *누적*해야 하고 왜
  그래디언트 클리핑이 필수인지를 설명한다.
- 그래디언트 곱적이 W_hh의 고유값이 1보다 작으면 지수적으로 0으로
  (소실), 크으면 발산(폭발)함을 숫자로 확인하고, LSTM/GRU가 "곱은
  조절 스위치, 덧셈은 통로"로 설계된 ResNet[^resnet] 스킵 연결의 시간 축 버전을
  원리 수준으로 서술한다.

![Residual building block (원 논문 Figure 2) — shortcut identity mapping과 residual mapping F(x)의 결합 구조 (F(x)+H(x))](../images/ref_resnet.png)

- 실전 상황에서 기본 RNN → GRU[^gru2014] → LSTM → Attention[^attention] 사이에서 언제 무엇을
  고를지 판단하고, 게이트가 그래디언트 소실을 "완화"할 뿐 "해결"하지
  않는 이유를 설명한다.

이 장을 관통하는 하나의 실은 **"기억과 그 감쇠**"다. 11.1에서 은닉 상태는
감쇠하며 갱신되는 요약이고, 11.2에서 그 감쇠가 그래디언트 소실이라는
이름으로 재등장하며, 11.3에서는 게이트가 감쇠를 늦추되 순차성이라는
근본 한계는 남긴다 — 그리고 그 실은 Chapter 12의 어텐션이 "순차적으로
압축하기보다 필요할 때마다 과거 전체를 직접 들여다보자"는 해답으로
이어진다. 11.2에서 "abcabc…"를 배우는 8차원 문자모델은 Chapter 13의
LLM "다음 토큰 예측"의 가장 축소된 형태 — 차이가 크기일 뿐 원리는
같다 — 고, 11.3의 "기본 RNN → GRU → LSTM → Attention" 사다리는 이 실을
실무 판단으로 직역한 것이다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [11.1 RNN: 은닉 상태와 순전파](chapter11/1.md) — "은닉 상태는 지금까지
  읽은 것의 요약"이라는 핵심 아이디어를 순전파 수식과 함께 다룬 뒤,
  스칼라 RNN의 3개 시점 순전파와 고정점 수렴을 손으로 계산한다. 누적합
  예제와 파라미터 수 계산을 통해 "w_hh가 기억의 길이를 결정한다"는
  감각과, 파라미터 공유가 CNN 필터 재사용의 시간 축 버전임을 확인한다.
- [11.2 BPTT와 numpy RNN 언어모델 실습](chapter11/2.md) — BPTT를 시점
  재귀식으로 유도하고 2시점 미니 RNN으로 숫자를 끝까지 따라간 뒤,
  numpy로 "abcabc…" 패턴을 배우는 문자 단위 언어모델을 학습시켜 LLM의
  "다음 토큰 예측"을 가장 축소된 형태로 재현한다. 그래디언트 클리핑과
  `+=` vs `=` 실수, 잘라내기 BPTT(truncated BPTT)까지 다룬다.
- [11.3 그래디언트 소실의 재발과 LSTM/GRU](chapter11/3.md) — 그래디언트
  곱적이 지수적으로 소실·폭발함을 숫자로 확인하고, LSTM/GRU의 "덧셈으로
  전달되는 셀 상태"가 왜 ResNet 스킵 연결의 시간 축 버전인지 보여
  준다. 길이 100 시퀀스의 첫 토크를 기억하는 "기억 과제" 실험으로
  게이트가 기억을 살리는 것을(GRU가 LSTM을 이기기도 한다는 점을
  포함해) 확인하고, RNN의 근본 한계인 순차성으로 마무리한다.

이 주제를 더 깊이 다루는 자료: [^cs224n].

[^elman1990]: Elman, J. L. (1990). "Finding Structure in Time." Cognitive Science 14(2), 179–211.
[^resnet]: He, K., Zhang, X., et al. (2015). "Deep Residual Learning for Image Recognition." arXiv:1512.03385.
[^gru2014]: Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP 2014. arXiv:1406.1078.
[^cs224n]: 더 깊이 보려면: Stanford CS224N: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/
[^transformer]: Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
[^pascanu2013]: Pascanu, R., Gulcehre, C., et al. (2013). "How to Construct Deep Recurrent Neural Networks." arXiv:1312.6026.
[^attention]: Bahdanau, D., Cho, K., Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015. arXiv:1409.0473.
[^numpy]: Harris, C. R. et al. (2020). "Array programming with NumPy." Nature 585, 357 (2020); arXiv:2006.10256.
