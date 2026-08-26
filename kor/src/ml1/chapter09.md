# Chapter 9. 신경망 기초, 역전파, 학습 기법 (Neural Network Basics, Backpropagation & Training Techniques)

1969년, 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)는 책
*Perceptrons*[^minskypapert]에서 단층 퍼셉트론(single-layer perceptron)이 **XOR** 하나조차
풀 수 없다는 것을 수학적으로 증명했다. XOR은 "두 입력이 서로 다르면 1, 같으면
0"이라는 아주 간단한 규칙이지만, 그 경계는 직선 하나로 그을 수 없다 — 지금까지
배운 로지스틱회귀(직선 하나로 나누는 모델)로는 근본적으로 풀리지 않는
문제였다. 이 결과는 신경망 연구에 대한 투자를 크게 위축시켰고, 이후 십수 년간
이어진 소위 "AI 겨울(AI winter)"의 한 원인이 됐다. 이번 장은 이 문제를 어떻게
풀었는지(은닉층 + 역전파)와, 그렇게 층을 깊게 쌓았을 때 새로 생기는 문제
(그래디언트 소실)를 어떻게 다스리는지를 한 흐름으로 다룬다.

Block A가 막 끝난 지금, 지난 6개 장에서 배운 회귀·생성·거리·마진·정규화·트리
모델들을 "정리하고 비교하는" 시간을 가졌다. 그 모델들은 하나같이 "학습할
가중치가 적거나, 구조가 사람 손으로 고르는 쪽"이었는데, 이번 장부터는 **가중치
를 데이터가 스스로 찾아내는** 모델 — 신경망 — 으로 넘어간다. 이제 Block B의
시작인 이번 장부터는 CNN(Chapter 10), RNN(Chapter 11), 트랜스포머(Chapter
12), LLM(Chapter 17)까지 이어지는 신경망 장의 서막이다. 이번 장에서 세우는
역전파와 활성함수, 초기화, 정규화, 학습률 스케줄링은 그 뒤의 모든 장에서 다시
쓰이는 도구들이므로, 이번 장이 흔들리면 뒷장이 흔들린다. 그리고 여기서는
데이터 형태에 구애받지 않는 가장 일반적인 형태(MLP)의 신경망을 배우지만, 다음
장(CNN)에서는 이미지의 특성에 맞는 구조(합성곱층)를 이 위에 올려놓을 것이다 —
먼저 모든 망에 공통인 "일반부"를 익히고, 특정 데이터에 맞는 "특수부"로 나아가
는 순서다.

**이 챕터를 마치면:**

- **XOR이 직선 하나로 안 풀리는 이유를 설명하고, 은닉층이 그걸 어떻게 푸는지
  보여줄 수 있다** — 부등식 모순으로 단층 퍼셉트론의 한계를 보여주고,
  은닉층이 "직선으로 분리되는 새로운 공간"을 만드는 과정을 구체적으로 확인
  한다.
- **역전파를 연쇄법칙으로부터 유도하고, 구현을 검증할 수 있다** — 오차 벡터
  δ가 출력층에서 은닉층으로 전파되며 각 층의 그래디언트를 만드는 역전파
  알고리즘을 손으로 유도하고, 손계산과 수치 미분으로 구현이 맞는지 확인
  한다.
- **그래디언트 소실·폭발을 진단하고, 그에 맞는 활성함수와 초기화를 고를
  수 있다** — 층이 깊어질수록 연쇄법칙의 곱이 어떻게 지수적으로 작아지거나
  커지는지 설명하고, 문제(활성함수, 층 크기)에 맞는 선택(ReLU 계열,
  Xavier/He)을 정당화한다.[^he][^xavier]
- **정규화와 학습률 스케줄링을 목적에 맞게 적용할 수 있다** — dropout[^dropout],
  BatchNorm[^batchnorm], 가중치 감쇠를 과적합·불안정한 학습 상황에 맞게 쓰고, "초반에
  크게, 후반에 작게"라는 학습률 스케줄링(StepLR, 코사인, warmup)[^sgdr]의 원리와
  초기 학습률 고르는 실무 기준을 적용한다.

세 절은 한 사슬로 이어진다: 9.1에서 역전파로 계산한 그래디언트는 "연쇄법칙의
곱"이라는 구조를 가진 값이고, 9.2는 그 곱이 깊어질수록 0으로 사라지거나(소실)
거꾸로 불어나는(폭발) 문제를 다루며, 9.3은 그래디언트를 살린 뒤 학습을 실제로
안정적으로 끝내는 도구들(정규화, 학습률 스케줄링)을 다룬다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [9.1 퍼셉트론의 한계, 순전파, 역전파](chapter09/1.md) — 퍼셉트론의
  역사와 XOR의 벽(민스키·페퍼트)에서 출발해, 순전파와 역전파 알고리즘
  (연쇄법칙의 재사용)을 다룬다. 2-2-1 신경망 손계산, 수치 미분 검증,
  로지스틱회귀 대비 numpy[^numpy]로 XOR을 푸는 실험까지 — 은닉층 하나면 "직선으로
  갈리는 공간"이 실제로 생긴다는 것을 확인한다.
- [9.2 그래디언트 소실·폭발, 활성함수, 초기화](chapter09/2.md) —
  연쇄법칙의 곱 형태(층마다 σ'와 가중치를 계속 곱)로 층이 깊어질수록
  그래디언트가 지수적으로 사라지거나(0.25의 10승 ≈ 10⁻⁶) 폭발하는
  원인을 보이고, 활성함수(sigmoid/tanh/ReLU 등)[^rectifier] 비교와 활성값의 스케일을
  층마다 보존하는 Xavier/He 초기화를 다룬 뒤, "깊은 망의 그래디언트를
  살리는 세 라인 방어선"으로 정리한다.
- [9.3 정규화 기법과 학습률 스케줄링](chapter09/3.md) — 신경망 전용
  정규화(dropout, BatchNorm, 가중치 감쇠)가 각각 과적합과 학습 불안정을
  어떻게 다스리는지 살펴본 뒤, 학습률 스케줄링(StepLR, 코사인, warmup)의
  "왜" — 골짜기 바닥에서 큰 스텝은 진동하고 작은 스텝만으로는 못 도착하는
  이유 — 과 초기 학습률(α₀)을 고르는 실무 기준을 PyTorch[^pytorch] 실험과 함께
  다룬다.

이 장의 정규화 도구(dropout·BatchNorm) 원 논문의 그림과, 이 절 전체를 더 깊이 다루는 자료: [^cs230].

[^he]: He, K., Zhang, X., Ren, S., Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." arXiv:1502.01852. — ReLU 네트워크를 위한 가중치 초기화 분포(He 초기화)를 제안한 논문.
[^xavier]: Glorot, X., Bengio, Y. (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." AISTATS 2010.
[^dropout]: Srivastava, N., Hinton, G. E., Krizhevsky, A., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR 15(2014) 1929-1958.
[^batchnorm]: Ioffe, S., Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." arXiv:1502.03167.
[^cs230]: 더 깊이 보려면: Stanford CS230: Deep Learning. https://cs230.stanford.edu/ — 이 절의 활성함수·초기화(Xavier/He), dropout, BatchNorm, 학습률 스케줄링 단원과 직접 겹친다.
[^sgdr]: Loshchilov, I., Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." arXiv:1608.03983. — 코사인 어닐링(코사인) + 웜 리스타트(warmup) 기반의 학습률 스케줄링을 제안한 원 논문.
[^pytorch]: Paszke, A., Gross, S., Massa, F., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." arXiv:1912.01703. — 9.3의 학습률 스케줄링 실험(및 9.1의 `.backward()` 자동 미분)이 전제로 하는 프레임워크의 원 논문.
[^numpy]: Harris, C. R. et al. (2020). "Array programming with NumPy." Nature 585, 357 (2020); arXiv:2006.10256.
[^rectifier]: Maas, A. L., Hannun, A. Y., Ng, A. Y. (2013). "Rectifier Nonlinearities Improve Neural Network Acoustic Models." Interspeech 2013 (ICML 워크숍: Machine Learning and Speech Processing). — 음성 인식 신경망에 rectifier(ReLU) 비선형성을 도입한 원 논문으로, Leaky ReLU·PReLU 등 ReLU 계열 변형의 직접적 선구작이다.
[^minskypapert]: Minsky, M., Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry in Perceptual Analysis." MIT Press. — 단층 퍼셉트론이 XOR(및 그 밖의 선형비분리 함수)를 표현할 수 없음을 수학적으로 증명한 원서로, 본문에서 언급한 1차 "AI 겨울"을 직접 촉발한 책이다.
