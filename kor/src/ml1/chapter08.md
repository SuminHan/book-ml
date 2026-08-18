# Chapter 8. 신경망 기초와 역전파 (Neural Network Basics & Backpropagation)

1969년, 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)는 책
*Perceptrons*에서 단층 퍼셉트론(single-layer perceptron)이 **XOR** 하나조차
풀 수 없다는 것을 수학적으로 증명했다. XOR은 "두 입력이 서로 다르면 1, 같으면
0"이라는 아주 간단한 규칙이지만, 그 경계는 직선 하나로 그을 수 없다 — 지금까지
배운 로지스틱회귀(직선 하나로 나누는 모델)로는 근본적으로 풀리지 않는
문제였다. 이 결과는 신경망 연구에 대한 투자를 크게 위축시켰고, 이후 십수 년간
이어진 소위 "AI 겨울(AI winter)"의 한 원인이 됐다.

## 8.1 퍼셉트론에서 다층 신경망으로

해법은 놀랍도록 단순했다: 직선 하나로 안 되면, **직선을 여러 개 겹쳐서 새로운
공간을 만든 뒤 그 공간에서 다시 나누면 된다.** 로지스틱회귀는 사실 입력층과
출력층만 있는(은닉층이 없는) 가장 단순한 "신경망"으로 볼 수 있다:
\\(a = \sigma(w^Tx)\\). **다층 퍼셉트론**(Multi-Layer Perceptron, MLP)은 그
사이에 은닉층(hidden layer)을 하나 이상 끼워 넣는다 — 은닉층 하나만
추가해도, XOR을 완벽히 풀 수 있는 경계를 만들 수 있다.

## 8.2 순전파 (Forward Propagation)

입력 \\(x\\), 은닉층 가중치 \\(W_1, b_1\\), 출력층 가중치 \\(W_2, b_2\\)를
가진 2층 신경망(입력 → 은닉 → 출력)의 순전파:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2,
\quad a_2 = \sigma(z_2)\\]

\\(z\\)는 활성화 함수를 적용하기 전(pre-activation), \\(a\\)는 적용한
후(activation)다. 최종 출력 \\(a_2\\)가 예측값이다.

## 8.3 역전파: 연쇄법칙을 거꾸로

은닉층을 추가하고 나니 새로운 문제가 생겼다: "그 은닉층의 가중치를 어떻게
학습시키는가"다 — 출력층은 Chapter 2의 로지스틱회귀와 똑같이 그래디언트를
계산하면 되지만, 은닉층은 정답을 직접 비교할 대상이 없다. 1986년 데이비드
럼멜하트(David Rumelhart), 제프리 힌턴(Geoffrey Hinton), 로널드
윌리엄스(Ronald Williams)가 대중화한 **역전파**(backpropagation) 알고리즘이
이 문제를 풀었다: 출력층의 오차를 연쇄법칙(chain rule)으로 거꾸로 흘려보내,
은닉층의 각 가중치가 최종 오차에 얼마나 책임이 있는지를 정확히 계산한다.

손실함수 \\(L\\)이 주어졌을 때, 모든 가중치에 대한 \\(\frac{\partial
L}{\partial W}\\)를 구해야 경사하강법을 적용할 수 있다. 문제는 \\(W_1\\)이
최종 손실에 미치는 영향이 \\(z_1 \to a_1 \to z_2 \to a_2 \to L\\)이라는 긴
사슬을 거친다는 것이다. 연쇄법칙은 이 사슬을 **출력에서 입력 방향으로 거꾸로**
따라가며 미분을 하나씩 곱해 나가면 된다고 말해준다:

1. **출력층 오차**: \\(\delta_2 = \frac{\partial L}{\partial z_2} = (a_2 - y)
   \sigma'(z_2)\\) (Chapter 2에서 본 것과 같은 형태 — 교차 엔트로피와
   시그모이드의 조합은 항상 이렇게 깔끔해진다.)
2. **은닉층 오차**: \\(\delta_1 = (W_2 \delta_2) \odot \sigma'(z_1)\\) —
   출력층 오차 \\(\delta_2\\)를 \\(W_2\\)를 거슬러 은닉층으로 "돌려보낸" 뒤,
   은닉층 자체의 미분(\\(\sigma'(z_1)\\))을 곱한다. (\\(\odot\\)는 원소별 곱.)
3. **그래디언트**: \\(\frac{\partial L}{\partial W_2} = \delta_2 \cdot
   a_1^T\\), \\(\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T\\)

"역전파"라는 이름의 의미가 바로 여기 있다: 오차(\\(\delta\\))가 출력층에서
계산되고, 그 값이 은닉층 방향으로 **거꾸로** 전파(propagate)되면서 각 층의
그래디언트를 만든다.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def two_layer_forward(x, W1, b1, W2, b2):
    z1 = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(len(b1))]
    a1 = [sigmoid(v) for v in z1]
    z2 = sum(W2[i] * a1[i] for i in range(len(a1))) + b2
    a2 = sigmoid(z2)
    return a2, (x, z1, a1, z2, a2)

def two_layer_backward(y_true, cache, W2):
    x, z1, a1, z2, a2 = cache
    delta2 = (a2 - y_true) * sigmoid_prime(z2)
    delta1 = [W2[i] * delta2 * sigmoid_prime(z1[i]) for i in range(len(z1))]
    grad_W2 = [delta2 * a1[i] for i in range(len(a1))]
    grad_b2 = delta2
    grad_W1 = [[delta1[i] * x[j] for j in range(len(x))] for i in range(len(delta1))]
    grad_b1 = delta1
    return grad_W1, grad_b1, grad_W2, grad_b2
```

## 8.4 왜 "autograd 없이" 손으로 해보는가

PyTorch/TensorFlow는 `.backward()` 한 줄로 이 모든 미분을 자동으로
계산해준다(자동 미분, automatic differentiation). 하지만 그 자동 미분이
내부에서 정확히 무엇을 하는지 모르면, 학습이 발산하거나 그래디언트가
사라지는(vanishing gradient, Chapter 9에서 다룸) 문제를 마주쳤을 때 원인을
진단할 방법이 없다. 이번 장 연습문제의 목적은 라이브러리가 대신 해주는 그
연쇄법칙을, 딱 한 번은 손으로 끝까지 따라가 보는 것이다.

**신경망이 "무엇이든 배울 수 있는" 이유는 층을 쌓아 표현력을 키운 데 있고,
그 층을 "실제로 학습시킬 수 있는" 이유는 연쇄법칙 하나에 있다.**

---

## 연습문제

**1. (코딩)** 입력층(2) → 은닉층(2, sigmoid) → 출력층(1, sigmoid)인 2층
신경망의 순전파/역전파(핵심 줄은 빈칸으로 남겨져 있다고 가정)를 완성하라:

```python
def two_layer_nn_forward(x, W1, b1, W2, b2):
    # ADD ADDITIONAL CODE HERE!!
    # 은닉층 pre-activation z1 = W1 @ x + b1, 활성화 a1 = sigmoid(z1)
    # 출력 pre-activation z2 = W2 . a1 + b2, 활성화 a2 = sigmoid(z2)

    return a2, (x, z1, a1, z2, a2)

def two_layer_nn_backward(y_true, cache, W2):
    x, z1, a1, z2, a2 = cache
    # ADD ADDITIONAL CODE HERE!!
    # 출력층 오차 delta2, 은닉층 오차 delta1, W2/b2/W1/b1 그래디언트

    return grads
```

**2. (개념 서술)** 은닉층이 **없는**(입력→출력 직결) 신경망에 시그모이드
출력을 쓰면 어떤 Chapter 2의 모델과 정확히 같아지는지 답하고, 은닉층을
추가했을 때 표현력이 왜 늘어나는지(XOR 예시를 들어) 두세 문장으로
설명하라.

**3. (손유도, Tier C — 폴백 준비 대상)** 위 신경망에 대해 손실함수
\\(L = \frac{1}{2}(a_2 - y)^2\\)에서 시작하여, **연쇄법칙만 사용하여**
\\(\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}\\)를
처음부터 끝까지 유도하라.

**빈칸채움형 폴백 버전** (자유 유도가 어려운 경우):

```
L = (1/2)(a2 - y)^2

Step 1: dL/da2 = ______________
Step 2: da2/dz2 = a2(1-a2)  [sigmoid 미분 공식 -- 이미 알려줌]
Step 3: dL/dz2 = dL/da2 * da2/dz2 = ______________  (이게 delta2)
Step 4: dz2/dW2 = a1
Step 5: dL/dW2 = delta2 * ______________

Step 6: dz2/da1 = W2
Step 7: dL/da1 = delta2 * ______________
Step 8: da1/dz1 = a1(1-a1)
Step 9: dL/dz1 = dL/da1 * da1/dz1 = ______________  (이게 delta1)
Step 10: dL/dW1 = delta1 * ______________  (x와의 외적)
```

**정확성 확인**: 완성한 수식으로 위 코드의 빈칸이 왜 그렇게 채워지는지
한 문장씩 연결하라.
