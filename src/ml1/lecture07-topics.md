# Topics Covered

## 퍼셉트론에서 다층 신경망으로

로지스틱회귀는 사실 입력층과 출력층만 있는(은닉층이 없는) 가장 단순한 "신경망"으로 볼 수
있다: \\(a = \sigma(w^Tx)\\). **다층 퍼셉트론**(Multi-Layer Perceptron, MLP)은 그
사이에 은닉층을 하나 이상 끼워 넣는다.

## 순전파(Forward Propagation)

입력 \\(x\\), 은닉층 가중치 \\(W_1, b_1\\), 출력층 가중치 \\(W_2, b_2\\)를 가진 2층
신경망(입력 → 은닉 → 출력)의 순전파:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2, \quad
a_2 = \sigma(z_2)\\]

\\(z\\)는 활성화 함수를 적용하기 전(pre-activation), \\(a\\)는 적용한 후
(activation)다. 최종 출력 \\(a_2\\)가 예측값이다.

## 역전파(Backpropagation): 연쇄법칙을 거꾸로

손실함수 \\(L\\)이 주어졌을 때, 모든 가중치에 대한 \\(\frac{\partial L}{\partial
W}\\)를 구해야 경사하강법을 적용할 수 있다. 문제는 \\(W_1\\)이 최종 손실에 미치는
영향이 \\(z_1 \to a_1 \to z_2 \to a_2 \to L\\)이라는 긴 사슬을 거친다는 것이다.
연쇄법칙은 이 사슬을 **출력에서 입력 방향으로 거꾸로** 따라가며 미분을 하나씩 곱해
나가면 된다고 말해준다:

1. **출력층 오차**: \\(\delta_2 = \frac{\partial L}{\partial z_2} = (a_2 - y)
   \sigma'(z_2)\\) (W03에서 본 것과 같은 형태 — 교차 엔트로피와 시그모이드의 조합은
   항상 이렇게 깔끔해진다.)
2. **은닉층 오차**: \\(\delta_1 = (W_2 \delta_2) \odot \sigma'(z_1)\\) — 출력층
   오차 \\(\delta_2\\)를 \\(W_2\\)를 거슬러 은닉층으로 "돌려보낸" 뒤, 은닉층 자체의
   미분(\\(\sigma'(z_1)\\))을 곱한다. (\\(\odot\\)는 원소별 곱.)
3. **그래디언트**: \\(\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1^T\\),
   \\(\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T\\)

"역전파"라는 이름의 의미가 바로 여기 있다: 오차(\\(\delta\\))가 출력층에서
계산되고, 그 값이 은닉층 방향으로 **거꾸로** 전파(propagate)되면서 각 층의 그래디언트를
만든다.

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

## 왜 "autograd 없이" 손으로 해보는가

PyTorch/TensorFlow는 `.backward()` 한 줄로 이 모든 미분을 자동으로 계산해준다(자동
미분, automatic differentiation). 하지만 그 자동 미분이 내부에서 정확히 무엇을 하는지
모르면, 학습이 발산하거나 그래디언트가 사라지는(vanishing gradient, W08에서 다룸) 문제를
마주쳤을 때 원인을 진단할 방법이 없다. 이번 주 손유도 과제의 목적은 라이브러리가 대신
해주는 그 연쇄법칙을, 딱 한 번은 손으로 끝까지 따라가 보는 것이다.
