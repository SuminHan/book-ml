# Problem Set

난이도 등급: **Tier C (폴백 준비 대상)** — 아래 두 버전을 모두 준비해두고, 학생 반응/Math
for ML 실제 커버리지 확인 후 선택한다.

**1.** (코딩) 입력층(2) → 은닉층(2, sigmoid) → 출력층(1, sigmoid)인 2층 신경망을
순전파(forward)와 역전파(backward) 둘 다 구현하라.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def two_layer_nn_forward(x, W1, b1, W2, b2):
    # x: 입력 벡터(길이2), W1: 2x2, b1: 길이2, W2: 길이2, b2: 스칼라
    # ADD ADDITIONAL CODE HERE!!
    # 은닉층 pre-activation z1 = W1 @ x + b1
    # 은닉층 활성화 a1 = sigmoid(z1)
    # 출력 pre-activation z2 = W2 . a1 + b2
    # 출력 활성화 a2 = sigmoid(z2)

    return a2, (x, z1, a1, z2, a2)

def two_layer_nn_backward(y_true, cache, W2):
    x, z1, a1, z2, a2 = cache
    # ADD ADDITIONAL CODE HERE!!
    # 출력층 오차 delta2 = (a2 - y_true) * sigmoid'(z2)
    # 은닉층 오차 delta1 = (delta2 * W2) * sigmoid'(z1)  (원소별)
    # W2, b2, W1, b1에 대한 그래디언트 계산

    return grads
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단)

### [버전 A] 자유 유도 (Math for ML이 편미분·연쇄법칙을 충분히 다뤘을 경우)

위 신경망에 대해 손실함수 \\(L = \frac{1}{2}(a_2 - y)^2\\)에서 시작하여, **연쇄법칙
(chain rule)만 사용하여** \\(\frac{\partial L}{\partial W_1}, \frac{\partial
L}{\partial W_2}\\)을 처음부터 끝까지 유도하라. autograd/라이브러리 없이, 각 단계에
어떤 연쇄법칙이 적용되는지 명시할 것.

### [버전 B] 빈칸채움형 유도 워크시트 (폴백)

아래 유도 과정의 빈칸만 채워라 (전체 구조는 이미 제공됨):

```
L = (1/2)(a2 - y)^2

Step 1: dL/da2 = ______________
Step 2: da2/dz2 = a2(1-a2)  [sigmoid 미분 공식 -- 이미 알려줌]
Step 3: dL/dz2 = dL/da2 * da2/dz2 = ______________  (이게 delta2)
Step 4: dz2/dW2 = a1  [출력층 pre-activation은 W2와 a1의 내적이므로]
Step 5: dL/dW2 = delta2 * ______________

Step 6: dz2/da1 = W2
Step 7: dL/da1 = delta2 * ______________
Step 8: da1/dz1 = a1(1-a1)  [sigmoid 미분 공식]
Step 9: dL/dz1 = dL/da1 * da1/dz1 = ______________  (이게 delta1, 원소별 계산)
Step 10: dL/dW1 = delta1 * ______________  (x와의 외적)
```

**정확성 확인**: 완성한 수식으로 코드의 순전파/역전파 빈칸이 왜 그렇게 채워지는지
한 문장씩 연결하라.

---

*교원 노트: 버전 A/B 중 선택은 Math for ML 실제 커버리지 확인 후 결정. 확인 전까지는
버전 B를 기본값으로 준비.*
