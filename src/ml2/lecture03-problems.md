# Problem Set

난이도 등급: **Tier C**(폴백 준비 대상) — 아래 두 버전을 모두 준비해두고, 학생 반응을
본 뒤 선택한다.

**1.** (코딩) 다음과 같은 함수 `scaled_dot_product_attention`을 작성하라:

- input parameter: `Q`, `K`, `V` (모두 리스트의 리스트, 각 행이 한 단어의 벡터),
  `d_k`(Key 벡터의 차원)
- return value: `(output, weights)` — 각 단어의 어텐션 결과 벡터들과, 어텐션 가중치
  행렬

```python
import math

def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]

def scaled_dot_product_attention(Q, K, V, d_k):
    # ADD ADDITIONAL CODE HERE!!
    # 1. scores[i][j] = (Q[i] . K[j]) / sqrt(d_k)
    # 2. weights = 각 행에 softmax 적용
    # 3. output[i] = weights[i]로 V의 가중합

    return output, weights

Q = [[1,0],[0,1]]
K = [[1,0],[0,1]]
V = [[10,0],[0,10]]
output, weights = scaled_dot_product_attention(Q, K, V, d_k=2)
print(weights)  # 각 단어가 "자기 자신"에 더 높은 가중치를 줌
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단)

### [버전 A] 자유 계산 (Math for ML이 행렬곱·softmax를 편하게 다룰 경우)

단어 2개짜리 문장에 대해, 다음 벡터가 주어졌다: \\(Q = \begin{pmatrix}1 & 0\\ 0 &
1\end{pmatrix}\\), \\(K = \begin{pmatrix}1 & 1\\ 1 & 0\end{pmatrix}\\), \\(V =
\begin{pmatrix}5 & 0\\ 0 & 5\end{pmatrix}\\) (\\(d_k=2\\)).

\\(QK^T\\)를 손으로 계산하고, \\(\sqrt{d_k}\\)로 나눈 뒤, 각 행에 softmax를 적용해
어텐션 가중치 행렬을 구하고, 마지막으로 \\(V\\)와의 가중합으로 최종 출력을 계산하라.

### [버전 B] 빈칸채움형 계산 워크시트 (폴백)

같은 \\(Q, K, V\\)를 이용해, 빈칸만 채워라:

```
Step 1: QK^T 계산 (2x2 행렬, 각 원소는 Q의 한 행과 K^T의 한 열의 내적)
  (QK^T)[0][0] = Q[0].K[0] = 1*1 + 0*1 = ______________
  (QK^T)[0][1] = Q[0].K[1] = 1*1 + 0*0 = ______________
  (QK^T)[1][0] = Q[1].K[0] = 0*1 + 1*1 = ______________
  (QK^T)[1][1] = Q[1].K[1] = 0*1 + 1*0 = ______________

Step 2: sqrt(d_k) = sqrt(2) ≈ 1.41 로 모든 원소를 나눈다
  scaled[0] = [______________, ______________]
  scaled[1] = [______________, ______________]

Step 3: 각 행에 softmax 적용 (첫 번째 행만 계산해보자)
  exp(scaled[0][0]) = ______________  (계산기 사용)
  exp(scaled[0][1]) = ______________
  weights[0] = [exp(scaled[0][0]), exp(scaled[0][1])]을 합이 1이 되도록 정규화
             = [______________, ______________]

Step 4: 출력[0] = weights[0][0] * V[0] + weights[0][1] * V[1]
       = ______________ * [5,0] + ______________ * [0,5]
       = [______________, ______________]
```

**정확성 확인**: 완성한 계산을 문제 1의 `scaled_dot_product_attention(Q, K, V,
d_k=2)` 출력과 대조해 일치하는지 확인하라.

---

*교원 노트: 버전 A/B 중 선택은 Math for ML의 행렬곱 커버리지 확인 후 결정. 확인
전까지는 버전 B를 기본값으로 준비.*
