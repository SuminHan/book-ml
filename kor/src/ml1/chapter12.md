# Chapter 12. 어텐션과 트랜스포머 (Attention & Transformer)

2017년, 구글 브레인의 연구자들은 논문 제목을 다소 도발적으로 지었다 —
**"Attention Is All You Need"**. 그때까지 시퀀스를 다루는 최고 성능의
모델들은 모두 RNN(또는 그 변형인 LSTM)을 기반으로 했는데, 이 논문은 순환
구조를 완전히 걷어내고 **어텐션**(attention) 메커니즘만으로 RNN보다 더
좋은 성능을 냈다. 이 구조가 **트랜스포머**(Transformer)이며, 지금 우리가
쓰는 거의 모든 대규모 언어모델(LLM)의 근간이다.

## 12.1 RNN의 순차성이라는 근본 문제

Chapter 11에서 본 RNN은 한 단어씩 순서대로 처리해야 했다 — 100번째
단어를 보려면 1번째부터 99번째까지 순서대로 거쳐야 한다. 이건 두 가지
문제를 만든다: (1) GPU는 병렬 연산에 특화됐는데, 순차적 구조는 그
병렬성을 활용하지 못해 느리다. (2) 먼 과거의 정보가 여러 단계를 거치며
흐려진다(Chapter 11의 그래디언트 소실과 정확히 같은 문제).

## 12.2 "어텐션"이라는 아이디어

어텐션의 직관은 단순하다: "지금 이 단어를 이해하려면, 문장의 **다른 모든
단어를 한 번에** 보고, 그중 관련 있는 단어에 더 집중(attend)한다." "그
동물은 도로를 건너지 않았다, 왜냐하면 **그것은** 너무 지쳐 있었기
때문이다"라는 문장에서, "그것은"이 가리키는 게 "그 동물"인지 "도로"인지는
문장 전체를 동시에 봐야 판단할 수 있다.

## 12.3 Query, Key, Value

각 단어(정확히는 각 단어의 임베딩 벡터)는 세 가지 벡터로 변환된다:
**Query**(Q) "내가 지금 무엇을 찾고 있는가", **Key**(K) "나는 어떤
정보를 갖고 있는가", **Value**(V) "실제로 전달할 내용". 세 벡터 모두
같은 입력 임베딩 \\(x\\)에 서로 다른 학습 가능한 가중치 행렬을 곱해서
얻는다:

\\[Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V\\]

## 12.4 Scaled Dot-Product Attention

한 단어의 Query가 모든 단어의 Key와 얼마나 "관련 있는지"를 내적(dot
product)으로 잰다:

\\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\\]

단계별로 뜯어보면:

1. \\(QK^T\\): 모든 단어 쌍의 Query-Key 내적을 한 번에 계산한 행렬(각
   행 = 한 단어가 다른 모든 단어와 얼마나 관련 있는지에 대한 원점수).
2. \\(\sqrt{d_k}\\)(Key 벡터의 차원의 제곱근)로 나눈다: 내적값이 차원이
   커질수록 커지는 경향이 있어, softmax가 극단적인 값(0 또는 1에 가까운)만
   내뱉게 되는 것을 막기 위한 스케일 조정이다.
3. **softmax**로 각 행을 확률분포(합이 1)로 바꾼다 — "이 단어가 다른
   단어들에 나눠 주는 주의(attention)의 비율".
4. 그 확률로 \\(V\\)(각 단어가 실제로 전달할 내용)의 가중합을 낸다 —
   결과는 "관련 있는 단어들의 내용을 그 관련도만큼 섞어 담은" 새로운
   벡터다.

```python
import math

def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]

def attention(Q, K, V, d_k):
    scores = [[sum(Q[i][t] * K[j][t] for t in range(d_k)) / math.sqrt(d_k)
               for j in range(len(K))] for i in range(len(Q))]
    weights = [softmax(row) for row in scores]
    output = [[sum(weights[i][j] * V[j][t] for j in range(len(V)))
               for t in range(len(V[0]))] for i in range(len(Q))]
    return output, weights
```

## 12.5 Self-Attention: 문장이 자기 자신을 본다

Q, K, V가 모두 **같은 문장**에서 나오면 "self-attention"이라 부른다 —
각 단어가 같은 문장 안의 다른 모든 단어(자기 자신 포함)와의 관련도를
계산한다. 이게 이번 장 도입부에서 본 "그것은"이 "그 동물"을 가리키는지
판단하는 메커니즘이다: "그것은"의 Query가 문장 안 모든 단어의 Key와
내적을 계산했을 때, "그 동물"의 Key와 가장 높은 점수가 나오도록
학습된다.

## 12.6 Multi-Head Attention

Q, K, V를 한 세트만 쓰는 대신, 여러 세트("헤드", head)를 병렬로 두고
각각 다른 관점의 관련성을 학습하게 한다 — 한 헤드는 문법적 관계(주어-동사)에,
다른 헤드는 의미적 관계(동의어)에 집중하는 식으로 역할이 나뉘는 경향이
관찰된다. 여러 헤드의 결과를 이어붙인(concatenate) 뒤 다시 한 번
선형변환해서 최종 출력을 만든다.

## 12.7 Positional Encoding: 순서 정보는 어떻게 넣는가

어텐션 연산 자체는 순서를 전혀 구분하지 않는다 — \\(QK^T\\)는 단어들의
순서를 뒤섞어도 각 쌍의 관련도 자체는 똑같이 계산된다(집합처럼 취급).
그런데 "강아지가 고양이를 쫓는다"에서는 순서가 의미를 바꾼다. 그래서
Transformer는 각 단어의 임베딩에 그 단어의 위치 정보를 담은 벡터
(positional encoding, 사인/코사인 함수로 만든 고정된 패턴)를 **더해서**
넣는다 — 이러면 같은 단어라도 위치가 다르면 입력 벡터 자체가 달라지므로,
어텐션이 간접적으로 순서를 활용할 수 있다.

## 12.8 RNN 대비 이점

Self-attention은 모든 단어 쌍을 **한 번에** 병렬로 계산한다 — 순차적으로
처리할 필요가 없어 GPU 병렬성을 최대한 활용하고, "1번째 단어와 100번째
단어의 관련성"도 중간 99단계를 거치지 않고 직접 계산된다(그래디언트 소실
문제가 구조적으로 훨씬 덜하다). 대신 문장 길이의 제곱에 비례하는
계산량(\\(QK^T\\)가 \\(n \times n\\) 행렬)이라는 새로운 비용이 생긴다 —
이건 아주 긴 문서를 다룰 때의 실무적 한계로 이어진다.

**"모든 단어를 한 번에 보고, 서로의 관련성을 직접 계산한다"는 이 아이디어
하나가 지난 몇 년간 딥러닝을 가장 크게 바꾼 변화다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `scaled_dot_product_attention`을
완성하라(핵심 줄은 빈칸으로 남겨져 있다고 가정):

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

**2. (개념 서술)** Self-attention이 RNN보다 그래디언트 소실에 덜 취약한
이유를, "1번째 단어와 100번째 단어 사이의 경로 길이"라는 관점에서
설명하라(Chapter 11의 BPTT와 비교).

**3. (손유도, Tier C — 폴백 준비 대상)** 단어 2개짜리 문장에 대해, 다음
벡터가 주어졌다: \\(Q = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}\\),
\\(K = \begin{pmatrix}1 & 1\\ 1 & 0\end{pmatrix}\\), \\(V =
\begin{pmatrix}5 & 0\\ 0 & 5\end{pmatrix}\\)(\\(d_k=2\\)).

\\(QK^T\\)를 손으로 계산하고, \\(\sqrt{d_k}\\)로 나눈 뒤, 각 행에
softmax를 적용해 어텐션 가중치 행렬을 구하고, 마지막으로 \\(V\\)와의
가중합으로 최종 출력을 계산하라.

**빈칸채움형 폴백 버전**(자유 계산이 어려운 경우):

```
Step 1: QK^T 계산 (2x2 행렬)
  (QK^T)[0][0] = Q[0].K[0] = 1*1 + 0*1 = ______________
  (QK^T)[0][1] = Q[0].K[1] = 1*1 + 0*0 = ______________
  (QK^T)[1][0] = Q[1].K[0] = 0*1 + 1*1 = ______________
  (QK^T)[1][1] = Q[1].K[1] = 0*1 + 1*0 = ______________

Step 2: sqrt(d_k) = sqrt(2) ≈ 1.41 로 모든 원소를 나눈다
  scaled[0] = [______________, ______________]
  scaled[1] = [______________, ______________]

Step 3: 첫 번째 행에 softmax 적용
  exp(scaled[0][0]) = ______________  (계산기 사용)
  exp(scaled[0][1]) = ______________
  weights[0] = [______________, ______________]  (합이 1이 되도록 정규화)

Step 4: 출력[0] = weights[0][0] * V[0] + weights[0][1] * V[1]
       = [______________, ______________]
```

**정확성 확인**: 완성한 계산을 문제 1의
`scaled_dot_product_attention(Q, K, V, d_k=2)` 출력과 대조해 일치하는지
확인하라.
