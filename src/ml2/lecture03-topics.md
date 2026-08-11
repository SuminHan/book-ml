# Topics Covered

## Query, Key, Value

각 단어(정확히는 각 단어의 임베딩 벡터)는 세 가지 벡터로 변환된다: **Query(Q)**
"내가 지금 무엇을 찾고 있는가", **Key(K)** "나는 어떤 정보를 갖고 있는가",
**Value(V)** "실제로 전달할 내용". 세 벡터 모두 같은 입력 임베딩 \\(x\\)에 서로
다른 학습 가능한 가중치 행렬을 곱해서 얻는다:

\\[Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V\\]

## Scaled Dot-Product Attention

한 단어의 Query가 모든 단어의 Key와 얼마나 "관련 있는지"를 내적(dot product)으로
잰다:

\\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\\]

단계별로 뜯어보면:

1. \\(QK^T\\): 모든 단어 쌍의 Query-Key 내적을 한 번에 계산한 행렬(각 행 = 한
   단어가 다른 모든 단어와 얼마나 관련 있는지에 대한 원점수).
2. \\(\sqrt{d_k}\\)(Key 벡터의 차원의 제곱근)로 나눈다: 내적값이 차원이 커질수록
   커지는 경향이 있어, softmax가 극단적인 값(0 또는 1에 가까운)만 내뱉게 되는 것을
   막기 위한 스케일 조정이다.
3. **softmax**로 각 행을 확률분포(합이 1)로 바꾼다 — "이 단어가 다른 단어들에
   나눠 주는 주의(attention)의 비율".
4. 그 확률로 \\(V\\)(각 단어가 실제로 전달할 내용)의 가중합을 낸다 — 결과는 "관련
   있는 단어들의 내용을 그 관련도만큼 섞어 담은" 새로운 벡터다.

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

## Self-Attention: 문장이 자기 자신을 본다

Q, K, V가 모두 **같은 문장**에서 나오면 "self-attention"이라 부른다 — 각 단어가
같은 문장 안의 다른 모든 단어(자기 자신 포함)와의 관련도를 계산한다. 이게 W03
opener에서 본 "그것은"이 "그 동물"을 가리키는지 판단하는 메커니즘이다: "그것은"의
Query가 문장 안 모든 단어의 Key와 내적을 계산했을 때, "그 동물"의 Key와 가장 높은
점수가 나오도록 학습된다.

## Multi-Head Attention

Q, K, V를 한 세트만 쓰는 대신, 여러 세트("헤드", head)를 병렬로 두고 각각 다른
관점의 관련성을 학습하게 한다 — 한 헤드는 문법적 관계(주어-동사)에, 다른 헤드는
의미적 관계(동의어)에 집중하는 식으로 역할이 나뉘는 경향이 관찰된다. 여러 헤드의
결과를 이어붙인(concatenate) 뒤 다시 한 번 선형변환해서 최종 출력을 만든다.

## Positional Encoding: 순서 정보는 어떻게 넣는가

어텐션 연산 자체는 순서를 전혀 구분하지 않는다 — \\(QK^T\\)는 단어들의 순서를
뒤섞어도 각 쌍의 관련도 자체는 똑같이 계산된다(집합처럼 취급). 그런데 "강아지가
고양이를 쫓는다"에서는 순서가 의미를 바꾼다. 그래서 Transformer는 각 단어의 임베딩에
그 단어의 위치 정보를 담은 벡터(positional encoding, 사인/코사인 함수로 만든 고정된
패턴)를 **더해서** 넣는다 — 이러면 같은 단어라도 위치가 다르면 입력 벡터 자체가
달라지므로, 어텐션이 간접적으로 순서를 활용할 수 있다.

## RNN 대비 이점

Self-attention은 모든 단어 쌍을 **한 번에** 병렬로 계산한다 — 순차적으로 처리할
필요가 없어 GPU 병렬성을 최대한 활용하고, "1번째 단어와 100번째 단어의 관련성"도
중간 99단계를 거치지 않고 직접 계산된다(그래디언트 소실 문제가 구조적으로 훨씬
덜하다). 대신 문장 길이의 제곱에 비례하는 계산량(\\(QK^T\\)가 \\(n \times n\\)
행렬)이라는 새로운 비용이 생긴다 — 이건 아주 긴 문서를 다룰 때의 실무적 한계로
이어진다.
