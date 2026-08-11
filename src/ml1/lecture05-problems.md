# Problem Set

난이도 등급: **Tier A (자유 유도)**

**1.** (코딩) 다음과 같은 함수 `gini`와 `best_split`을 작성하라:

- `gini(labels)`: 라벨 리스트를 받아 지니불순도를 반환
- `best_split(X, y, feature_idx)`: 하나의 특징(`feature_idx`번째 열)에 대해, 가능한
  모든 분기점(각 샘플 값) 중 지니불순도 감소량이 가장 큰 분기점과 그 감소량을 반환

```python
def gini(labels):
    # ADD ADDITIONAL CODE HERE!!

def weighted_gini(left_labels, right_labels):
    # ADD ADDITIONAL CODE HERE!!
    # 왼쪽/오른쪽 노드 크기로 가중평균한 지니불순도

def best_split(X, y, feature_idx):
    # ADD ADDITIONAL CODE HERE!!
    # 후보 분기점마다: threshold보다 작은/큰 샘플로 나누고
    # (부모 지니 - 가중평균 지니)가 최대인 threshold를 찾는다
    best_threshold, best_gain = None, -1
    parent_gini = gini(y)
    for threshold in sorted(set(row[feature_idx] for row in X)):
        left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > threshold]
        if not left_y or not right_y:
            continue
        gain = parent_gini - weighted_gini(left_y, right_y)
        if gain > best_gain:
            best_threshold, best_gain = threshold, gain
    return best_threshold, best_gain

X = [[2.0],[3.0],[4.0],[7.0],[8.0],[9.0]]
y = ["A","A","A","B","B","B"]
print(best_split(X, y, 0))  # (4.0, 1.0) -- 완벽하게 나뉨
```

**2.** 랜덤 포레스트가 개별 결정 트리보다 과적합에 강한 이유를, 배깅(bagging)과 특징
무작위화 **두 가지**를 모두 언급하며 한 문단으로 설명하라.

---

## 손유도 과제 (실습시간, Tier A — 자유 유도)

### 정보이득/지니불순도 손계산

다음 8개 샘플이 있다: 클래스는 `[A,A,A,A,B,B,B,B]`이고, 특징 \\(x\\) 값은
`[1,2,3,4,5,6,7,8]`이다.

**1단계**: 분기 전체(8개)의 지니불순도와 엔트로피를 각각 손으로 계산하라.

**2단계**: \\(x \le 4\\)와 \\(x > 4\\)로 나누는 분기(threshold=4)에 대해, 왼쪽 노드
(4개)와 오른쪽 노드(4개)의 지니불순도를 각각 계산하고, 가중평균 지니불순도를 구하라.

**3단계**: threshold=4의 정보이득(지니불순도 감소량)을 계산하라. 그리고 threshold=2로
나눴을 때(왼쪽: `[A,A]`, 오른쪽: `[A,A,B,B,B,B]`)의 정보이득과 비교하라 — 어느 쪽
분기가 데이터를 더 잘 나누는가?

**정확성 확인**: 손계산한 값을 문제 1에서 작성한 `best_split(X, y, 0)`의 출력과
비교해 일치하는지 확인하라.
