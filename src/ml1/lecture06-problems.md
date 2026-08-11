# Problem Set

난이도 등급: **Tier C (폴백 준비 대상)** — 아래 두 버전을 모두 준비해두고, 학생 반응을
본 뒤 선택한다.

**1.** (코딩) 결정 트리 스텁 함수 `fit_stump`(깊이 1짜리 트리, 즉 분기 하나만 있는
트리)가 이미 주어졌을 때, 다음과 같은 함수 `gbdt_fit`을 작성하라:

- input parameter: 특징행렬 `X`, 타겟 `y`, 트리 개수 `n_trees`, 학습률 `learning_rate`
- return value: 학습된 트리(stump) 리스트

```python
def gbdt_fit(X, y, n_trees, learning_rate):
    trees = []
    predictions = [0.0] * len(y)
    for t in range(n_trees):
        # ADD ADDITIONAL CODE HERE!!
        # 1. 잔차 = y - predictions 계산
        # 2. fit_stump(X, residuals)로 트리 하나 학습
        # 3. predictions에 learning_rate * 트리 예측값을 더해 누적

    return trees

def gbdt_predict(trees, x, learning_rate):
    # ADD ADDITIONAL CODE HERE!!
    # 모든 트리의 예측을 learning_rate 곱해서 합산

    return total
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단)

### [버전 A] SHAP값 자유 계산 (Math for ML이 조합론 기초를 충분히 다뤘을 경우)

특징이 2개(\\(A, B\\))인 장난감 모델의 예측함수 \\(f(S)\\)(부분집합 \\(S\\)를 입력으로
받아 예측값을 내는 함수)가 다음과 같이 주어졌다:

\\[f(\{\}) = 10, \quad f(\{A\}) = 16, \quad f(\{B\}) = 13, \quad f(\{A,B\}) = 20\\]

가능한 두 추가 순서(\\(A \to B\\), \\(B \to A\\))에서 각 특징의 한계 기여(marginal
contribution)를 모두 구하고, \\(\phi_A\\)와 \\(\phi_B\\)를 계산하라. \\(\phi_0 +
\phi_A + \phi_B = f(\{A,B\})\\)가 정확히 성립하는지 확인하라.

### [버전 B] 빈칸채움형 계산 워크시트 (폴백)

같은 \\(f\\) 값들을 이용해, 아래 표의 빈칸만 채워라:

```
순서 A -> B:
  A의 한계 기여 = f({A}) - f({}) = 16 - 10 = ______________
  B의 한계 기여 = f({A,B}) - f({A}) = 20 - 16 = ______________

순서 B -> A:
  B의 한계 기여 = f({B}) - f({}) = 13 - 10 = ______________
  A의 한계 기여 = f({A,B}) - f({B}) = 20 - 13 = ______________

phi_A = (순서 A->B에서 A의 기여 + 순서 B->A에서 A의 기여) / 2 = ______________
phi_B = (순서 A->B에서 B의 기여 + 순서 B->A에서 B의 기여) / 2 = ______________

검산: phi_0 + phi_A + phi_B = f({}) + phi_A + phi_B = ______________
      (f({A,B}) = 20과 같아야 함)
```

**정확성 확인**: 검산 결과가 20과 정확히 일치하는지 확인하고, 만약 특징이 3개였다면
가능한 순서가 몇 가지였을지(\\(3! = 6\\)) 계산하라.

---

*교원 노트: 버전 A/B 중 선택은 Math for ML의 조합/순열 커버리지 확인 후 결정. 확인
전까지는 버전 B를 기본값으로 준비.*
