# Chapter 6. 정규화와 모델 선택 (Regularization & Model Selection)

1996년, 통계학자 로버트 팁시라니(Robert Tibshirani)는 "Lasso"(Least Absolute
Shrinkage and Selection Operator)라는 방법을 제안했다. 아이디어는 단순했다 —
회귀의 손실함수에 가중치 절댓값의 합을 페널티로 더하면, 놀랍게도 중요하지 않은
특징의 가중치가 **정확히 0**이 되어버린다. 특징이 수천 개인 문제에서 "어떤
특징이 진짜 중요한가"를 사람이 일일이 고르는 대신, 손실함수 하나를 바꾸는 것만
으로 모델이 스스로 걸러낸다는 뜻이다. 이번 장은 Chapter 4.4에서 본 편향-분산
트레이드오프를 실제로 **조절하는 손잡이**를 다룬다.

## 6.1 정규화: 손실함수에 페널티를 더한다

Chapter 4.4에서 봤듯, 모델이 너무 유연하면(파라미터가 크고 자유로우면)
분산이 커져 과적합한다. **정규화**(regularization)는 손실함수에 "가중치가
너무 커지지 않도록" 페널티 항을 더해서, 모델의 유효 유연성을 인위적으로
낮추는 기법이다:

\\[J(w) = \underbrace{\frac{1}{2m}\sum_{i=1}^m (h_w(x^{(i)})-y^{(i)})^2}\_{\text{원래 손실(적합도)}} +
\underbrace{\lambda R(w)}\_{\text{정규화 항(단순함을 요구)}}\\]

\\(\lambda\\)(정규화 강도)가 0이면 원래 손실함수와 같고(정규화 없음),
\\(\lambda\\)가 커질수록 "데이터에 딱 맞추기"보다 "가중치를 작게 유지하기"를
더 중요하게 여긴다 — Chapter 4.4의 언어로는, \\(\lambda\\)를 올릴수록
분산은 줄고 편향은 늘어난다.

## 6.2 L2(Ridge)와 L1(Lasso): 페널티의 모양이 다르면 결과도 다르다

가장 흔한 두 페널티는 다음과 같다:

- **L2 정규화(Ridge)**: \\(R(w) = \|w\|_2^2 = \sum_j w_j^2\\)
- **L1 정규화(Lasso)**: \\(R(w) = \|w\|_1 = \sum_j |w_j|\\)

둘 다 "가중치를 작게 만든다"는 목표는 같지만, **모양이 다르면 결과도
다르다**. L2는 모든 가중치를 매끄럽게 0 쪽으로 축소시키지만(shrinkage),
좀처럼 정확히 0이 되지는 않는다. L1은 반대로 중요하지 않은 가중치를
**정확히 0**으로 만들어버리는 경향이 있다 — 그래서 Lasso는 정규화와
동시에 **특징 선택**(feature selection)을 자동으로 해내는 셈이다.

**기하학적 직관**: 제약 \\(R(w) \le t\\) 아래에서 원래 손실을 최소화하는
문제로 다시 쓰면, L2의 제약 영역은 원(구)이고 L1의 제약 영역은 마름모(각
축 위에 뾰족한 꼭짓점이 있는 다면체)다. 손실함수의 등고선이 이 영역과
만나는 최적점을 찾을 때, 마름모는 꼭짓점(즉 어떤 좌표가 정확히 0인 지점)에서
만날 확률이 원보다 훨씬 높다 — 이게 L1이 정확히 0을 만들어내고 L2는 그렇지
않은 기하학적 이유다.

```python
def ridge_gradient_descent(X, y, lam, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)  # w[0] = bias
    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            pred = w[0] + sum(w[j+1] * X[i][j] for j in range(n))
            error = pred - y[i]
            grad[0] += error
            for j in range(n):
                grad[j+1] += error * X[i][j]
        for j in range(n + 1):
            reg = lam * w[j] if j > 0 else 0  # bias(w[0])는 관례적으로 정규화에서 제외
            w[j] -= alpha * (grad[j] / m + reg)
    return w
```

`lam`(\\(\lambda\\))을 0에서 점점 키우면 학습된 가중치가 점점 0에 가까워지는
것을 직접 확인할 수 있다 — 같은 데이터에 \\(\lambda=0, 0.5, 5.0\\)을 넣으면
기울기 \\(w_1\\)이 대략 \\(2.0 \to 1.4 \to 0.4\\)로 줄어든다.

## 6.3 교차검증: \\(\lambda\\)를 데이터로 정한다

\\(\lambda\\)는 사람이 직접 정해야 하는 **하이퍼파라미터**(hyperparameter) —
학습으로 정해지는 \\(w\\)와 달리, 학습 시작 전에 값을 정해줘야 하는
파라미터다. 학습 데이터에 대한 손실만 보고 \\(\lambda\\)를 고르면 항상
\\(\lambda=0\\)(정규화 없음)이 제일 낮은 손실을 내므로 의미가 없다 —
**검증 데이터**(validation set)로 성능을 재야 한다.

데이터가 넉넉하지 않을 때는 **k-겹 교차검증**(k-fold cross-validation)을
쓴다: 데이터를 \\(k\\)개로 쪼갠 뒤, 매번 한 조각을 검증용으로 남기고
나머지로 학습하는 것을 \\(k\\)번 반복해서 성능을 평균낸다 — 데이터
전체를 한 번씩은 검증에도, 학습에도 써보는 셈이다.

```python
def k_fold_split(data, k, fold_idx):
    n = len(data)
    fold_size = n // k
    start = fold_idx * fold_size
    end = start + fold_size if fold_idx < k - 1 else n
    val = data[start:end]
    train = data[:start] + data[end:]
    return train, val
```

**모델 선택의 원칙**: 학습 데이터로 파라미터(\\(w\\))를 정하고, 검증
데이터로 하이퍼파라미터(\\(\lambda\\), \\(k\\)의 kNN, 트리 깊이 등)를
정하고, **테스트 데이터**로는 최종 성능만 딱 한 번 확인한다 — 테스트
데이터를 하이퍼파라미터 튜닝에 쓰면 "부정행위"(테스트 데이터를 몰래
봐서 고른 모델)와 다를 게 없다. 이 3분할(train/validation/test) 원칙은
이번 학기 팀 프로젝트(Chapter 8, 16)와 ML2의 팀 프로젝트에서도 그대로
지켜야 한다.

**정규화는 "모델을 덜 유연하게 만들어서 분산을 줄인다"는 Chapter 4.4의
편향-분산 원리를, 손실함수에 항 하나를 더하는 것만으로 실제로 조절
가능하게 만든 도구다 — 그리고 그 조절 손잡이(\\(\lambda\\)) 자체는
교차검증이라는 또 다른 절차로 정해야 한다.**

---

## 연습문제

**1. (코딩)** 위 `ridge_gradient_descent`와 `k_fold_split`(핵심 줄은
빈칸으로 남겨져 있다고 가정)을 완성하라:

```python
def ridge_gradient_descent(X, y, lam, alpha, epochs):
    # ADD ADDITIONAL CODE HERE!!
    # Chapter 2의 gradient_descent에 L2 정규화 항(bias 제외)만 추가

def k_fold_split(data, k, fold_idx):
    # ADD ADDITIONAL CODE HERE!!
    # data를 k등분해서 fold_idx번째를 검증용, 나머지를 학습용으로 반환

X = [[1.0],[2.0],[3.0],[4.0]]
y = [3.0,5.0,7.0,9.0]
print(ridge_gradient_descent(X, y, lam=0.0, alpha=0.01, epochs=2000))  # 대략 [1.0, 2.0]
print(ridge_gradient_descent(X, y, lam=5.0, alpha=0.01, epochs=2000))  # w[1]이 0쪽으로 눌림

print(k_fold_split(list(range(10)), k=5, fold_idx=2))  # ([...], [4, 5])
```

**2. (개념 서술)** 다음 상황에서 L1과 L2 중 어느 쪽이 더 적합할지 이유와
함께 답하라: 특징이 10,000개인데 그중 실제로 결과에 영향을 주는 특징은
20개 정도일 것이라 추정되는 유전자 발현 데이터 분석 문제.

**3. (손유도, Tier A — 자유 유도)** L2 정규화가 있는 선형회귀(Ridge
회귀)의 비용함수 \\(J(w) = \frac{1}{2m}\|Xw-y\|^2 + \frac{\lambda}{2}\|w\|^2\\)를
\\(w\\)에 대해 미분해 0으로 놓고, 닫힌 형태의 해가

\\[w^* = (X^TX + \lambda I)^{-1}X^Ty\\]

가 됨을 유도하라(Chapter 2의 정규방정식 유도와 같은 방식을 재사용하되,
정규화 항의 미분 \\(\nabla_w \frac{\lambda}{2}\|w\|^2 = \lambda w\\)를
추가로 이용한다). \\(\lambda \to 0\\)일 때 이 식이 Chapter 2의 정규방정식
\\(w^*=(X^TX)^{-1}X^Ty\\)로 돌아감을 확인하고, \\(\lambda \to \infty\\)일
때 \\(w^*\\)가 어떤 값에 가까워지는지 한 문장으로 설명하라.
