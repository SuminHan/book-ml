# Chapter 3. 생성 모델 관점의 분류: 나이브베이즈와 GDA (Generative Classifiers: Naive Bayes & GDA)

2002년, 프로그래머 폴 그레이엄(Paul Graham)은 "스팸에 대한 계획"(A Plan for
Spam)이라는 글에서, 당시 대부분의 스팸 필터가 손으로 짠 규칙("제목에 '무료'가
있으면 스팸")에 의존해 쉽게 뚫리던 문제를, 훨씬 단순한 통계적 방법으로 풀 수
있다고 제안했다 — 스팸 메일과 정상 메일 각각에서 단어가 나타나는 빈도를 세고,
베이즈 정리로 뒤집어 "이 단어들이 나왔을 때 스팸일 확률"을 계산하는 것이다. 이
방법(나이브베이즈)은 널리 채택되어 초기 스팸 필터의 표준이 됐다. 이번 장은
Chapter 2와는 근본적으로 다른 방향에서 분류 문제에 접근한다.

## 3.1 베이즈 정리, 그리고 생성 vs 판별

Chapter 2의 로지스틱회귀는 \\(P(y|x)\\)(입력이 주어졌을 때 정답의 확률)를
**곧바로** 모델링했다 — 이런 접근을 **판별적**(discriminative) 모델이라
부른다. 이번 장의 접근은 정반대다: 각 클래스가 데이터를 어떻게
"만들어내는지"(\\(P(x|y)\\))를 먼저 모델링하고, **베이즈 정리**로 뒤집어서
원하는 \\(P(y|x)\\)를 얻는다 — 이런 접근을 **생성적**(generative) 모델이라
부른다.

\\[P(y|x) = \frac{P(x|y)P(y)}{P(x)}\\]

분류에서는 \\(x\\)가 고정된 채 \\(y\\)의 각 값(클래스)을 비교하므로, 분모
\\(P(x)\\)(모든 클래스에 걸쳐 동일)는 무시하고 분자만 최대화하는 클래스를
고르면 된다:

\\[\hat{y} = \arg\max_y P(x|y)P(y)\\]

여기서 \\(P(y)\\)는 **사전확률**(prior, 예: 전체 메일 중 스팸 비율)이고,
\\(P(x|y)\\)는 **가능도**(likelihood, 예: 스팸 메일이 이런 단어들을 포함할
확률)다. "생성적"이라는 이름은, 이 모델이 사실상 "클래스 \\(y\\)가 주어지면
데이터 \\(x\\)를 어떻게 생성하는가"를 학습한다는 데서 온다 — 실제로 학습이
끝난 \\(P(x|y)\\)에서 샘플을 뽑으면, 그 클래스의 "전형적인" 가짜 데이터를
만들어낼 수도 있다(Chapter 15의 EM/GMM·생성형 모델과 같은 계열의
아이디어다).

## 3.2 가우시안 판별분석 (GDA)

입력 \\(x\\)가 연속값(실수 벡터)일 때, 가장 자연스러운 선택은 각 클래스의
데이터가 **정규분포**를 따른다고 가정하는 것이다 — 이게 **가우시안
판별분석**(Gaussian Discriminant Analysis, GDA)이다. 이진 분류라면:

\\[y \sim \text{Bernoulli}(\phi), \qquad x \mid y{=}0 \sim
\mathcal{N}(\mu_0, \Sigma), \qquad x \mid y{=}1 \sim \mathcal{N}(\mu_1, \Sigma)\\]

두 클래스가 **같은 공분산 행렬** \\(\Sigma\\)를 공유하되, 평균 \\(\mu_0,
\mu_1\\)만 다르다고 가정한다(두 클래스의 데이터가 같은 "모양"으로 퍼져
있지만 중심 위치만 다르다는 뜻). 파라미터 \\(\phi, \mu_0, \mu_1, \Sigma\\)는
학습 데이터의 평균·공분산을 그대로 계산해서 구한다(최대우도추정, maximum
likelihood estimation) — 경사하강법 없이 닫힌 형태로 바로 구해진다는 점이
로지스틱회귀와의 실전적 차이다.

**놀라운 사실**: 이렇게 구한 \\(P(y=1|x)\\)를 베이즈 정리로 전개하면, 정확히
Chapter 2의 로지스틱회귀와 **같은 시그모이드 형태** \\(P(y=1|x) =
\sigma(w^Tx+b)\\)가 나온다(유도는 이번 장 연습문제 2번). 즉 GDA는 "로지스틱
회귀와 같은 결정 경계(직선)에 도달하는 또 다른 길"이다 — 다만 가는
경로(데이터가 정규분포를 따른다고 먼저 가정하는가, 아니면 결정 경계를
바로 학습하는가)가 다르다. 데이터가 실제로 가우시안에 가깝다면 GDA가 더
적은 데이터로도 잘 맞고, 그 가정이 틀렸다면 판별적 모델(로지스틱회귀)이
더 안정적인 경향이 있다 — "모델을 데이터에 맞출 것인가, 데이터가 모델을
따른다고 가정할 것인가"라는, 생성적 모델과 판별적 모델의 근본적인
트레이드오프다.

## 3.3 나이브베이즈: 독립을 가정하고 차원의 저주를 피한다

GDA는 \\(x\\)가 저차원 연속값일 때는 잘 맞지만, 스팸 필터처럼 \\(x\\)가
"어휘 사전 크기(수만 개)만큼의 차원을 가진, 각 단어의 등장 여부" 같은
고차원 데이터라면 공분산 행렬 \\(\Sigma\\)(어휘 크기 × 어휘 크기)를 추정하는
것 자체가 감당이 안 된다. **나이브베이즈**(Naive Bayes)는 과감한 단순화로
이 문제를 피한다 — 클래스 \\(y\\)가 주어지면 **각 특징(단어)이 서로
독립**이라고 가정해버린다:

\\[P(x|y) = \prod_{j=1}^n P(x_j|y)\\]

"나이브(순진)"라는 이름은 이 가정이 현실적으로 거의 항상 틀리기 때문이다 —
"무료"라는 단어와 "당첨"이라는 단어는 스팸 메일에서 실제로 같이 나타나는
경향이 있어 독립이 아니다. 그런데도 나이브베이즈는 실전에서(특히 텍스트
분류) 놀랄 만큼 잘 작동한다 — 각 \\(P(x_j|y)\\)는 데이터에서 단어 \\(j\\)의
빈도만 세면 바로 추정되므로, 어휘가 아무리 커도 파라미터 수가 어휘
크기에 **선형**으로만 늘어난다(GDA의 공분산 행렬처럼 제곱으로 늘지 않는다).

**라플라스 스무딩**: 학습 데이터에 한 번도 안 나온 단어가 새 메일에 등장하면
\\(P(x_j|y)=0\\)이 되어 곱 전체가 0이 돼버린다. 모든 카운트에 1을 더해서
이 문제를 피한다:

\\[P(x_j{=}1|y{=}k) = \frac{(\text{클래스 } k\text{에서 단어 } j\text{가 등장한 문서 수}) + 1}{(\text{클래스 } k\text{의 전체 문서 수}) + 2}\\]

```python
import math

def train_naive_bayes(emails, labels):
    # emails: 단어 리스트의 리스트, labels: 0(정상)/1(스팸) 리스트
    vocab = set(w for email in emails for w in email)
    n_spam = sum(1 for l in labels if l == 1)
    n_ham = len(labels) - n_spam
    word_counts = {0: {}, 1: {}}
    for email, label in zip(emails, labels):
        for w in set(email):  # 등장 여부만 세는 Bernoulli 나이브베이즈
            word_counts[label][w] = word_counts[label].get(w, 0) + 1
    return {"vocab": vocab, "word_counts": word_counts,
            "n_spam": n_spam, "n_ham": n_ham, "n_total": len(labels)}

def classify(email, model):
    words = set(email)
    log_prob = {}
    for label, n_docs in [(0, model["n_ham"]), (1, model["n_spam"])]:
        log_p = math.log(n_docs / model["n_total"])  # log P(y)
        for w in model["vocab"]:
            p_present = (model["word_counts"][label].get(w, 0) + 1) / (n_docs + 2)
            log_p += math.log(p_present) if w in words else math.log(1 - p_present)
        log_prob[label] = log_p
    return 1 if log_prob[1] > log_prob[0] else 0
```

**왜 로그를 더하는가**: 단어가 수천 개면 \\(P(x_j|y)\\)를 수천 번 곱해서
극도로 작은 수(0에 가까운)가 되고, 컴퓨터의 부동소수점 정밀도로는 구분이
안 될 수 있다(underflow). Chapter 2에서 교차 엔트로피가 확률의 곱 대신 로그의
합을 쓴 것과 정확히 같은 이유로, 여기서도 \\(\log \prod_j P(x_j|y) = \sum_j
\log P(x_j|y)\\)로 곱을 합으로 바꿔서 계산한다.

**GDA와 나이브베이즈는 접근 방식(연속값에 정규분포 vs. 이산값에 독립 가정)은
다르지만, 둘 다 "먼저 각 클래스가 데이터를 어떻게 만드는지 모델링하고,
베이즈 정리로 뒤집는다"는 같은 생성적 철학을 공유한다.**

---

## 연습문제

**1. (코딩)** 위 `train_naive_bayes`와 `classify`(핵심 줄은 빈칸으로 남겨져
있다고 가정)를 완성하라:

```python
def train_naive_bayes(emails, labels):
    # ADD ADDITIONAL CODE HERE!!
    # vocab 구성, 클래스별 문서 수 계산, 클래스별 단어 등장 문서 수 카운트

def classify(email, model):
    # ADD ADDITIONAL CODE HERE!!
    # 클래스 0, 1 각각에 대해 log P(y) + sum(log P(x_j|y))를 계산해 비교

emails = [["free", "money", "now"], ["meeting", "tomorrow", "project"],
          ["free", "prize", "click"], ["project", "deadline", "meeting"]]
labels = [1, 0, 1, 0]
model = train_naive_bayes(emails, labels)
print(classify(["free", "prize"], model))  # 1 (스팸)
print(classify(["project", "meeting"], model))  # 0 (정상)
```

**2. (손유도, Tier B — 힌트 제공)** \\(x|y{=}0 \sim \mathcal{N}(\mu_0,
\Sigma)\\), \\(x|y{=}1 \sim \mathcal{N}(\mu_1, \Sigma)\\)(공분산 공유),
\\(y \sim \text{Bernoulli}(\phi)\\)일 때,

\\[P(y{=}1|x) = \sigma(w^Tx+b), \qquad w = \Sigma^{-1}(\mu_1-\mu_0)\\]

가 됨을 유도하라(즉 GDA의 사후확률이 Chapter 2의 로지스틱회귀와 정확히
같은 시그모이드-선형 형태임을 보여라).

**힌트**(세 단계로 나눠서): (1) \\(z = \log\frac{P(x|y{=}1)P(y{=}1)}{P(x|y{=}0)P(y{=}0)}\\)로
정의하면 \\(P(y{=}1|x) = \sigma(z)\\)임을 먼저 확인하라(베이즈 정리와
시그모이드 정의를 이용). (2) 다변량 정규분포의 로그를 전개하면
\\(\log P(x|y{=}k) = -\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k) + \text{상수}\\)
형태다 — 이걸 \\(k=0,1\\) 각각에 대해 \\(z\\)에 대입하고 전개하라. (3) 두
클래스가 **같은** \\(\Sigma\\)를 쓰므로 \\(x^T\Sigma^{-1}x\\) 이차항이
정확히 상쇄되어 사라진다 — 무엇이 남는지 정리해서 \\(z\\)가 \\(x\\)에
대한 **일차식**(선형)임을 확인하고, \\(w\\)와 \\(b\\)를 \\(\mu_0, \mu_1,
\Sigma, \phi\\)로 표현하라.

**정확성 확인**: 만약 두 클래스의 공분산이 서로 다르다면(\\(\Sigma_0 \ne
\Sigma_1\\)) 이차항이 상쇄되지 않는다 — 이 경우 결정 경계가 더 이상
직선이 아니라 어떤 모양이 될지(힌트: 이차식) 한 문장으로 설명하라.
