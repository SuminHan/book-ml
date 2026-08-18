# Chapter 13. EM 알고리즘과 가우시안 혼합모델 (EM Algorithm & Gaussian Mixture Models)

1977년, 통계학자 아서 뎀스터(Arthur Dempster), 낸 레어드(Nan Laird), 도널드
루빈(Donald Rubin)은 "불완전한 데이터로부터의 최대우도추정"이라는 논문에서,
겉보기엔 서로 다른 여러 통계 문제들이 사실 하나의 공통된 구조 — **일부
정보가 관측되지 않았을 때(잠재변수), 그 정보를 알았다면 쉬웠을 계산을
반복적으로 근사한다** — 를 공유한다는 것을 보였다. 그들이 정리한 알고리즘,
**EM**(Expectation-Maximization)은 지금도 Chapter 12의 k-means부터 ML2
Chapter 10의 VAE까지, "관측되지 않은 구조를 가정하고 데이터를 설명한다"는
모든 비지도학습의 밑바탕에 깔려 있다.

## 13.1 닭이 먼저냐 달걀이 먼저냐: 잠재변수 문제

Chapter 12의 k-means는 각 점을 **딱 하나의** 클러스터에 배정했다(하드
할당, hard assignment). 그런데 실제 데이터는 클러스터 경계가 모호한
경우가 많다 — 어떤 점은 클러스터 A에 60%, B에 40% 정도 속한다고 보는 게
더 자연스러울 수 있다. **가우시안 혼합모델**(Gaussian Mixture Model,
GMM)은 각 클러스터를 정규분포로 표현하고(Chapter 3의 GDA와 같은 가정),
각 점이 어느 클러스터에서 나왔는지를 **확률**로 표현한다.

문제는 이렇다 — 각 점이 어느 클러스터에서 나왔는지를 나타내는 변수
\\(z\\)는 데이터에 **관측되지 않는다**(잠재변수, latent variable):

- 만약 각 점의 \\(z\\)(진짜 클러스터)를 **안다면**, 각 클러스터의 평균과
  분산을 구하는 건 쉽다(Chapter 3의 GDA와 똑같이, 그 클러스터에 속한
  점들의 평균·분산을 계산하면 된다).
- 만약 각 클러스터의 평균과 분산을 **안다면**, 각 점이 어느 클러스터에
  속할 확률(\\(z\\)의 사후확률)을 구하는 것도 쉽다(베이즈 정리).

문제는 **둘 다 모른다**는 것이다 — 닭이 먼저냐 달걀이 먼저냐의 순환이다.
EM은 이 순환을 반복으로 풀어낸다: 하나를 (아무 값으로) 고정하고 다른
하나를 구하고, 그걸로 다시 처음 것을 고치고, 이걸 값이 더 이상 바뀌지
않을 때까지 반복한다.

## 13.2 EM 알고리즘: E-step과 M-step

**E-step (Expectation)**: 현재 각 클러스터의 파라미터(평균 \\(\mu_k\\),
분산 \\(\sigma_k^2\\), 클러스터 비율 \\(\pi_k\\))가 주어졌다고 가정하고,
각 데이터 점 \\(x_i\\)가 클러스터 \\(k\\)에 속할 사후확률(**책임값**,
responsibility) \\(\gamma_{ik}\\)을 베이즈 정리로 계산한다:

\\[\gamma_{ik} = P(z_i{=}k \mid x_i) = \frac{\pi_k \, \mathcal{N}(x_i;
\mu_k,\sigma_k^2)}{\sum_{j} \pi_j \, \mathcal{N}(x_i;\mu_j,\sigma_j^2)}\\]

(Chapter 3에서 GDA의 사후확률을 구한 것과 정확히 같은 계산이다 — 다만
거기서는 \\(z\\)를 라벨로 이미 알고 있었고, 여기서는 \\(z\\)가 없어서
이 확률 자체를 구하는 게 목적이다.)

**M-step (Maximization)**: 방금 구한 책임값을 "가중치"로 삼아, 각
클러스터의 파라미터를 다시 추정한다 — 클러스터 \\(k\\)에 대한 책임값이
클수록 그 점이 \\(\mu_k, \sigma_k^2\\) 추정에 더 크게 기여한다:

\\[\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \qquad
\sigma_k^2 = \frac{\sum_i \gamma_{ik}(x_i-\mu_k)^2}{\sum_i \gamma_{ik}},
\qquad \pi_k = \frac{\sum_i \gamma_{ik}}{m}\\]

```python
import math

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def em_step(data, means, sigmas, weights):
    K, n = len(means), len(data)
    # E-step: 책임값(responsibility) 계산
    resp = []
    for x in data:
        probs = [weights[k] * gaussian_pdf(x, means[k], sigmas[k]) for k in range(K)]
        total = sum(probs)
        resp.append([p / total for p in probs])
    # M-step: 책임값으로 가중평균한 파라미터 재추정
    Nk = [sum(resp[i][k] for i in range(n)) for k in range(K)]
    new_means = [sum(resp[i][k] * data[i] for i in range(n)) / Nk[k] for k in range(K)]
    new_sigmas = [math.sqrt(sum(resp[i][k] * (data[i] - new_means[k]) ** 2
                                  for i in range(n)) / Nk[k]) for k in range(K)]
    new_weights = [Nk[k] / n for k in range(K)]
    return new_means, new_sigmas, new_weights
```

E-step과 M-step을 번갈아 반복하면, 두 클러스터의 평균·분산·비율이 점점
실제 값에 수렴한다 — 평균이 0과 10 근처인 두 정규분포에서 뽑은 장난감
데이터로 이 반복을 30번만 돌려도, 추정된 평균이 거의 정확히 0과 10에,
분산은 거의 1에 수렴하는 것을 확인할 수 있다.

## 13.3 GMM은 "소프트 k-means"다

k-means와 나란히 놓고 보면 구조가 똑같다는 게 보인다:

| | k-means | GMM/EM |
|---|---|---|
| 할당 단계 | 가장 가까운 중심점 **하나**로 확정(하드) | 모든 클러스터에 대한 확률(소프트) |
| 갱신 단계 | 배정된 점들의 **평균**만 계산 | 책임값으로 **가중평균**한 평균·분산 계산 |
| 반복 | 배정이 안 바뀔 때까지 | 파라미터가 안 바뀔 때까지 |

사실 k-means는 GMM에서 모든 클러스터의 분산을 아주 작게(그래서 책임값이
사실상 0 또는 1로 확정되게) 고정한 특수한 경우로 볼 수 있다. Chapter
12에서 본 "값을 반복해서 갱신하다 보면 고정점에 수렴한다"는 패턴이
여기서도 그대로 반복된다.

**EM에서 VAE로**: GMM은 "잠재변수 \\(z\\)(어느 클러스터인가)가 유한하고
이산적인" 가장 단순한 경우다. ML2 Chapter 10에서 배울 **VAE**(Variational
Autoencoder)는 정확히 같은 질문 — "관측되지 않은 잠재변수가 데이터를
어떻게 만들어내는가" — 을 던지되, \\(z\\)를 유한한 클러스터 번호가 아니라
**연속적인 벡터**로, E-step/M-step의 명시적 반복 계산을 **신경망**으로
근사한다는 점이 다르다. GMM에서 익힌 "관측 안 된 변수를 가정하고, 그
사후확률을 구하고, 파라미터를 갱신한다"는 이 반복 구조 자체가 VAE를
이해하는 가장 좋은 준비 운동이다.

**EM은 "잠재변수만 알았다면 문제가 쉬웠을 것"이라는 상황마다 재사용되는
범용 도구다 — GMM은 그중 가장 손으로 만져볼 수 있는 형태일 뿐이고, 같은
아이디어가 형태를 바꿔 이 학기 마지막까지 계속 등장한다.**

---

## 연습문제

**1. (코딩)** 위 `em_step`(핵심 줄은 빈칸으로 남겨져 있다고 가정)을
완성하라:

```python
def em_step(data, means, sigmas, weights):
    # ADD ADDITIONAL CODE HERE!!
    # E-step: 각 데이터 x에 대해 클러스터별 책임값(responsibility) 계산
    # M-step: 책임값으로 가중평균한 새 means, sigmas, weights 계산

    return new_means, new_sigmas, new_weights

data = [1.0, 1.5, 0.8, 9.2, 10.1, 9.8]
means, sigmas, weights = [2.0, 8.0], [1.0, 1.0], [0.5, 0.5]
for _ in range(20):
    means, sigmas, weights = em_step(data, means, sigmas, weights)
print(means)  # 대략 [1.1, 9.7] 근처로 수렴
```

**2. (개념 서술)** GMM에서 각 클러스터의 분산 \\(\sigma_k^2\\)을 아주
작은 값으로 고정하고 학습하지 않는다면, E-step에서 계산되는 책임값
\\(\gamma_{ik}\\)이 어떤 모양(0과 1 사이 여러 값 vs. 거의 0 또는 1)이
될지 설명하고, 왜 이 극단적인 경우가 k-means와 같아지는지 논하라.

**3. (손유도, Tier B — 힌트 제공)** 데이터 3개 \\(x_1=1, x_2=9, x_3=10\\),
2개 클러스터, 현재 파라미터가 \\(\mu_1=0, \sigma_1=1, \mu_2=9,
\sigma_2=1, \pi_1=\pi_2=0.5\\)라 하자. \\(x_1\\)에 대한 책임값
\\(\gamma_{1,1}\\)과 \\(\gamma_{1,2}\\)를 손으로 계산하라.

**힌트**: 정규분포 확률밀도 \\(\mathcal{N}(x;\mu,\sigma^2) =
\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}\\)를 \\(x_1=1\\)에
대해 두 클러스터 각각 계산한 뒤(\\(\sigma=1\\)이므로 지수의 분모가
2임을 이용), \\(\gamma_{1,1} = \frac{\pi_1 \mathcal{N}(x_1;\mu_1,1)}
{\pi_1\mathcal{N}(x_1;\mu_1,1) + \pi_2\mathcal{N}(x_1;\mu_2,1)}\\)에
대입하라. \\(\pi_1=\pi_2\\)이므로 실제로는 \\(\pi\\)가 약분되어
사라진다는 것도 확인하라.

**정확성 확인**: \\(x_1=1\\)이 \\(\mu_1=0\\)에 훨씬 가까우므로
\\(\gamma_{1,1}\\)이 \\(\gamma_{1,2}\\)보다 훨씬 커야 한다 — 계산
결과가 이 직관과 맞는지 확인하라.
