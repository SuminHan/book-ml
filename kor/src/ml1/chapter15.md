# Chapter 15. 잠재변수 생성모델: EM/GMM에서 VAE, GAN, Diffusion까지 (Latent-Variable Generative Models)

1977년, 통계학자 아서 뎀스터(Arthur Dempster), 낸 레어드(Nan Laird), 도널드
루빈(Donald Rubin)은 "불완전한 데이터로부터의 최대우도추정"이라는 논문에서,
겉보기엔 서로 다른 여러 통계 문제들이 사실 하나의 공통된 구조 — **일부
정보가 관측되지 않았을 때(잠재변수), 그 정보를 알았다면 쉬웠을 계산을
반복적으로 근사한다** — 를 공유한다는 것을 보였다. 이번 장은 그들이 정리한
**EM**(Expectation-Maximization) 알고리즘에서 시작해, 정확히 같은 질문("관측
안 된 잠재변수가 데이터를 어떻게 만들어내는가")을 신경망으로 확장한 VAE,
그리고 완전히 다른 두 원리(적대적 학습, 점진적 노이즈 제거)로 같은 목표에
도달하는 GAN·Diffusion까지, 생성형 모델의 네 가지 얼굴을 한 장에서 훑는다.

## 15.1 닭이 먼저냐 달걀이 먼저냐: 잠재변수 문제

Chapter 14의 k-means는 각 점을 **딱 하나의** 클러스터에 배정했다(하드
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

## 15.2 EM 알고리즘: E-step과 M-step

**E-step (Expectation)**: 현재 각 클러스터의 파라미터(평균 \\(\mu_k\\),
분산 \\(\sigma_k^2\\), 클러스터 비율 \\(\pi_k\\))가 주어졌다고 가정하고,
각 데이터 점 \\(x_i\\)가 클러스터 \\(k\\)에 속할 사후확률(**책임값**,
responsibility) \\(\gamma_{ik}\\)을 베이즈 정리로 계산한다:

\\[\gamma_{ik} = P(z_i{=}k \mid x_i) = \frac{\pi_k \, \mathcal{N}(x_i;
\mu_k,\sigma_k^2)}{\sum_{j} \pi_j \, \mathcal{N}(x_i;\mu_j,\sigma_j^2)}\\]

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
실제 값에 수렴한다. k-means와 나란히 놓고 보면 구조가 똑같다는 게
보인다 — 사실 k-means는 GMM에서 모든 클러스터의 분산을 아주 작게(그래서
책임값이 사실상 0 또는 1로 확정되게) 고정한 특수한 경우로 볼 수 있다.

## 15.3 EM에서 VAE로: 잠재변수를 연속적으로, 신경망으로

GMM은 "잠재변수 \\(z\\)(어느 클러스터인가)가 유한하고 이산적인" 가장
단순한 경우다. 지금부터 배울 **VAE**(Variational Autoencoder, 2013년
디데릭 킹마와 맥스 웰링)는 정확히 같은 질문 — "관측되지 않은 잠재변수가
데이터를 어떻게 만들어내는가" — 을 던지되, \\(z\\)를 유한한 클러스터
번호가 아니라 **연속적인 벡터**로, E-step/M-step의 명시적 반복 계산을
**신경망**으로 근사한다는 점이 다르다.

일반 오토인코더의 잠재 공간은 "어떤 값이 그럴듯한 데이터로 복원되는지"에
대한 보장이 없다 — 학습 데이터들이 잠재 공간의 여기저기 흩어져 있을 수
있고, 그 사이 빈 공간에서 무작위로 값을 뽑으면 무엇이 나올지 알 수 없다.
VAE의 해법은 인코더가 하나의 점 \\(z\\)를 내놓는 대신, **확률분포**(보통
정규분포의 평균과 분산)를 내놓도록 강제하는 것이다. 그리고 그 분포가
표준 정규분포(평균 0, 분산 1)에 가깝도록 추가 제약을 건다 — 그러면
잠재 공간 전체가 매끄럽고 빈틈없이 채워져서, 임의의 지점에서 샘플링해도
그럴듯한 데이터가 나올 가능성이 높아진다.

**구조**: 인코더 \\(q_\phi(z|x)\\)는 입력 \\(x\\)를 받아 \\(z\\)의 확률
분포(평균·분산)를 내놓고, 그 분포에서 \\(z\\)를 하나 샘플링한 뒤,
디코더 \\(p_\theta(x|z)\\)가 \\(z\\)로부터 \\(x\\)를 복원(생성)한다.

VAE는 "생성형 모델을 만드는 방식" 중 첫 번째 — **우도 기반**
(likelihood-based) 접근이다: 모델이 학습 데이터를 만들어낼 확률(우도)
\\(P(x)\\)를 직접(또는 그 근사치를) 최대화하도록 학습한다. 문제는
\\(p_\theta(x) = \int p_\theta(x|z)p(z)\,dz\\)는 가능한 모든 \\(z\\)에
대한 적분이라, 일반적으로 계산이 불가능하다(intractable).

## 15.4 ELBO: 계산 가능한 하한으로 우회하기

직접 계산할 수 없는 \\(\log p_\theta(x)\\) 대신, 다음이 성립함을 보일
수 있다:

\\[\log p_\theta(x) \ge \mathbb{E}\_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] -
D\_{KL}\big(q_\phi(z|x) \,\|\, p(z)\big)\\]

우변을 **ELBO**(Evidence Lower BOund)라 부른다. 두 항의 의미:

- 첫 항 \\(\mathbb{E}[\log p_\theta(x|z)]\\): **복원 항** — 인코더가
  만든 \\(z\\)로 디코더가 원본 \\(x\\)를 얼마나 잘 복원하는가(일반
  오토인코더의 복원 오차와 본질적으로 같다).
- 둘째 항 \\(D_{KL}(q_\phi(z|x)\|p(z))\\): **정규화 항** — 인코더가
  만든 분포 \\(q_\phi(z|x)\\)가, 목표로 하는 사전분포 \\(p(z)\\)(보통
  표준정규분포)에서 얼마나 벗어나 있는지(KL divergence, Chapter 2.6에서
  섀넌의 정보이론으로 소개한 그 개념). 이 항이 바로 잠재 공간을 매끄럽게
  만드는 힘이다.

\\(\log p_\theta(x) \ge \text{ELBO}\\)이므로, **ELBO를 최대화하면 실제
우도의 하한도 함께 올라간다** — 직접 계산 못 하는 목표를, 계산 가능한
대리(surrogate) 목표로 바꿔치기한 것이다.

```python
def vae_loss(x, x_reconstructed, mu, log_var):
    # 복원 손실 (여기서는 MSE로 근사)
    recon_loss = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x)))
    # KL divergence: 정규분포 q(mu, sigma^2)와 표준정규분포 N(0,1) 사이의 닫힌 형태 공식
    kl_loss = -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                          for i in range(len(mu)))
    return recon_loss + kl_loss  # ELBO를 최대화 = 이 손실(음의 ELBO)을 최소화
```

**Reparameterization Trick(참고)**: \\(z\\)를 확률분포에서 직접
샘플링하면, "샘플링"이라는 연산은 미분이 안 돼서 역전파가 인코더까지
흘러가지 못한다. VAE는 \\(z = \mu + \sigma \odot \epsilon\\)
(\\(\epsilon \sim \mathcal{N}(0,1)\\)은 무작위성을 밖으로 빼낸 상수
취급)으로 샘플링을 다시 쓰는 트릭으로 이 문제를 우회한다.

## 15.5 GAN: 생성자와 판별자의 min-max 게임

2014년, 당시 대학원생이던 이언 굿펠로우(Ian Goodfellow)는 아이디어
하나를 떠올렸다: 진짜 같은 가짜 이미지를 만드는 모델을 학습시키는 대신,
**두 개의 신경망이 서로 경쟁하게 만들면 어떨까?** 하나는 가짜를 만드는
위조범(생성자, Generator), 다른 하나는 진짜와 가짜를 구별하는
감정사(판별자, Discriminator)로 두고 서로 계속 겨루게 하면, 위조범의
실력이 점점 진짜에 가깝게 늘어나지 않을까? VAE가 "우도를 최대화한다"는
명확한 단일 목표함수가 있었던 것과 달리, GAN은 두 신경망이 서로 다른,
심지어 **반대되는** 목표를 갖고 동시에 학습된다.

- **생성자**(Generator) \\(G\\): 무작위 노이즈 \\(z \sim p(z)\\)를
  입력받아 가짜 데이터 \\(G(z)\\)를 만든다.
- **판별자**(Discriminator) \\(D\\): 데이터를 입력받아, 그것이 진짜일
  확률 \\(D(x) \in [0,1]\\)을 출력한다(Chapter 2의 로지스틱회귀와
  정확히 같은 형태).

목적함수(min-max game):

\\[\min_G \max_D V(D,G) = \mathbb{E}\_{x \sim p_{\text{data}}}[\log D(x)] +
\mathbb{E}\_{z \sim p(z)}[\log(1 - D(G(z)))]\\]

```python
def discriminator_loss(D_real, D_fake):
    return -(math.log(D_real) + math.log(1 - D_fake))  # 판별자가 최소화할 손실

def generator_loss(D_fake):
    return -math.log(D_fake)  # 생성자가 최소화할 손실 (D_fake를 1에 가깝게)
```

**왜 "균형점"을 논해야 하는가**: 두 네트워크가 서로 반대 목표로 동시에
학습되므로, "학습이 끝났다"는 것이 무슨 의미인지부터 다시 정의해야
한다. 게임이론에서 이런 안정 상태를 **내쉬 균형**(Nash equilibrium)이라
부른다 — 어느 쪽도 혼자서 전략을 바꿔서 더 나아질 수 없는 상태다.
이론적으로 이 게임의 내쉬 균형은 \\(D(x) = 0.5\\)(판별자가 진짜와
가짜를 전혀 구별하지 못함)이고, \\(G\\)가 만드는 분포가 실제 데이터
분포와 정확히 같아지는 지점임이 증명돼 있다.

실전에서는 이론과 달리 GAN 학습이 자주 불안정하다 — 생성자가 판별자를
속이는 몇 가지 패턴에만 몰려서 다양성을 잃는 현상(**모드 붕괴, mode
collapse**)이 대표적인 문제다. VAE가 이론적으로 안정적이지만 생성
품질이 다소 흐릿한(blurry) 경향이 있는 반면, GAN은 선명한 결과를 내지만
학습이 까다롭다.

## 15.6 Diffusion: 노이즈를 점진적으로 되돌리기

**Diffusion 모델**은 세 번째 완전히 다른 접근이다. **정방향
과정**(forward process)은 진짜 이미지 \\(x_0\\)에 아주 작은 노이즈를
\\(T\\)번 반복해서 더해, 결국 \\(x_T\\)가 순수한 무작위 노이즈가 되게
만든다(이 과정은 고정된 절차이지 학습 대상이 아니다). **역방향
과정**(reverse process)은 신경망이 "한 단계 전 노이즈가 어땠는지"를
예측하도록 학습된다 — 학습이 끝나면, 순수 노이즈 \\(x_T\\)에서 시작해
이 역방향 단계를 \\(T\\)번 반복하면 새로운 이미지가 만들어진다.

GAN이 "한 번에" 노이즈를 이미지로 바꾸려 하는 것과 달리, Diffusion은
그 어려운 문제를 아주 작은 단계 \\(T\\)개로 잘게 쪼갠다 — 각 단계는
"노이즈를 아주 조금만 제거하는" 훨씬 쉬운 문제이므로, 전체적으로 훨씬
안정적으로 학습된다. 대신 이미지 하나를 생성하는 데 \\(T\\)번의 반복이
필요해 GAN보다 생성 속도가 느리다는 대가가 있다. 지금 가장 널리 쓰이는
이미지 생성 모델들(Stable Diffusion 등)이 이 원리를 쓴다.

## 15.7 네 원리 비교

| | EM/GMM | VAE (우도 기반) | GAN (적대적) | Diffusion (스코어 기반) |
|---|---|---|---|---|
| 잠재변수 | 이산(클러스터 번호) | 연속 벡터 | 없음(노이즈를 직접 변환) | 연속(노이즈 단계) |
| 학습 방법 | E-step/M-step 반복 | ELBO 최대화(역전파) | min-max 게임의 균형 | 각 단계 노이즈 예측 오차 최소화 |
| 학습 안정성 | 안정적 | 비교적 안정적 | 불안정하기 쉬움 | 안정적 |
| 생성 속도 | — | 빠름(한 번에) | 빠름(한 번에) | 느림(반복 필요) |

**하나의 원리만 배우고 끝냈다면 "생성형 모델 = 이런 것"이라고 좁게
오해했을 것이다 — 네 원리를 나란히 보고 나서야, "관측되지 않은 구조가
데이터를 만들어낸다"는 하나의 질문이 얼마나 다양한 방식으로 풀릴 수
있는지가 보인다.**

---

## 연습문제

**1. (코딩)** 위 `em_step`과, 다음 `kl_divergence_gaussian`(정규분포
\\(\mathcal{N}(\mu, \sigma^2)\\)와 표준정규분포 사이의 KL divergence)을
완성하라(핵심 줄은 빈칸으로 남겨져 있다고 가정):

```python
def em_step(data, means, sigmas, weights):
    # ADD ADDITIONAL CODE HERE!!
    return new_means, new_sigmas, new_weights

data = [1.0, 1.5, 0.8, 9.2, 10.1, 9.8]
means, sigmas, weights = [2.0, 8.0], [1.0, 1.0], [0.5, 0.5]
for _ in range(20):
    means, sigmas, weights = em_step(data, means, sigmas, weights)
print(means)  # 대략 [1.1, 9.7] 근처로 수렴

def kl_divergence_gaussian(mu, log_var):
    # log_var = log(sigma^2), D_KL = -0.5*(1 + log_var - mu^2 - exp(log_var))
    # ADD ADDITIONAL CODE HERE!!

print(kl_divergence_gaussian(mu=0.0, log_var=0.0))  # 0.0
print(kl_divergence_gaussian(mu=2.0, log_var=0.0))  # 2.0
```

**2. (코딩)** 위 `discriminator_loss`와 `generator_loss`를 완성하라:

```python
def discriminator_loss(D_real, D_fake):
    # ADD ADDITIONAL CODE HERE!!

def generator_loss(D_fake):
    # ADD ADDITIONAL CODE HERE!!

print(discriminator_loss(D_real=0.9, D_fake=0.1))
print(generator_loss(D_fake=0.9))
```

**3. (개념 서술)** GMM에서 각 클러스터의 분산을 아주 작은 값으로 고정하고
학습하지 않는다면 왜 k-means와 같아지는지 설명하고, VAE(우도 기반)와
GAN(적대적)의 생성 품질·학습 안정성 트레이드오프를 두세 문장으로
비교하라.

**4. (손유도, Tier B — 힌트 제공)** 데이터 3개 \\(x_1=1, x_2=9, x_3=10\\),
2개 클러스터, 현재 파라미터가 \\(\mu_1=0, \sigma_1=1, \mu_2=9,
\sigma_2=1, \pi_1=\pi_2=0.5\\)라 하자. \\(x_1\\)에 대한 책임값
\\(\gamma_{1,1}\\)과 \\(\gamma_{1,2}\\)를 손으로 계산하라.

**힌트**: 정규분포 확률밀도 \\(\mathcal{N}(x;\mu,\sigma^2) =
\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}\\)를 \\(x_1=1\\)에
대해 두 클러스터 각각 계산한 뒤 대입하라. \\(\pi_1=\pi_2\\)이므로
실제로는 \\(\pi\\)가 약분되어 사라진다는 것도 확인하라.

**5. (손유도, Tier C — 폴백 준비 대상)** \\(\log p_\theta(x) = \log
\int p_\theta(x,z)\,dz\\)에서 시작해서, 옌센 부등식(\\(\log
\mathbb{E}[X] \ge \mathbb{E}[\log X]\\))을 이용해 15.4절의 ELBO 부등식을
유도하라.

**빈칸채움형 폴백 버전** (자유 유도가 어려운 경우):

```
Step 1: log p(x) = log integral[ q(z|x) * (p(x,z) / q(z|x)) dz ]
                  = log E_q[ ______________ ]

Step 2: 옌센 부등식(log가 오목함수) 적용:
  log E_q[ p(x,z)/q(z|x) ] >= E_q[ log(______________) ]

Step 3: p(x,z) = p(x|z) * p(z)로 분해하면:
  = E_q[ log p(x|z) ] + E_q[ ______________ ]

Step 4: E_q[ log p(z) - log q(z|x) ] = -D_KL(q(z|x) || p(z))

결론: log p(x) >= E_q[log p(x|z)] - D_KL(q(z|x) || p(z))   [ELBO]
```

**정확성 확인**: Step 2에서 부등호가 등호가 아니라 부등식인 이유를 한
문장으로 설명하라(옌센 부등식이 언제 등식이 되는지: \\(X\\)가 상수일 때).
