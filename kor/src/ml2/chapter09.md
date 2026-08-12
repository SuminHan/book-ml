# Chapter 9. 생성형 모델 I: 우도 기반 (Generative Models I: Likelihood-Based)

ML1의 마지막 장에서 오토인코더가 남긴 질문을 기억하는가 — "병목의 값을
무작위로 하나 골라서 디코더에 넣으면, 존재한 적 없는 새로운 데이터를
만들 수 있지 않을까?" 2013년 디데릭 킹마(Diederik Kingma)와 맥스
웰링(Max Welling)이 발표한 **변분 오토인코더**(Variational
Autoencoder, VAE)가 바로 이 질문에 대한 정식 답이다.

## 9.1 오토인코더를 그냥 쓰면 안 되는 이유

일반 오토인코더의 잠재 공간(latent space)은 "어떤 값이 그럴듯한
데이터로 복원되는지"에 대한 보장이 없다 — 학습 데이터들이 잠재 공간의
여기저기 흩어져 있을 수 있고, 그 사이 빈 공간에서 무작위로 값을 뽑으면
무엇이 나올지 알 수 없다. VAE의 해법은 인코더가 하나의 점 \\(z\\)를
내놓는 대신, **확률분포**(보통 정규분포의 평균과 분산)를 내놓도록
강제하는 것이다. 그리고 그 분포가 표준 정규분포(평균 0, 분산 1)에
가깝도록 추가 제약을 건다 — 그러면 잠재 공간 전체가 매끄럽고 빈틈없이
채워져서, 임의의 지점에서 샘플링해도 그럴듯한 데이터가 나올 가능성이
높아진다.

## 9.2 VAE의 구조

- **인코더** \\(q_\phi(z|x)\\): 입력 \\(x\\)를 받아, 잠재변수 \\(z\\)의
  확률분포를 내놓는다(보통 정규분포 \\(\mathcal{N}(\mu(x),
  \sigma^2(x))\\)의 평균과 분산을 출력).
- **샘플링**: 그 분포에서 \\(z\\)를 하나 뽑는다.
- **디코더** \\(p_\theta(x|z)\\): \\(z\\)로부터 \\(x\\)를 복원(또는
  생성)한다.

## 9.3 우도라는 관점

VAE는 "생성형 모델을 만드는 세 가지 원리" 중 첫 번째 — **우도
기반**(likelihood-based) 접근이다: 모델이 학습 데이터를 만들어낼
확률(우도) \\(P(x)\\)를 직접(또는 그 근사치를) 최대화하도록 학습한다.
문제는 \\(p_\theta(x) = \int p_\theta(x|z)p(z)\,dz\\)는 가능한 모든
\\(z\\)에 대한 적분이라, 일반적으로 계산이 불가능하다(intractable).

## 9.4 ELBO: 계산 가능한 하한으로 우회하기

직접 계산할 수 없는 \\(\log p_\theta(x)\\) 대신, 다음이 성립함을 보일
수 있다:

\\[\log p\_\theta(x) \ge \mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)] -
D\_{KL}\big(q\_\phi(z|x) \\,\\|\\, p(z)\big)\\]

우변을 **ELBO**(Evidence Lower BOund)라 부른다. 두 항의 의미:

- 첫 항 \\(\mathbb{E}[\log p_\theta(x|z)]\\): **복원 항** — 인코더가
  만든 \\(z\\)로 디코더가 원본 \\(x\\)를 얼마나 잘 복원하는가(일반
  오토인코더의 복원 오차와 본질적으로 같다).
- 둘째 항 \\(D_{KL}(q_\phi(z|x)\|p(z))\\): **정규화 항** — 인코더가
  만든 분포 \\(q_\phi(z|x)\\)가, 목표로 하는 사전분포 \\(p(z)\\)(보통
  표준정규분포)에서 얼마나 벗어나 있는지(KL divergence, 두 확률분포
  사이의 "거리"). 이 항이 바로 잠재 공간을 매끄럽게 만드는 힘이다.

\\(\log p_\theta(x) \ge \text{ELBO}\\)이므로, **ELBO를 최대화하면 실제
우도의 하한도 함께 올라간다** — 직접 계산 못 하는 목표를, 계산 가능한
대리(surrogate) 목표로 바꿔치기한 것이다. 이 부등식 자체를 옌센
부등식(Jensen's inequality)으로 유도하는 것이 이번 장 연습문제의
핵심이다.

```python
def vae_loss(x, x_reconstructed, mu, log_var):
    # 복원 손실 (여기서는 MSE로 근사)
    recon_loss = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x)))
    # KL divergence: 정규분포 q(mu, sigma^2)와 표준정규분포 N(0,1) 사이의 닫힌 형태 공식
    kl_loss = -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                          for i in range(len(mu)))
    return recon_loss + kl_loss  # ELBO를 최대화 = 이 손실(음의 ELBO)을 최소화
```

## 9.5 Reparameterization Trick (참고)

\\(z\\)를 확률분포에서 직접 샘플링하면, "샘플링"이라는 연산은 미분이
안 돼서 역전파가 인코더까지 흘러가지 못한다. VAE는 \\(z = \mu + \sigma
\odot \epsilon\\)(\\(\epsilon \sim \mathcal{N}(0,1)\\)은 무작위성을
밖으로 빼낸 상수 취급)으로 샘플링을 다시 쓰는 트릭(reparameterization
trick)으로 이 문제를 우회한다 — 이제 \\(\mu, \sigma\\)에 대한 미분이
가능해져 역전파가 정상적으로 흐른다. 자세한 유도는 이 학기 범위를
넘어서지만, "왜 그냥 샘플링하면 안 되는가"라는 질문 자체는 기억해둘
만하다.

**다음 장에는 완전히 다른 두 원리(적대적, 스코어 기반)를 배운다 — 세
원리를 나란히 놓고 비교하면, "그럴듯한 새 데이터를 만든다"는 같은
목표에 얼마나 다른 방식으로 접근할 수 있는지가 뚜렷하게 드러난다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `kl_divergence_gaussian`(정규분포
\\(\mathcal{N}(\mu, \sigma^2)\\)와 표준정규분포 사이의 KL divergence,
\\(D_{KL} = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)\\))과
`vae_loss`를 완성하라(핵심 줄은 빈칸으로 남겨져 있다고 가정):

```python
import math

def kl_divergence_gaussian(mu, log_var):
    # log_var = log(sigma^2)
    # ADD ADDITIONAL CODE HERE!!

print(kl_divergence_gaussian(mu=0.0, log_var=0.0))  # 0.0
print(kl_divergence_gaussian(mu=2.0, log_var=0.0))  # 2.0

def vae_loss(recon_loss, mu_list, log_var_list):
    # ADD ADDITIONAL CODE HERE!!
    # 여러 잠재 차원에 대한 KL divergence를 모두 더한 뒤, 복원 손실과 합산

print(vae_loss(recon_loss=5.0, mu_list=[0.5, -0.3], log_var_list=[0.1, -0.2]))
```

**2. (손유도, Tier C — 폴백 준비 대상)** \\(\log p_\theta(x) = \log
\int p_\theta(x,z)\,dz\\)에서 시작해서, 옌센 부등식(\\(\log
\mathbb{E}[X] \ge \mathbb{E}[\log X]\\))을 이용해

\\[\log p\_\theta(x) \ge \mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)] -
D\_{KL}(q\_\phi(z|x)\\|p(z))\\]

를 유도하라(힌트: \\(\log p\_\theta(x) = \log \mathbb{E}\_{q\_\phi}
\left[\frac{p\_\theta(x,z)}{q\_\phi(z|x)}\right]\\)로 먼저 바꾼 뒤 옌센
부등식을 적용하고, \\(p_\theta(x,z) = p_\theta(x|z)p(z)\\)로 분해해서
KL divergence의 정의가 나오도록 정리하라).

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

**정확성 확인**: Step 2에서 부등호(`>=`)가 등호(`=`)가 아니라 부등식인
이유를 한 문장으로 설명하고(옌센 부등식이 언제 등식이 되는지:
\\(X\\)가 상수일 때), ELBO를 최대화하는 것이 왜 실제 우도
\\(\log p(x)\\)를 "간접적으로" 최대화하는 셈이 되는지 설명하라.
