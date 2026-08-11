# Topics Covered

## VAE의 구조

- **인코더** \\(q_\phi(z|x)\\): 입력 \\(x\\)를 받아, 잠재변수 \\(z\\)의 확률분포를
  내놓는다(보통 정규분포 \\(\mathcal{N}(\mu(x), \sigma^2(x))\\)의 평균과 분산을
  출력).
- **샘플링**: 그 분포에서 \\(z\\)를 하나 뽑는다.
- **디코더** \\(p_\theta(x|z)\\): \\(z\\)로부터 \\(x\\)를 복원(또는 생성)한다.

## 우도를 직접 계산할 수 없는 이유

생성형 모델의 목표는 데이터 \\(x\\)가 나올 확률(우도) \\(p_\theta(x)\\)를 최대화하는
것이다. 그런데 \\(p_\theta(x) = \int p_\theta(x|z)p(z)\,dz\\)는 가능한 모든
\\(z\\)에 대한 적분이라, 일반적으로 계산이 불가능하다(intractable).

## ELBO: 계산 가능한 하한으로 우회하기

직접 계산할 수 없는 \\(\log p_\theta(x)\\) 대신, 다음이 성립함을 보일 수 있다:

\\[\log p_\theta(x) \ge \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] -
D_{KL}\big(q_\phi(z|x) \,\|\, p(z)\big)\\]

우변을 **ELBO**(Evidence Lower BOund)라 부른다. 두 항의 의미:

- 첫 항 \\(\mathbb{E}[\log p_\theta(x|z)]\\): **복원 항** — 인코더가 만든 \\(z\\)로
  디코더가 원본 \\(x\\)를 얼마나 잘 복원하는가 (일반 오토인코더의 복원 오차와
  본질적으로 같다).
- 둘째 항 \\(D_{KL}(q_\phi(z|x)\|p(z))\\): **정규화 항** — 인코더가 만든 분포
  \\(q_\phi(z|x)\\)가, 목표로 하는 사전분포 \\(p(z)\\)(보통 표준정규분포)에서
  얼마나 벗어나 있는지(KL divergence, 두 확률분포 사이의 "거리"). 이 항이 바로
  잠재 공간을 매끄럽게 만드는 힘이다.

\\(\log p_\theta(x) \ge \text{ELBO}\\)이므로, **ELBO를 최대화하면 실제 우도의 하한도
함께 올라간다** — 직접 계산 못 하는 목표를, 계산 가능한 대리(surrogate) 목표로
바꿔치기한 것이다. 이번 주 손유도 과제는 이 부등식 자체를 옌센 부등식(Jensen's
inequality)으로 유도하는 것이다.

```python
def vae_loss(x, x_reconstructed, mu, log_var):
    # 복원 손실 (여기서는 MSE로 근사)
    recon_loss = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x)))
    # KL divergence: 정규분포 q(mu, sigma^2)와 표준정규분포 N(0,1) 사이의 닫힌 형태 공식
    kl_loss = -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                          for i in range(len(mu)))
    return recon_loss + kl_loss  # ELBO를 최대화 = 이 손실(음의 ELBO)을 최소화
```

## Reparameterization Trick (참고)

\\(z\\)를 확률분포에서 직접 샘플링하면, "샘플링"이라는 연산은 미분이 안 돼서
역전파가 인코더까지 흘러가지 못한다. VAE는 \\(z = \mu + \sigma \odot \epsilon\\)
(\\(\epsilon \sim \mathcal{N}(0,1)\\)은 무작위성을 밖으로 빼낸 상수 취급)으로
샘플링을 다시 쓰는 트릭(reparameterization trick)으로 이 문제를 우회한다 — 이제
\\(\mu, \sigma\\)에 대한 미분이 가능해져 역전파가 정상적으로 흐른다. 자세한
유도는 이 학기 범위를 넘어서지만, "왜 그냥 샘플링하면 안 되는가"라는 질문 자체는
기억해둘 만하다.
