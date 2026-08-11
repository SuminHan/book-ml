# Problem Set

난이도 등급: **Tier C (폴백 준비 대상)** — 아래 두 버전을 모두 준비해두고, 학생 반응을
본 뒤 선택한다.

**1.** (코딩) 다음과 같은 함수 `kl_divergence_gaussian`을 작성하라 — 정규분포
\\(\mathcal{N}(\mu, \sigma^2)\\)와 표준정규분포 \\(\mathcal{N}(0,1)\\) 사이의 KL
divergence를 닫힌 형태 공식으로 계산한다:

\\[D_{KL} = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)\\]

```python
import math

def kl_divergence_gaussian(mu, log_var):
    # log_var = log(sigma^2)
    # ADD ADDITIONAL CODE HERE!!

print(kl_divergence_gaussian(mu=0.0, log_var=0.0))  # 0.0 (표준정규분포와 동일하면 거리 0)
print(kl_divergence_gaussian(mu=2.0, log_var=0.0))  # 2.0 (평균이 멀수록 거리 커짐)
```

**2.** (코딩) 문제 1을 이용해 `vae_loss(recon_loss, mu_list, log_var_list)`를
작성하라 — 여러 잠재 차원에 대한 KL divergence를 모두 더한 뒤, 복원 손실과 합산한다.

```python
def vae_loss(recon_loss, mu_list, log_var_list):
    # ADD ADDITIONAL CODE HERE!!

print(vae_loss(recon_loss=5.0, mu_list=[0.5, -0.3], log_var_list=[0.1, -0.2]))
```

---

## 손유도 과제 — 두 가지 버전 중 택1 (교원 판단)

### [버전 A] ELBO 부등식 자유 유도 (Math for ML이 옌센 부등식/로그 성질을 다룰 경우)

\\(\log p_\theta(x) = \log \int p_\theta(x,z)\,dz\\)에서 시작해서, 옌센 부등식
(\\(\log \mathbb{E}[X] \ge \mathbb{E}[\log X]\\), \\(\log\\)가 오목함수이므로 성립)을
이용해

\\[\log p_\theta(x) \ge \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] -
D_{KL}(q_\phi(z|x)\|p(z))\\]

를 유도하라. (힌트: \\(\log p_\theta(x) = \log \int q_\phi(z|x) \frac{p_\theta(x,z)}
{q_\phi(z|x)}\,dz = \log \mathbb{E}_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z|x)}
\right]\\)로 먼저 바꾼 뒤 옌센 부등식을 적용하고, \\(p_\theta(x,z) =
p_\theta(x|z)p(z)\\)로 분해해서 KL divergence의 정의가 나오도록 정리하라.)

### [버전 B] 빈칸채움형 유도 워크시트 (폴백)

```
목표: log p(x)를 직접 계산할 수 없으니, 계산 가능한 하한(ELBO)을 찾고 싶다.

Step 1: q(z|x)를 인위적으로 곱하고 나눠서 식을 바꾼다 (값은 그대로 유지됨):
  log p(x) = log integral[ p(x,z) dz ]
           = log integral[ q(z|x) * (p(x,z) / q(z|x)) dz ]
           = log E_q[ ______________ ]   [기댓값의 정의를 이용해 정리]

Step 2: 옌센 부등식 log E[X] >= E[log X] (log가 오목함수이므로 성립)을 적용한다:
  log E_q[ p(x,z)/q(z|x) ] >= E_q[ log(______________) ]
                            = E_q[ log p(x,z) - log q(z|x) ]

Step 3: p(x,z) = p(x|z) * p(z)로 분해하면:
  E_q[ log p(x,z) - log q(z|x) ] = E_q[ log p(x|z) + log p(z) - log q(z|x) ]
                                  = E_q[ log p(x|z) ] + E_q[ ______________ ]

Step 4: E_q[ log p(z) - log q(z|x) ] = -E_q[ log q(z|x) - log p(z) ]
       이 항이 정확히 -D_KL(q(z|x) || p(z))의 정의다 (KL divergence의 정의:
       D_KL(q||p) = E_q[log q - log p])

결론: log p(x) >= E_q[log p(x|z)] - D_KL(q(z|x) || p(z))   [ELBO]
```

**정확성 확인**: Step 2에서 부등호(`>=`)가 등호(`=`)가 아니라 부등식인 이유를 한
문장으로 설명하고(옌센 부등식이 언제 등식이 되는지: \\(X\\)가 상수일 때), ELBO를
최대화하는 것이 왜 실제 우도 \\(\log p(x)\\)를 "간접적으로" 최대화하는 셈이
되는지 설명하라.

---

*교원 노트: 버전 A/B 중 선택은 Math for ML의 확률·부등식 커버리지 확인 후 결정.
확인 전까지는 버전 B를 기본값으로 준비.*
