# Chapter 15. 잠재변수 생성모델: EM/GMM에서 VAE, GAN, Diffusion까지 (Latent-Variable Generative Models)

[![Open In Colab: VAE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_vae_elbo.ipynb)
[![Open In Colab: GAN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_gan.ipynb)

1977년, 통계학자 아서 뎀스터(Arthur Dempster), 낸 레어드(Nan Laird), 도널드
루빈(Donald Rubin)은 "불완전한 데이터로부터의 최대우도추정"이라는 논문에서,
겉보기엔 서로 다른 여러 통계 문제들이 사실 하나의 공통된 구조 — **일부
정보가 관측되지 않았을 때(잠재변수), 그 정보를 알았다면 쉬웠을 계산을
반복적으로 근사한다** — 를 공유한다는 것을 보였다. 이번 장은 그들이 정리한
**EM**(Expectation-Maximization) 알고리즘에서 시작해, 정확히 같은 질문("관측
안 된 잠재변수가 데이터를 어떻게 만들어내는가")을 신경망으로 확장한 VAE,
그리고 완전히 다른 두 원리(적대적 학습, 점진적 노이즈 제거)로 같은 목표에
도달하는 GAN·Diffusion까지, 생성형 모델의 네 가지 얼굴을 한 장에서 훑는다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [15.1 EM 알고리즘과 GMM](chapter15/1.md)
- [15.2 VAE: ELBO와 실습](chapter15/2.md)
- [15.3 GAN, Diffusion, 그리고 네 원리 비교](chapter15/3.md)
