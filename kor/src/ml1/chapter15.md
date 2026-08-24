# Chapter 15. 잠재변수 생성모델: EM/GMM에서 VAE, GAN, Diffusion까지 (Latent-Variable Generative Models)

[![Open In Colab: VAE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_vae_elbo.ipynb)
[![Open In Colab: GAN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_gan.ipynb)

1977년, 통계학자 아서 뎀스터(Arthur Dempster), 낸 레어드(Nan Laird), 도널드
루빈(Donald Rubin)[^em1977]은 "불완전한 데이터로부터의 최대우도추정"이라는 논문에서,
겉보기엔 서로 다른 여러 통계 문제들이 사실 하나의 공통된 구조 — **일부
정보가 관측되지 않았을 때(잠재변수), 그 정보를 알았다면 쉬웠을 계산을
반복적으로 근사한다** — 를 공유한다는 것을 보였다. 이번 장은 그들이 정리한
**EM**(Expectation-Maximization) 알고리즘에서 시작해, 정확히 같은 질문("관측
안 된 잠재변수가 데이터를 어떻게 만들어내는가")을 신경망으로 확장한 VAE[^vae],
그리고 완전히 다른 두 원리(적대적 학습, 점진적 노이즈 제거)로 같은 목표에
도달하는 GAN·Diffusion까지, 생성형 모델의 네 가지 얼굴을 한 장에서 훑는다.

![VAE의 방향적 그래프 모델 (원 논문 Figure 1) — 데이터 𝐱, 잠재 변수 𝐳, 파라미터 𝜃 노드와 생성 모델 p_𝜃(𝐳), p_𝜃(𝐱|𝐳) 및 인식 모델 q_𝜙(𝐳|𝐱)의 구조.](../images/ref_vae.png)

![두 개의 새로운 모델 아키텍처 — CBOW는 컨텍스트 단어로부터 현재 단어를, Skip-gram은 현재 단어로부터 주변 단어를 예측 (원 논문 Figure 1).](../images/ref_word2vec.png)

이 질문은 Chapter 14 "표현학습"의 연장선이다. 거기서 우리는 "정보를 담는
벡터로 데이터를 압축한다"는 직관을 배웠다 — PCA의 주성분, word2vec[^word2vec]의 단어
벡터. 이번 장은 그 압축된 표현에 **확률**을 입힌다. 잠재변수 \\(z\\)는
단 하나의 점사상이 아니라 데이터의 "설명"이 되고, 질문은 둘로 갈라진다 —
"관측된 \\(x\\)로부터 보이지 않는 \\(z\\)를 어떻게 추론할 것인가", 그리고
"알려진 분포에서 \\(z\\)를 뽑아 어떻게 새로운 \\(x\\)를 만들어낼
것인가". VAE의 인코더가 \\(z\\) 위의 한 점이 아니라 **분포**를 내놓는
것은, Chapter 14의 압축 직관을 딥러닝 버전으로 확률화한 모습이다. 반대로
이 장에서 배울 네 원리(EM·VAE·GAN·Diffusion)는 다음 장인 "Block B
캡스톤: 팀 프로젝트와 ML1 총정리"(Chapter 16)에서 팀 프로젝트의 모델을
고를 때 안정성·생성 속도·품질의 기준으로 선택하고 정당화할 수 있는
도구 상자가 된다. 동시에 이 장은 Block B(Chapter 9~15)를 관통해온
신경망 도구가 데이터를 설명하는 확률 모델로 완성되는 지점이다 —
이제까지 배운 모든 역전파가 이번 장에서는 잠재변수의 분포를
최적화하는 데 쓰인다.[^cs230]

이 챕터를 마치면 다음을 할 수 있다:

- EM 알고리즘의 E-step/M-step을 "잠재변수를 안다면 MLE가 쉽다"는 두 줄
  절차로 설명하고, 옌센 부등식으로 "매 반복마다 우도가 절대 줄지
  않는다"는 보증을 보일 수 있다.
- 계산 불가능한 우도를 대체하는 계산 가능한 하한 ELBO를 복원 항과 KL
  정규화 항 두 부분으로 분해하고, 직접 학습한 VAE의 손실 곡선에서 두 항의
  줄다리기가 균형점에 이르는 과정을 읽을 수 있다.
- GAN의 min-max 게임을 내쉬 균형으로 설명하고, "판별자 정확도 50%가 목표
  운영점"이라는 실전 판독과, 목적함수에 다양성 항이 없어 생기는 모드
  붕괴의 원인을 1차원 장난감 예제로 진단할 수 있다.
- EM/GMM·VAE·GAN·Diffusion 네 원리를 잠재변수의 종류, 학습 방법, 안정성,
  생성 속도의 네 축으로 비교하고, Stable Diffusion이 VAE의 압축된 잠재
  공간에서 Diffusion을 돌리는 구조임을 설명할 수 있다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [15.1 EM 알고리즘과 GMM](chapter15/1.md) — 가우시안 혼합모델(GMM)의
  "닭이 먼저냐 달걀이 먼저냐"(어느 클러스터에서 나왔는지 잠재변수 \\(z\\)와
  클러스터 모수를 둘 다 모른다)는 순환을 E-step(책임값 \\(\gamma\\)로
  소프트 라벨 채우기)/M-step(가중평균으로 모수 재추정)이 풀어내는 과정을
  손계산과 코드로 따라가고, 옌센 부등식으로 우도 단조증가 보증을 유도한
  뒤, k-means와의 차이(타원형 클러스터), BIC으로 하는 \\(K\\) 선택,
  분산 붕괴 실패 사례까지 다룬다.
- [15.2 VAE: ELBO와 실습](chapter15/2.md) — 잠재변수를 연속 벡터로,
  계산 불가능한 E-step을 신경망으로 대체한 VAE[^vae]를 배운다. 옌센 부등식 한
  번으로 ELBO를 유도하고 복원 항·KL 정규화 항의 의미를 확인한 뒤, 두
  봉우리 데이터로 작은 VAE를 실제로 학습해 표준정규분포에서 \\(z\\)를 뽑아
  디코더만 통과시키는 생성 모드까지 확인한다(리파라미터화 트릭, β-VAE[^beta_vae],
  "흐릿한" 생성 이미지의 원인 포함).
- [15.3 GAN, Diffusion, 그리고 네 원리 비교](chapter15/3.md) — 우도
  기반과 다른 두 패러다임을 만난다. GAN[^gan]은 생성자·판별자의 min-max 게임
  (내쉬 균형, 1차원 장난감의 균형점 계산, 모드 붕괴, WGAN[^wgan]·CycleGAN[^cyclegan]·
  StyleGAN[^stylegan]까지)이고, Diffusion은 노이즈를 한 단계씩 제거하는 MSE 회귀
  문제(DDPM[^ddpm]/DDIM[^ddim], VAE 잠재 공간을 쓰는 Stable Diffusion[^stablediffusion])다. 마지막으로
  네 원리를 잠재변수/학습 방법/안정성/생성 속도로 나란히 비교해 장을
  마무리한다.

![Figure 1: GAN이 학습되면서 샘플 품질이 개선되는 과정을 MNIST로 보여준다 — 판별기 D와 생성기 G가 동시에 갱신되며 (a)에서 (d)로 갈수록 생성 샘플이 실제 데이터에 가까워진다.](../images/ref_gan.png)

[^word2vec]: Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781.
[^vae]: Kingma, D. P., Welling, M. (2013). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
[^gan]: Goodfellow, I. J. et al. (2014). "Generative Adversarial Networks." arXiv:1406.2661.
[^ddpm]: Ho, J., Jain, A., Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020. arXiv:2006.11239.
[^stablediffusion]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. arXiv:2112.10752.
[^beta_vae]: Higgins, I., Matthey, L., Pal, A., et al. (2017). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR 2017. arXiv:1806.03916.
[^ddim]: Song, J., Meng, C., Ermon, S. (2021). "Denoising Diffusion Implicit Models." ICLR 2021. arXiv:2010.02502.
[^wgan]: Arjovsky, M., Chintala, S., Bottou, L. (2017). "Wasserstein GAN." arXiv:1701.07875.
[^stylegan]: Karras, T., Laine, S., Aila, T. (2018). "A Style-Based Generator Architecture for Generative Adversarial Networks." arXiv:1812.04948.
[^cs230]: Stanford CS230: Deep Learning. https://cs230.stanford.edu/
[^cyclegan]: Zhu, J.-Y., Park, T., Isola, P., Efros, A. A. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." CVPR 2017. arXiv:1703.10593.
[^em1977]: Dempster, A. P., Laird, N. M., Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." Journal of the Royal Statistical Society, Series B (Methodological), 39(1), 1-38. https://doi.org/10.1111/j.2517-6161.1977.tb01682.x
