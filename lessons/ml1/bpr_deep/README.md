<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# BPR과 딥러닝

*딥러닝은 BPR의 어디를 바꿨을까*

> BPR + 딥러닝, 추천 연구는 어디를 바꿨을까?

## 영상 내용

- 2009년의 BPR은 두 부품으로 볼 수 있어요. 사용자와 영화를 숫자 묶음(벡터)으로 바꿔 곱하는 '점수 계산기', 그리고 본 영화를 안 본 영화보다 위로 올리는 '순서 학습(BPR 손실)'. 딥러닝 연구들은 주로 점수 계산기를 바꿨어요.

- • VBPR (He & McAuley, AAAI 2016): 미리 학습된 CNN으로 상품 이미지의 특징을 뽑아 점수에 더해요. 관객이 없는 신작도 이미지로 추천할 수 있어 콜드 스타트를 완화해요.
- • NCF (He 외, WWW 2017): 두 벡터를 곱하는 대신, 신경망(MLP)에 넣어 점수를 계산해요.
- • NGCF (Wang 외, SIGIR 2019) · LightGCN (He 외, SIGIR 2020): 사용자–아이템 연결 그래프에서 이웃 정보를 모아 벡터를 만들어요. 학습은 여전히 BPR 손실이에요.
- • 반전: LightGCN은 NGCF에서 특징 변환과 비선형 활성화를 덜어냈는데 오히려 더 좋아졌어요. Rendle 외(RecSys 2020)도 하이퍼파라미터를 잘 고른 내적(곱셈)이 NCF의 MLP보다 낫다고 보였어요.

- → 순서 학습은 BPR 그대로, 계산기는 진화, 그리고 깊다고 늘 좋은 건 아니에요.

- 📚 References
- [1] S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.
- [2] R. He, J. McAuley, "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback", AAAI 2016. https://arxiv.org/abs/1510.01784
- [3] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T.-S. Chua, "Neural Collaborative Filtering", WWW 2017. https://arxiv.org/abs/1708.05031
- [4] X. Wang, X. He, M. Wang, F. Feng, T.-S. Chua, "Neural Graph Collaborative Filtering", SIGIR 2019. https://arxiv.org/abs/1905.08108
- [5] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020. https://arxiv.org/abs/2002.02126
- [6] S. Rendle, W. Krichene, L. Zhang, J. Anderson, "Neural Collaborative Filtering vs. Matrix Factorization Revisited", RecSys 2020. https://arxiv.org/abs/2005.09683

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 딥러닝 시대, 추천 연구는 / BPR의 어디를 바꿨을까? | 딥러닝 시대, 추천 연구는 비피알의 어디를 바꿨을까요? |
| 2 | 점수 계산기 + 순서 학습(BPR) / 연구들은 주로 계산기를 바꿨다 | 비피알은 두 부품이에요. 사용자와 영화를 숫자 묶음으로 곱하는 점수 계산기, 본 영화를 위로 올리는 순서 학습. 연구들은 주로 계산기를 바꿨어요. |
| 3 | VBPR (2016): 이미지를 읽는 신경망(CNN) / 관객 0명인 신작도 포스터로 추천 | 2016년 브이비피알은, 포스터를 씨엔엔으로 읽어 점수에 더했어요. 관객 0명인 신작도 추천할 수 있죠. |
| 4 | NCF (2017) / 곱셈 대신 신경망으로 점수 계산 | 2017년 엔씨에프는, 곱하는 대신 신경망으로 점수를 계산했어요. |
| 5 | NGCF (2019) · LightGCN (2020) / 연결 그래프에서 이웃 정보 모으기 + BPR 학습 | 2019년 엔지씨에프와 2020년 라이트 지씨엔은, 연결 그래프에서 이웃 정보를 모아요. 학습은 여전히 비피알. |
| 6 | 반전: 덜어낸 LightGCN이 더 좋았다 / 잘 다듬은 곱셈 > 신경망 (Rendle 외, 2020) | 반전은, 신경망 부품을 덜어낸 라이트 지씨엔이 더 좋았다는 것. 렌들도, 잘 다듬은 곱셈이 신경망보다 낫다고 보였죠. |
| 7 | 순서 학습은 BPR 그대로 · 계산기는 진화 / 깊다고 늘 좋은 건 아니다 | 정리하면, 순서 학습은 비피알 그대로, 계산기는 진화했고, 깊다고 늘 좋은 건 아니에요. |

## 출처

포스터: 협업 필터링·BPR 편과 같음 (각 영화 위키백과 문서 포스터, 사용자 판단으로 사용)
사실 출처
- S. Rendle 외, "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
- R. He, J. McAuley, "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback", AAAI 2016 — 사전 학습 CNN 이미지 특징, 콜드 스타트 완화
- X. He 외, "Neural Collaborative Filtering", WWW 2017 — 내적 대신 MLP
- X. Wang 외, "Neural Graph Collaborative Filtering", SIGIR 2019 — 사용자-아이템 그래프 전파, BPR 손실
- X. He 외, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020 — 특징 변환·비선형 활성화 제거, BPR 손실
- S. Rendle, W. Krichene, L. Zhang, J. Anderson, "Neural Collaborative Filtering vs. Matrix Factorization Revisited", RecSys 2020 — 잘 튜닝한 내적 > MLP

## 파일

- 영상: `~/book-ml/videos/주제별/BPR과딥러닝/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
