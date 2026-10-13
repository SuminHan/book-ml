# slides/v2 — 차시별 수업 슬라이드 (새 버전)

`slides/kor/` 는 주 단위(3시간) 덱이라 길고 글이 많다. v2 는 **차시(50분) 하나 = 개념 하나 = tex 한 파일 = PDF 한 개**.
쇼츠를 만들며 굳힌 방식(한 장 한 생각, 직접 세기 → 공식, 비유 하나로 통일)을 슬라이드에 그대로 쓴다.
작업 규칙·사용자 취향은 스킬 `slide-v2` (`book-ml/.claude/skills/slide-v2/SKILL.md`) 에 모여 있다.

## 폴더 규칙 (파일 이름은 영어, 주제 이름을 앞에 붙인다)
```
slides/v2/<ml1|ml2>/weekNN/<주제>/      # 주차 폴더 아래 주제명. 폴더 이름 = 주제 = tex 이름
├── <주제>.tex                  덱 전체, 한 파일 (pages/ 로 쪼개지 않는다). 프레임마다 \note{발표자 노트}
├── <주제>.pdf                  빌드 결과
├── <주제>_speaker_notes.md     \note 를 하단 번호별로 모은 것 (deck.py build 가 자동 생성)
├── data.py                     슬라이드·그림이 쓰는 숫자 (한 곳에서만 정한다)
├── make_figs.py                figs/*.png 원본 (그림은 손으로 고치지 말고 여기를 고쳐 다시 만든다)
├── figs/
└── lab/
    ├── make_lab.py             노트북 원본 (여기를 고치고 실행)
    ├── <주제>.ipynb            학생용 (빈칸 = `return None  # TODO: …`)
    └── <주제>_solution.ipynb   정답
slides/v2/tools/deck.py         빌드·쪽 찾기·렌더·전체 미리보기·노트 생성
```
- 테마는 공용 `slides/theme/ksa-theme.tex` (tex 에서 `../../../../theme/ksa-theme.tex`). 시키지 않으면 건드리지 않는다.
- 로고 때문에 `\graphicspath{{figs/}{../../../../assets/}}` 가 있어야 한다. 폴더 깊이가 바뀌면 이 두 경로도 바꾼다.

## 도구 (`slides/v2/tools/deck.py`, 인자는 덱 폴더)
```
python3 slides/v2/tools/deck.py build  <덱폴더>        # 빌드 + 오류/Overfull(어느 프레임인지) + 쪽 수 + 노트 md 재생성
python3 slides/v2/tools/deck.py where  <덱폴더> 8 12   # 하단 번호 8/N, 12/N 의 프레임 제목과 tex 줄
python3 slides/v2/tools/deck.py show   <덱폴더> 8      # 하단 번호 8 쪽 png -> /tmp/deck/<주제>-8.png
python3 slides/v2/tools/deck.py sheet  <덱폴더>        # 전체를 12장씩 모은 미리보기 (리뷰용)
```
- **쪽 번호**: 사용자가 말하는 "8/24" 는 하단 번호. 표지는 번호가 없어서 PDF 쪽 = 하단 번호 + 1. 도구는 전부 하단 번호를 받는다.
- 그림·노트북은 시스템 python 에 matplotlib/PIL 이 없으므로 `~/Documents/manim_shorts/_env/v/bin/python` 으로 돌린다.
  (`deck.py sheet` 는 알아서 이 파이썬으로 다시 실행한다.)
  ```
  cd <덱폴더> && ~/Documents/manim_shorts/_env/v/bin/python make_figs.py
  cd <덱폴더>/lab && ~/Documents/manim_shorts/_env/v/bin/python make_lab.py   # 노트북 2개 생성 + 정답 실행 검증
  ```
- 학생 환경 확인용 conda 환경 `lab` (JupyterLab): `~/miniconda3/envs/lab/bin/jupyter lab <덱폴더>/lab/`

## 만드는 순서 (새 차시)
1. **숫자·비유를 먼저 `data.py` 에 고정**한다. 슬라이드·그림·실습이 같은 숫자를 쓴다. 표에 나오는 개수는 **정수**가 되게 고른다 (0.7통 X).
2. **장면 설계**: 한 장 한 생각. 작은 구체 예 → 직접 세기 → 공식 → 문제 → 해결 → 유도 → 결과(크게) → 문제풀이 → 실습 → 정리.
3. **초안**: 반복·대량 부분만 `lq` 에 맡긴다 (사용자가 시킬 때만). 수치·수식은 원문과 대조한다.
4. 그림은 `make_figs.py`, 노트북은 `lab/make_lab.py` (실제 공개 데이터, 처음 실행 때 내려받기).
5. `deck.py build` → 고친 쪽만 `deck.py show` 로 확인 → 마지막에 `deck.py sheet` 로 전체 훑기.

## 기존 자료와의 관계
- `lessons/ml1/<주제>/` 는 쇼츠·개념영상·리허설 파이프라인(`lessons/_tools`, `src/lesson.py` 의 `SLIDES`).
  v2 `ml1/week03/naive_bayes` 는 거기서 옮겨 와 손으로 고친 것이고, 지금은 v2 쪽이 원본이다.
  `lessons/` 와 달라진 점: 베르누이 모델로 통일, '당첨' 정상 확률 0.02 → 0.20, 실습은 영어 SMS 실데이터.

## 진행 현황

강의 단위 = 쇼츠 주제 하나: **강의 20~30분 + 문제풀이 / Colab** (2026-10-13 개편). 주차는 대략적인 배치이고,
배정 근거는 `course-prep/ML1_2027_주차별_쇼츠배분.md` (v4). 폴더마다 README 가 원본 쇼츠(`lessons/ml1/<주제>/`)를 가리킨다.

| 주 | 주제 폴더 | 내용 | 상태 |
|---|---|---|---|
| 1 | `week01/course_intro` | 첫 강의: AI는 과학인가 공학인가 · 역사 · 이 수업의 방식 (50분, 이 덱은 그대로 둔다) | 29쪽 (표지 + 28) |
| 1 | `week01/overfitting` | 과대적합과 과소적합 | — |
| 1 | `week01/dirty_data` | 더티 데이터 (1주 3차시) | 14쪽 · 실습: 더럽힌 iris 청소 |
| 2 | `week02/linear_regression` | 선형회귀 | 16쪽 · 실습: 경사하강법 구현 + Auto MPG |
| 2 | `week02/sgd` | SGD 확률적 경사하강법 | 23쪽 · 실습: diabetes SGD + PyTorch MNIST |
| 2 | `week02/cross_entropy` | 교차 엔트로피 | 15쪽 · 실습: 유방암 CE vs MSE + 셰익스피어 다음 글자 |
| 2 | `week02/precision_recall` | 정밀도와 재현율 | — |
| 2 | `week02/pr_auc` | PR-AUC | — |
| 3 | `week03/gaussian` | 가우시안 분포 | — |
| 3 | `week03/naive_bayes` | 나이브 베이즈 | 25쪽 · 실습: UCI SMS Spam |
| 4 | `week04/knn` | kNN 최근접 이웃 | — |
| 4 | `week04/kmeans` | k-means 클러스터링 | — |
| 4 | `week04/em_algorithm` | EM 알고리즘 | — |
| 5 | `week05/svm` | 서포트 벡터 머신 | — |
| 7 | `week07/decision_tree` | 결정 트리 | — |
| 7 | `week07/entropy` | 섀넌 엔트로피 | — |
| 7 | `week07/random_forest` | 랜덤 포레스트 | — |
| 7 | `week07/gbdt_shap` | GBDT와 SHAP | — |
| 8 | `week08/ablation` | Ablation 테스트 | — |
| 8 | `week08/good_researcher` | 좋은연구자가되려면 | — |
| 8 | `week08/peer_review` | AI 논문 리뷰 | — |
| 9 | `week09/xor_backprop` | XOR과 역전파 | — |
| 9 | `week09/vanishing_gradient` | 기울기 소실 Vanishing Gradient | — |
| 9 | `week09/dropout` | Dropout 드롭아웃 | — |
| 9 | `week09/batchnorm` | BatchNorm 배치 정규화 | — |
| 9 | `week09/momentum_adam` | 모멘텀과 Adam | — |
| 10 | `week10/cnn` | CNN 합성곱 신경망 | — |
| 10 | `week10/resnet` | ResNet 잔차 신경망 | — |
| 10 | `week10/data_augmentation` | 데이터 증강 Data Augmentation | — |
| 11 | `week11/rnn` | RNN 순환 신경망 | — |
| 11 | `week11/lstm` | LSTM 장단기 메모리 | — |
| 12 | `week12/attention` | 어텐션 | — |
| 12 | `week12/seq2seq` | Seq2Seq 시퀀스투시퀀스 | — |
| 12 | `week12/transformer` | 트랜스포머 | — |
| 13 | `week13/gcn` | GCN 그래프 신경망 | — |
| 14 | `week14/pca` | PCA 주성분 분석 | — |
| 14 | `week14/tsne` | t-SNE 데이터 지도 | — |
| 14 | `week14/word2vec` | word2vec | — |
| 14 | `week14/embedding` | 표상과 임베딩 | — |
| 14 | `week14/bert` | BERT 빈칸 채우기 | — |
| 15 | `week15/vae` | VAE 변분 오토인코더 | — |
| 15 | `week15/kl_divergence` | KL 발산 | — |
| 15 | `week15/gan` | GAN 생성적 적대 신경망 | — |
| 15 | `week15/diffusion` | Diffusion 확산 모델 | — |
| 16 | `week16/review_validity` | 리뷰어 팩트체크 | — |
| 16 | `week16/review_cutoff` | ICLR 커트라인 | — |
| 17 | `week17/gpt` | GPT 다음 단어 맞히기 | — |
| 17 | `week17/tokenizer_bpe` | 토큰화 BPE | — |
| 17 | `week17/cot` | Chain-of-Thought 생각의 사슬 | — |
| 17 | `week17/temperature` | 온도 Temperature | — |
| 17 | `week17/rlhf` | RLHF 사람 피드백 강화학습 | — |

6주차(정규화·교차검증·분리)는 맞는 쇼츠가 없어 비어 있다. 10주의 alexnet 복습은 `week01/course_intro` 로 대신한다.
