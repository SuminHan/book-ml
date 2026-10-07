<!-- make_readme.py 로 자동 생성: 고치려면 lines.json · 캡션 · SOURCES.md 를 고치고 다시 실행 -->
# 약물 재창출

*추천 기술로 병에 맞는 약 찾기*

> 약물 재창출(Drug Repositioning), 영화 추천 기술로 병에 맞는 약을 찾는다?

## 영상 내용

- 추천 시스템에서 사용자를 질병으로, 영화를 약으로 바꾸면 '이 병에 아직 안 써 본 약 중 효과 있을 만한 약'을 찾는 문제가 돼요. 이미 알려진 치료 관계가 '본 영화' 역할을 하죠.

- • 새 약을 처음부터 개발하면 보통 10년 이상, 수조 원이 들어요. 혈압약이던 미녹시딜이 탈모약이 된 것처럼, 이미 안전성이 확인된 약의 새 쓰임을 찾는 게 약물 재창출이에요.
- • 약끼리의 유사도, 병끼리의 유사도, 알려진 약–질병 관계를 하나의 그래프로 묶고, 질병에서 출발해 걷다가 가끔 출발점으로 돌아오는 재시작 랜덤 워크(RWR)로 가까운 약을 찾아요. 예시 그래프에서 질병 A의 1순위 후보는 비슷한 병 B에 쓰는 약 3 (1,000걸음마다 127번).
- • 실제 연구(TP-NRWRH)에서는 알츠하이머 후보 상위 10개 중 9개가 이미 신경퇴행성 질환에 승인됐거나 시험 중인 약이었어요.
- • 한국의 유전자동의보감사업단(2012년부터, KAIST 이도헌 교수 중심)은 한 걸음 더 들어가, 천연물(한약재) 속 여러 성분이 몸속 여러 표적에 동시에 작용하는 원리(다성분-다표적)를 가상 인체 모델 등으로 밝히려 했어요. 그 성과로 디지털 가상인체(CODA)를 비롯한 5대 원천기술을 개발했고, 2022년 7월 10여 년의 연구성과를 공개했어요.

**참고**: 질병 A·B·C, 약 1~4는 설명용 가상 예시예요. 특정 약의 효능을 말하는 영상이 아니에요.

- 📚 References
- [1] H. Liu 외, "Inferring new indications for approved drugs via random walk on drug-disease heterogenous networks", BMC Bioinformatics 17 (2016).
- [2] X. Chen 외, "Drug–target interaction prediction by random walk on the heterogeneous network", Molecular BioSystems 8 (2012).
- [3] H. Yu 외, "CODA: Integrating multi-level context-oriented directed associations for analysis of drug effects", Scientific Reports 7 (2017). https://doi.org/10.1038/s41598-017-07448-6
- [4] (재)유전자동의보감사업단 소개, biosynergy.re.kr

## 장면별 자막과 대사

| # | 화면 자막 | 대사 |
|---|---|---|
| 1 | 영화 추천 기술로 / 병에 맞는 약도 찾을 수 있을까? | 영화를 추천하던 기술로, 병에 맞는 약도 찾을 수 있을까요? |
| 2 | 사용자 → 질병 · 영화 → 약 / 본 영화 → 이미 알려진 치료 관계 | 사용자 대신 질병, 영화 대신 약. 이미 알려진 치료 관계가 본 영화예요. |
| 3 | 신약 개발: 10년 이상 · 수조 원 / 있는 약의 새 쓰임 찾기 = 약물 재창출 | 신약은 10년 넘게, 수조 원이 들어요. 혈압약이던 미녹시딜이, 탈모약이 된 것처럼, 있는 약의 새 쓰임을 찾는 게 약물 재창출. |
| 4 | 약–약 · 병–병 · 알려진 치료를 한 그래프로 / 질병에서 걷다 돌아오기 (재시작 랜덤 워크) | 약끼리, 병끼리 비슷함과 알려진 치료를 한 그래프로 묶고, 질병에서 걷다 돌아오는 랜덤 워크로 가까운 약을 찾아요. |
| 5 | 질병 A의 1순위 후보: 약 3 / 실제 연구: 알츠하이머 후보 10개 중 9개 이미 사용·시험 중 | 계산하면 질병 A에는, 비슷한 병 B에 쓰는 약 3이 1순위. 실제 연구에서도 알츠하이머 후보 10개 중 9개가, 이미 쓰이거나 시험 중이었어요. |
| 6 | 유전자동의보감사업단 (2012~) / 가상 인체 CODA 등 5대 원천기술 | 한국의 유전자동의보감 사업단은, 한약재 속 여러 성분이 여러 표적에 닿는 원리를 가상 인체로 풀어, 코다 등 다섯 원천기술을 내놓았어요. |
| 7 | 질병·약도 추천 문제 → 그래프로 묶어 걷기 / → 있는 약의 새 쓰임 찾기 | 정리하면, 질병과 약도 추천 문제, 그래프로 묶어 걷고, 있는 약의 새 쓰임을 찾아요. |

## 출처

사실 출처
- 미녹시딜: 경구 혈압약으로 개발 → 털이 자라는 부작용에서 바르는 탈모약으로 (대표적 약물 재창출 사례)
- 신약 개발 기간·비용: 통상 10~15년, 수조 원 규모 (예: DiMasi 외 2016 추정 약 26억 달러)
- TP-NRWRH: H. Liu 외, "Inferring new indications for approved drugs via random walk on drug-disease heterogenous networks", BMC Bioinformatics 2016 — 알츠하이머 사례, 예측 상위 10개 중 9개가 신경퇴행성 질환에 승인 또는 시험 중
- NRWRH: X. Chen 외, Molecular BioSystems 2012 (약물-표적 상호작용 예측)
- 유전자동의보감사업단: 2012년 6월부터 10년, 과기정통부 바이오·의료기술개발사업, KAIST 이도헌 교수 중심. 천연물 복합 성분의 다성분-다표적(MCMT) 작용을 가상 인체 모델 등으로 규명 (biosynergy.re.kr)
예시 그래프(질병 A·B·C, 약 1~4)는 설명용 가상 데이터
성과 출처: 사업단 5대 원천기술(디지털 가상인체 CODA, 초고속 소재발굴 iHTac, 천연물 분자표적 발굴 LARIAT, 천연물 바이오마커 Synergy Marker, 인체적용 중개기술 ICAB) — 사업단 소개 자료·한국강사신문 기사; 2022년 7월 10여 년 연구성과 세미나·전시회 (KAIST 바이오및뇌공학과 학과뉴스)
CODA 논문: H. Yu, J. Jung, S. Yoon, M. Kwon, S. Bae, S. Yim, J. Lee, S. Kim, Y. Kang, D. Lee, "CODA: Integrating multi-level context-oriented directed associations for analysis of drug effects", Scientific Reports 7 (2017), doi:10.1038/s41598-017-07448-6
(PharmDB-K는 이도헌 교수 저자 목록에 없어 제외)

## 파일

- 영상: `~/book-ml/videos/주제별/약물재창출/` (영상·제작 원본은 git 에 올리지 않음)
- 다시 만들기: `~/Documents/manim_shorts/_env/v/bin/python src/make.py shorts`
