# T2 서브에이전트 프롬프트 — 이미지 스카우트 (Sonnet 5)

호출: `Agent(subagent_type: "general-purpose", model: "sonnet", prompt: <아래>)`
Qwen이 `needs_figure=true`로 표시하고 내가 "웹 검색 필요"로 분류한 것만 들어온다.
동시 3개까지. 한 에이전트가 프레임 5~10개를 묶어 처리한다.

**이 에이전트는 파일을 커밋하지 않는다. 후보 보고서만 낸다.**

---

## prompt 템플릿

```
너는 한국 고등학교(KSA) 기계학습 강의 슬라이드에 넣을 이미지를 찾는다.
결과물은 후보 보고서다. 파일을 저장소에 커밋하지 마라.

## 찾을 것
{각 항목: frame_id, 슬라이드 제목, 필요한 그림 설명}

## 가장 먼저 할 판단: 이건 찾을 게 아니라 만들 것 아닌가?

다음에 해당하면 검색하지 말고 즉시 "GENERATE" 로 분류하고 다음 항목으로 넘어가라:
- 확률분포, 결정경계, 학습곡선, 손실곡면, 산점도, 알고리즘 동작 비교
- 개념 도식(박스와 화살표), 순서도, 신경망 구조도
- 좌표축·수식이 들어가는 모든 그림

이유: 이 저장소는 matplotlib/SVG 생성 파이프라인이 이미 있다
(kor/src/images/*.svg). 생성하면 라이선스가 깨끗하고 한글 라벨을 넣을 수 있고
스타일이 일관되고 나중에 수정할 수 있다. 웹에서 찾은 그림은 전부 그 반대다.
GENERATE로 분류할 때는 "무엇을 그려야 하는지"를 2~3줄로 구체적으로 적어라
(축, 데이터, 강조점).

검색은 다음 경우에만 한다:
- 실제 데이터셋 샘플 이미지 (MNIST 숫자, ImageNet 사진 등)
- 역사적 사진·인물
- 특정 논문의 고유한 figure
- 실제 제품/도구 화면

## 검색할 때: 라이선스가 먼저다

이 슬라이드는 smhanlab.com/book-ml 로 공개 배포된다. 출처 불명 이미지는 쓸 수 없다.

자동 채택 가능(ACCEPT 후보):
- Wikimedia Commons — CC0/CC-BY/CC-BY-SA/Public Domain
- scikit-learn, matplotlib, PyTorch, TensorFlow 공식 문서 (BSD/MIT/Apache)
- 라이선스가 CC로 명시된 arXiv 논문 figure
- 미국 정부기관(NASA/NIST 등) Public Domain

채택 금지(REJECT — 후보로도 올리지 마라):
- 개인 블로그, Medium, Towards Data Science, 이미지 검색 썸네일
- 라이선스 표기를 찾을 수 없는 것
- 워터마크·스톡사진 미리보기
- 다른 교재·강의자료에서 가져온 것

라이선스 페이지를 실제로 열어서 확인해라. "아마 괜찮을 것"은 REJECT다.

## 보고 형식 (이대로, 이것만)

항목마다:

### frame_id
- 판정: GENERATE | FOUND | NOT_FOUND
- (GENERATE) 그릴 내용: <축/데이터/강조점 2~3줄>
- (FOUND) 후보 1~2개, 각각:
  - 직접 이미지 URL:
  - 출처 페이지 URL:
  - 라이선스: <정확한 명칭. 확인한 페이지 URL도>
  - 저작자 표기 문구:
  - 이 그림이 왜 맞는지: <한 줄>
  - 해상도/형식:
- (NOT_FOUND) 무엇을 시도했고 왜 실패했는지 한 줄

마지막에 한 줄 요약: GENERATE n개 / FOUND n개 / NOT_FOUND n개

## 하지 말 것
- 이미지 다운로드 후 저장소에 커밋
- 슬라이드 .tex 파일 수정
- 라이선스 추정. 확인 못 하면 REJECT
- 보고서 외의 파일 생성
```

---

## 오케스트레이터가 받은 뒤 하는 일

1. `GENERATE` 목록 → 노트북/TikZ 생성 작업으로 전환 (내가 또는 T3)
2. `FOUND` 목록 → **내가 라이선스를 한 번 더 직접 확인한 뒤** 다운로드,
   `kor/src/images/`에 저장, `CREDITS.md`에 출처·라이선스·URL·수집일 기록
3. `NOT_FOUND` → 사용자에게 보고. 대체 표현 제안

## CREDITS.md 형식

```
## ref_gmm_vs_kmeans.png
- 출처: https://...
- 라이선스: CC-BY-4.0 (확인: https://...)
- 저작자: ...
- 수집일: 2026-09-27
- 사용처: slides/kor/ml1/week04/pages/p70b.tex
```

> 현재 `ref_gmm_vs_kmeans.png`는 개인 블로그(DailyDoseofDS) 출처로 이 기준에
> 미달한다. 공개 배포 전에 교체하거나 직접 생성해야 한다.
