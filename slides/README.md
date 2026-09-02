# slides/ — KSA 강의 슬라이드 (beamer, 한국어판)

`kor/src/{ml1,ml2}/*.md` 교재 본문을 주차별 LaTeX beamer 슬라이드로 변환한다.

## 구조

```
slides/
  theme/ksa-theme.tex   — 공용 beamer 테마 (XeLaTeX). 모든 주차 파일이 \input.
  kor/weekNN.tex         — 주차별 덱. \documentclass + \input{../theme/ksa-theme.tex}.
  figs/                  — 슬라이드용 그림 (build_figs.sh 가 생성; 배경 흰색 flatten)
  assets/                — 원본 자산. ksa-logo.pdf (업로드된 .ai = PDF), ksa-logo.png
  build_figs.sh          — kor/src/images/*.svg → figs/*.png (rsvg-convert -b white),
                           투명 PNG → 흰 배경 flatten, KSA 로고도 여기서 처리
```

## 빌드

```bash
cd slides && bash build_figs.sh          # 그림 먼저 (한 번, 또는 그림 바뀔 때)
cd slides/kor && xelatex week01.tex       # 2회 돌리면 섹션 참조까지 갱신
```

- **엔진: XeLaTeX** (한글). 이 서버엔 ctex/xeCJK/kotex 이 없어서 fontspec 만으로
  `Noto Sans CJK KR` 를 본문·제목·모노에 직접 지정한다. `theme/ksa-theme.tex` 참고.
- 그림 경로는 테마의 `\graphicspath{{figs/}{../figs/}}` 로 해결 → `\includegraphics{이름.png}`.

## 테마 (KSA 아이덴티티)

공식 KSA 심벌마크에서 추출한 브랜드 색:

| 색 | HEX | 의미 |
|---|---|---|
| Deep Prussian Blue | `#2E3192` | 지성 (structure / 제목 / 강조) |
| Golden Gray | `#D6CBB1` (짙게 `#A9976B`) | 감성 (룰 · 악센트) |

- 프레임 제목: 파랑 볼드 + (금색 tick + 파랑) 밑줄
- 푸터: `[KSA 로고] Machine Learning 1 · N주차` | `Sumin Han · suminhan@ksa.hs.kr` | `n / N`
- 섹션 구분 슬라이드 자동 삽입 (`\AtBeginSection`, 번호 제외)
- 매크로: `\kb{...}` 파랑 키워드, `\kq{...}` 빨강 핵심질문
- 코드: `listings` (`style=ksa`) — 파랑 키워드 · 왼쪽 파랑 룰 · 회색 배경

## 주차별 파일 만들 때

1. `kor/src/ml1/chapterNN*.md` (개요 + 각 절) 읽기
2. `kor/weekNN.tex` 작성: `\renewcommand{\ksaweek}{N주차}`, `\title{...}`,
   절 = `\section{N.x ...}`, 슬라이드는 개념 단위로 쪼개기 (프레임당 6~8줄)
3. 그림이 필요하면 `build_figs.sh` 의 `svgs=()` / `rasters=()` 에 이름 추가 후 재실행
4. `xelatex` 2회, PDF 확인

## 상태

- `kor/week01.tex` — ML1 Chapter 1 (샘플, 20 프레임). 테마 검토용.
- 나머지 주차: 미작성.
