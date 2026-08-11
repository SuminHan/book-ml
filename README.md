# book-ml

KSA Machine Learning 1/2 교재. **한국어판과 영문판이 완전히 분리된 두 개의
mdBook**으로 빌드된다(하나의 책 안에 4개 파트를 넣던 예전 구조 대신 —
사이드바/목차가 언어별로 따로 보이게 하려고 분리함).

**Live:**
- 랜딩(언어 선택): https://smhanlab.com/book-ml/
- 한국어: https://smhanlab.com/book-ml/kor/
- English: https://smhanlab.com/book-ml/eng/

(개인 계정 커스텀 도메인 하위로 자동 서빙됨 — 대소문자까지 저장소 이름과
정확히 일치해야 함. GitHub Pages가 "Deploy from a branch → main / docs"로
켜져 있어야 함.)

## 구조

```
kor/
  book.toml       — 한국어판 mdBook 설정 (build-dir = "../docs/kor")
  src/
    SUMMARY.md     — 목차 (Machine Learning 1 / Machine Learning 2, Introduction)
    README.md      — Introduction 페이지
    ml1/chapterNN.md
    ml2/chapterNN.md
  theme/
    custom.css     — fold-toggle 숨김 + lang-switch-badge 스타일
    lang-switch.js — 우상단에 "🌐 English" 배지 삽입(영문판으로 링크)

eng/               — kor/와 완전히 동일한 구조, 영문 콘텐츠 + 한국어판 링크 배지

docs/              — 두 mdBook의 빌드 결과물이 합쳐지는 곳. 커밋 대상.
  index.html       — 언어 선택 랜딩페이지 (수동 작성, mdbook 생성 아님) + 자동 리다이렉트
  kor/             — kor/ 빌드 출력
  eng/             — eng/ 빌드 출력
```

각 장은 하나의 페이지 안에 번호가 매겨진 절(챕터 자체 주차 번호 기준,
예: 5.1, 5.2...)로 구성된 진짜 교과서 형태를 따른다. 자세한 설명은
`kor/src/README.md` / `eng/src/README.md` (Introduction 페이지) 참고.

## 왜 두 개의 mdBook인가

mdBook은 한 책 안에서 언어별 스위처를 지원하는 기능이 없다(파트 제목으로
나눠도 사이드바에 전부 한 번에 표시됨). `book-cs2`/`book-cs2-intl`처럼
레포 자체를 통째로 나누는 대신, **레포 하나 안에서 mdBook 프로젝트 2개
(`kor/`, `eng/`)를 두고 각각 다른 `build-dir`로 빌드**해서 `docs/kor/`,
`docs/eng/`라는 별도 사이트를 만드는 방식을 택했다 — 소스 관리(커밋 이력,
이슈트래커 등)는 하나로 유지하면서 목차만 언어별로 분리된다.

두 책 사이 이동은 SUMMARY.md에 외부 링크를 못 넣어서(mdBook이 항상 로컬
챕터 파일로 해석하려 시도해 빌드 에러 발생 — 아래 참고) `lang-switch.js`로
모든 페이지 우상단에 고정 배지를 주입하는 방식으로 처리했다.

## 콘텐츠 현황 (2026-08-12 기준)

각 언어판 ML1/ML2 12개 챕터씩 총 48개 챕터. 원본 기획 문서는
`KSA-CS/03_ML1_ML2_기획/`(로컬, 이 저장소 밖) 참고.

**⚠️ 다음 검증이 아직 안 됨** — 실제 수업에 쓰기 전에 확인 필요:
- 새로 작성된 대부분의 연습문제 코드는 book-cs2/book-cs2-intl과 달리
  **하나하나 직접 실행해서 검증되지 않았다**.
- 도입부의 역사적 일화는 WebSearch로 건건이 사실 확인을 거치지 않았다.
- 손유도 과제의 Tier 등급(A/B/C)은 원 커리큘럼 문서의 등급을 그대로
  따랐을 뿐, 새로 쓴 분량의 실제 난이도가 그 등급에 맞는지는 검증 안 됨.

## 로컬에서 빌드

```bash
cd kor && mdbook build   # ../docs/kor 에 생성
cd eng && mdbook build   # ../docs/eng 에 생성
# 미리보기는 각 폴더 안에서 mdbook serve (포트가 겹치지 않게 --port 지정)
```

## 배포

GitHub Pages: **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
`docs/`를 커밋하고 `main`에 push하면 몇 분 안에 반영된다. 별도 GitHub
Action 없음. `docs/index.html`은 mdbook이 아니라 손으로 쓴 정적 랜딩
페이지이므로, 두 책을 다시 빌드해도 덮어써지지 않는다(각 build-dir이
`docs/kor`, `docs/eng` 하위이기 때문).

## ⚠️ 주의

- 이 저장소(`SuminHan/book-ml`)와 개인 홈페이지 저장소
  (`suminhan/suminhan.github.io`, Hugo 기반 실제 블로그)는 **완전히 별개**다.
  절대 혼동해서 개인 홈페이지 저장소에 push하지 말 것.
- 새 챕터를 채울 땐 `kor/src/{ml1,ml2}/*.md` 또는 `eng/src/{ml1,ml2}/*.md`만
  수정 → 해당 폴더 안에서 `mdbook build` → `docs/`도 같이 커밋. `docs/`를
  손으로 직접 고치지 말 것(`docs/index.html` 제외 — 이건 mdbook이 안 건드림).
