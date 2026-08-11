# book-ml

KSA Machine Learning 1/2 교재 mdBook.

**Live:** https://smhanlab.com/book-ml/
(개인 계정 커스텀 도메인 하위로 자동 서빙됨 — 대소문자까지 저장소 이름과 정확히 일치해야 함.
GitHub Pages를 아직 "Deploy from a branch → main / docs"로 켜지 않았다면 Settings에서
설정 필요.)

## 구조

```
book.toml       — mdBook 설정 (build-dir = "docs", GitHub Pages가 이 폴더를 서빙)
src/
  SUMMARY.md    — 목차 (Machine Learning 1 / Machine Learning 2 2부 + Introduction)
  README.md     — Introduction 페이지
  ml1/          — lecture01~12 (Machine Learning 1, 12주)
  ml2/          — lecture01~12 (Machine Learning 2, 12주)
docs/           — mdbook build 결과물. 커밋 대상 (Pages가 직접 서빙하므로 gitignore 금지).
```

각 주차는 3개 파일: `lectureNN.md`(Opener) + `lectureNN-topics.md`(Topics Covered) +
`lectureNN-problems.md`(Problem Set, 있는 주차만 — Orientation/Review/총정리/팀
프로젝트 발표 주차는 Problem Set 없음).

## 콘텐츠 현황 (2026-08-12 기준)

24주 전체(ML1 12주 + ML2 12주)의 초안이 작성돼 있다. 원본 기획 문서는
`KSA-CS/03_ML1_ML2_기획/`(로컬, 이 저장소 밖) 참고. 아래 4개 주차의 Problem Set은
`KSA-CS/03_ML1_ML2_기획/ML1_ML2_교재개발/`에 미리 작성 및 코드로 실행 검증된
문제를 그대로 포팅한 것이고, 나머지 20주는 이번에 새로 작성됨:

- ML1 W02 선형회귀 (Tier A), W07 역전파 (Tier C)
- ML2 W05 MDP/정책평가 (Tier B), W08 Policy Gradient (Tier C)

**⚠️ 다음 검증이 아직 안 됨** — 실제 수업에 쓰기 전에 확인 필요:
- 새로 작성된 20주 분량의 Problem Set 코드는 book-cs2/book-cs2-intl과 달리
  **하나하나 직접 실행해서 검증되지 않았다** (시간 제약으로 스킵). 특히 정답
  예시로 적어둔 출력값(`print(...) # 예상 출력`)이 실제로 그 값이 나오는지 확인
  필요.
- Opener의 역사적 일화(허블·비셀 실험, 페르휠스트의 로지스틱함수, 굿펠로우의
  GAN 술자리 일화 등)는 book-cs2 essay들처럼 WebSearch로 건건이 사실 확인을
  거치지 않았다 — 잘 알려진 사실 위주로 적었지만, 실제 수업 자료로 쓰기 전
  한 번씩 재확인 권장.
- 손유도 과제의 Tier 등급(A/B/C)은 `ML1_ML2_커리큘럼_및_연간계획.md`의 등급을
  그대로 따랐지만, 새로 쓴 20주 분량의 실제 난이도가 그 등급에 맞는지는
  검증 안 됨.
- Machine Learning 1 / Machine Learning 2 두 파트가 한 책 안에 있어서, 사이드바
  챕터 번호가 1~12(ML1) 다음 13~24(ML2)로 이어진다 — mdBook이 한 책 안에서 파트별
  번호를 재시작하는 기능을 지원하지 않아서 생기는 현상. 각 페이지 안에서는
  "W01~W12"로 표기돼 있어 실제 강의 주차와는 무관하니 참고.

## 로컬에서 빌드

```bash
mdbook build        # docs/ 에 정적 사이트 생성
mdbook serve         # 로컬 미리보기 (기본 http://localhost:3000)
```

## 배포

GitHub Pages: **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
`docs/`를 커밋하고 `main`에 push하면 몇 분 안에 반영된다. 별도 GitHub Action 없음.

## ⚠️ 주의

- 이 저장소(`SuminHan/book-ml`)와 개인 홈페이지 저장소
  (`suminhan/suminhan.github.io`, Hugo 기반 실제 블로그)는 **완전히 별개**다.
  절대 혼동해서 개인 홈페이지 저장소에 push하지 말 것.
- 새 챕터를 채울 땐 `src/{ml1,ml2}/*.md`만 수정 → `mdbook build`로 `docs/` 재생성
  → `docs/`도 같이 커밋. `docs/`를 손으로 직접 고치지 말 것 (다음 build 때
  덮어써짐).
