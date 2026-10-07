---
name: slide-fix
description: "book-ml/slides 의 한글 beamer 덱(kor/ml*-weekNN.tex, ml2-*)을 사용자가 하나씩 지적하는 대로 고치고 PDF를 다시 빌드하는 반복 작업용 스킬. 핵심: 출력을 극단적으로 줄인다 — 지적당 한 줄 보고. 트리거: 'slides 수정', '슬라이드 이거 고쳐', 'week01 이 부분', 'PDF 다시 빌드', 그리고 같은 덱을 연달아 조금씩 고치는 대화 흐름."
version: "v1.0 (2026-09-07)"
---

# slide-fix — 슬라이드 반복 수정 (저출력 모드)

사용자가 슬라이드를 **하나씩 지적**한다. 매번: 고치고 → PDF 다시 빌드 → **한 줄로만** 보고.
목적은 토큰 절약이므로 단계별 나열·중간 로그·검증 이미지·서론을 전부 생략한다.

## 대상 / 환경 (이미 세팅됨 — 재설치·재설명 금지)
- 덱: 2026-09-11부터 각 덱이 **자기 폴더**를 가짐 —
  `/Users/hansumin/book-ml/slides/kor/<deck>/<deck>.tex` (+ 같은 폴더에 `<deck>.pdf`,
  그 덱이 쓰는 이미지 전부). 공용 테마는 `slides/theme/ksa-theme.tex`
  (`\input{../../theme/ksa-theme.tex}`로 두 단계 위 참조).
- 빌드: `tectonic -X compile`. 폰트(Noto Sans CJK KR + **Mono** CJK KR), `librsvg`,
  `poppler`, `imagemagick`, `graphviz` 설치 완료.
- 그림: 덱 폴더 안에 이미 flatten된 png/pdf로 들어 있음(`\graphicspath{{./}{../../figs/}}`,
  뒤쪽은 구버전 공용 캐시 fallback). 소스 SVG/원본은 여전히
  **`/Users/hansumin/book-ml/kor/src/images/`**. 새 `\includegraphics{name}`을
  추가했을 때만: `cd slides && bash build_figs.sh kor/<deck>/<deck>.tex`로
  `slides/figs/`에 만든 뒤 `cp figs/<name> kor/<deck>/<name>`으로 덱 폴더에 복사.

## 루프 (지적 1건당)

1. **고친다.** 사용자가 짚은 부분만 Edit. 확인차 파일 되읽기 금지(Edit가 실패하면 알려준다).
   여러 건을 한꺼번에 지적하면 전부 Edit 후 빌드는 1회.
2. **빌드한다.**
   ```bash
   cd /Users/hansumin/book-ml/slides/kor/<deck> && tectonic -X compile <deck>.tex 2>&1 | tail -3; echo "exit=$?"
   ```
3. **한 줄 보고.** 예: `✅ p.7 문구 교체 · build OK (72p)` / `✅ 3곳 수정 · build OK`.
   - PDF는 워크트리에 그대로 둔다. git 손대지 말 것(그림은 실제 파일이라 커밋 가능 상태).
   - 커밋은 사용자가 명시할 때만.

## 보고 규칙 (엄수)
- **기본은 한 줄.** 무엇을 바꿨는지 + `build OK` + 페이지 수. 그 이상 쓰지 않는다.
- **이미지 렌더 안 함.** 예외: (a) 빌드 에러, (b) 사용자가 "보여줘"라고 함,
  (c) 표·그림·긴 block 을 새로 넣어 레이아웃이 깨질 만한 편집 → 그 페이지 1장만 `pdftoppm -r 110` 렌더해 확인 후에도 보고는 한 줄.
- **빌드 에러(exit≠0)** 만 예외적으로 여러 줄: 에러 메시지 + 해당 라인 + 고칠 방법.
- **Overfull 경고**: 아래 기존 목록은 **재보고 금지**. 내가 방금 건드린 프레임이나 새로 생긴 것만 언급.
- 추측·요약·다음 제안 금지. 다음 지적을 기다린다.

### ml1-week01.tex 기존 Overfull (내 수정과 무관 — 무시)
라인 271(hbox), 582, 1234, 1330, 1369, 1493, 1617 (vbox). 다른 덱은 첫 빌드 로그로 베이스라인 잡고 동일 원칙.

## 한 줄 요약
> 짚어준 데만 Edit → `tectonic -X compile` → `✅ <바뀐 것> · build OK (Np)` 한 줄. 이미지·로그·서론 없음. 에러일 때만 자세히.
