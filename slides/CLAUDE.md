# slides/ — 한글 beamer 슬라이드 작업 규칙

> **새 버전 `slides/v2/` (차시별 덱, tex 한 파일)** 은 아래 규칙이 아니라 `slides/v2/README.md` 와 스킬 `slide-v2` 를 따른다.
> 이 파일의 규칙은 주 단위 덱 `slides/kor/` 용이다.

## 구조
- 덱 폴더: `slides/kor/<ml1|ml2>/weekNN/` (예: `slides/kor/ml1/week03/`)
  - `ml1-week03.tex` — 본체. `\input{pages/p01}` … 순서만 들어 있다.
  - `pages/pNN.tex` — **프레임 하나 = 파일 하나**. 고칠 때는 이 파일을 연다.
  - `figs/` — 이 덱이 쓰는 그림. `ml1-week03.pdf` — 빌드 결과.
- 공용 테마: `slides/theme/ksa-theme.tex` — 사용자가 시키지 않으면 건드리지 않는다.
- 퀴즈·시험·유인물: `slides/kor/<ml1|ml2>/{quizzes,exams,handouts}/`.

## 고칠 프레임 찾기
- 문구로: `grep -rn '찾을 문구' pages/`
- PDF 페이지 번호로: pNN 번호는 PDF 페이지와 **다를 수 있다**(p03b 같은 파일이 끼어 있음).
  `pdftotext -f N -l N ml1-week03.pdf - | head` 로 그 페이지 글자를 본 뒤 grep으로 파일을 찾는다.

## 빌드 (덱 폴더에서)
```
cd ~/book-ml/slides/kor/ml1/week03 && tectonic -X compile ml1-week03.tex > /tmp/build.log 2>&1; echo exit=$?
grep -E '^(error|!)|Overfull|l\.[0-9]' /tmp/build.log | head -20
pdfinfo ml1-week03.pdf | grep Pages
```
- 결과 확인(필요할 때 한 장만): `pdftoppm -f N -l N -r 80 -png ml1-week03.pdf /tmp/slide` → 생긴 png를 Read.

## 규칙
- 지적받은 부분만 Edit. `\begin{frame}…\end{frame}`, `{ }` 짝, `\item` 구조를 깨지 않게.
- 여러 곳을 고쳐도 빌드는 마지막에 한 번.
- 빌드 실패 시: 로그의 `!` 줄과 `l.<줄번호>`를 보고 그 파일·줄을 고친다.
- 새 그림을 넣을 때: 원본은 `~/book-ml/kor/src/images/`. 덱의 `figs/`에 png로 복사해 `\includegraphics[width=0.8\linewidth]{name}`.
  svg는 `rsvg-convert -z 3 -o figs/name.png ~/book-ml/kor/src/images/name.svg`.
- Overfull 경고는 내가 고친 프레임에서 새로 생긴 것만 보고한다.
- 보고: `✅ p.7 문구 교체 · build OK (72p)` 처럼 한 줄.
