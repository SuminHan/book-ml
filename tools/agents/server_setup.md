# 연구실 서버 세팅 — 붙여넣을 프롬프트

서버에서 `git pull` 한 뒤, 그 서버의 Claude Code 세션에 아래 블록을 그대로 붙여넣는다.

---

```
이 저장소(book-ml)의 tools/agents/README.md 를 먼저 읽어라.
지금 할 일은 그 문서의 "Qwen 서빙 설정" 을 이 서버에서 실제로 띄우는 것이다.
슬라이드 파일은 건드리지 마라. 커밋하지 마라.

1) 하드웨어 확인
   nvidia-smi 로 GPU 개수와 장착 VRAM을 확인하고, 아래 기준으로 모델을 고른다.
   (고른 이유를 한 줄로 보고할 것)
   - 총 VRAM 16GB  : Qwen/Qwen3-8B-FP8
   - 총 VRAM 24GB  : Qwen/Qwen3-8B            (bf16, 여유 있음)
   - 총 VRAM 40GB+ : Qwen/Qwen3-30B-A3B-FP8   (MoE, 활성 3B — 이게 있으면 이걸로)
   - 총 VRAM 80GB+ : Qwen/Qwen3-30B-A3B       (bf16)

2) vLLM 설치 (없으면). 시스템 파이썬을 오염시키지 말고 venv 나 uv 를 써라.

3) 서빙. 포트 8000. 고른 모델로:
   vllm serve <모델> --port 8000 --max-model-len 8192 \
     --guided-decoding-backend xgrammar
   백그라운드로 띄우고 로그는 파일로 남겨라.

4) 스모크 테스트 — 이게 핵심이다. 반드시 통과시켜라.
   tools/agents/qwen_split.md 안의 system 프롬프트와 few-shot user 입력을
   그대로 써서 /v1/chat/completions 를 호출한다.
   temperature 0, chat_template_kwargs {"enable_thinking": false},
   그리고 그 문서의 guided_json 스키마를 response_format 으로 넘긴다.
   확인할 것:
   - 응답이 스키마에 맞는 JSON 인가 (guided decoding 이 실제로 걸렸는가)
   - labels 의 id 집합이 입력 항목 번호와 정확히 일치하는가
   통과 못 하면 원인을 찾아 고쳐라. 이게 안 되면 파이프라인 전체가 무의미하다.

5) 처리량 측정: 같은 요청을 동시 16으로 32번 보내고 초당 프레임 수를 재라.

마지막에 이것만 보고해라:
   - 고른 모델과 이유
   - GPU 구성
   - guided_json 통과 여부 (예/아니오)
   - 동시 16에서 초당 프레임 수
   - 서버가 외부에서 접속 가능한지, 아니면 SSH 터널이 필요한지
```

---

## Claude Code 없이 손으로 할 때 최소 경로

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv   # 먼저 VRAM 확인
pip install -U vllm
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 8192 \
  --guided-decoding-backend xgrammar
```

## 주의

- **30B-A3B는 MoE지만 가중치 30B 전체를 VRAM에 올려야 한다.** 활성 파라미터가
  3B라 *속도*가 빠른 것이지 *메모리*가 적게 드는 게 아니다. FP8로도 40GB급이
  필요하므로 24GB에서는 Qwen3-8B가 맞다.
- **`--host 0.0.0.0` 을 쓰지 마라.** vLLM에는 인증이 없어서 랩 네트워크에
  그대로 노출된다. 기본값(localhost)을 유지하고, 맥에서 붙을 때 터널을 뚫는다:
  ```bash
  ssh -N -L 8000:localhost:8000 <lab-server>
  ```

## 이 단계의 통과/실패 기준

4번의 `guided_json` 이 안 걸리면 8B가 스키마를 자주 어기고,
그러면 "Qwen은 라벨만 붙인다"는 설계 전제가 무너진다.
여기서 한 번 갈린다 — 실패하면 T1을 폐기하고 오케스트레이터가 직접 처리한다.
