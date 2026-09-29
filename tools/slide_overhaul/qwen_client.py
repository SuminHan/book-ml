import os, re, json, pathlib, urllib.request
D = pathlib.Path(__file__).parent / "agents" / "qwen_split.md"
md = D.read_text(encoding="utf-8")
blocks = re.findall(r"```(\w*)\n(.*?)```", md, re.S)
SYSTEM   = blocks[0][1].strip()          # ## system
SCHEMA   = json.loads(blocks[2][1])      # guided_json 스키마
if os.environ.get("SYSTEM_FILE"): SYSTEM = open(os.environ["SYSTEM_FILE"], encoding="utf-8").read().strip()
FS_USER  = blocks[3][1].strip()          # few-shot user
FS_ASSIS = blocks[4][1].strip()          # few-shot assistant
URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-27B"

def build_user(title, items):
    return f"제목: {title}\n항목 수: {len(items)}\n\n" + "\n".join(f"[{i}] {t}" for i, t in enumerate(items, 1))

def call(user, schema=SCHEMA, extra_msgs=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": FS_USER},
            {"role": "assistant", "content": FS_ASSIS},
            {"role": "user", "content": user}]
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "split", "schema": schema, "strict": True}}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"]["content"], r["usage"]
