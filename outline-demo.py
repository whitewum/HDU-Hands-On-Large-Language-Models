# pip install outlines transformers torch
# pip install --upgrade pytorch-lightning
# pip install --upgrade outlines
# pip list | grep -E "outlines|transformers|torch" # outlines                          0.2.3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, warnings, outlines, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

warnings.filterwarnings("ignore")
os.environ.update(
    TRANSFORMERS_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
)

MODEL_NAME = "qwen/Qwen2.5-0.5B-Instruct" 
MODEL_DIR = snapshot_download(MODEL_NAME)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    local_files_only=True,
    trust_remote_code=True,
)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = outlines.models.Transformers(llm, tok)

# ---------------------------------------------------------------------------
# 全局生成器
# ---------------------------------------------------------------------------

yesno_gen      = outlines.generate.choice(model, ["是", "否"])
score_gen      = outlines.generate.choice(model, list("12345"))
sentiment_gen  = outlines.generate.choice(model, ["积极", "中性", "消极"])
abcd_gen       = outlines.generate.choice(model, list("ABCD"))

int_gen   = outlines.generate.format(model, int)
float_gen = outlines.generate.format(model, float)

IP_PATTERN = r"((25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d\\d?)"
ip_gen    = outlines.generate.regex(model, IP_PATTERN, sampler=outlines.samplers.greedy())

CHARACTER_SCHEMA = """{
  "title": "Character",
  "type": "object",
  "properties": {
    "name":  {"type": "string",  "maxLength": 10},
    "age":   {"type": "integer"},
    "armor": {"type": "string",  "enum": ["leather", "chainmail", "plate"]},
    "weapon":{"type": "string",  "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"]},
    "strength": {"type": "integer"}
  },
  "required": ["name", "age", "armor", "weapon", "strength"]
}"""
json_gen = outlines.generate.json(model, CHARACTER_SCHEMA)

# --- SQL Grammar (spaces baked‑in) ----------------------------------------

SQL_GRAMMAR = r"""
?start: select_stmt

select_stmt: "SELECT " column_list "FROM " table_name where_clause? order_by_clause? limit_clause? ";"?

column_list: "*" | column (", " column)*

where_clause: "WHERE " condition

condition: column " " comparator " " value

comparator: "=" | ">" | "<" | ">=" | "<=" | "<>"

order_by_clause: " ORDER BY " column (" ASC" | " DESC")?

limit_clause: " LIMIT " NUMBER

column: CNAME
table_name: CNAME
value: NUMBER | ESCAPED_STRING

%import common.CNAME
%import common.NUMBER
%import common.ESCAPED_STRING
"""

sql_gen = outlines.generate.cfg(model, SQL_GRAMMAR, sampler=outlines.samplers.greedy())

# ---------------------------------------------------------------------------
# 包装函数
# ---------------------------------------------------------------------------

def ask_yesno(q: str) -> str:
    return yesno_gen(f"{q}\n答案（是/否）：")

def score(item: str) -> str:
    return score_gen(f"请给 {item} 打 1–5 分：")

def sentiment(text: str) -> str:
    return sentiment_gen(f"请判断情感（积极/中性/消极）：{text}\n情感：")

def mcq(q: str, opts: str) -> str:
    return abcd_gen(f"{q}\n{opts}\n请选择正确答案：")

def calc_int(expr: str) -> int:
    return int_gen(f"{expr} = ", max_tokens=10)

def calc_float(expr: str) -> float:
    return float_gen(f"{expr} = ", max_tokens=10)

def google_dns() -> str:
    return ip_gen("Google DNS server IP address is ", max_tokens=20)

def random_character(desc: str = "Give me a character description") -> dict:
    return json_gen(desc, max_tokens=120)

def generate_sql(task: str) -> str:
    """根据简化 SQL CFG 生成查询语句"""
    prompt = f"## 任务: {task}\nSQL: "
    sql = sql_gen(prompt, max_tokens=60)
    # 如果末尾没有分号则补上
    return sql if sql.strip().endswith(";") else sql + ";"

# ---------------------------------------------------------------------------
# CLI 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo 开始 ===")
    print("情感：", sentiment("这部电影很好看，剧情紧凑。"))
    print("评分：", score("星巴克拿铁"))
    print("是非：", ask_yesno("北京是中国首都吗？"))
    print("选择：", mcq("以下哪个是编程语言？", "A.HTML B.Python C.CSS D.JSON"))

    print("整数计算 1+2=", calc_int("1 + 2"))
    print("浮点计算 sqrt(2)=", calc_float("sqrt(2)"))
    print("Google DNS IP: ", google_dns())
    print("随机角色 JSON:\n", json.dumps(random_character(), ensure_ascii=False, indent=2))

    print("SQL 查询示例:")
    print(generate_sql("列出 orders 表中 id 和 total, where total > 100"))

    # GPU 清理
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("=== Demo 结束 ===")


