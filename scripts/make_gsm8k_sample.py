# scripts/make_gsm8k_sample.py
from datasets import load_dataset
import os, re, json, random
from typing import Optional

OUT_DIR = os.path.join("data", "datasets")
OUT_PATH = os.path.join(OUT_DIR, "gsm8k_500.jsonl")
SEED = 42
N = 500

# 从 GSM8K 的 "main" 配置加载训练集（更大），也可以换 test
ds = load_dataset("gsm8k", "main", split="train")  # train: ~7473 条; test: ~1319 条

random.seed(SEED)
ds = ds.shuffle(seed=SEED)

num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_final_number(ans: str) -> Optional[str]:
    """
    GSM8K 的答案字段一般末尾一行形如 '#### 42'
    这里先找 '#### ' 之后的片段，再用正则抓第一个数字。
    """
    if not ans:
        return None
    # 找到 #### 后的内容
    if "####" in ans:
        tail = ans.split("####", 1)[1]
    else:
        tail = ans
    # 抓数字（去掉逗号分隔）
    tail = tail.replace(",", " ")
    m = num_re.search(tail)
    return m.group(0) if m else None

# 过滤出有明确数值答案的样本，并取前 N 个
items = []
for ex in ds:
    q = ex.get("question", "").strip()
    a = ex.get("answer", "").strip()
    gold = extract_final_number(a)
    if q and gold is not None:
        items.append({"question": q, "answer": gold})
    if len(items) >= N:
        break

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

print(f"[OK] wrote {len(items)} examples → {OUT_PATH}")
