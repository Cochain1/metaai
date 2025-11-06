# src/metagen_ai/datasets/loader.py
from __future__ import annotations
import os, json
from typing import Dict, Iterable, List, Any

DATA_DIR = os.path.join("data", "datasets")

def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _builtin_gsm8k_tiny() -> List[Dict[str, Any]]:
    # 最小可用的几条样例：保持 question + answer 字段
    return [
        {"question": "John has 3 apples and buys 4 more. How many apples does he have now?", "answer": "7"},
        {"question": "A notebook costs 5 dollars. If Mary buys 6 of them, how much does she pay in total?", "answer": "30"},
        {"question": "There are 12 cookies. Tom eats 3 and Anna eats 2. How many are left?", "answer": "7"},
        {"question": "A box holds 8 oranges. How many oranges do 5 boxes hold?", "answer": "40"},
        {"question": "Add 12 and 21. Provide the final result.", "answer": "33"},
    ]

def _builtin_arith_grid() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in [1, 2, 3, 4, 5]:
        for b in [6, 7, 8]:
            out.append({"question": f"Compute {a} + {b}.", "answer": str(a + b)})
    for a in [2, 3, 4]:
        for b in [3, 4, 5]:
            out.append({"question": f"Compute {a} * {b}.", "answer": str(a * b)})
    return out

def load_dataset(name: str) -> Iterable[Dict[str, Any]]:
    """
    返回一个可迭代对象，每个元素为一个 task 字典。
    兼容三类格式：
      1) GSM8K/算术类：{"question": str, "answer": str}
      2) MMLU：        {"question": str, "choices": ["A. ...","B. ...","C. ...","D. ..."], "answer": "A"}
      3) HumanEval：   {"task_id": str, "prompt": str, "tests": str, "entry_point": str}

    解析顺序：
      - 若存在 data/datasets/<name>.jsonl 则直接加载；
      - 否则匹配内置名称；
      - 否则若 name 本身是文件路径（相对/绝对），也允许加载；
      - 找不到则报错。
    """
    name = str(name).strip()
    # 1) 明确路径：data/datasets/<name>.jsonl
    jsonl_path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if os.path.isfile(jsonl_path):
        return _iter_jsonl(jsonl_path)

    # 2) 预置数据集
    if name == "gsm8k_tiny":
        return _builtin_gsm8k_tiny()
    if name == "arith_grid":
        return _builtin_arith_grid()

    # 常用别名映射（方便命令行）
    # 你可以将 mmlu_val 指向 data/datasets/mmlu_val.jsonl（若已准备）
    alias_to_file = {
        "gsm8k_test": os.path.join(DATA_DIR, "gsm8k_test.jsonl"),
        "gsm8k_500":  os.path.join(DATA_DIR, "gsm8k_500.jsonl"),
        "mmlu_test":  os.path.join(DATA_DIR, "mmlu_test.jsonl"),
        "mmlu_val":   os.path.join(DATA_DIR, "mmlu_val.jsonl"),
        "humaneval":  os.path.join(DATA_DIR, "humaneval.jsonl"),
    }
    if name in alias_to_file and os.path.isfile(alias_to_file[name]):
        return _iter_jsonl(alias_to_file[name])

    # 3) 若 name 本身是一个现有文件（绝对或相对），也允许加载
    if os.path.isfile(name):
        return _iter_jsonl(name)

    raise FileNotFoundError(
        f"Dataset '{name}' not found.\n"
        f"- Put a JSONL at {jsonl_path}\n"
        f"- Or use builtins: gsm8k_tiny, arith_grid\n"
        f"- Or prepared aliases if files exist: {', '.join(alias_to_file.keys())}\n"
        f"JSONL schema:\n"
        f"  * GSM8K-like: {{'question': str, 'answer': str}}\n"
        f"  * MMLU:       {{'question': str, 'choices': ['A. ...','B. ...','C. ...','D. ...'], 'answer': 'A'}}\n"
        f"  * HumanEval:  {{'task_id': str, 'prompt': str, 'tests': str, 'entry_point': str}}\n"
    )
