# src/metagen_ai/eval/metrics.py
from __future__ import annotations
import os, re, sys, tempfile, subprocess
from typing import Dict, Any, Optional

# -----------------------
# 通用抽取（GSM8K 等）
# -----------------------
_NUM_PAT = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)|Final answer\s*:\s*([-+]?\d+(?:\.\d+)?)", re.I)
_ONLY_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _gold_to_letter(gold: Any) -> Optional[str]:
    """将 gold 转为 'A'|'B'|'C'|'D'；兼容 0/1/2/3（0基）、1/2/3/4（1基）、或字母本身。"""
    if gold is None:
        return None
    # 字母
    if isinstance(gold, str) and gold.strip().upper() in {"A","B","C","D"}:
        return gold.strip().upper()
    # 纯数字或数字字符串
    try:
        idx = int(str(gold).strip())
    except Exception:
        return None
    # 先假设 0 基
    if 0 <= idx <= 3:
        return "ABCD"[idx]
    # 再兼容 1 基
    if 1 <= idx <= 4:
        return "ABCD"[idx - 1]
    return None

def _extract_option_letter(txt: str) -> Optional[str]:
    """从模型输出里抽取 A/B/C/D：取“最后一个”匹配，防止前文提到其它选项干扰。"""
    if not txt:
        return None
    cand = re.findall(r"\b([ABCD])\b", txt.upper())
    return cand[-1] if cand else None

def _extract_number(txt: str) -> Optional[str]:
    if not txt:
        return None
    m = _NUM_PAT.search(txt)
    if m:
        return (m.group(1) or m.group(2)).strip()
    nums = _ONLY_NUM.findall(txt)
    return nums[-1].strip() if nums else None

def _extract_option_letter(txt: str) -> Optional[str]:
    if not txt:
        return None
    m = re.search(r"\b([ABCD])\b", txt.upper())
    return m.group(1) if m else None

# -----------------------
# HumanEval：官方与简易两个后端
# -----------------------
_PY_RUN = sys.executable

def _extract_solution_code(final_text: str) -> str:
    # 优先提取 ```python ... ```；否则 ```...```；否则全量
    if not final_text:
        return ""
    m = re.search(r"```python(.*?)```", final_text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", final_text, re.S)
    if m:
        return m.group(1).strip()
    return final_text.strip()

def _run_humaneval_simple(solution_text: str, tests: str, timeout_s: int = 20) -> Dict[str, Any]:
    code = f"# --- Solution ---\n{solution_text}\n\n# --- Tests ---\n{tests}\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "heval_run.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run([_PY_RUN, path],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout_s, check=False, text=True)
            return {
                "passed": proc.returncode == 0,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "returncode": proc.returncode,
                "backend": "simple"
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stderr": "Timeout", "returncode": -1, "backend": "simple"}

def _run_humaneval_official(task_id: str, completion: str, timeout_s: int = 20) -> Dict[str, Any]:
    # 延迟导入官方适配器
    try:
        from .humaneval_official import evaluate_one_official
    except Exception as e:
        raise RuntimeError(f"Official HumanEval adapter not available: {e}")

    res = evaluate_one_official(task_id, completion, timeout=timeout_s, n_workers=1, k=1)
    return {"passed": bool(res.get("passed", False)), "backend": res.get("backend", "official"), "raw": res}

def _should_use_official() -> bool:
    # 优先：显式开启；或者安装了官方包（evalplus 或 human-eval）
    env = os.environ.get("METAGEN_USE_OFFICIAL_HUMANEVAL", "").strip()
    if env in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    try:
        import importlib
        if importlib.util.find_spec("evalplus") is not None:
            return True
        if importlib.util.find_spec("human_eval") is not None:
            return True
    except Exception:
        pass
    return False

# -----------------------
# 统一判分
# -----------------------
def judge_correct(task: Dict[str, Any], final_text: str) -> bool:
    """
    统一判分入口：
      - HumanEval：默认优先使用官方（EvalPlus/human-eval），不可用时退回简易执行器；
      - MMLU：选项字母与 gold 比较；
      - 默认（GSM8K 等）：抽数值与 gold 比较。
    """
    # 1) HumanEval：识别字段
    is_he = ("task_id" in task) and ("prompt" in task)
    if is_he:
        # “官方”路径：使用 task_id + completion
        if _should_use_official():
            # completion：只传代码文本（不包含 fences 更稳）
            completion = _extract_solution_code(final_text)
            if not completion.strip():
                completion = str(task.get("prompt", ""))  # 兜底
            try:
                res = _run_humaneval_official(task_id=str(task["task_id"]), completion=completion, timeout_s=20)
                return bool(res.get("passed", False))
            except Exception:
                # 官方失败时安全回退到简易执行器
                pass

        # 简易执行器路径（不泄漏 tests 给 LLM，只在评测端拼接执行）
        completion = _extract_solution_code(final_text)
        if not completion.strip():
            completion = str(task.get("prompt", ""))  # 兜底
        res = _run_humaneval_simple(completion, str(task.get("tests", "")), timeout_s=20)
        return bool(res.get("passed", False))

    # 2) MMLU：choices 存在即视为多选任务
    if isinstance(task.get("choices"), list) and task.get("choices"):
        gold_letter = _gold_to_letter(task.get("answer"))
        if gold_letter is None:
            return False
        pred = _extract_option_letter(final_text)
        return (pred is not None) and (pred == gold_letter)

    # 3) 默认：数值题（GSM8K 等）
    gold = task.get("answer")
    if gold is None:
        return False
    pred = _extract_number(final_text or "")
    return (pred is not None) and (str(pred).strip() == str(gold).strip())
