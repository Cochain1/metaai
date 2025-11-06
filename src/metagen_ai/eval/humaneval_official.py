# src/metagen_ai/eval/humaneval_official.py
from __future__ import annotations

from typing import Dict, Any, Optional
import json
import os
import tempfile

"""
修复要点：
- 同时向官方评测器传入“单题”的 problem_file，避免单样本 samples.jsonl 与全量题库长度不等而报错；
- 兼容不同版本的 evalplus/human-eval（有的版本不接受 problem_file；再退回单题的 check_correctness）；
- 返回统一格式：{"passed": bool, "backend": "evalplus"|"human-eval"|"human-eval-single", "raw": <官方原始返回或最小结果>}
"""


# -----------------------
# 工具：读取 HumanEval 题库 & 写出单题 problem.jsonl
# -----------------------
def _read_problems() -> Dict[str, Dict[str, Any]]:
    """
    先尝试从 evalplus 读取（若存在自带修补/扩展），否则退回 human-eval。
    返回一个 dict: {task_id: {prompt, entry_point, test, ...}}
    """
    # 优先 evalplus （如果安装）
    try:
        from evalplus.data import read_problems as ep_read_problems  # type: ignore
        return ep_read_problems()
    except Exception:
        pass

    # 退回 human-eval
    from human_eval.data import read_problems  # type: ignore
    return read_problems()


def _dump_single_problem_jsonl(task_id: str) -> str:
    """
    把单个 task 的 problem 记录写到临时 jsonl，供官方评测器以 problem_file 方式读取。
    """
    problems = _read_problems()
    if task_id not in problems:
        raise KeyError(f"Unknown HumanEval task_id: {task_id}")

    record = problems[task_id].copy()
    record["task_id"] = task_id

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return path


# -----------------------
# 官方评测后端调用（优先 EvalPlus → 再 human-eval）
# -----------------------
def _try_evalplus(
    sample_path: str,
    problem_path: str,
    k: int = 1,
    timeout: int = 20,
    n_workers: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    统一调用 evalplus.evaluate.evaluate_functional_correctness：
      - 优先传 problem_file；
      - 兼容旧版本形参（用位置参数兜底）；
      - 若完全失败则返回 None。
    """
    try:
        from evalplus.evaluate import evaluate_functional_correctness as evalplus_eval  # type: ignore
    except Exception:
        return None

    # 优先使用带 problem_file 的调用
    try:
        return evalplus_eval(sample_file=sample_path, problem_file=problem_path, k=k, n_workers=n_workers, timeout=timeout)
    except TypeError:
        # 某些老版本可能不接受 problem_file；尝试“仅 sample_file”
        try:
            return evalplus_eval(sample_file=sample_path, k=k, n_workers=n_workers, timeout=timeout)
        except Exception:
            return None
    except Exception:
        return None


def _try_humaneval(
    sample_path: str,
    problem_path: str,
    k: int = 1,
    timeout: int = 20,
    n_workers: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    统一调用 human_eval.evaluation.evaluate_functional_correctness：
      - 优先传 problem_file；
      - 兼容旧版本形参（用位置参数兜底）；
      - 若失败则返回 None（后续还有“单题 check_correctness”兜底）。
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness as heval_eval  # type: ignore
    except Exception:
        return None

    # 优先使用带 problem_file 的调用
    try:
        return heval_eval(sample_file=sample_path, problem_file=problem_path, k=k, n_workers=n_workers, timeout=timeout)
    except TypeError:
        # 老版本不支持 problem_file；尝试仅 sample_file（可能触发“Some problems are not attempted.”）
        try:
            return heval_eval(sample_file=sample_path, k=k, n_workers=n_workers, timeout=timeout)
        except Exception:
            return None
    except Exception:
        return None


def _try_humaneval_single_check(
    task_id: str,
    completion: str,
    timeout: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    最后兜底：直接调用 human_eval.evaluation.check_correctness 对单题做评测。
    注意：这是“官方实现的单题原语”，不是我们自己拼测试代码；属于合规“官方后端”。
    """
    try:
        from human_eval.evaluation import check_correctness  # type: ignore
    except Exception:
        return None

    problems = _read_problems()
    if task_id not in problems:
        return None

    problem = problems[task_id]
    try:
        # check_correctness 返回 dict: {"passed": bool, "result": "...", "task_id": "...", ...}
        out = check_correctness(problem=problem, completion=completion, timeout=timeout)
        return {"results": {task_id: {"passed": bool(out.get("passed", False))}}, "raw_single": out}
    except Exception:
        return None


# -----------------------
# 结果解析：尽量从官方返回里定位到单题 passed
# -----------------------
def _parse_official_result(res: Dict[str, Any], task_id: str) -> bool:
    """
    兼容多种返回形态：
      - {"results": {"<task_id>": {"passed": bool, ...}}, "pass@1": ...}
      - {"<task_id>": {"passed": bool}, ...}
      - {"results": [{"task_id": "...", "passed": bool}, ...]}
      - 兜底：无法解析则 False
    """
    if not isinstance(res, dict):
        return False

    # 常见/新式：res["results"][task_id]["passed"]
    results = res.get("results")
    if isinstance(results, dict):
        item = results.get(task_id)
        if isinstance(item, dict) and "passed" in item:
            return bool(item["passed"])

    # 顶层直接以 task_id 为键
    item = res.get(task_id)
    if isinstance(item, dict) and "passed" in item:
        return bool(item["passed"])

    # 列表形态
    if isinstance(results, list):
        for it in results:
            if isinstance(it, dict) and it.get("task_id") == task_id:
                return bool(it.get("passed", False))

    # 再退一步：某些实现只给出 pass@1（单题时 ==1.0 算通过）
    try:
        p1 = results.get("pass@1") if isinstance(results, dict) else res.get("pass@1")
        if p1 is not None:
            return float(p1) >= 1.0
    except Exception:
        pass

    return False


# -----------------------
# 对外主函数：评测“单题”样本
# -----------------------
def evaluate_one_official(
    task_id: str,
    completion: str,
    *,
    timeout: int = 20,
    n_workers: int = 1,
    k: int = 1,
) -> Dict[str, Any]:
    """
    使用官方（优先 EvalPlus，退回 human-eval）评测“单个” HumanEval 样本。
    返回：{"passed": bool, "backend": "evalplus"|"human-eval"|"human-eval-single", "raw": <官方或单题检查结果>}
    """
    # 写临时 samples.jsonl（只包含该样本）
    with tempfile.TemporaryDirectory() as td:
        sample_path = os.path.join(td, "samples.jsonl")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

        # 同时准备“单题” problem.jsonl
        problem_path = _dump_single_problem_jsonl(task_id)

        # 1) EvalPlus（若可用）
        res = _try_evalplus(sample_path, problem_path, k=k, timeout=timeout, n_workers=n_workers)
        if res is not None:
            passed = _parse_official_result(res, task_id)
            return {"passed": passed, "backend": "evalplus", "raw": res}

        # 2) human-eval（若可用）
        res = _try_humaneval(sample_path, problem_path, k=k, timeout=timeout, n_workers=n_workers)
        if res is not None:
            passed = _parse_official_result(res, task_id)
            return {"passed": passed, "backend": "human-eval", "raw": res}

        # 3) 兜底：单题 check_correctness（human-eval 官方原语）
        res = _try_humaneval_single_check(task_id, completion, timeout=timeout)
        if res is not None:
            passed = _parse_official_result(res, task_id)
            return {"passed": passed, "backend": "human-eval-single", "raw": res}

        # 4) 全部失败
        raise RuntimeError(
            "Neither EvalPlus nor human-eval are usable for single-sample evaluation. "
            "Please `pip install --upgrade evalplus human-eval`."
        )
