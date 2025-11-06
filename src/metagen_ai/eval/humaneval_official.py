# src/metagen_ai/eval/humaneval_official.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json, os, tempfile, sys

# 说明：
# - 优先使用 EvalPlus（更严格、覆盖更全），找不到就退回 human-eval 官方脚本。
# - 两条路径都按“写入 samples.jsonl → 调用 evaluate → 解析结果”的方式执行。
# - 为了和你的框架无缝衔接，这里提供“单样本评测”的便捷包装。

def _try_evalplus(sample_path: str, k: int = 1, timeout: int = 20, n_workers: int = 1) -> Optional[Dict[str, Any]]:
    try:
        from evalplus.evaluate import evaluate_functional_correctness as evalplus_eval
    except Exception:
        return None
    # EvalPlus 的 API 与 human-eval 保持一致（sample_file[, problem_file]...）
    try:
        # 有些版本参数名不同，这里统一用位置参数和常见关键字名容错
        res = evalplus_eval(sample_file=sample_path, k=k, n_workers=n_workers, timeout=timeout)
        return res
    except TypeError:
        # 旧版本可能只接受位置参数
        res = evalplus_eval(sample_path, k, n_workers, timeout)
        return res

def _try_humaneval(sample_path: str, k: int = 1, timeout: int = 20, n_workers: int = 1) -> Optional[Dict[str, Any]]:
    try:
        from human_eval.evaluation import evaluate_functional_correctness as heval_eval
    except Exception:
        return None
    try:
        res = heval_eval(sample_file=sample_path, k=k, n_workers=n_workers, timeout=timeout)
        return res
    except TypeError:
        res = heval_eval(sample_path, k, n_workers, timeout)
        return res

def evaluate_one_official(task_id: str, completion: str, *, timeout: int = 20, n_workers: int = 1, k: int = 1) -> Dict[str, Any]:
    """
    使用官方（优先 EvalPlus，退回 human-eval）评测单个 HumanEval 样本。
    返回字典：{"passed": bool, "backend": "evalplus"|"human-eval", "raw": <原始返回>}
    """
    # 写临时 samples.jsonl（官方接口要求）
    with tempfile.TemporaryDirectory() as td:
        sample_path = os.path.join(td, "samples.jsonl")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")

        # 1) 尝试 EvalPlus
        res = _try_evalplus(sample_path, k=k, timeout=timeout, n_workers=n_workers)
        if res is not None:
            passed = _parse_official_result(res, task_id)
            return {"passed": passed, "backend": "evalplus", "raw": res}

        # 2) 尝试 human-eval 官方
        res = _try_humaneval(sample_path, k=k, timeout=timeout, n_workers=n_workers)
        if res is not None:
            passed = _parse_official_result(res, task_id)
            return {"passed": passed, "backend": "human-eval", "raw": res}

        # 都不可用
        raise RuntimeError(
            "Neither EvalPlus nor human-eval is available. "
            "Please `pip install evalplus human-eval`."
        )

def _parse_official_result(res: Dict[str, Any], task_id: str) -> bool:
    """
    兼容不同版本返回结构：
    常见结构：
      {
        "results": { "<task_id>": {"passed": bool, ...}, ... },
        "pass@1": 0.xx,
        ...
      }
    有的版本可能是 { "<task_id>": {"passed": bool}, ... } 或追加别的层级。
    """
    if not isinstance(res, dict):
        return False

    # 新版/常见：res["results"][task_id]["passed"]
    results = res.get("results")
    if isinstance(results, dict):
        item = results.get(task_id)
        if isinstance(item, dict) and "passed" in item:
            return bool(item["passed"])

    # 退路：顶层直接按 task_id
    item = res.get(task_id)
    if isinstance(item, dict) and "passed" in item:
        return bool(item["passed"])

    # 再退路：某些实现会给出列表
    if isinstance(results, list):
        for it in results:
            if isinstance(it, dict) and it.get("task_id") == task_id:
                val = it.get("passed")
                return bool(val)
    return False
