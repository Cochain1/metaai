# src/metagen_ai/utils/role_stats.py
from __future__ import annotations
import json, os
from typing import Dict

def load_role_stats(path: str) -> Dict[str, dict]:
    """
    读取角色历史统计：
      {
        "role_name": {"seen": int, "ok": int},
        ...
      }
    若文件不存在或损坏，返回 {}。
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
            return {}
    except Exception:
        return {}

def save_role_stats(path: str, stats: Dict[str, dict]) -> None:
    """写回统计 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def bump_role_outcome(stats: Dict[str, dict], role_name: str, ok: bool) -> None:
    """
    在内存中更新某个角色的一次结果：
      seen += 1, ok += int(ok)
    """
    rec = stats.setdefault(role_name, {"seen": 0, "ok": 0})
    rec["seen"] = int(rec.get("seen", 0)) + 1
    rec["ok"]  = int(rec.get("ok", 0)) + (1 if ok else 0)

def trust_of(role_name: str, stats: Dict[str, dict],
             prior: float = 0.5, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    计算角色的“信任度”，默认用 Beta 后验均值：
      (ok + alpha) / (seen + alpha + beta)
    若无记录，返回 prior（默认 0.5）。
    """
    rec = stats.get(role_name)
    if not rec:
        return float(prior)
    seen = max(0, int(rec.get("seen", 0)))
    ok   = max(0, int(rec.get("ok", 0)))
    return (ok + alpha) / (seen + alpha + beta)
