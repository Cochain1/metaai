# src/metagen_ai/utils/metrics.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import statistics

def summarize_usage(usages: List[Dict[str, int]]) -> Dict[str, float]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for u in usages:
        totals["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        totals["completion_tokens"] += int(u.get("completion_tokens", 0))
        totals["total_tokens"] += int(u.get("total_tokens", 0))
        totals["calls"] += int(u.get("calls", 1)) if "calls" in u else 1
    return totals

def accuracy_from_final(final_text: str, gold: str) -> float:
    if gold is None:
        return 0.0
    return 1.0 if str(gold).strip() in str(final_text) else 0.0

def cost_from_usage(usage: Dict[str, int], w_tokens: float = 1.0, w_calls: float = 5.0) -> float:
    """
    Simple scalarized cost: w_tokens * total_tokens + w_calls * calls.
    Calls are not tracked per node in our runner, so we approximate with 1 call per node output.
    For global run usage we just use total_tokens, and assume single call unit.
    """
    tokens = float(usage.get("total_tokens", 0))
    calls = float(usage.get("calls", 1))
    return w_tokens * tokens + w_calls * calls
