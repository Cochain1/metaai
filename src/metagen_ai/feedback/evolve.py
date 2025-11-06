from __future__ import annotations
from typing import Dict, Any, Optional
import re

def build_meta_evolver_hook(program, cfg, llm):
    """
    返回一个可作为 hooks.textual_gradient_hook 使用的函数：
    - 选一个低贡献角色，向 LLM 请求“微调模板”（更确定/更简洁）；
    - 随机 gating 一条非关键边（非 evaluator/judge 入边）；
    - 如需 HumanEval 题内进化，可在标注处把失败摘要写入 program._last_evolve_summary。
    """
    # 简单“贡献度”度量：输出长度 + 是否包含数字（对 GSM8K 有用）
    def _score(text: str) -> float:
        if not text:
            return 0.0
        s = len(text)
        has_num = any(ch.isdigit() for ch in text)
        return s * (1.25 if has_num else 1.0)

    def _choose_low_contrib(node_outputs: Dict[str, str]) -> Optional[str]:
        items = [(nid, _score(txt)) for nid, txt in node_outputs.items()]
        if not items:
            return None
        items.sort(key=lambda x: x[1])
        for nid, _ in items:
            if nid not in ("task_hub", "evaluator", "judge"):
                return nid
        return None

    def _rewrite_role(role, task, facts: Dict[str, Any]):
        sys = (
            "You are refining an agent role definition to improve determinism and brevity.\n"
            "Return ONLY JSON with keys: system_template, user_template.\n"
            "Keep instructions concise; enforce single-line / exact-format outputs when needed."
        )
        usr = f"""Task payload (stringified): {task}
Current role name: {role.name}
Current description: {role.description}
Current system_template: {role.system_template}
Current user_template: {role.user_template}
Observations (short): {facts}

Goal: Slightly refine the templates to be more crisp/deterministic.
Output pure JSON (no extra text, no code fences)."""
        try:
            resp = llm.chat(
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                temperature=0.2, max_tokens=400
            )
            raw = resp.get("text", "")
            # 允许直接 JSON；若用户加了围栏，也兜一层
            m = re.search(r"\{.*\}", raw, re.S)
            js = m.group(0) if m else raw
            import json
            data = json.loads(js)
            st = str(data.get("system_template", "")).strip() or role.system_template
            ut = str(data.get("user_template", "")).strip() or role.user_template
            # 适度截断，避免提示过长
            role.system_template = st[:1600]
            role.user_template = ut[:1600]
        except Exception:
            # 静默失败
            pass

    def _gate_one_edge(G):
        # 轻量 gating：随机挑一条非关键边、若还处于 active 则置为 inactive
        # 关键边：任何指向 evaluator/judge 的边
        import random
        edges = [(u, v, d) for (u, v, d) in G.edges(data=True) if d.get("active", True)]
        random.shuffle(edges)
        for (u, v, d) in edges:
            if v in ("evaluator", "judge"):
                continue
            # 小概率 gating，避免扰动过大
            if random.random() < 0.2:
                d["active"] = False
                break

    def _maybe_humaneval_feedback(payload: Dict[str, Any]):
        # 如需把 HumanEval 失败摘要写回下一轮，可在此集成你的执行器：
        # 示例（伪代码）：
        #
        # task = payload.get("task", {})
        # if not (isinstance(task, dict) and "tests" in task and "entry_point" in task):
        #     return
        # node_outputs = payload.get("node_outputs", {}) or {}
        # latest_code = None
        # for nid in ["Decider", "BugFixer", "ProgrammingExpert"]:
        #     if nid in node_outputs:
        #         m = re.search(r"```python\s*(.*?)```", node_outputs[nid], re.S|re.I)
        #         if m:
        #             latest_code = m.group(1).strip()
        #             break
        # if latest_code:
        #     try:
        #         from metagen_ai.eval.humaneval_exec import run_one  # 如果你有这个执行器
        #         fail_report = run_one(latest_code, task["tests"], task["entry_point"])
        #     except Exception:
        #         fail_report = ""
        #     if fail_report:
        #         program._last_evolve_summary = "[TEST_FAILURES]\n" + fail_report
        #     else:
        #         program._last_evolve_summary = "[TESTS_PASS]\nAll tests passed."
        return

    def _hook(payload: Dict[str, Any]):
        try:
            # 1) 选一个低贡献节点微调角色模板（小改，不会破坏 GSM8K 主链）
            node_outputs = payload.get("node_outputs", {}) or {}
            nid = _choose_low_contrib(node_outputs)
            if nid and nid in program.role_library:
                _rewrite_role(program.role_library[nid], str(payload.get("task", ""))[:1200], {
                    "node_id": nid,
                    "summary": (payload.get("summary", "") or "")[:400],
                    "output": (node_outputs.get(nid, "") or "")[:600]
                })
            # 2) 图结构轻量 gating
            _gate_one_edge(program.G)
            # 3) （可选）HumanEval 失败摘要回灌
            _maybe_humaneval_feedback(payload)
        except Exception:
            # 静默
            return
    return _hook
