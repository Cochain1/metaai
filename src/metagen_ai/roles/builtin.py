# src/metagen_ai/roles/builtin.py
from __future__ import annotations
import re, ast
from typing import Dict, Any, List, Optional
from metagen_ai.roles.schema import RoleProfile

__all__ = ["BUILTIN_ROLES"]

# ---------- Helpers ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _extract_numbers(s: str):
    return [float(x) if "." in x else int(x) for x in _NUMBER_RE.findall(str(s))]

def _gather_text(ctx: Dict[str, Any]) -> str:
    """Concatenate task text, previous summary, and upstream node outputs."""
    task = ctx.get("task", {})
    ins = ctx.get("inputs", {})
    prev = ctx.get("prev_summary", "")
    parts = [str(task.get("question") or task.get("prompt") or task), str(prev)]
    if isinstance(ins, dict):
        parts.extend([str(v) for v in ins.values()])
    return "\n".join(p for p in parts if p)


def _extract_code_blocks(text: str) -> List[str]:
    """Extract ```python ...``` or ```...``` blocks."""
    if not text:
        return []
    blocks = []
    for m in re.finditer(r"```(?:python)?\s*(.*?)```", text, re.S | re.I):
        code = m.group(1).strip()
        if code:
            blocks.append(code)
    return blocks

def _first_func_name_from_prompt(prompt: str) -> Optional[str]:
    """
    Try to infer function name from HumanEval prompt.
    """
    if not prompt:
        return None
    # Try "def name("
    m = re.search(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(", prompt)
    if m:
        return m.group(1)
    return None


# ---------- Local handlers (deterministic) ----------

def _calc_handler(ctx: Dict[str, Any]) -> str:
    text = _gather_text(ctx).lower()
    nums = _extract_numbers(text)
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if any(k in text for k in ["add", "plus", "sum", "+"]):
            return f"{a} + {b} = {a + b}"
        if any(k in text for k in ["subtract", "minus", "difference", "−", "- "]):
            return f"{a} - {b} = {a - b}"
        if any(k in text for k in ["multiply", "times", "product", "×", "*"]):
            return f"{a} * {b} = {a * b}"
        if any(k in text for k in ["divide", "quotient", "÷", "/"]):
            if b == 0:
                return "Division by zero is undefined."
            return f"{a} / {b} = {a / b}"
        return f"{a} + {b} = {a + b}"
    return "No numbers found."

def _simplify_handler(ctx: Dict[str, Any]) -> str:
    text = _gather_text(ctx)
    m = re.search(r"=\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return f"Candidate result: {m.group(1)}"
    return text

def _safe_eval(expr: str):
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
               ast.Constant, ast.Tuple)
    node = ast.parse(expr, mode="eval")

    def _check(n):
        if not isinstance(n, allowed):
            raise ValueError(f"Unsafe node: {type(n).__name__}")
        for child in ast.iter_child_nodes(n):
            _check(child)
    _check(node)

    return eval(compile(node, "<safe-eval>", "eval"), {"__builtins__": {}}, {})

def _equation_extractor_handler(ctx: Dict[str, Any]) -> str:
    """
    从上游 reasoner 输出中抽取 'EQUATION: ...' 并安全求值。
    """
    text = _gather_text(ctx)
    lines = [ln.strip() for ln in str(text).splitlines()]
    eq = None
    for ln in lines:
        m = re.match(r"(?i)EQUATION\s*:\s*(.+)$", ln)
        if m:
            eq = m.group(1).strip()

    if eq is None:
        m = re.search(r"(?i)(?:Equation|Compute)\s*:\s*(.+)", text)
        if m:
            eq = m.group(1).strip()

    if eq is None:
        nums = _extract_numbers(text)
        if len(nums) >= 2:
            return f"Candidate result: {nums[0] + nums[1]}"
        return "No equation found."

    try:
        val = _safe_eval(eq)
        if isinstance(val, float) and abs(val - round(val)) < 1e-9:
            val = int(round(val))
        return f"Candidate result: {val}"
    except Exception as e:
        return f"Equation error: {e}"

def _committee_agg_handler(ctx: Dict[str, Any]) -> str:
    """
    多数票汇聚：从上游收集 'Candidate result: x' 或最后一个数字，做众数投票。
    """
    ins = ctx.get("inputs", {}) or {}
    cands: List[str] = []
    for v in ins.values():
        m = re.search(r"(?:Candidate result|Final answer)\s*:\s*([-+]?\d+(?:\.\d+)?)", str(v), re.I)
        if m:
            cands.append(m.group(1))
        else:
            nums = _extract_numbers(v)
            if nums:
                cands.append(str(nums[-1]))
    if not cands:
        return "No candidate."
    from collections import Counter
    vote = Counter(cands).most_common(1)[0][0]
    return f"Candidate result: {vote}"

def _evaluator_handler(ctx: Dict[str, Any]) -> str:
    """
    GSM8K/数值类：产出 Final answer: <number>...
    """
    task = ctx.get("task", {})
    gold = task.get("answer")
    text = _gather_text(ctx)
    m = re.search(r"(?:Candidate result|Final answer)\s*:\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    nums = _extract_numbers(text)
    cand = m.group(1) if m else (str(nums[-1]) if nums else None)
    if cand is None:
        return "No confident answer; request another round or more evidence."
    if gold is not None:
        ok = (str(cand).strip() == str(gold).strip())
        status = "verified" if ok else "unverified"
        return f"Final answer: {cand} ({status})."
    else:
        return f"Final answer: {cand} (unverified)."

# ----- HumanEval 专用：只输出代码的“评审器” -----

def _code_evaluator_handler(ctx: Dict[str, Any]) -> str:
    """
    从上游节点输出里，挑选/整合一个【唯一的 Python 代码块】作为最终结果。
    规则：
      1) 若 task含 entry_point，则优先选择包含 `def <entry_point>(` 的代码块；
      2) 否则尝试从 prompt 中推断函数名；
      3) 多个候选时，选择包含目标函数名且行数最多的那个；
      4) 若仍无，选择最长的代码块；若完全找不到代码块，返回空壳函数（保函数名）。
    """
    task = ctx.get("task", {}) or {}
    ins = ctx.get("inputs", {}) or {}
    prompt_text = str(task.get("prompt") or task.get("question") or "")
    entry_point = task.get("entry_point") or _first_func_name_from_prompt(prompt_text)

    # 汇总所有上游文本并提取代码块
    upstream_text = "\n".join([str(v) for v in ins.values()]) if ins else ""
    blocks = _extract_code_blocks(upstream_text)
    # 在某些情况下，final也会被并入 inputs；这里不再重复提取

    def _has_entry(code: str, name: Optional[str]) -> bool:
        if not name:
            return False
        pat = rf"(?m)^\s*def\s+{re.escape(name)}\s*\("
        return re.search(pat, code) is not None

    picked = None
    if blocks:
        # 1) 先筛选包含 entry_point 的
        if entry_point:
            with_entry = [b for b in blocks if _has_entry(b, entry_point)]
            if with_entry:
                # 行数多 ≈ 更完整
                with_entry.sort(key=lambda x: x.count("\n"), reverse=True)
                picked = with_entry[0]
        # 2) 没有就挑最长的
        if picked is None:
            blocks.sort(key=lambda x: len(x), reverse=True)
            picked = blocks[0]

    if picked is None:
        # 最后兜底：构造空壳函数（至少不会跑题到文字）
        name = entry_point or "solution"
        picked = f"def {name}(*args, **kwargs):\n    # TODO: implement\n    raise NotImplementedError"

    return f"```python\n{picked}\n```"


# ---------- Built-in roles (Hybrid-ready) ----------

BUILTIN_ROLES: Dict[str, RoleProfile] = {
    # 让 LLM 专心把文字题转成表达式（不计算）
    "reasoner": RoleProfile(
        name="reasoner",
        description="Parse the word problem and output one arithmetic expression only.",
        system_template=(
            "You are a math reasoning agent. Convert the word problem into a single arithmetic expression.\n"
            "Rules:\n"
            "1) DO NOT compute the final value.\n"
            "2) Output exactly one line: 'EQUATION: <expression>'.\n"
            "3) Use only numbers and + - * / ** and parentheses."
        ),
        user_template=(
            "Problem: {question}\n"
            "Useful upstream: {inputs}\n"
            "Output format (single line): EQUATION: <expression>"
        ),
        local_handler=None,   # 纯 LLM
        temperature=0.1,
        capabilities=["reason"]
    ),

    # 把方程转为候选数值（本地确定性）
    "equation_extractor": RoleProfile(
        name="equation_extractor",
        description="Extract 'EQUATION: ...' from upstream and compute its value safely.",
        system_template="You extract the equation and compute deterministically using a safe evaluator.",
        user_template="Use upstream to find 'EQUATION:' line and compute its value.",
        local_handler=_equation_extractor_handler,
        temperature=0.0,
        capabilities=["compute"]
    ),

    # 多数票聚合
    "committee_aggregator": RoleProfile(
        name="committee_aggregator",
        description="Aggregate multiple candidate numeric answers via majority voting.",
        system_template="Aggregate upstream numeric candidates by majority vote.",
        user_template="Read upstream candidates and output 'Candidate result: <number>'.",
        local_handler=_committee_agg_handler,
        temperature=0.0,
        capabilities=["aggregate"]
    ),

    # GSM8K/数值类终判
    "evaluator": RoleProfile(
        name="evaluator",
        description="Verify the final numeric answer against the gold answer if provided.",
        system_template="You check the final numeric answer against the gold answer if provided.",
        user_template="Task: {question} (gold: {answer})\nPrev: {prev_summary}\nUpstream: {inputs}",
        local_handler=_evaluator_handler,
        temperature=0.0,
        capabilities=["verify"]
    ),

    # 可选：演示用计算器
    "calculator": RoleProfile(
        name="calculator",
        description="Deterministic arithmetic parser and solver for simple expressions.",
        system_template="You compute on extracted numbers deterministically.",
        user_template=(
            "Task: {question}\nPrev: {prev_summary}\nUpstream: {inputs}\n"
            "Compute the required operation."
        ),
        local_handler=_calc_handler,
        temperature=0.0,
        capabilities=["compute"]
    ),

    "math_simplifier": RoleProfile(
        name="math_simplifier",
        description="Extract the numeric result from upstream outputs.",
        system_template="You extract the numeric result on the right side of equations.",
        user_template=(
            "Task: {question}\nPrev: {prev_summary}\nUpstream: {inputs}\n"
            "Return 'Candidate result: <number>' if possible."
        ),
        local_handler=_simplify_handler,
        temperature=0.0,
        capabilities=["aggregate"]
    ),

    # --------- HumanEval 专用角色 ---------

    "programmer": RoleProfile(
        name="programmer",
        description="Writes the full Python implementation for the given function signature and docstring.",
        system_template=(
            "You are a senior Python engineer.\n"
            "Return ONLY a Python code block implementing the requested function.\n"
            "Do not include any explanations or text outside the code block."
        ),
        user_template=(
            "Implement the function described below.\n"
            "Constraints:\n"
            "1) Keep the SAME function name and parameters as given.\n"
            "2) Pure Python; no external packages.\n"
            "3) Include any helpers inside the same file.\n"
            "4) If tests imply edge cases, handle them.\n"
            "\n"
            "Function prompt:\n{task}\n"
        ),
        local_handler=None,
        temperature=0.1,
        capabilities=["compute"]
    ),

    "code_evaluator": RoleProfile(
        name="code_evaluator",
        description="Selects the best code implementation from upstream candidates or synthesizes a stub; outputs a single Python code block.",
        system_template=(
            "You are a strict code selector. "
            "Your output MUST be exactly one Python code block with the final solution. "
            "Prefer a block that defines the required entry point function if known."
        ),
        user_template=(
            "Task:\n{task}\n"
            "Upstream candidates (may include multiple code blocks):\n{inputs}\n"
            "Return ONLY one Python code block. No extra text."
        ),
        local_handler=_code_evaluator_handler,
        temperature=0.0,
        capabilities=["verify"]
    ),
}
