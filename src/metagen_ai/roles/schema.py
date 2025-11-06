from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import re

# Local handler signature:
# ctx: {
#   "task": Dict[str, Any],
#   "inputs": Dict[str, str],
#   "prev_summary": str,
#   "node_id": str
# }
LocalHandler = Callable[[Dict[str, Any]], str]


@dataclass
class RoleProfile:
    name: str
    description: str
    system_template: str
    user_template: str
    local_handler: Optional[LocalHandler] = None
    temperature: Optional[float] = None
    # 新增：角色能力标签，用于多样化选择与能力覆盖
    # 可用标签示例：["reason"], ["compute"], ["aggregate"], ["verify"]
    capabilities: List[str] = field(default_factory=list)


@dataclass
class NodeOutput:
    text: str
    usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        }
    )


@dataclass
class RunTraceItem:
    node_id: str
    role: str
    prompt_preview: str
    output_preview: str
    usage: Dict[str, int]


@dataclass
class Hooks:
    # All hooks are optional callables; return None.
    before_node: Optional[Callable[[str, Dict[str, Any]], None]] = None
    after_node: Optional[Callable[[str, NodeOutput], None]] = None
    textual_gradient_hook: Optional[Callable[[Dict[str, Any]], None]] = None


# 兼容多种写法的 task 占位符
_BRACKET_Q = re.compile(r"\{task\[['\"]?question['\"]?\]\}")
_BRACKET_A = re.compile(r"\{task\[['\"]?answer['\"]?\]\}")
_BRACKET_QUERY = re.compile(r"\{task\[['\"]?query['\"]?\]\}")
_BRACKET_PROMPT = re.compile(r"\{task\[['\"]?prompt['\"]?\]\}")


class _SafeDict(dict):
    """format_map 兜底：缺键给空串，避免 KeyError。"""
    def __missing__(self, key):
        return ""


def _sanitize_placeholders(tmpl: str) -> str:
    """
    把 {task[question]} / {task['question']} / {task["question"]} 等替换成 {question}，
    同理处理 answer / query / prompt。
    """
    if not isinstance(tmpl, str):
        return ""
    s = tmpl
    s = _BRACKET_Q.sub("{question}", s)
    s = _BRACKET_A.sub("{answer}", s)
    s = _BRACKET_QUERY.sub("{query}", s)
    s = _BRACKET_PROMPT.sub("{prompt}", s)
    return s


def render_messages(role: RoleProfile,
                    task: Dict[str, Any],
                    inputs: Dict[str, Any],
                    prev_summary: str):
    """
    将角色模板渲染为 OpenAI/DeepSeek 兼容的 messages 列表。
    支持占位符：
      {task} / {inputs} / {prev_summary}
      {question} / {answer} / {query} / {prompt}
    并自动把 {task[...]} 写法替换成扁平字段以避免 KeyError。
    """
    # 扁平上下文
    question = ""
    answer = ""
    query = ""
    prompt = ""

    if isinstance(task, dict):
        question = str(task.get("question", ""))
        # HumanEval 等代码补全类任务常用 prompt
        prompt = str(task.get("prompt", ""))
        answer = str(task.get("answer", ""))
        query = str(task.get("query", ""))
    else:
        # task 是字符串时，按 question 处理
        question = str(task)

    ctx = _SafeDict({
        "task": task,
        "inputs": inputs,
        "prev_summary": prev_summary,
        "question": question,
        "answer": answer,
        "query": query,
        "prompt": prompt,
    })

    sys_t = _sanitize_placeholders(getattr(role, "system_template", "") or "")
    usr_t = _sanitize_placeholders(getattr(role, "user_template", "") or "")

    try:
        sys = sys_t.format_map(ctx)
    except Exception:
        sys = sys_t

    try:
        usr = usr_t.format_map(ctx)
    except Exception:
        usr = usr_t

    # 没有模板也要给出最低限度提示，避免空 messages
    if not sys and not usr:
        usr = f"Task: {question or prompt or task}\nInputs: {inputs}\nPrev: {prev_summary}"

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]
