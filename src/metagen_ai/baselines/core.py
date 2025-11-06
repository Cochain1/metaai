# src/metagen_ai/baselines/core.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re, time, math, random
from collections import Counter

from metagen_ai.utils.llm import LLMClient
from metagen_ai.eval.metrics import judge_correct

_ANS_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)|Final answer:\s*([-+]?\d+(?:\.\d+)?)", re.I)

def _extract_num(s: str) -> str | None:
    m = _ANS_RE.search(s)
    if m:
        return (m.group(1) or m.group(2)).strip()
    # 回退：抓最后一个数字
    m2 = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return m2[-1] if m2 else None

def _prompt_cot(question: str) -> List[Dict[str,str]]:
    sys = "You are a careful math tutor. Think step by step, then give the final numeric answer as: Final answer: <number>."
    usr = f"Solve the problem step by step, then end with one line 'Final answer: <number>'.\nProblem: {question}"
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

def run_cot(llm: LLMClient, task: Dict[str,Any], temperature=0.2, max_tokens=512) -> Tuple[str, Dict[str,int], float]:
    t0 = time.time()
    msgs = _prompt_cot(task["question"])
    r = llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
    dt = time.time() - t0
    txt = r["text"]
    usage = r.get("usage", {"total_tokens":0})
    ans = _extract_num(txt) or ""
    return (f"Final answer: {ans}", usage, dt)

def run_self_consistency(llm: LLMClient, task: Dict[str,Any], k=5, temperature=0.7, max_tokens=512) -> Tuple[str, Dict[str,int], float]:
    t0 = time.time()
    votes: List[str] = []
    total_tok = 0
    for _ in range(k):
        r = llm.chat(_prompt_cot(task["question"]), temperature=temperature, max_tokens=max_tokens)
        votes.append(_extract_num(r["text"]) or "")
        total_tok += r.get("usage", {}).get("total_tokens", 0)
    dt = time.time() - t0
    cnt = Counter(votes)
    cand, _ = cnt.most_common(1)[0]
    return (f"Final answer: {cand}", {"total_tokens": total_tok}, dt)

def run_debate(llm: LLMClient, task: Dict[str,Any], rounds=2, temperature=0.5, max_tokens=512) -> Tuple[str, Dict[str,int], float]:
    """
    两个辩手分别推理，然后互评若干轮，最后交给裁判（同一模型）做裁决。
    轻量实现，保证和 CoT 同样的 API。
    """
    q = task["question"]
    t0 = time.time()
    usage = 0

    # 初稿
    debaters = []
    for side in ("A","B"):
        sys = f"You are Debater {side}. Reason step by step and propose a numeric answer for the problem."
        usr = f"Problem: {q}\nEnd with 'Final answer: <number>'."
        r = llm.chat([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=temperature, max_tokens=max_tokens)
        debaters.append(r["text"]); usage += r.get("usage",{}).get("total_tokens",0)

    # 互评 rounds 轮
    for _ in range(max(0, rounds-1)):
        na = llm.chat(
            [{"role":"system","content":"You critique briefly, then refine. End with 'Final answer: <number>'."},
             {"role":"user","content":f"Problem: {q}\nOpponent's reasoning:\n{debaters[1]}\nNow produce your improved reasoning and final answer."}],
            temperature=temperature, max_tokens=max_tokens)
        nb = llm.chat(
            [{"role":"system","content":"You critique briefly, then refine. End with 'Final answer: <number>'."},
             {"role":"user","content":f"Problem: {q}\nOpponent's reasoning:\n{debaters[0]}\nNow produce your improved reasoning and final answer."}],
            temperature=temperature, max_tokens=max_tokens)
        debaters = [na["text"], nb["text"]]
        usage += na.get("usage",{}).get("total_tokens",0) + nb.get("usage",{}).get("total_tokens",0)

    # 裁判
    judge = llm.chat(
        [{"role":"system","content":"You are a fair judge. Read both solutions and choose ONE numeric answer. Output exactly one line 'Final answer: <number>'."},
         {"role":"user","content":f"Problem: {q}\nSolution A:\n{debaters[0]}\nSolution B:\n{debaters[1]}"}],
        temperature=0.0, max_tokens=256)
    usage += judge.get("usage",{}).get("total_tokens",0)
    dt = time.time() - t0
    ans = _extract_num(judge["text"]) or (_extract_num(debaters[0]) or _extract_num(debaters[1]) or "")
    return (f"Final answer: {ans}", {"total_tokens": usage}, dt)

def run_tot(llm: LLMClient, task: Dict[str,Any], breadth=3, depth=2, temperature=0.5, max_tokens=256) -> Tuple[str, Dict[str,int], float]:
    """
    简化版 Tree-of-Thought：宽度=3，深度=2，打分选优。
    每个节点生成若干“想法”，用 self-eval 打分，选 top-1 向下展开。
    """
    q = task["question"]
    t0 = time.time()
    usage = 0

    frontier = [""]
    best_path, best_score, best_ans = "", -1e9, ""
    for d in range(depth):
        new_frontier: List[Tuple[str,float,str]] = []
        for path in frontier:
            # 生成 breadth 个想法
            for _ in range(breadth):
                r = llm.chat(
                    [{"role":"system","content":"Generate one brief reasoning step continuing the solution. Then propose 'Final answer: <number>' if you are confident."},
                     {"role":"user","content":f"Problem: {q}\nContext so far:\n{path}\nContinue with one short step."}],
                    temperature=temperature, max_tokens=max_tokens)
                usage += r.get("usage",{}).get("total_tokens",0)
                step = r["text"]

                # 自评打分
                s = llm.chat(
                    [{"role":"system","content":"Score 0-10: how promising is the above step toward a correct numeric answer? Output only a number."},
                     {"role":"user","content":step}],
                    temperature=0.0, max_tokens=8)
                usage += s.get("usage",{}).get("total_tokens",0)
                try:
                    score = float(re.findall(r"[-+]?\d+(?:\.\d+)?", s["text"])[0])
                except Exception:
                    score = 5.0
                new_frontier.append((path + "\n" + step, score, _extract_num(step) or ""))

        # 选 top-1 进入下一层
        new_frontier.sort(key=lambda x: x[1], reverse=True)
        if new_frontier:
            path, score, cand = new_frontier[0]
            frontier = [path]
            if score > best_score and cand:
                best_score, best_path, best_ans = score, path, cand

    dt = time.time() - t0
    final = best_ans or ""
    return (f"Final answer: {final}", {"total_tokens": usage}, dt)

def run_star_lite(llm: LLMClient, task: Dict[str,Any], temperature=0.3, max_tokens=512) -> Tuple[str, Dict[str,int], float]:
    """
    STAR 风格的轻量版：先让“学生”解，再让“老师”点评改进，最后学生复写一遍。
    """
    q = task["question"]
    t0 = time.time(); usage = 0

    student1 = llm.chat(
        [{"role":"system","content":"You are a student. Solve step by step, end with 'Final answer: <number>'."},
         {"role":"user","content":q}],
        temperature=temperature, max_tokens=max_tokens)
    usage += student1.get("usage",{}).get("total_tokens",0)

    teacher = llm.chat(
        [{"role":"system","content":"You are a teacher. Briefly point out mistakes or improvements. Suggest the correct final numeric answer if possible."},
         {"role":"user","content":f"Problem: {q}\nStudent solution:\n{student1['text']}"}],
        temperature=0.2, max_tokens=384)
    usage += teacher.get("usage",{}).get("total_tokens",0)

    student2 = llm.chat(
        [{"role":"system","content":"Rewrite a clean, corrected solution, and end with exactly one line 'Final answer: <number>'."},
         {"role":"user","content":f"Problem: {q}\nTeacher feedback:\n{teacher['text']}"}],
        temperature=0.2, max_tokens=max_tokens)
    usage += student2.get("usage",{}).get("total_tokens",0)

    dt = time.time() - t0
    ans = _extract_num(student2["text"]) or _extract_num(teacher["text"]) or _extract_num(student1["text"]) or ""
    return (f"Final answer: {ans}", {"total_tokens": usage}, dt)
