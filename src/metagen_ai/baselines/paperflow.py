# src/metagen_ai/baselines/paperflow.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import re, time, json
from collections import defaultdict, Counter

from metagen_ai.utils.llm import LLMClient

_NUM_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)|Final answer:\s*([-+]?\d+(?:\.\d+)?)", re.I)
def extract_number(text: str) -> Optional[str]:
    m = _NUM_RE.search(text or "")
    if m:
        return (m.group(1) or m.group(2)).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")
    return nums[-1].strip() if nums else None

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@dataclass
class RoleSpec:
    name: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 512

@dataclass
class PaperFlow:
    roles: Dict[str,RoleSpec]
    rounds: int
    edges: List[Dict[str,str]]
    aggregate: Dict[str,Any]

class PaperFlowRunner:
    def __init__(self, flow: PaperFlow, llm: LLMClient):
        self.flow = flow
        self.llm = llm

    def _render(self, role: RoleSpec, question: str, context: str) -> List[Dict[str,str]]:
        sys = role.system_prompt
        usr = role.user_prompt.format(question=question, context=context)
        return [{"role":"system","content":sys},{"role":"user","content":usr}]

    def run_one(self, task: Dict[str,Any]) -> Tuple[str, Dict[str,int], float, Dict[str,Any]]:
        q = str(task.get("question",""))
        t0 = time.time()
        total_tokens = 0
        history: Dict[int, Dict[str, str]] = defaultdict(dict)

        for r in range(1, self.flow.rounds + 1):
            for name, role in self.flow.roles.items():
                vis = []
                if history.get(r-1, {}).get(name):
                    vis.append(f"[Self prev]\n{history[r-1][name]}")
                for e in self.flow.edges:
                    if e.get("to") == name:
                        src = e.get("from")
                        if history.get(r-1, {}).get(src):
                            tag = e.get("mode","msg")
                            vis.append(f"[{src} prev {tag}]\n{history[r-1][src]}")
                context = "\n\n".join(vis).strip()
                msgs = self._render(role, q, context)
                resp = self.llm.chat(msgs, temperature=role.temperature, max_tokens=role.max_tokens)
                history[r][name] = resp["text"]
                total_tokens += resp.get("usage",{}).get("total_tokens",0)

        agg = self.flow.aggregate or {"type":"judge"}
        final = ""
        if agg["type"] == "judge":
            judge_name = agg.get("judge","judge")
            judge = self.flow.roles[judge_name]
            bundle = []
            for n, txt in history[self.flow.rounds].items():
                if n != judge_name:
                    bundle.append(f"Solution {n}:\n{txt}")
            msgs = [
                {"role":"system","content":judge.system_prompt},
                {"role":"user","content":f"Problem: {q}\n\n" + "\n\n".join(bundle)}
            ]
            resp = self.llm.chat(msgs, temperature=0.0, max_tokens=min(256, judge.max_tokens))
            total_tokens += resp.get("usage",{}).get("total_tokens",0)
            final = resp["text"]
        elif agg["type"] == "majority":
            votes = []
            for n, txt in history[self.flow.rounds].items():
                ans = extract_number(txt)
                if ans: votes.append(ans)
            if votes:
                final = f"Final answer: {Counter(votes).most_common(1)[0][0]}"
            else:
                final = "Final answer: "
        elif agg["type"] == "last_judge":
            judge_name = agg.get("judge","judge")
            final = history[self.flow.rounds].get(judge_name,"")
        else:
            raise ValueError(f"Unknown aggregate.type: {agg['type']}")

        dt = time.time() - t0
        return final, {"total_tokens": total_tokens}, dt, {"history": history}

def load_paperflow_from_yaml(yaml_path: str) -> PaperFlow:
    import yaml, os
    data = yaml.safe_load(_load_text(yaml_path))
    roles = {}
    base = os.path.dirname(yaml_path)
    def join(p): return p if os.path.isabs(p) else os.path.join(base, p)
    for r in data["roles"]:
        roles[r["name"]] = RoleSpec(
            name=r["name"],
            system_prompt=_load_text(join(r["system_prompt"])),
            user_prompt=_load_text(join(r["user_prompt"])),
            temperature=float(r.get("temperature", 0.2)),
            max_tokens=int(r.get("max_tokens", 512)),
        )
    return PaperFlow(
        roles=roles,
        rounds=int(data.get("rounds", 1)),
        edges=list(data.get("edges", [])),
        aggregate=dict(data.get("aggregate", {"type":"judge"})),
    )
