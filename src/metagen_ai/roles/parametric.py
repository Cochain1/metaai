# src/metagen_ai/roles/parametric.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from metagen_ai.roles.schema import RoleProfile

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class TempConfig:
    delta_min: float = -0.7   # allowed temperature offset range
    delta_max: float = 0.7
    base_min: float = 0.0     # absolute clamp for final temperature
    base_max: float = 1.0
    lr: float = 0.1           # learning rate for temp updates
    wd: float = 1e-3          # weight decay on deltas

@dataclass
class PromptConfig:
    # Predefined prompt “hints” we can attach as small strings to system/user templates
    # You can extend/modify these safely.
    hints: Dict[str, str] = field(default_factory=lambda: {
        "concise": "Be concise and return only what is asked.",
        "cot_short": "Think step by step briefly, then provide the final answer.",
        "verify": "Cross-check consistency with upstream messages before finalizing.",
        "reflect": "If uncertain, reflect on the most likely mistake and fix it.",
    })
    max_hints_per_role: int = 2

@dataclass
class ParamManager:
    """
    Soft-parameter store for roles:
      - temperature deltas (role → ΔT)
      - prompt hint selections (role → [hint_ids])
    It mutates RoleProfile in-place before each node runs.
    """
    temp_cfg: TempConfig = field(default_factory=TempConfig)
    prompt_cfg: PromptConfig = field(default_factory=PromptConfig)

    # state
    temp_delta: Dict[str, float] = field(default_factory=dict)
    prompt_hints: Dict[str, List[str]] = field(default_factory=dict)

    def get_final_temp(self, role: RoleProfile, role_name: str, default_temperature: float) -> float:
        base = role.temperature if role.temperature is not None else default_temperature
        delta = self.temp_delta.get(role_name, 0.0)
        temp = _clip(base + delta, self.temp_cfg.base_min, self.temp_cfg.base_max)
        return temp

    def _render_hints_suffix(self, role_name: str) -> str:
        ids = self.prompt_hints.get(role_name, [])
        if not ids:
            return ""
        # Deduplicate and cap
        uniq = []
        for h in ids:
            if h not in uniq and h in self.prompt_cfg.hints:
                uniq.append(h)
            if len(uniq) >= self.prompt_cfg.max_hints_per_role:
                break
        if not uniq:
            return ""
        joined = " ".join(self.prompt_cfg.hints[h] for h in uniq)
        return f"\n[INSTRUCTIONS] {joined}"

    # --------- Public API: apply before a node runs ----------
    def apply_before_node(self, role: RoleProfile, role_name: str, default_temperature: float) -> None:
        """
        In-place adjust role.temperature and append hint suffix to system_template.
        """
        # 1) Temperature
        role.temperature = self.get_final_temp(role, role_name, default_temperature)

        # 2) Prompt hints: append a small instruction suffix to system_template
        suffix = self._render_hints_suffix(role_name)
        if suffix:
            # Avoid duplicating the suffix during multiple rounds: store a guard tag.
            tag = f"<!--pm-{role_name}-hint-->"
            if tag not in role.system_template:
                role.system_template = role.system_template + "\n" + tag + suffix

    # --------- Update rules from “textual gradients” ----------
    def step_temperature(self, grads: Dict[str, float]) -> None:
        """
        grads: role_name → dL/dT (positive increases loss; we move temp in the negative direction)
        """
        for r, g in grads.items():
            cur = self.temp_delta.get(r, 0.0)
            g_total = float(g) + self.temp_cfg.wd * cur
            nxt = cur - self.temp_cfg.lr * g_total
            self.temp_delta[r] = _clip(nxt, self.temp_cfg.delta_min, self.temp_cfg.delta_max)

    def add_hint(self, role_name: str, hint_id: str) -> None:
        if hint_id not in self.prompt_cfg.hints:
            return
        lst = self.prompt_hints.setdefault(role_name, [])
        if hint_id not in lst:
            lst.append(hint_id)
            # keep small
            if len(lst) > self.prompt_cfg.max_hints_per_role:
                self.prompt_hints[role_name] = lst[-self.prompt_cfg.max_hints_per_role:]

    def remove_hint(self, role_name: str, hint_id: str) -> None:
        lst = self.prompt_hints.get(role_name, [])
        if hint_id in lst:
            lst.remove(hint_id)
            self.prompt_hints[role_name] = lst
