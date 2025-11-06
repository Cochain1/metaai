# src/metagen_ai/feedback/textual_grad.py
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Tuple, Optional
import re

@dataclass
class TGConfig:
    # Whether to enable textual-gradient logic (kept for config compatibility)
    textual_gradient: bool = True

    # Which channels to touch: "prompt" | "temperature" | "topology"
    channels: List[str] = field(default_factory=lambda: ["prompt", "temperature", "topology"])

    # Temperature channel knobs
    temp_step: float = 0.1
    temp_min: float = 0.0
    temp_max: float = 1.0

    # Topology channel knobs
    reinforce_edge_on_success: bool = True
    weaken_edge_on_failure: bool = True

    # --- helper ---
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "TGConfig":
        """Create TGConfig from a dict, ignoring unknown keys."""
        if cfg is None:
            return cls()
        known = {f.name for f in fields(cls)}
        safe_kwargs = {k: v for k, v in dict(cfg).items() if k in known}
        return cls(**safe_kwargs)

# --------- tiny helpers ---------
_SUCCESS_RE = re.compile(r"\b(final answer:|correct|verified)\b", re.I)

def _is_success(summary: str, final: str) -> bool:
    s = (summary or "") + " " + (final or "")
    return _SUCCESS_RE.search(s) is not None

def _find_exit_node(program) -> Optional[str]:
    for n, d in program.G.nodes(data=True):
        if d.get("is_exit", False):
            return n
    # fallbacks
    if "evaluator" in program.G.nodes:
        return "evaluator"
    if "judge" in program.G.nodes:
        return "judge"
    return None

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# --------- public API used by demos ---------
def apply_textual_update(program, result: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    Rule-based textual gradient that can touch:
      - topology: edge active flags (light reinforcement/weakening)
      - temperature: nudge role temperatures a bit
    `result` is expected to contain keys: "final", "summary", "node_outputs"
    """
    tg = TGConfig.from_cfg(cfg.get("feedback", {}) if isinstance(cfg, dict) else {})
    if not tg.textual_gradient:
        return

    summary = result.get("summary", "")
    final = result.get("final", "")
    ok = _is_success(summary, final)

    exit_node = _find_exit_node(program)
    if exit_node is None:
        # nothing to update structurally; can still tweak temps
        pass

    # ----- Topology channel -----
    if "topology" in tg.channels and exit_node is not None:
        # collect active incoming edges
        incoming: List[Tuple[str, str]] = []
        for u in program.G.predecessors(exit_node):
            if not program.G.nodes[u].get("active", True):
                continue
            eattr = program.G.get_edge_data(u, exit_node) or {}
            if eattr.get("active", True):
                incoming.append((u, exit_node))

        if ok and tg.reinforce_edge_on_success:
            # reinforce: keep all active incoming edges as active (idempotent),
            # optional: could store a confidence counter if needed.
            for (u, v) in incoming:
                program.G.edges[u, v]["active"] = True
        elif (not ok) and tg.weaken_edge_on_failure and len(incoming) > 1:
            # weaken one edge heuristically（最早的上游）
            incoming.sort(key=lambda e: e[0])
            (u, v) = incoming[0]
            program.G.edges[u, v]["active"] = False

    # ----- Temperature channel -----
    if "temperature" in tg.channels:
        # Make evaluator colder on success, warmer on failure; solvers反向
        roles = set(d.get("role", "") for _, d in program.G.nodes(data=True))
        judge = "evaluator" if "evaluator" in roles else ("judge" if "judge" in roles else None)
        solvers = [r for r in ["math_simplifier", "calculator"] if r in roles]

        def _nudge_role(role_name: str, delta: float):
            rp = program.role_library.get(role_name)
            if rp is None:
                return
            base = rp.temperature if rp.temperature is not None else program.default_temperature
            rp.temperature = _clip(base + delta, tg.temp_min, tg.temp_max)

        if ok:
            if judge: _nudge_role(judge, -tg.temp_step)
            for s in solvers: _nudge_role(s, -tg.temp_step * 0.5)
        else:
            if judge: _nudge_role(judge, +tg.temp_step * 0.5)
            for s in solvers: _nudge_role(s, +tg.temp_step)

def build_textual_gradient_hook(program, cfg: Dict[str, Any]):
    """
    Returns a simple hook(payload) that applies textual updates using the program's
    current graph and role library. Compatible with runner.run(..., hooks=...).
    """
    def _hook(payload: Dict[str, Any]):
        # payload contains: task, round, node_outputs, summary
        fake_result = {
            "final": payload.get("summary", ""),
            "summary": payload.get("summary", ""),
            "node_outputs": payload.get("node_outputs", {}),
            "usage": {},
        }
        apply_textual_update(program, fake_result, cfg)
    return _hook
