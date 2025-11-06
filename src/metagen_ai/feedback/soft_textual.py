# src/metagen_ai/feedback/soft_textual.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import re

from metagen_ai.graph_ops.softmask import SoftMaskManager

def _success(summary: str) -> bool:
    s = (summary or "").lower()
    return ("verified" in s) or (re.search(r"\bfinal answer:\b", s) is not None)

def _pick_weaken_target(program) -> Optional[Tuple[str, str]]:
    """Pick one incoming active edge to the exit node (judge/evaluator) to weaken if many exist."""
    G = program.G
    exit_node = None
    for n, d in G.nodes(data=True):
        if d.get("is_exit", False):
            exit_node = n
            break
    if exit_node is None:
        exit_node = "evaluator" if "evaluator" in G.nodes else None
    if exit_node is None:
        return None

    incoming = [(u, exit_node) for u in G.predecessors(exit_node) if G.edges[u, exit_node].get("active", True)]
    if len(incoming) <= 1:
        return None
    # Heuristic: weaken the earliest predecessor (arbitrary but deterministic)
    incoming.sort(key=lambda e: e[0])
    return incoming[0]

def build_softmask_textual_hook(smm: SoftMaskManager):
    """
    Returns a hook usable as GraphProgram.run(..., hooks=...).
    Behavior:
      - If success: reinforce edges that fed the exit node in this round (negative gradients).
      - If failure: weaken one incoming edge to exit node (positive gradient).
      - Nodes are reinforced/penalized in sync with their incident edges.
    """
    def _hook(payload: Dict[str, Any]):
        program = smm.program
        node_grad: Dict[str, float] = {}
        edge_grad: Dict[Tuple[str, str], float] = {}

        summary = payload.get("summary", "")
        ok = _success(summary)

        G = program.G
        # Find exit node
        exit_node = None
        for n, d in G.nodes(data=True):
            if d.get("is_exit", False):
                exit_node = n
                break
        if exit_node is None and "evaluator" in G.nodes:
            exit_node = "evaluator"

        if exit_node:
            active_in = [(u, exit_node) for u in G.predecessors(exit_node) if G.edges[u, exit_node].get("active", True)]

            if ok:
                # Reinforce: negative grads on used incoming edges and their sources
                for (u, v) in active_in:
                    edge_grad[(u, v)] = edge_grad.get((u, v), 0.0) - 1.0
                    node_grad[u] = node_grad.get(u, 0.0) - 0.5
                # Make exit more deterministic
                node_grad[exit_node] = node_grad.get(exit_node, 0.0) - 0.25
            else:
                # Weaken: positive grad on one incoming edge (if multiple)
                target = _pick_weaken_target(program)
                if target is not None:
                    edge_grad[target] = edge_grad.get(target, 0.0) + 1.0
                    node_grad[target[0]] = node_grad.get(target[0], 0.0) + 0.5
                # Encourage exploration on upstream
                for (u, v) in active_in:
                    node_grad[u] = node_grad.get(u, 0.0) - 0.1  # mild exploration

        # Apply pseudo-gradient update to alphas and rewrite active flags for next round
        smm.step(node_grad=node_grad, edge_grad=edge_grad)
        smm.forward(write_active=True)

    return _hook
