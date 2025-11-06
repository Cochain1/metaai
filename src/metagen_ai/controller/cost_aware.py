# src/metagen_ai/controller/cost_aware.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_MODEL = None
except Exception:
    SentenceTransformer = None
    _SBERT_MODEL = None

from metagen_ai.graph_ops.runner import GraphProgram
from metagen_ai.roles.schema import RoleProfile

# ---------------- Embedding (with offline fallback) ----------------
def _lazy_embedder():
    global _SBERT_MODEL
    if SentenceTransformer is not None and _SBERT_MODEL is None:
        try:
            _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT_MODEL = None
    if _SBERT_MODEL is not None:
        def _emb(text: str):
            v = _SBERT_MODEL.encode([text], normalize_embeddings=True)[0]
            return v.tolist()
        return _emb

    import hashlib
    def _emb(text: str, dim: int = 256):
        vec = [0.0] * dim
        for tok in str(text).lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
    return _emb

def _cos(a, b) -> float:
    return float(sum(x*y for x, y in zip(a, b)))

def _role_repr(role: RoleProfile) -> str:
    return f"{role.name}. {role.description}"

# ---------------- Cost model ----------------
DEFAULT_ROLE_COSTS = {
    # Rough relative costs; local handlers ~0, LLM-based ~1 by default
    "task_hub": 0.05,
    "math_simplifier": 0.0,   # local
    "calculator": 0.0,        # local
    "evaluator": 0.0,         # local (can be LLM if you remove local_handler)
    # generated roles default to 1.0 unless overridden
}

def _estimate_role_cost(role: RoleProfile) -> float:
    # Local handlers approximate zero cost; LLM-backed roles cost 1.0 unit by default
    if role.local_handler is not None:
        return 0.0
    return 1.0

def _node_cost(role_name: str, role_lib: Dict[str, RoleProfile]) -> float:
    if role_name in DEFAULT_ROLE_COSTS:
        return float(DEFAULT_ROLE_COSTS[role_name])
    role = role_lib.get(role_name)
    if role is None:
        return 1.0
    return _estimate_role_cost(role)

# ---------------- Scoring ----------------
def _relevance_scores(G: nx.DiGraph, role_lib: Dict[str, RoleProfile], task: Dict[str, Any]) -> Dict[str, float]:
    emb = _lazy_embedder()
    q = str(task.get("question") or task.get("query") or task)
    qv = emb(q)

    scores: Dict[str, float] = {}
    for n, data in G.nodes(data=True):
        rname = data.get("role", "")
        role = role_lib.get(rname)
        if role is None:
            scores[n] = 0.0
            continue
        rv = emb(_role_repr(role))
        scores[n] = max(0.0, _cos(qv, rv))
    return scores

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return d
    mx = max(d.values()) or 1e-8
    mn = min(d.values())
    if mx == mn:
        return {k: 0.0 for k in d}  # all equal
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def _combined_score(relevance_n: Dict[str, float],
                    cost_n: Dict[str, float],
                    lambda_cost: float,
                    alpha: float = 1.0) -> Dict[str, float]:
    # Higher is better: alpha * rel - lambda * cost
    comb = {}
    for n in relevance_n:
        r = relevance_n.get(n, 0.0)
        c = cost_n.get(n, 0.0)
        comb[n] = alpha * r - lambda_cost * c
    return comb

def _select_by_threshold(scores: Dict[str, float], threshold: float, must_include: List[str]) -> List[str]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(max(0.0, v) for _, v in items) or 1e-8
    target = max(0.0, min(1.0, threshold)) * total

    selected: List[str] = []
    csum = 0.0
    for n, s in items:
        if csum >= target and selected:
            break
        selected.append(n)
        csum += max(0.0, s)

    for n in must_include:
        if n in scores and n not in selected:
            selected.append(n)
    return selected

def _activate(program: GraphProgram, kept_nodes: List[str]) -> None:
    kept = set(kept_nodes)
    for n in program.G.nodes:
        program.G.nodes[n]["active"] = (n in kept)
    for u, v in program.G.edges:
        program.G.edges[u, v]["active"] = (u in kept and v in kept)

def _mark_exit(program: GraphProgram, exit_candidates: List[str]) -> None:
    for n in program.G.nodes:
        program.G.nodes[n]["is_exit"] = False
    for cand in exit_candidates:
        if cand in program.G.nodes:
            program.G.nodes[cand]["is_exit"] = True
            return
    # fallback: first sink
    sinks = [n for n in program.G.nodes if program.G.out_degree(n) == 0]
    if sinks:
        program.G.nodes[sinks[0]]["is_exit"] = True

# ---------------- Public API ----------------
def sample_architecture_costaware(program: GraphProgram, task: Dict[str, Any], cfg: Dict[str, Any]) -> GraphProgram:
    """
    Cost-aware MoE gating:
      score(n) = alpha * relevance(n) - lambda_cost * normalized_cost(n)
    Then select nodes by cumulative-threshold and activate subgraph; mark exit.
    """
    ctrl = cfg.get("controller", {}) if isinstance(cfg, dict) else {}
    threshold = float(ctrl.get("threshold", 0.7))
    lambda_cost = float(ctrl.get("cost_weight", 0.3))  # stronger → more frugal
    alpha = 1.0

    rel = _relevance_scores(program.G, program.role_library, task)
    # Cost dictionary (pre-normalized to [0,1])
    raw_cost = {n: _node_cost(program.G.nodes[n].get("role", n), program.role_library) for n in program.G.nodes}
    rel_n = _normalize(rel)
    cost_n = _normalize(raw_cost)
    comb = _combined_score(rel_n, cost_n, lambda_cost=lambda_cost, alpha=alpha)

    selected = _select_by_threshold(comb, threshold=threshold, must_include=["evaluator", "judge"] if "judge" in program.G.nodes else ["evaluator"])
    _activate(program, selected)
    _mark_exit(program, ["evaluator", "judge"])

    program.G.graph["gating_relevance"] = rel_n
    program.G.graph["gating_cost"] = cost_n
    program.G.graph["gating_combined"] = comb
    program.G.graph["selected_nodes"] = selected
    program.G.graph["lambda_cost"] = lambda_cost
    return program
