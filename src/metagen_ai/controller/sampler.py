# src/metagen_ai/controller/sampler.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math
import networkx as nx

try:
    # Optional, improves gating quality. Falls back gracefully if unavailable.
    from sentence_transformers import SentenceTransformer
    _SBERT_MODEL = None  # lazy init
except Exception:
    SentenceTransformer = None
    _SBERT_MODEL = None

from metagen_ai.graph_ops.runner import GraphProgram
from metagen_ai.roles.schema import RoleProfile

def _lazy_embedder():
    """
    Returns an embed(text)->List[float] function.
    Prefers SentenceTransformer('all-MiniLM-L6-v2'), falls back to bag-of-words hashing.
    """
    global _SBERT_MODEL
    if SentenceTransformer is not None:
        if _SBERT_MODEL is None:
            try:
                _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                _SBERT_MODEL = None

    if _SBERT_MODEL is not None:
        def _emb(text: str):
            v = _SBERT_MODEL.encode([text], normalize_embeddings=True)[0]
            return v.tolist()
        return _emb

    # Fallback: deterministic bag-of-words hashing to a fixed-size vector
    import hashlib
    def _emb(text: str, dim: int = 256):
        vec = [0.0] * dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            i = h % dim
            vec[i] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
    return _emb

def _cosine(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(min(len(a), len(b))):
        s += a[i] * b[i]
    # vectors are already normalized in sbert path; fallback path normalized above
    return float(s)

def _role_repr(role: RoleProfile) -> str:
    return f"{role.name}. {role.description}"

def _score_nodes(G: nx.DiGraph, role_lib: Dict[str, RoleProfile], task: Dict[str, Any]) -> Dict[str, float]:
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
        scores[n] = max(0.0, _cosine(qv, rv))  # cosine in [0, 1] when normalized
    return scores

def _select_nodes_by_threshold(scores: Dict[str, float], threshold: float, must_include: List[str]) -> List[str]:
    """
    Select nodes by descending score until the cumulative sum reaches `threshold * total_sum`,
    always including `must_include` if present.
    """
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in items) or 1e-8
    target = max(0.0, min(1.0, threshold)) * total

    selected: List[str] = []
    csum = 0.0
    for n, s in items:
        if csum >= target and len(selected) > 0:
            break
        selected.append(n)
        csum += s

    # Ensure required nodes are included
    for n in must_include:
        if n in scores and n not in selected:
            selected.append(n)

    return selected

def _activate_subgraph(G: nx.DiGraph, kept_nodes: List[str]) -> None:
    kept = set(kept_nodes)
    for n in G.nodes:
        G.nodes[n]["active"] = (n in kept)

    for u, v in G.edges:
        G.edges[u, v]["active"] = (u in kept and v in kept)

def _mark_exit(G: nx.DiGraph, exit_candidates: List[str]) -> Optional[str]:
    """
    Mark a single exit node with 'is_exit'=True.
    Preference order: first existing candidate present & active; else try a sink.
    """
    for cand in exit_candidates:
        if cand in G.nodes:
            G.nodes[cand]["is_exit"] = True
            return cand

    # Fallback: any sink
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if sinks:
        G.nodes[sinks[0]]["is_exit"] = True
        return sinks[0]
    return None

def sample_architecture(program: GraphProgram, task: Dict[str, Any], cfg: Dict[str, Any]) -> GraphProgram:
    """
    In-place gating over program.G using task-aware scores.
    Returns the same program object for chaining.
    """
    ctrl = cfg.get("controller", {}) if isinstance(cfg, dict) else {}
    threshold = float(ctrl.get("threshold", 0.7))
    # Nodes that we strongly prefer to keep (e.g., final judge)
    must_include = ["judge"]  # can be extended in future
    scores = _score_nodes(program.G, program.role_library, task)
    selected = _select_nodes_by_threshold(scores, threshold=threshold, must_include=must_include)

    _activate_subgraph(program.G, selected)
    _mark_exit(program.G, exit_candidates=["judge"])

    # (Optional) Log for inspection
    program.G.graph["gating_scores"] = scores
    program.G.graph["selected_nodes"] = selected
    return program
