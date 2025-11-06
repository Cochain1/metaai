# src/metagen_ai/pruning/one_shot.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set
import networkx as nx
import math
import random

def _active_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        if d.get("active", True):
            H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if d.get("active", True) and H.has_node(u) and H.has_node(v):
            H.add_edge(u, v, **d)
    return H

def _find_exits(G: nx.DiGraph) -> List[str]:
    exits = [n for n, d in G.nodes(data=True) if d.get("is_exit", False) and d.get("active", True)]
    if exits:
        return exits
    # fallbacks
    for name in ("evaluator", "judge"):
        if name in G.nodes and G.nodes[name].get("active", True):
            return [name]
    # if no explicit exit, treat zero-outdegree active nodes as sinks
    return [n for n in G.nodes if G.out_degree(n) == 0 and G.nodes[n].get("active", True)]

def _find_sources(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes if G.in_degree(n) == 0 and G.nodes[n].get("active", True)]

def _shortest_keep_edges(G: nx.DiGraph, sources: List[str], exits: List[str]) -> Set[Tuple[str, str]]:
    """Keep at least one path from any source to any exit (shortest path heuristic)."""
    keep: Set[Tuple[str, str]] = set()
    for s in sources:
        for t in exits:
            if s not in G.nodes or t not in G.nodes:
                continue
            try:
                path = nx.shortest_path(G, s, t)
                for i in range(len(path) - 1):
                    keep.add((path[i], path[i+1]))
            except nx.NetworkXNoPath:
                continue
    return keep

def _edge_betweenness_scores(H: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Low betweenness ≈ more prune-able."""
    if H.number_of_edges() == 0:
        return {}
    U = H.to_undirected(as_view=False)
    try:
        eb = nx.edge_betweenness_centrality(U, normalized=True)
    except Exception:
        eb = {}
        for u, v in H.edges:
            eb[(u, v)] = (H.degree(u) * H.degree(v)) / max(1.0, (2.0 * H.number_of_edges()))
    out: Dict[Tuple[str, str], float] = {}
    for u, v in H.edges:
        key = (u, v) if (u, v) in eb else (v, u)
        out[(u, v)] = float(eb.get(key, 0.0))
    return out

def prune_once(program, task: Dict[str, Any], cfg: Dict[str, Any]):
    """
    One-shot pruning with 'keep-alive path' constraint:
      - Preserve at least one path from any active source to any active exit node (is_exit=True/evaluator/judge).
      - Also preserve the main reasoning chain if present: reasoner->equation_extractor->(evaluator/judge).
      - Rank candidate edges by (low) edge betweenness; prune the lowest-scored edges first.
    """
    pcfg = (cfg.get("pruning", {}) or {}) if isinstance(cfg, dict) else {}
    enable = bool(pcfg.get("enable", True))
    if not enable:
        return program

    strategy = str(pcfg.get("strategy", "similarity"))
    prune_ratio = float(pcfg.get("prune_ratio", 0.2))
    prune_ratio = max(0.0, min(1.0, prune_ratio))

    G = program.G
    H = _active_subgraph(G)

    if H.number_of_edges() == 0:
        G.graph["kept_edges"] = []
        return program

    # compute keep set (paths from sources to exits)
    sources = _find_sources(H)
    exits = _find_exits(H)
    keep: Set[Tuple[str, str]] = _shortest_keep_edges(H, sources, exits)

    # --- 保活主链（若存在） ---
    if "reasoner" in H.nodes and "equation_extractor" in H.nodes:
        if H.has_edge("reasoner", "equation_extractor"):
            keep.add(("reasoner", "equation_extractor"))
        sink = exits[0] if exits else ("evaluator" if "evaluator" in H.nodes else None)
        if sink and H.has_edge("equation_extractor", sink):
            keep.add(("equation_extractor", sink))

    # list candidate edges (active & not in keep)
    candidates: List[Tuple[str, str]] = [(u, v) for u, v in H.edges if (u, v) not in keep]

    if not candidates:
        G.graph["kept_edges"] = list(keep)
        return program

    # score candidates
    if strategy == "similarity":
        scores = _edge_betweenness_scores(H)
        ranked = sorted(candidates, key=lambda e: scores.get(e, 0.0))  # ascending
    elif strategy == "degree":
        ranked = sorted(candidates, key=lambda e: (H.degree(e[0]) + H.degree(e[1])))
    else:  # random
        ranked = list(candidates)
        random.shuffle(ranked)

    target = int(math.floor(len(H.edges) * prune_ratio))
    pruned = 0
    for (u, v) in ranked:
        if pruned >= target:
            break
        # temporarily remove and check connectivity
        H.remove_edge(u, v)
        still_ok = any(nx.has_path(H, s, t) for s in sources for t in exits if s in H and t in H)
        if still_ok:
            if G.has_edge(u, v):
                G.edges[u, v]["active"] = False
            pruned += 1
        else:
            # critical edge: restore and mark as kept
            H.add_edge(u, v)
            keep.add((u, v))

    G.graph["kept_edges"] = list(keep)
    return program
