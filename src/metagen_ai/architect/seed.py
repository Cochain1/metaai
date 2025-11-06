from __future__ import annotations
import networkx as nx
from metagen_ai.graph_ops.runner import GraphProgram
from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.utils.llm import build_llm_from_cfg

def seed_graph(task, cfg):
    """
    Build a tiny DAG: solver_a -> solver_b -> judge
    Returns a GraphProgram ready to run.
    """
    G = nx.DiGraph()
    G.add_node("solver_a", role="math_simplifier", active=True)
    G.add_node("solver_b", role="calculator", active=True)
    G.add_node("judge", role="evaluator", active=True)
    G.add_edge("solver_a", "solver_b", kind="space", active=True)
    G.add_edge("solver_b", "judge", kind="space", active=True)

    # LLM client is optional because builtin roles have local handlers
    llm = None
    try:
        llm = build_llm_from_cfg(cfg)
    except Exception:
        # No API key or endpoint; will still work for builtin local roles
        llm = None

    return GraphProgram(G=G, role_library=BUILTIN_ROLES, llm=llm, default_temperature=cfg.get("llm", {}).get("temperature", 0.2))
