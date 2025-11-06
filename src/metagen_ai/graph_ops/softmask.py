# src/metagen_ai/graph_ops/softmask.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional
import math
import networkx as nx

_SIGMOID_CLIP = 10.0  # numerical stability for extreme alphas

def _sigmoid(x: float, tau: float = 1.0) -> float:
    # temperatured logistic
    x = max(-_SIGMOID_CLIP, min(_SIGMOID_CLIP, x))
    return 1.0 / (1.0 + math.exp(-x / max(1e-6, tau)))

@dataclass
class SoftMaskConfig:
    # Sigmoid temperature; lower → harder gate
    tau_node: float = 1.0
    tau_edge: float = 1.0
    # Threshold for hard decision during forward (discrete run)
    thresh_node: float = 0.5
    thresh_edge: float = 0.5
    # Learning rates for pseudo-gradient updates
    lr_node: float = 0.5
    lr_edge: float = 0.5
    # Optional L2 weight decay on alphas to avoid drift to infinity
    wd: float = 1e-3

@dataclass
class SoftMaskState:
    node_alpha: Dict[str, float] = field(default_factory=dict)
    edge_alpha: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # caches for the last forward pass
    node_prob: Dict[str, float] = field(default_factory=dict)
    edge_prob: Dict[Tuple[str, str], float] = field(default_factory=dict)

class SoftMaskManager:
    """
    Maintains continuous soft gates (alphas) for nodes and edges in a DAG, and
    provides:
      - forward(): compute probabilities and set active flags using STE-style binarization
      - step(): apply pseudo-gradients dL/dalpha from textual/meta signals
      - binarize(): commit current probs to hard structure (write active flags)
    """
    def __init__(self, program, cfg: Optional[SoftMaskConfig] = None):
        self.program = program
        self.cfg = cfg or SoftMaskConfig()
        self.state = SoftMaskState()

        # Initialize alphas from current active flags (active→positive, inactive→negative)
        for n, data in self.program.G.nodes(data=True):
            self.state.node_alpha[n] = 2.0 if data.get("active", True) else -2.0
        for u, v, data in self.program.G.edges(data=True):
            self.state.edge_alpha[(u, v)] = 2.0 if data.get("active", True) else -2.0

    # ---------------- Forward / Apply ----------------
    def forward(self, write_active: bool = True) -> None:
        """Compute probabilities and optionally write active flags (hard via threshold)."""
        # Node probs
        self.state.node_prob = {
            n: _sigmoid(a, tau=self.cfg.tau_node) for n, a in self.state.node_alpha.items()
        }
        # Edge probs
        self.state.edge_prob = {
            e: _sigmoid(a, tau=self.cfg.tau_edge) for e, a in self.state.edge_alpha.items()
        }
        if write_active:
            self._write_active_from_probs()

    def _write_active_from_probs(self) -> None:
        G = self.program.G
        # nodes
        for n, p in self.state.node_prob.items():
            G.nodes[n]["active"] = (p >= self.cfg.thresh_node)
        # edges
        for (u, v), p in self.state.edge_prob.items():
            G.edges[u, v]["active"] = (p >= self.cfg.thresh_edge) and G.nodes[u].get("active", True) and G.nodes[v].get("active", True)

    # ---------------- STE Update ----------------
    def step(self,
             node_grad: Optional[Dict[str, float]] = None,
             edge_grad: Optional[Dict[Tuple[str, str], float]] = None) -> None:
        """
        Apply pseudo-gradients (textual/meta signals) to alphas:
          alpha <- alpha - lr * (grad + wd * alpha)
        Convention: positive grad increases loss → decreases probability.
        If you want to reinforce an edge/node, pass a negative grad.
        """
        node_grad = node_grad or {}
        edge_grad = edge_grad or {}

        # weight decay
        for n, a in self.state.node_alpha.items():
            g = float(node_grad.get(n, 0.0)) + self.cfg.wd * a
            self.state.node_alpha[n] = a - self.cfg.lr_node * g

        for e, a in self.state.edge_alpha.items():
            g = float(edge_grad.get(e, 0.0)) + self.cfg.wd * a
            self.state.edge_alpha[e] = a - self.cfg.lr_edge * g

    # ---------------- Hard Commit ----------------
    def binarize(self) -> None:
        """Write current discrete structure into the graph (same as forward with writing)."""
        self.forward(write_active=True)

    # ---------------- Diagnostics ----------------
    def counts(self) -> Dict[str, int]:
        G = self.program.G
        nodes = sum(1 for n in G.nodes if G.nodes[n].get("active", True))
        edges = sum(1 for u, v in G.edges if G.edges[u, v].get("active", True))
        return {"active_nodes": nodes, "active_edges": edges}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "node_alpha": dict(self.state.node_alpha),
            "edge_alpha": {f"{u}->{v}": a for (u, v), a in self.state.edge_alpha.items()},
            "node_prob": dict(self.state.node_prob),
            "edge_prob": {f"{u}->{v}": p for (u, v), p in self.state.edge_prob.items()},
        }
