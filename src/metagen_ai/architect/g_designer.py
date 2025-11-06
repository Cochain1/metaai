# src/metagen_ai/architect/g_designer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import random, os
import networkx as nx

from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.roles.schema import RoleProfile
from metagen_ai.graph_ops.runner import GraphProgram
from metagen_ai.utils.llm import build_llm_from_cfg

# ----- Optional torch-geometric VGAE backend (kept but unused by default) -----
_TG_OK = False
try:
    import torch
    from torch import nn
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
    _TG_OK = True
except Exception:
    _TG_OK = False

# ----- Optional semantic embedder with offline fallback -----
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = None
except Exception:
    SentenceTransformer = None
    _SBERT = None


def _lazy_embedder(cfg: Optional[Dict[str, Any]] = None):
    """
    enc(text) -> List[float]; supports fully-offline hashed-BoW fallback.
    """
    global _SBERT
    sbert_path = None
    local_only = False
    if isinstance(cfg, dict):
        emb = (cfg.get("embeddings") or {})
        sbert_path = emb.get("sbert_path")
        local_only = bool(emb.get("local_only", False))
    sbert_path = os.environ.get("SBERT_LOCAL_PATH", sbert_path)

    def _offline_fallback():
        import hashlib, math as _math
        def enc(text: str, dim: int = 256):
            vec = [0.0] * dim
            for tok in str(text).lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            s = _math.sqrt(sum(x * x for x in vec)) or 1.0
            return [x / s for x in vec]
        return enc

    if local_only:
        if sbert_path and os.path.isdir(sbert_path):
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                if SentenceTransformer is not None and _SBERT is None:
                    _SBERT = SentenceTransformer(sbert_path)
                if _SBERT is not None:
                    def enc(text: str):
                        v = _SBERT.encode([text], normalize_embeddings=True)[0]
                        return v.tolist()
                    return enc
            except Exception:
                pass
        return _offline_fallback()

    if SentenceTransformer is not None and _SBERT is None:
        try:
            if sbert_path and os.path.isdir(sbert_path):
                _SBERT = SentenceTransformer(sbert_path)
            else:
                _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT = None

    if _SBERT is not None:
        def enc(text: str):
            v = _SBERT.encode([text], normalize_embeddings=True)[0]
            return v.tolist()
        return enc

    return _offline_fallback()


def _cos(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


@dataclass
class GDesignerConfig:
    max_nodes: int = 6
    k_top: int = 3
    sparsity: float = 0.5
    use_vgae: bool = False
    seed: int = 42
    vgae_hidden: int = 16
    vgae_epochs: int = 80
    vgae_lr: float = 1e-2


# -------------------- VGAE (optional, not used in fast paths) --------------------
if _TG_OK:
    class VGAE(nn.Module):
        def __init__(self, in_dim: int, hid: int):
            super().__init__()
            self.enc1 = GCNConv(in_dim, hid)
            self.mu = GCNConv(hid, hid)
            self.logvar = GCNConv(hid, hid)

        def forward(self, x, edge_index):
            h = torch.relu(self.enc1(x, edge_index))
            mu = self.mu(h, edge_index)
            logvar = self.logvar(h, edge_index)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            adj = torch.sigmoid(torch.matmul(z, z.t()))
            return adj, mu, logvar

    def _vgae_refine(adj_init, x_init, hid, epochs, lr) -> List[List[float]]:
        device = torch.device("cpu")
        x = torch.tensor(x_init, dtype=torch.float32, device=device)
        adj = torch.tensor(adj_init, dtype=torch.float32, device=device)
        edge_index = dense_to_sparse((adj > 0).to(torch.float32))[0]

        model = VGAE(in_dim=x.shape[1], hid=hid).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(max(1, epochs)):
            opt.zero_grad()
            adj_hat, mu, logvar = model(x, edge_index)
            recon = torch.nn.functional.binary_cross_entropy(adj_hat, adj)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 1e-3 * kl
            loss.backward()
            opt.step()

        return adj_hat.detach().cpu().tolist()
else:
    def _vgae_refine(*args, **kwargs):
        raise RuntimeError("torch-geometric not available")


# -------------------- Public API --------------------
def build_task_graph(task: Dict[str, Any], cfg: Dict[str, Any],
                     role_library: Optional[Dict[str, RoleProfile]] = None) -> GraphProgram:
    """
    HumanEval：固定最小稳链
        task_hub → programmer → code_evaluator  (code_evaluator 为出口)
    其它（如 GSM8K）：保守主链
        task_hub → reasoner → equation_extractor → evaluator  (evaluator 为出口)
    """
    rcfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    seed = int(rcfg.get("seed", 42))
    random.seed(seed)

    try:
        llm = build_llm_from_cfg(cfg)
    except Exception:
        llm = None

    role_lib = dict(role_library or BUILTIN_ROLES)

    # 提供 task_hub 的虚拟角色（仅路由，不求解）
    task_hub_profile = RoleProfile(
        name="task_hub",
        description="A virtual node that distributes the task context.",
        system_template="You are a router; do not solve the task.",
        user_template="Task: {task}\nPrev: {prev_summary}\nUpstream: {inputs}\nRewrite the task briefly for downstream agents.",
        local_handler=lambda ctx: f"Context: {ctx['task']}",
        capabilities=[]
    )
    role_lib_aug = dict(role_lib)
    role_lib_aug["task_hub"] = task_hub_profile

    # ---- Detect HumanEval-format task ----
    is_humaneval = isinstance(task, dict) and (("entry_point" in task) or ("tests" in task))

    if is_humaneval:
        # 确保 programmer / code_evaluator 存在（由 builtin.py 提供）
        if "programmer" not in role_lib_aug:
            role_lib_aug["programmer"] = BUILTIN_ROLES["programmer"]
        if "code_evaluator" not in role_lib_aug:
            role_lib_aug["code_evaluator"] = BUILTIN_ROLES["code_evaluator"]

        # 最小稳链：只连必要边，保证输出唯一代码块
        G = nx.DiGraph()
        for n in ["task_hub", "programmer", "code_evaluator"]:
            G.add_node(n, role=n, active=True)
        G.add_edge("task_hub", "programmer", kind="space", active=True)
        G.add_edge("programmer", "code_evaluator", kind="space", active=True)
        G.nodes["code_evaluator"]["is_exit"] = True

        return GraphProgram(
            G=nx.DiGraph(G),
            role_library=role_lib_aug,
            llm=llm,
            default_temperature=cfg.get("llm", {}).get("temperature", 0.2) if isinstance(cfg, dict) else 0.2
        )

    # ---------------- Non-HumanEval: 保守 GSM8K 主链 ----------------
    G = nx.DiGraph()
    for n in ["task_hub", "reasoner", "equation_extractor", "evaluator"]:
        if n not in role_lib_aug:
            role_lib_aug[n] = BUILTIN_ROLES[n]
        G.add_node(n, role=n, active=True)

    G.add_edge("task_hub", "reasoner", kind="space", active=True)
    G.add_edge("reasoner", "equation_extractor", kind="space", active=True)
    G.add_edge("equation_extractor", "evaluator", kind="space", active=True)
    G.nodes["evaluator"]["is_exit"] = True

    return GraphProgram(
        G=nx.DiGraph(G),
        role_library=role_lib_aug,
        llm=llm,
        default_temperature=cfg.get("llm", {}).get("temperature", 0.2) if isinstance(cfg, dict) else 0.2
    )
