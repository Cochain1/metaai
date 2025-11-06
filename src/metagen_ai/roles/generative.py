# src/metagen_ai/roles/generative.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import math, re, random, uuid

from metagen_ai.roles.schema import RoleProfile
from metagen_ai.utils.llm import LLMClient, build_llm_from_cfg

# ---------- Embedding with offline fallback ----------
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = None
except Exception:
    SentenceTransformer = None
    _SBERT = None

def _lazy_embedder():
    """
    Returns an embedding function with normalized outputs.
    Prefers sentence-transformers; falls back to hashed BoW to work fully offline.
    """
    global _SBERT
    if SentenceTransformer is not None and _SBERT is None:
        try:
            _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _SBERT = None

    if _SBERT is not None:
        def enc(text: str):
            v = _SBERT.encode([text], normalize_embeddings=True)[0]
            return v.tolist()
        return enc

    import hashlib
    def enc(text: str, dim: int = 256):
        vec = [0.0] * dim
        for tok in str(text).lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        # L2 normalize
        s = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / s for x in vec]
    return enc

def _cos(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x, y in zip(a, b)))

def _dist(a: List[float], b: List[float]) -> float:
    # Diversity distance in [0, 2] for normalized vectors → map to [0,1]
    # Using 1 - cosine as a simple dissimilarity
    return max(0.0, 1.0 - _cos(a, b))


# ---------- Generative role proposal ----------
_SYSTEM_PROMPT = (
    "You are an architect LLM that invents new agent roles for a multi-agent system.\n"
    "Each role MUST include: name, 1-line description, a concise system prompt, a concise user prompt template,\n"
    "and a temperature between 0.0 and 1.0. Roles should be functional and non-overlapping.\n"
    "Return a JSON list with keys: name, description, system_template, user_template, temperature.\n"
)

_USER_PROMPT_TEMPLATE = (
    "Task description:\n{task_text}\n\n"
    "Existing roles (names only): {existing}\n"
    "Invent {k} NEW roles that help with the task by adding distinct capabilities.\n"
    "Avoid duplicates and do NOT repeat existing roles. Keep templates short."
)

@dataclass
class GeneratedRole:
    profile: RoleProfile
    raw_name: str
    diversity_score: float = 0.0
    utility_score: float = 0.0

# Fallback local generator to run fully offline
_FALLBACK_POOL = [
    {
        "name": "retrieval_specialist",
        "description": "Find and quote relevant snippets from context or memory.",
        "system_template": "You retrieve concise, relevant snippets; do not speculate.",
        "user_template": "Task: {task[question]}\nUpstream: {inputs}\nReturn 3 short relevant facts if available.",
        "temperature": 0.0,
    },
    {
        "name": "planner",
        "description": "Draft a 2-4 step plan before solving.",
        "system_template": "You create minimal, effective plans.",
        "user_template": "Produce a numbered mini-plan for: {task[question]}. Keep it under 4 steps.",
        "temperature": 0.2,
    },
    {
        "name": "consistency_checker",
        "description": "Check if candidate answers agree and flag conflicts.",
        "system_template": "You detect contradictions crisply.",
        "user_template": "Candidates: {inputs}\nList any conflicts and pick a consistent candidate if possible.",
        "temperature": 0.0,
    },
    {
        "name": "tool_router",
        "description": "Decide whether to call a tool or forward to a solver.",
        "system_template": "You are a cautious tool router.",
        "user_template": "Given: {task[question]}\nSuggest 'use_tool:X' or 'pass_to:solver' with one sentence rationale.",
        "temperature": 0.2,
    },
    {
        "name": "self_reflection",
        "description": "Reflect on mistakes and propose one correction.",
        "system_template": "You reflect briefly and concretely.",
        "user_template": "Given upstream reasoning {inputs}, propose one likely mistake and fix.",
        "temperature": 0.3,
    },
]

def _unique_name(base: str, existing: Dict[str, RoleProfile]) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_")
    if not name: 
        name = "role"
    candidate = name
    i = 2
    while candidate in existing:
        candidate = f"{name}_{i}"
        i += 1
    return candidate

def _coerce_profile(item: Dict[str, Any], existing: Dict[str, RoleProfile]) -> RoleProfile:
    name = _unique_name(item.get("name", f"role_{uuid.uuid4().hex[:6]}"), existing)
    desc = str(item.get("description", "A helpful role."))
    sys_t = str(item.get("system_template", "You are a helpful role."))
    usr_t = str(item.get("user_template", "Task: {task[question]}\nUpstream: {inputs}"))
    temp = float(item.get("temperature", 0.2))
    return RoleProfile(
        name=name,
        description=desc,
        system_template=sys_t,
        user_template=usr_t,
        temperature=max(0.0, min(1.0, temp)),
        local_handler=None,   # generated roles default to LLM; can be swapped later
    )

def propose_roles(task: Dict[str, Any], cfg: Dict[str, Any], 
                  role_library: Dict[str, RoleProfile],
                  k: int) -> List[RoleProfile]:
    """
    Propose k new RoleProfile objects using LLM if available, else fallback to offline pool.
    """
    # Try to build an LLM client
    llm: Optional[LLMClient] = None
    try:
        llm = build_llm_from_cfg(cfg)
    except Exception:
        llm = None

    existing_names = list(role_library.keys())
    task_text = str(task.get("question") or task.get("query") or task)

    if llm is not None:
        msg = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(task_text=task_text, existing=existing_names, k=k)}
        ]
        try:
            out = llm.chat(msg, temperature=0.4, max_tokens=800)
            text = out["text"]
            # Try to extract JSON array
            import json
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                arr = json.loads(text[start:end+1])
                cands = []
                for item in arr:
                    try:
                        rp = _coerce_profile(item, role_library)
                        cands.append(rp)
                    except Exception:
                        continue
                if cands:
                    return cands[:k]
        except Exception:
            pass

    # Fallback path: sample from fixed pool and lightly adapt to task
    rnd = random.Random((cfg.get("runtime", {}) or {}).get("seed", 42))
    picked = rnd.sample(_FALLBACK_POOL, k=min(k, len(_FALLBACK_POOL)))
    adjusted = []
    for item in picked:
        rp = _coerce_profile(item, role_library)
        # Optionally, inject task keyword into templates
        rp.user_template = rp.user_template.replace("{task[question]}", str(task.get("question", "")))
        adjusted.append(rp)
    return adjusted

# ---------- Diversity selection ----------
def select_diverse_roles(candidates: List[RoleProfile],
                         role_library: Dict[str, RoleProfile],
                         max_new_roles: int,
                         min_semantic_distance: float) -> List[GeneratedRole]:
    """
    Greedy selection maximizing diversity distance to both existing roles and already-selected ones.
    """
    emb = _lazy_embedder()

    # Precompute embeddings
    def rp_vec(rp: RoleProfile) -> List[float]:
        text = f"{rp.name}. {rp.description}"
        return emb(text)

    existing_vecs = [(name, rp_vec(rp)) for name, rp in role_library.items()]
    cand_vecs = [rp_vec(rp) for rp in candidates]

    selected: List[GeneratedRole] = []
    used = [False] * len(candidates)

    for _ in range(max_new_roles):
        best_idx, best_score = -1, -1.0
        for i, rp in enumerate(candidates):
            if used[i]:
                continue
            v = cand_vecs[i]
            # Distance to existing roles
            dist_exist = min((_dist(v, ev) for _, ev in existing_vecs), default=1.0)
            # Distance to already selected
            dist_sel = min((_dist(v, rp_vec(s.profile)) for s in selected), default=1.0)
            score = 0.5 * dist_exist + 0.5 * dist_sel  # simple aggregation
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        if best_score < min_semantic_distance:
            break
        used[best_idx] = True
        selected.append(GeneratedRole(profile=candidates[best_idx],
                                      raw_name=candidates[best_idx].name,
                                      diversity_score=float(best_score),
                                      utility_score=0.0))
    return selected

# ---------- Adapter stubs (no heavy deps) ----------
@dataclass
class AdapterSpec:
    method: str = "lora"    # "lora" | "prefix" | etc.
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05

def fit_adapter_stub(role: RoleProfile, spec: AdapterSpec, task_examples: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    Placeholder for quick role instantiation on a small model.
    Currently a no-op to keep the stack dependency-light.
    """
    # You can later replace this with real PEFT/LoRA finetuning if needed.
    return

# ---------- Public API ----------
def generate_and_register_roles(task: Dict[str, Any], cfg: Dict[str, Any],
                                role_library: Dict[str, RoleProfile]) -> Tuple[Dict[str, RoleProfile], List[GeneratedRole]]:
    """
    Main entrypoint:
      1) Propose K candidate roles.
      2) Select up to max_new_roles with diversity constraints.
      3) Optionally fit adapters (stubbed).
      4) Register them into role_library with unique names.
    """
    rcfg = cfg.get("role_space", {}) if isinstance(cfg, dict) else {}
    k = int(rcfg.get("max_new_roles", 3))
    min_dist = float(rcfg.get("min_semantic_distance", 0.25))
    adapter_method = str(rcfg.get("adapter", "lora"))

    # 1) Propose
    cand = propose_roles(task, cfg, role_library, k=k)
    if not cand:
        return role_library, []

    # 2) Select diverse
    selected = select_diverse_roles(cand, role_library, max_new_roles=k, min_semantic_distance=min_dist)
    if not selected:
        return role_library, []

    # 3) Adapter fitting (stub) and 4) Register
    updated = dict(role_library)
    for item in selected:
        spec = AdapterSpec(method=adapter_method)
        fit_adapter_stub(item.profile, spec, task_examples=None)
        # Ensure unique name in library
        unique = item.profile.name
        idx = 2
        while unique in updated:
            unique = f"{item.profile.name}_{idx}"
            idx += 1
        item.profile.name = unique
        updated[unique] = item.profile

    return updated, selected
