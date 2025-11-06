# scripts/tri_self_evolve_demo.py
# Three-fold self-evolution demo:
# 1) Structure soft-gating with SoftMask + STE
# 2) Parametric roles (temperature + prompt hints) with textual gradients
# 3) One-shot pruning after several online rounds

from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.cost_aware import sample_architecture_costaware
from metagen_ai.graph_ops.softmask import SoftMaskManager, SoftMaskConfig
from metagen_ai.feedback.soft_textual import build_softmask_textual_hook
from metagen_ai.roles.parametric import ParamManager
from metagen_ai.feedback.param_textual import build_parametric_hooks
from metagen_ai.pruning.one_shot import prune_once

def toy_task():
    return {"question": "What is 19 + 24? Output only the number.", "answer": "43"}

class CombinedHooks:
    """
    A tiny adapter that merges 'before_node' and 'textual_gradient_hook'
    from multiple hook providers into one object compatible with GraphProgram.run.
    """
    def __init__(self, *hook_objs):
        self._before = []
        self._tgrad = []
        for h in hook_objs:
            if hasattr(h, "before_node") and callable(h.before_node):
                self._before.append(h.before_node)
            if hasattr(h, "textual_gradient_hook") and callable(h.textual_gradient_hook):
                self._tgrad.append(h.textual_gradient_hook)

    def before_node(self, node_id, ctx):
        for fn in self._before:
            fn(node_id, ctx)

    def textual_gradient_hook(self, payload):
        for fn in self._tgrad:
            fn(payload)

def main():
    cfg = bootstrap("configs/default.yaml")
    task = toy_task()

    # 1) Task-adaptive graph + cost-aware gating
    program = build_task_graph(task, cfg)
    program = sample_architecture_costaware(program, task, cfg)

    # 2) Initialize structure soft masks (continuous gates)
    smm = SoftMaskManager(
        program,
        cfg=SoftMaskConfig(
            tau_node=0.8, tau_edge=0.8,      # softer at start
            thresh_node=0.5, thresh_edge=0.5,
            lr_node=0.25, lr_edge=0.25, wd=1e-3
        )
    )
    smm.forward(write_active=True)
    softmask_hook = build_softmask_textual_hook(smm)

    # 3) Initialize parametric roles (temperature + prompt hints)
    pm = ParamManager()
    param_hooks = build_parametric_hooks(program, pm)

    # 4) Merge hooks so runner can call both seamlessly
    hooks = CombinedHooks(param_hooks, type("H", (), {"textual_gradient_hook": softmask_hook})())

    # Diagnostics before evolution
    init_nodes = sum(1 for n in program.G.nodes if program.G.nodes[n].get("active", True))
    init_edges = sum(1 for u, v in program.G.edges if program.G.edges[u, v].get("active", True))
    print(f"[Init] active: nodes={init_nodes} edges={init_edges}")

    # 5) Online self-evolution for multiple rounds (same task for clarity)
    ROUNDS = 6
    for r in range(ROUNDS):
        res = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
        counts = smm.counts()
        print(f"[Round {r}] Final={res['final']} | nodes={counts['active_nodes']} edges={counts['active_edges']}")
        # Show evolving temps and hints of key roles
        for role_name in ["math_simplifier", "calculator", "evaluator"]:
            if role_name in program.role_library:
                rp = program.role_library[role_name]
                print(f"   - {role_name}: T={rp.temperature} hints={pm.prompt_hints.get(role_name, [])}")

    # 6) One-shot pruning after soft evolution
    program = prune_once(program, task, cfg)
    kept = program.G.graph.get("kept_edges")
    kept = kept if kept is not None else []
    print("[After Pruning] kept edges:", kept)

    # 7) Final run post-pruning
    res_final = program.run(task=task, rounds=1, early_exit=True, hooks=hooks)
    counts = smm.counts()
    print(f"[Post-Prune] Final={res_final['final']} | nodes={counts['active_nodes']} edges={counts['active_edges']}")

if __name__ == "__main__":
    main()
