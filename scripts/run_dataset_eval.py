# scripts/run_dataset_eval.py
from __future__ import annotations

# —— 提前关闭 tokenizers 并行告警（必须放在一切 HF/transformers 相关导入之前） ——
import os as _os
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, csv, os, time, random, json, re
from typing import Dict, Any, List, Optional, Tuple, Iterable

from tqdm import tqdm

from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.architect.g_designer import build_task_graph
from metagen_ai.controller.sampler import sample_architecture
from metagen_ai.pruning.one_shot import prune_once
from metagen_ai.feedback.textual_grad import build_textual_gradient_hook
from metagen_ai.feedback.evolve import build_meta_evolver_hook   # ← 回合内自进化
from metagen_ai.utils.llm import build_llm_from_cfg

# 数据集与评测
from metagen_ai.datasets.loader import load_dataset       # -> Iterable[task dict]
from metagen_ai.eval.metrics import judge_correct         # -> bool

# 角色对象与内置库（用于识别哪些是“生成角色”）
from metagen_ai.roles.schema import RoleProfile
from metagen_ai.roles.builtin import BUILTIN_ROLES


# ---------------------------
# 角色缓存（题间进化，固化到磁盘）
# ---------------------------

class RoleCache:
    """
    JSONL 缓存格式（每行）:
      {"name": "...", "description": "...", "system_template": "...", "user_template": "..."}
    - 去重按 name
    - 只持久化必要字段（不会保存 local_handler/温度等）
    """
    def __init__(self, path: str):
        self.path = path
        self._roles: Dict[str, RoleProfile] = {}
        self._loaded = False

    def ensure_file(self) -> None:
        """
        确保缓存文件所在目录存在；若文件不存在则创建一个**空文件**并打印日志。
        无论是否创建，均打印当前缓存统计信息。
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                pass
            print(f"[CACHE] created empty cache file -> {self.path}")
        # 打印当前内存已知条目数（如果还没 load，这里先尝试加载一次）
        roles = self.load()
        print(f"[CACHE] current cached roles: {len(roles)} -> {self.path}")

    def load(self) -> Dict[str, RoleProfile]:
        if self._loaded:
            return self._roles
        self._roles = {}
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        rp = RoleProfile(
                            name=str(obj["name"]),
                            description=str(obj.get("description", "")),
                            system_template=str(obj.get("system_template", "")),
                            user_template=str(obj.get("user_template", "")),
                            local_handler=None,
                            temperature=None,
                        )
                        # 合法性最小校验
                        if rp.name and rp.system_template and rp.user_template:
                            self._roles[rp.name] = rp
                    except Exception:
                        continue
        self._loaded = True
        return self._roles

    def _serialize(self, rp: RoleProfile) -> Dict[str, str]:
        return {
            "name": rp.name,
            "description": rp.description or "",
            "system_template": rp.system_template or "",
            "user_template": rp.user_template or "",
        }

    def save_append(self, roles: List[RoleProfile]) -> None:
        """追加写入（去重：若 name 已存在则跳过重写磁盘，仅更新内存）。打印追加与总量日志。"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not roles:
            # 也打印日志，方便观察
            print(f"[CACHE] nothing to append. total={len(self._roles)} -> {self.path}")
            return

        new_to_write: List[RoleProfile] = []
        for r in roles:
            if not r.name or not r.system_template or not r.user_template:
                continue
            if r.name in self._roles:
                # 内存内更新（若模板变化，这里覆盖）
                self._roles[r.name] = RoleProfile(
                    name=r.name,
                    description=r.description,
                    system_template=r.system_template,
                    user_template=r.user_template,
                    local_handler=None,
                    temperature=None,
                )
                continue
            self._roles[r.name] = r
            new_to_write.append(r)

        wrote = 0
        if new_to_write:
            with open(self.path, "a", encoding="utf-8") as f:
                for r in new_to_write:
                    f.write(json.dumps(self._serialize(r), ensure_ascii=False) + "\n")
                    wrote += 1

        print(f"[CACHE] appended {wrote} roles (total={len(self._roles)}) -> {self.path}")


# ---------------------------
# 调试与 Trace 工具（不会影响准确率）
# ---------------------------

def _extract_code_blocks(text: str) -> List[str]:
    """提取 ```python ... ``` 或 ``` ... ``` 代码块；返回 list[str]"""
    if not text:
        return []
    blocks = []
    for m in re.finditer(r"```(?:python)?\s*(.*?)```", text, re.S | re.I):
        blocks.append(m.group(1).strip())
    return blocks

def _dump_trace(dump_dir: str, dataset: str, seed: int, idx: int, ok: bool,
                task: dict, program, out: dict, extra: dict=None) -> str:
    os.makedirs(dump_dir, exist_ok=True)
    node_out = out.get("node_outputs", {})
    final_text = out.get("final", "")
    kept_edges = program.G.graph.get("kept_edges", [])
    code_candidates = []
    # 1) 从 final 提取
    code_candidates += _extract_code_blocks(final_text)
    # 2) 从各节点输出再补充（如 programmer/bug_fixer 等）
    for nid, txt in node_out.items():
        code_candidates += _extract_code_blocks(txt)
    # 去重
    seen, code_snips = set(), []
    for c in code_candidates:
        if c not in seen:
            seen.add(c); code_snips.append(c)

    payload = {
        "dataset": dataset,
        "seed": seed,
        "index": idx,
        "ok": ok,
        "task": task,
        "final": final_text,
        "node_outputs": node_out,
        "usage": out.get("usage", {}),
        "kept_edges": kept_edges,
        "active_edges": [(u, v) for (u, v, d) in program.G.edges(data=True) if d.get("active", True)],
        "active_nodes": [n for n, d in program.G.nodes(data=True) if d.get("active", True)],
        "code_blocks": code_snips[:5],  # 最多存 5 个，防止太大
    }
    if extra:
        payload.update(extra)
    tag = "ok" if ok else "fail"
    fn = f"{dataset}_seed{seed}_{idx:04d}_{tag}.json"
    path = os.path.join(dump_dir, fn)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def _print_failure_brief(i: int, task: dict, out: dict, code_blocks: list):
    q = task.get("question") or task.get("prompt") or task.get("task_id") or ""
    print("\n=== ❌ FAIL @ ex#{} ===".format(i))
    print("Q/Prompt:", (q[:400] + "...") if len(q) > 400 else q)
    final = out.get("final", "")
    print("Final (trunc):", (final[:400] + "...") if len(final) > 400 else final)
    if code_blocks:
        print("Top code block (trunc):")
        head = code_blocks[0]
        head = head[:1000] + "..." if len(head) > 1000 else head
        print("```python\n" + head + "\n```")
    print("==============\n")


# ---------------------------
# 运行辅助
# ---------------------------

def disable_local_handlers(program):
    for rp in program.role_library.values():
        rp.local_handler = None

def ensure_exit_mark(program):
    if not any(d.get("is_exit", False) for _, d in program.G.nodes(data=True)):
        for k in ("evaluator", "judge"):
            if k in program.G.nodes:
                program.G.nodes[k]["is_exit"] = True
                break

def contributing_nodes_from_kept_edges(program) -> List[str]:
    """
    从 one-shot 剪枝后保留的边估计“贡献节点”，优先保活 evaluator 路径上的节点。
    """
    kept: List[Tuple[str, str]] = program.G.graph.get("kept_edges", []) or []
    if not kept:
        # 回退：取所有非 task_hub 的活跃节点
        return [n for n, d in program.G.nodes(data=True) if d.get("active", True) and n not in ("task_hub",)]
    nodes = set()
    for u, v in kept:
        nodes.add(u); nodes.add(v)
    # 去除虚拟节点
    nodes.discard("task_hub")
    return list(nodes)

def select_roles_to_cache(program, cache_topk: int) -> List[RoleProfile]:
    """
    选择要固化到缓存里的新角色：
    - 非内置（不在 BUILTIN_ROLES）
    - 优先：在 kept_edges 路径上（近似视为有贡献）
    - 兜底：如为空，则缓存所有活跃节点里的非内置（最多 top-k）
    """
    builtin_names = set(BUILTIN_ROLES.keys()) | {"task_hub"}

    # 优先：保留边路径上的节点
    contrib_nodes = contributing_nodes_from_kept_edges(program)
    picked: List[RoleProfile] = []
    for nid in contrib_nodes:
        if nid in builtin_names:
            continue
        rp = program.role_library.get(nid)
        if rp and rp.system_template and rp.user_template:
            picked.append(rp)
        if len(picked) >= cache_topk:
            return picked

    # 兜底：全图里找非内置、活跃节点
    fallback: List[RoleProfile] = []
    for nid, d in program.G.nodes(data=True):
        if not d.get("active", True) or nid in builtin_names:
            continue
        rp = program.role_library.get(nid)
        if rp and rp.system_template and rp.user_template:
            fallback.append(rp)
            if len(fallback) >= cache_topk:
                break
    return picked or fallback

def merge_role_library_with_cache(cache_roles: Dict[str, RoleProfile]) -> Dict[str, RoleProfile]:
    """把内置角色与缓存角色合并为一个 role_library（缓存可覆盖同名）"""
    role_lib = dict(BUILTIN_ROLES)
    for name, rp in cache_roles.items():
        role_lib[name] = rp
    return role_lib


# ---------------------------
# 单题运行（含剪枝与进化钩子）
# ---------------------------

def run_one_task(
    task: Dict[str, Any],
    cfg: Dict[str, Any],
    mode: str,
    rounds: int,
    use_prune: bool,
    use_feedback: bool,
    llm_client,                     # 复用同一 LLM 客户端
    role_cache: Optional[RoleCache],
    cache_topk: int,
    cache_only_if_correct: bool
) -> Tuple[Dict[str, Any], Any, List[RoleProfile]]:
    """
    返回: (out, program, cached_roles_to_append)
    """
    # 1) 预合并缓存角色 → 构图（G-Designer 里还会做“AI生成角色”与合并）
    cache_roles = role_cache.load() if role_cache is not None else {}
    role_library = merge_role_library_with_cache(cache_roles)

    program = build_task_graph(task, cfg, role_library=role_library)
    program = sample_architecture(program, task, cfg)
    ensure_exit_mark(program)

    # 2) 运行模式：pure-llm / hybrid
    if mode == "pure-llm":
        disable_local_handlers(program)

    # 3) 构造“回合内自进化”钩子（以及可选的 textual-grad）
    hooks = None
    evolver = build_meta_evolver_hook(program, cfg, llm_client) if use_feedback else None
    if evolver is not None:
        hooks = type("H", (), {"textual_gradient_hook": staticmethod(evolver)})()

    # 4) 剪枝/运行
    if use_prune:
        # 剪枝前热身一轮（可选）
        _ = program.run(task=task, rounds=1, early_exit=True, hooks=None)
        program = prune_once(program, task, cfg)
        out = program.run(task=task, rounds=rounds, early_exit=True, hooks=hooks)
    else:
        out = program.run(task=task, rounds=rounds, early_exit=True, hooks=hooks)

    # 统一以 evaluator 的文本作为最终输出（若存在）
    if "evaluator" in program.G.nodes:
        if "evaluator" in out.get("node_outputs", {}):
            out["final"] = out["node_outputs"]["evaluator"]

    # 5) 选择要固化的角色（题间进化）
    to_cache: List[RoleProfile] = select_roles_to_cache(program, cache_topk=cache_topk)

    return out, program, to_cache


# ---------------------------
# 主程序
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", default="gsm8k_tiny")
    ap.add_argument("--mode", default="pure-llm", choices=["hybrid","pure-llm"])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--no_prune", action="store_true")
    ap.add_argument("--no_feedback", action="store_true")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="inter-task sleep to avoid rate limit")
    ap.add_argument("--out", default=None, help="output CSV path; default auto under logs/metrics/")

    # 题间进化（角色缓存）相关
    ap.add_argument("--role_cache_path", default="data/roles/generated_roles.jsonl",
                    help="JSONL 文件；用于在任务之间持久化/复用生成角色")
    ap.add_argument("--cache_topk", type=int, default=3, help="每题最多固化的生成角色数")
    ap.add_argument("--cache_save_only_correct", action="store_true", help="仅当答对时才固化角色（默认：否）")
    ap.add_argument("--cache_debug", action="store_true", help="调试：打印每题的非内置节点与缓存候选")

    # 调试与可视化
    ap.add_argument("--dump_traces_dir", default=None,
                    help="若设置，则把每道题的节点输出/最终答案等保存为 JSON（不影响推理结果）")
    ap.add_argument("--print_failures", action="store_true",
                    help="打印失败样本的简要信息（题干/最终输出/首个代码块片段）")

    args = ap.parse_args()

    cfg = bootstrap(args.config)
    os.makedirs("logs/metrics", exist_ok=True)
    out_csv = args.out or f"logs/metrics/eval_{args.mode}.csv"

    # 角色缓存
    role_cache = RoleCache(args.role_cache_path)
    # 启动即确保路径与空文件存在，并打印现有条目数
    role_cache.ensure_file()

    rows = []
    for seed in range(args.seeds):
        random.seed(seed)
        for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            # 读取数据
            all_tasks = list(load_dataset(ds))
            tasks = all_tasks if args.max_examples <= 0 else all_tasks[: args.max_examples]
            if not tasks:
                print(f"[WARN] dataset empty: {ds}")
                continue

            # 预创建 LLM 客户端（回合内进化也会用到）
            llm = build_llm_from_cfg(cfg)

            n_ok, tot_tokens, tot_calls, lat_sum = 0, 0, 0, 0.0
            bar = tqdm(total=len(tasks), desc=f"{ds} seed={seed}", unit="ex")

            for i, task in enumerate(tasks, 1):
                t0 = time.time()

                out, program, to_cache = run_one_task(
                    task=task,
                    cfg=cfg,
                    mode=args.mode,
                    rounds=args.rounds,
                    use_prune=(not args.no_prune),
                    use_feedback=(not args.no_feedback),
                    llm_client=llm,
                    role_cache=role_cache,
                    cache_topk=args.cache_topk,
                    cache_only_if_correct=args.cache_save_only_correct,
                )

                dt = time.time() - t0

                # 判分（容错：final 取不到则给空串）
                final_text = out.get("final", "")
                ok = bool(judge_correct(task, final_text))
                n_ok += int(ok)
                tot_tokens += out.get("usage", {}).get("total_tokens", 0)
                tot_calls += 1
                lat_sum += dt

                # —— 题间进化：固化角色到磁盘（可选“仅正确时”） ——
                if to_cache:
                    if (not args.cache_save_only_correct) or ok:
                        role_cache.save_append(to_cache)
                else:
                    # 即使没有可写，也打印一行，便于诊断
                    print(f"[CACHE] no roles selected for caching at ex#{i}")

                # 调试：打印非内置活跃节点与最终入选缓存的角色名
                if args.cache_debug:
                    builtin_names = set(BUILTIN_ROLES.keys()) | {"task_hub"}
                    non_builtin = [n for n, d in program.G.nodes(data=True)
                                   if d.get("active", True) and n not in builtin_names]
                    picked_names = [r.name for r in (to_cache or [])]
                    print(f"[CACHE-DEBUG] non-builtin nodes: {non_builtin} | picked: {picked_names} | ok={ok}")

                # 可选：落盘 trace
                if args.dump_traces_dir:
                    trace_path = _dump_trace(
                        dump_dir=args.dump_traces_dir,
                        dataset=ds, seed=seed, idx=i, ok=ok,
                        task=task, program=program, out=out
                    )
                    if i <= 3:
                        print(f"[TRACE] saved -> {trace_path}")

                # 可选：失败样本打印
                if args.print_failures and not ok:
                    codes = _extract_code_blocks(final_text)
                    for txt in out.get("node_outputs", {}).values():
                        codes += _extract_code_blocks(txt)
                    _print_failure_brief(i, task, out, codes)

                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

                bar.set_postfix({
                    "acc": f"{(n_ok/i):.3f}",
                    "avg_tok": f"{(tot_tokens/max(1,tot_calls)):.1f}",
                    "avg_lat": f"{(lat_sum/max(1,tot_calls)):.2f}s"
                })
                bar.update(1)

                # 里程碑打印
                if i % 50 == 0 or i == len(tasks):
                    print(f"[{ds} seed={seed}] {i}/{len(tasks)} "
                          f"acc={(n_ok/i):.3f} avg_tokens={(tot_tokens/max(1,tot_calls)):.1f} "
                          f"avg_latency={lat_sum/max(1,tot_calls):.2f}s")

            bar.close()

            rows.append({
                "mode": args.mode,
                "dataset": ds,
                "seed": seed,
                "accuracy": n_ok / max(1, len(tasks)),
                "avg_tokens": tot_tokens / max(1, tot_calls),
                "avg_latency_s": lat_sum / max(1, tot_calls),
                "count": len(tasks),
                "rounds": args.rounds,
                "no_prune": int(args.no_prune),
                "no_feedback": int(args.no_feedback),
            })

    # 写 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else \
            ["mode","dataset","seed","accuracy","avg_tokens","avg_latency_s","count","rounds","no_prune","no_feedback"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved metrics -> {out_csv}")

    # 结束前再次确保缓存文件存在，并打印最终统计
    role_cache.ensure_file()
    print(f"[OK] role cache at -> {role_cache.path}")

if __name__ == "__main__":
    main()
