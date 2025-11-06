from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx

from metagen_ai.roles.schema import RoleProfile, NodeOutput, RunTraceItem, Hooks, render_messages
from metagen_ai.utils.llm import LLMClient

@dataclass
class GraphProgram:
    """
    Executable multi-agent program represented as a DAG.
    Node attributes expected:
        - role: str (must exist in role_library)
        - active: bool (optional, default True)    # for pruning/gating
    Edge attributes (u, v):
        - active: bool (optional, default True)    # for pruning/gating
        - kind: "space" | "time" (default "space")
    """
    G: nx.DiGraph
    role_library: Dict[str, RoleProfile]
    llm: Optional[LLMClient] = None
    default_temperature: float = 0.2

    def __post_init__(self):
        if not nx.is_directed_acyclic_graph(self.G):
            raise ValueError("GraphProgram requires a DAG.")
        # Initialize defaults
        for n, data in self.G.nodes(data=True):
            data.setdefault("active", True)
        for u, v, data in self.G.edges(data=True):
            data.setdefault("active", True)
            data.setdefault("kind", "space")

    def run(self,
            task: Dict[str, Any],
            rounds: int = 1,
            hooks: Optional[Hooks] = None,
            early_exit: bool = False) -> Dict[str, Any]:
        """
        Execute the DAG for a number of rounds. Space edges pass same-round messages;
        time edges bring summaries from the previous round.

        Returns:
            {
              "final": str,
              "node_outputs": Dict[str, str],
              "traces": List[RunTraceItem],
              "usage": Dict[str, int],
              "rounds": int
            }
        """
        # Make sure hooks is at least an empty object
        hooks = hooks or type("Hooks", (), {})()

        order = list(nx.topological_sort(self.G))
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        traces: List[RunTraceItem] = []
        last_round_summary = ""

        node_text: Dict[str, str] = {}

        for r in range(rounds):
            round_outputs: Dict[str, NodeOutput] = {}
            for node_id in order:
                nattr = self.G.nodes[node_id]
                if not nattr.get("active", True):
                    continue
                role_name = nattr.get("role", "")
                if role_name not in self.role_library:
                    raise KeyError(f"Role '{role_name}' not found in role_library.")
                role = self.role_library[role_name]

                # Gather inputs from active predecessors via space edges
                inputs = {}
                for u in self.G.predecessors(node_id):
                    if not self.G.nodes[u].get("active", True):
                        continue
                    eattr = self.G.get_edge_data(u, node_id) or {}
                    if not eattr.get("active", True) or eattr.get("kind", "space") != "space":
                        continue
                    if u in round_outputs:
                        inputs[u] = round_outputs[u].text
                    else:
                        # If predecessor skipped (inactive), ignore
                        pass

                # Add time-edge derived summary
                prev_summary = last_round_summary

                # Hooks: before_node (safe call)
                before = getattr(hooks, "before_node", None)
                if callable(before):
                    before(node_id, {"task": task, "inputs": inputs, "prev_summary": prev_summary})

                # Execute: local handler preferred; else LLM
                if role.local_handler is not None:
                    text = role.local_handler({
                        "task": task, "inputs": inputs, "prev_summary": prev_summary, "node_id": node_id
                    })
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                else:
                    if self.llm is None:
                        raise RuntimeError(f"No LLM client configured and role '{role.name}' has no local handler.")
                    messages = render_messages(role, task, inputs, prev_summary)
                    temp = role.temperature if role.temperature is not None else self.default_temperature
                    resp = self.llm.chat(messages, temperature=temp)
                    text = resp["text"]
                    usage = resp.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

                out = NodeOutput(text=text, usage=usage)
                round_outputs[node_id] = out

                # Accumulate usage
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)

                # Trace (compact)
                prompt_preview = ""  # can be populated if needed
                traces.append(RunTraceItem(
                    node_id=node_id,
                    role=role.name,
                    prompt_preview=prompt_preview,
                    output_preview=(text[:160] + ("..." if len(text) > 160 else "")),
                    usage=usage,
                ))

                # Hooks: after_node (safe call)
                after = getattr(hooks, "after_node", None)
                if callable(after):
                    after(node_id, out)

                # Early-exit: if this node is tagged as an exit, stop the round
                if early_exit and nattr.get("is_exit", False):
                    break

            # End of round summary: concat sink node outputs (active sinks)
            sinks = [n for n in order if self.G.out_degree(n) == 0 and self.G.nodes[n].get("active", True)]
            last_round_summary = "\n".join(round_outputs[s].text for s in sinks if s in round_outputs)
            # Store for return after final round
            node_text = {nid: ro.text for nid, ro in round_outputs.items()}

            # Textual gradient hook after each round (safe call)
            tgh = getattr(hooks, "textual_gradient_hook", None)
            if callable(tgh):
                tgh({
                    "task": task,
                    "round": r,
                    "node_outputs": node_text,
                    "summary": last_round_summary,
                })

        # Choose final output: prefer "judge"; else any sink; else last node
        final_text = self._select_final(node_text, prefer="judge")
        return {
            "final": final_text,
            "node_outputs": node_text,
            "traces": traces,
            "usage": total_usage,
            "rounds": rounds,
        }

    def _select_final(self, node_text: Dict[str, str], prefer: str = "judge") -> str:
        if prefer in self.G.nodes and prefer in node_text:
            return node_text[prefer]
        # otherwise choose a sink with output
        sinks = [n for n in self.G.nodes if self.G.out_degree(n) == 0]
        for s in sinks:
            if s in node_text:
                return node_text[s]
        # fallback: last available
        for n in reversed(list(self.G.nodes)):
            if n in node_text:
                return node_text[n]
        return ""
