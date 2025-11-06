# src/metagen_ai/roles/seed_humaneval.py
from __future__ import annotations
from metagen_ai.roles.schema import RoleProfile

def seed_roles_humaneval():
    return {
        "ProjectManager": RoleProfile(
            name="ProjectManager",
            description="Plan minimal structure for the function; ≤50 words.",
            system_template=("You are a project manager. Propose the minimal, correct structure to implement the "
                             "given Python function (≤50 words). No over-engineering."),
            user_template=("Signature + docstring:\n{question}\n"
                           "Return a concise plan (bullets allowed)."),
            local_handler=None, temperature=0.1),
        "AlgorithmDesigner": RoleProfile(
            name="AlgorithmDesigner",
            description="Design concise algorithm / short pseudocode; ≤50 words.",
            system_template=("You are an algorithm designer. Provide a concise algorithm design and, if needed, short "
                             "pseudocode (≤50 words)."),
            user_template=("Signature + docstring:\n{question}\n"
                           "If logic is complex, include 3-5 lines pseudocode."),
            local_handler=None, temperature=0.1),
        "ProgrammingExpert": RoleProfile(
            name="ProgrammingExpert",
            description="Write full implementation; output only one Python fenced block; keep signature.",
            system_template=("You are a programming expert. Write the full implementation. "
                             "Output ONLY ONE Python fenced block. Do NOT change the function name or params."),
            user_template=("Implement this function:\n{question}\n"
                           "Output ONLY ONE fenced Python code block."),
            local_handler=None, temperature=0.1),
        "TestAnalyst": RoleProfile(
            name="TestAnalyst",
            description="Read failing messages; list pinpointed issues & edge cases; ≤50 words.",
            system_template=("You are a test analyst. Given code and failing feedback, list minimal concrete issues and "
                             "edge cases (≤50 words)."),
            user_template=("Code/notes:\n{inputs}\nFailing feedback:\n{prev_summary}\n"
                           "List precise problems and concrete fixes."),
            local_handler=None, temperature=0.1),
        "BugFixer": RoleProfile(
            name="BugFixer",
            description="Produce corrected full implementation; only one Python block; keep signature.",
            system_template=("You are a bug fixer. Produce the corrected full implementation. "
                             "Output ONLY ONE Python fenced block. Keep the original signature."),
            user_template=("Original prompt:\n{question}\nIssues to fix:\n{prev_summary}\n"
                           "Return ONLY ONE fenced Python code block."),
            local_handler=None, temperature=0.1),
        # 可选最终决策者：如果你要多候选合并
        "Decider": RoleProfile(
            name="Decider",
            description="Adopt the most reliable implementation; output only Python block.",
            system_template=("You adopt the best implementation and output ONLY ONE Python fenced block."),
            user_template=("Consider previous candidates in inputs; keep signature; output only code."),
            local_handler=None, temperature=0.1),
    }
