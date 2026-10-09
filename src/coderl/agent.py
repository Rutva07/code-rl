from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict, total=False):
    task: dict
    best: str
    best_id: str
    original_id: str
    proposal: dict
    compiled: dict
    feedback: dict
    history: list
    attempt: int
    failed: bool


def build_agent(tools, planner, max_attempts=3, trials=7, threshold=0.05):
    from .schema import Task

    def baseline(s):
        task = Task.model_validate(s["task"])
        c = tools.compile(task.source)
        if not c["ok"]:
            return {"failed": True, "feedback": {"baseline_compile_error": c["stderr"]}, "history": [], "attempt": 0}
        check = tools.compare(c["artifact_id"], c["artifact_id"], task.tests, trials, threshold)
        if not check["correct"]:
            return {"failed": True, "feedback": check, "history": [], "attempt": 0}
        return {"best": task.source, "best_id": c["artifact_id"], "original_id": c["artifact_id"],
                "failed": False, "attempt": 0, "history": []}

    def plan(s):
        return {"proposal": planner.propose(s["best"], s["history"], s["attempt"])}

    def compile_candidate(s):
        if not s["proposal"]["candidate"].strip():
            return {"compiled": {"ok": False, "stderr": "generator returned empty source"}}
        return {"compiled": tools.compile(s["proposal"]["candidate"])}

    def evaluate(s):
        c = s["compiled"]
        if not c["ok"]:
            return {"feedback": {"correct": False, "accepted": False, "reward": -1.0,
                                 "compile_failed": True, "compiler_diagnostics": c["stderr"]}}
        task = Task.model_validate(s["task"])
        f = tools.compare(s["best_id"], c["artifact_id"], task.tests, trials, threshold)
        f["test_failed"] = not f["correct"]
        f["last_reduction"] = f.get("runtime_reduction", 0)
        return {"feedback": f}

    def refine(s):
        entry = {"attempt": s["attempt"], **s["proposal"], "compiled": s["compiled"], "feedback": s["feedback"]}
        changes = {"history": s["history"] + [entry], "attempt": s["attempt"] + 1}
        if s["feedback"].get("accepted"):
            changes.update(best=s["proposal"]["candidate"], best_id=s["compiled"]["artifact_id"])
        return changes

    graph = StateGraph(State)
    for name, fn in [("baseline", baseline), ("plan", plan), ("compile", compile_candidate), ("evaluate", evaluate), ("refine", refine)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "baseline")
    graph.add_conditional_edges("baseline", lambda s: "stop" if s["failed"] else "go", {"stop": END, "go": "plan"})
    graph.add_edge("plan", "compile")
    graph.add_edge("compile", "evaluate")
    graph.add_edge("evaluate", "refine")
    graph.add_conditional_edges("refine", lambda s: "stop" if s["attempt"] >= max_attempts or s["proposal"]["action"] == "identity" else "go", {"stop": END, "go": "plan"})
    return graph.compile()
