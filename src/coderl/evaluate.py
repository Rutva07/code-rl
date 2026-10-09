import json
import math
import hashlib
from pathlib import Path
import numpy as np
from .agent import build_agent


def evaluate(tasks, tools, planner, output, attempts=1, trials=11, threshold=0.05):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = build_agent(tools, planner, attempts, trials, threshold)
    records = []
    with (output / "results.jsonl").open("w") as f:
        for task in tasks:
            # The graph and planner only receive input code, tests, and metadata.
            visible = task.model_dump(exclude={"target"})
            state = graph.invoke({"task": visible}, {"recursion_limit": 6*attempts+5})
            record = {"task_id": task.task_id, "problem_id": task.problem_id,
                      "baseline_valid": not state.get("failed", False),
                      "history": state.get("history", [])}
            if record["baseline_valid"]:
                final = tools.compare(state["original_id"], state["best_id"], task.tests, trials, threshold)
                if state["original_id"] != state["best_id"] and not final.get("accepted", False):
                    record["final_candidate_check"] = final
                    record["reverted_after_recheck"] = True
                    state["best"], state["best_id"] = task.source, state["original_id"]
                    final = tools.compare(state["original_id"], state["best_id"], task.tests, trials, threshold)
                record.update(final=final, optimized=state["original_id"] != state["best_id"] and final["accepted"],
                              selected_changed=state["original_id"] != state["best_id"],
                              selected_source_sha256=hashlib.sha256(state['best'].encode()).hexdigest())
                (output / f"{task.problem_id}_{task.provenance.get('src_id', 'program')}.cpp").write_text(state["best"])
            else:
                record["failure"] = state.get("feedback")
            records.append(record)
            f.write(json.dumps(record)+"\n")
            f.flush()
    report = summarize(records, len(tasks), attempts, trials, threshold)
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def summarize(records, requested, attempts, trials, threshold):
    valid = [r for r in records if r["baseline_valid"]]
    ratios = [r["final"].get("speedup", 1) if r.get("optimized") else 1 for r in valid]
    successful = sum(bool(r.get("optimized")) for r in valid)
    geomean = math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
    report = {"requested_programs": requested, "baseline_valid_programs": len(valid),
              "excluded_invalid_baselines": requested-len(valid), "correct_optimizations": successful,
              "correct_optimization_rate": successful/len(valid) if valid else None,
              "geometric_mean_speedup_all_valid_programs": geomean,
              "runtime_reduction_from_geomean": 1-1/geomean if geomean else None,
              "output_correctness_selected": sum(r['final'].get('correct',False) for r in valid)/len(valid) if valid else None,
              "threshold": threshold, "attempt_budget": attempts, "trials": trials,
              "correctness_scope": "all cases in the supplied task split; see dataset provenance; not formal equivalence",
              "measurement": "paired randomized native wall-clock timings including process startup"}
    return report


def compare_reports(before, after, output):
    a = json.loads((Path(before)/"summary.json").read_text())
    b = json.loads((Path(after)/"summary.json").read_text())
    read = lambda p: {r['task_id']:r for r in map(json.loads,(Path(p)/'results.jsonl').read_text().splitlines()) if r['baseline_valid']}
    ar, br = read(before), read(after)
    ids = sorted(set(ar) & set(br))
    if not ids:
        raise ValueError("no shared valid tasks")
    initial_rate = sum(bool(ar[i].get('optimized')) for i in ids)/len(ids)
    final_rate = sum(bool(br[i].get('optimized')) for i in ids)/len(ids)
    speedup = lambda r: r['final'].get('speedup',1) if r.get('optimized') else 1
    def source_hash(directory, record):
        if record.get('selected_source_sha256'):
            return record['selected_source_sha256']
        files = list(Path(directory).glob(record['problem_id']+'_*.cpp'))
        return hashlib.sha256(files[0].read_bytes()).hexdigest() if len(files) == 1 else None
    identical = {i for i in ids if source_hash(before, ar[i]) and source_hash(before, ar[i]) == source_hash(after, br[i])}
    # An unchanged selected program cannot establish a model-caused runtime gain.
    # Independent timings of the same code differ through noise and host load.
    runtime_ratios = [1.0 if i in identical else speedup(ar[i])/speedup(br[i]) for i in ids]
    relative_runtime = math.exp(np.log(runtime_ratios).mean())
    report = {"shared_valid_tasks":len(ids), "before_optimization_rate":initial_rate,
              "after_optimization_rate":final_rate,
              "optimization_rate_gain_percentage_points":100*(final_rate-initial_rate),
              "optimization_rate_relative_gain_percent":100*(final_rate/initial_rate-1) if initial_rate else None,
              "runtime_reduction_vs_initial_percent":100*(1-relative_runtime),
              "identical_selected_programs":len(identical),
              "identical_program_runtime_ratio":1.0,
              "small_sample_warning": "pilot estimates; native subprocess timings may be noisy",
              "matching_attempt_budget":a['attempt_budget']==b['attempt_budget'],
              "matching_trials":a['trials']==b['trials']}
    Path(output).write_text(json.dumps(report,indent=2))
    return report
