import copy
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from .edits import ACTIONS, EditPolicy, apply_edit, features
from .grpo import grpo_loss


def supervised_edit_policy(tasks, output, steps=400, seed=42):
    """Warm-start action selection from transformations observed in real TRAIN pairs."""
    from .edits import PAUSE
    if not tasks or any(t.split != 'train' or not t.target for t in tasks):
        raise ValueError('real train source/target pairs required')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    policy = EditPolicy()
    x = torch.stack([features(t.source) for t in tasks])
    labels = []
    for task in tasks:
        pause = bool(PAUSE.search(task.source)) and not PAUSE.search(task.target)
        fast = 'sync_with_stdio' not in task.source and 'sync_with_stdio' in task.target
        endl = bool(re.search(r'<<\s*(?:std::)?endl\b', task.source)) and not re.search(r'<<\s*(?:std::)?endl\b', task.target)
        action = 'all_edits' if pause and endl and fast else 'fast_io_remove_pause' if pause and fast else 'remove_pause' if pause else 'fast_io_avoid_endl' if fast and endl else 'fast_io' if fast else 'avoid_endl' if endl else 'identity'
        labels.append(ACTIONS.index(action))
    y = torch.tensor(labels)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'labels.jsonl').open('w') as f:
        for t, idx in zip(tasks, labels):
            f.write(json.dumps({'task_id':t.task_id, 'problem_id':t.problem_id, 'split':'train', 'observed_action':ACTIONS[idx]})+'\n')
    with (output/'training.jsonl').open('w') as f:
        for step in range(steps):
            loss = torch.nn.functional.cross_entropy(policy(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            f.write(json.dumps({'step':step, 'loss':float(loss.detach())})+'\n')
    policy.save(output/'policy.pt')
    report = {'mode':'supervised_edit_policy', 'steps':steps, 'real_training_pairs':len(tasks),
              'label_counts':{a:labels.count(i) for i,a in enumerate(ACTIONS)}, 'seed':seed,
              'label_rule':'source-to-target removal of pause/endl or addition of sync_with_stdio; otherwise preserve',
              'target_code_visible_to_actor_at_inference':False}
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    return report


def train_edit_policy(tasks, tools, output, steps=8, group_size=4, seed=42,
                      lr=0.003, beta=0.02, trials=3, checkpoint=None):
    """On-policy compiler-scored rollouts. No PIE target is passed to the actor."""
    if not tasks or any(t.split != "train" for t in tasks):
        raise ValueError("nonempty train split required")
    if group_size < 2:
        raise ValueError("group_size must be >=2")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = EditPolicy.load(checkpoint) if checkpoint else EditPolicy()
    policy.train()
    policy.save(output / "initial.pt")
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0)
    cache = {}
    input_tasks = len(tasks)
    baselines = {}
    invalid = []
    for task in tasks:
        baseline = tools.compile(task.source)
        check = tools.compare(baseline["artifact_id"], baseline["artifact_id"], task.tests, trials) if baseline["ok"] else {"correct": False, "compiler_diagnostics": baseline["stderr"]}
        if check["correct"]:
            baselines[task.task_id] = baseline
        else:
            invalid.append({"task_id": task.task_id, "feedback": check})
    tasks = [task for task in tasks if task.task_id in baselines]
    (output / "invalid_baselines.json").write_text(json.dumps(invalid, indent=2))
    if not tasks:
        raise ValueError("no baseline passes its public tests")
    trajectories = 0
    informative_groups = 0
    began = time.time()
    with (output / "trajectories.jsonl").open("w") as logfile, (output / "training.jsonl").open("w") as metrics:
        for step in range(steps):
            task = tasks[step % len(tasks)]
            x = features(task.source)
            with torch.no_grad():
                old_dist = torch.distributions.Categorical(logits=policy(x))
                actions = old_dist.sample((group_size,))
                old_logp = old_dist.log_prob(actions)
                ref_logp = torch.distributions.Categorical(logits=reference(x)).log_prob(actions)
            baseline = baselines[task.task_id]
            rewards = []
            for action_idx in actions.tolist():
                action = ACTIONS[action_idx]
                code = apply_edit(task.source, action)
                digest = hashlib.sha256(code.encode()).hexdigest()
                key = (task.task_id, digest)
                cached = key in cache
                if not cached:
                    compiled = tools.compile(code)
                    result = tools.compare(baseline["artifact_id"], compiled["artifact_id"], task.tests, trials) if compiled["ok"] else {"correct": False, "reward": -1.0, "compile_failed": True}
                    cache[key] = result
                result = cache[key]
                rewards.append(result["reward"])
                logfile.write(json.dumps({"trajectory_id": trajectories, "step": step,
                    "task_id": task.task_id, "problem_id": task.problem_id, "split": task.split,
                    "action": action, "candidate_sha256": digest, "measurement_reused": cached,
                    "old_logp": float(old_logp[len(rewards)-1]), "feedback": result}) + "\n")
                trajectories += 1
            reward_tensor = torch.tensor(rewards)
            informative = bool(reward_tensor.std(unbiased=False) > 1e-6)
            informative_groups += int(informative)
            last = {}
            for _ in range(2):
                current = torch.distributions.Categorical(logits=policy(x)).log_prob(actions)
                loss, last = grpo_loss(current, old_logp, ref_logp, reward_tensor, beta=beta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
                optimizer.step()
            metrics.write(json.dumps({"step": step, "loss": float(loss.detach()),
                          "reward_mean": float(reward_tensor.mean()), "informative": informative, **last}) + "\n")
            logfile.flush()
            metrics.flush()
    policy.save(output / "policy.pt")
    change = sum((a-b).abs().sum().item() for a,b in zip(policy.parameters(), reference.parameters()))
    report = {"mode": "edit_policy_grpo", "steps": steps, "group_size": group_size,
              "sampled_trajectories": trajectories, "distinct_measured_candidates": len(cache),
              "unique_training_tasks": len(tasks), "informative_groups": informative_groups,
              "input_training_tasks": input_tasks, "excluded_invalid_baselines": len(invalid),
              "parameter_l1_change": change, "seed": seed, "seconds": time.time()-began,
              "checkpoint": "policy.pt", "target_code_visible_to_actor": False,
              "measurement_cache": "same task and exact candidate reuse; each sampled action remains a rollout"}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report
