"""Bounded source transformations; tests, not regexes, decide correctness."""
import re
import math

import torch

ACTIONS = ["identity", "fast_io", "avoid_endl", "fast_io_avoid_endl",
           "remove_pause", "fast_io_remove_pause", "all_edits"]
PAUSE = re.compile(r'(?<![\w:])(?:std::)?system\s*\(\s*"pause"\s*\)\s*;')


def apply_edit(source, action):
    if action not in ACTIONS:
        raise ValueError(action)
    result = source
    if action in ("avoid_endl", "fast_io_avoid_endl", "all_edits"):
        # Only replace streamed manipulators; subsequent verification catches interactive semantics.
        result = re.sub(r"<<\s*(?:std::)?endl\b", lambda _: "<< '\\n'", result)
    if action in ("fast_io", "fast_io_avoid_endl", "fast_io_remove_pause", "all_edits"):
        # Mixed C/C++ I/O can change semantics; leave it alone.
        mixed = re.search(r"\b(?:scanf|printf|puts|getchar|putchar|fread|fwrite)\s*\(", source)
        if not mixed and re.search(r"\b(?:cin|cout)\b", source) and "sync_with_stdio" not in source:
            result = re.sub(r"(\b(?:int|signed)\s+main\s*\([^)]*\)\s*\{)",
                            r"\1\nstd::ios::sync_with_stdio(false); std::cin.tie(nullptr);\n",
                            result, count=1)
    if action in ("remove_pause", "fast_io_remove_pause", "all_edits"):
        # Linux-specific cleanup learned from PIE pairs; verification remains mandatory.
        result = PAUSE.sub("(void)0;", result)
    return result


def features(source, feedback=None):
    feedback = feedback or {}
    vals = [math.log1p(len(source)) / 12,
            float(bool(re.search(r"\bcin\b", source))),
            float(bool(re.search(r"\bcout\b", source))),
            float("endl" in source), float("sync_with_stdio" in source),
            float(bool(re.search(r"\b(?:scanf|printf)\s*\(", source))),
            min(len(re.findall(r"\b(?:for|while)\b", source)) / 10, 1),
            float(feedback.get("compile_failed", False)),
            float(feedback.get("test_failed", False)),
            float(feedback.get("last_reduction", 0)), float(bool(PAUSE.search(source)))]
    return torch.tensor(vals, dtype=torch.float32)


class EditPolicy(torch.nn.Module):
    """Small neural planner selecting one source edit; distinct from a code LLM."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.Tanh(), torch.nn.Linear(64, len(ACTIONS)))

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "actions": ACTIONS}, path)

    @classmethod
    def load(cls, path):
        result = cls()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["actions"] == ACTIONS[:4]:
            # Preserve the original planner's logits when loading its old checkpoint.
            state, old = result.state_dict(), saved["state_dict"]
            state["net.0.weight"].zero_()
            state["net.0.weight"][:, :10] = old["net.0.weight"]
            state["net.0.bias"] = old["net.0.bias"]
            state["net.2.weight"].zero_()
            state["net.2.weight"][:4] = old["net.2.weight"]
            state["net.2.bias"].fill_(-30)
            state["net.2.bias"][:4] = old["net.2.bias"]
            saved["state_dict"] = state
        elif saved["actions"] != ACTIONS:
            raise ValueError("checkpoint action schema mismatch")
        result.load_state_dict(saved["state_dict"])
        return result.eval()


class EditPlanner:
    def __init__(self, checkpoint=None):
        self.policy = EditPolicy.load(checkpoint) if checkpoint else None

    def propose(self, source, history, attempt):
        used = {x.get("action") for x in history}
        if self.policy:
            with torch.no_grad():
                prior = history[-1].get("feedback", {}) if history else {}
                order = self.policy(features(source, prior)).argsort(descending=True).tolist()
        else:
            order = [6, 5, 4, 3, 1, 2, 0]
        for idx in order:
            action = ACTIONS[idx]
            candidate = apply_edit(source, action)
            if action == "identity" and action not in used:
                return {"plan": "The policy selects preservation of the current best program.", "action": action, "candidate": source}
            if action not in used and candidate != source:
                return {"plan": f"Apply {action}; check all outputs and paired runtimes.",
                        "action": action, "candidate": candidate}
        return {"plan": "No remaining applicable edit; preserve current best.", "action": "identity", "candidate": source}
