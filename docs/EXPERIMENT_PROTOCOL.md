# V2 experiment protocol

The revised experiment compares the old trained planner with the revised system.
It evaluates 24 fixed test programs selected before timing candidate edits, with
2,379 real merged benchmark input/output cases. Selection is the first unique
test problem with source length at most 1,800 characters, excluding the old
training problems and validation problems. All metadata `usable_case_ids` are
used for held-out evaluation. This is a small, short-program pilot; it does not
represent every PIE task. Existing generated AlphaCode cases are part of the
authors' real benchmark archive. This project generated no replacement cases.

Training uses 64 separate train problems: 32 containing `system("pause")` and
32 without it, ordered by a seeded hash within each stratum. This deliberate
training balance teaches a new Linux cleanup from the real source/target pairs.
It does not determine the held-out test selection. One baseline failing exact
output checking is recorded and excluded. Eight metadata-usable cases per task
(or all if fewer exist) are selected by hash for training throughput.

The supervised edit labels reflect removal of `pause` or streamed `endl`, or
addition of `sync_with_stdio` in a human target; other pairs teach preservation.
This is a bounded action policy, with seven actions and eleven input features.
The human faster target is never available at inference or to the GRPO actor.
The policy is warm-started with 400 supervised steps and then receives 12,501
groups of four on-policy samples. Exact task/candidate feedback is cached;
50,004 rollouts do not mean 50,004 unique programs or independent measurements.

The old and revised policies each get three proposals, the same compiler flags,
the same cases, eleven randomized paired timing trials and the same 5% minimum
per-program reduction. Both must pass a positive local bootstrap lower bound.
A final timing recheck reverts an apparent gain that does not repeat. Separate
supervised-only results test whether GRPO adds anything beyond the warm start.

The native timer measures fork/exec/wait4 with CLOCK_MONOTONIC, including C++
program startup and any child processes. It excludes Python polling overhead.
Both systems are rerun with this timer, so a gain cannot come merely from the
measurement change. Dataset gem5 timings and human target timings are never
substituted for actual local measurements.

After all model training ends, `confirm_outputs.py` retests the frozen before
and after programs with 21 fresh trials. It does not search for new edits or
retune the model. Gains failing this confirmation are reverted. The final report
uses geometric mean speedup over ALL valid programs; unchanged programs count
as speedup 1. Optimization-rate changes are percentage points when the old rate
is zero. Exact-output checking is evidence on supplied cases, not equivalence
on hidden inputs or interactive problems.

The separate pretrained pilot downloads a pinned Qwen2.5-Coder-0.5B-Instruct,
trains rank-eight q_proj/v_proj LoRA for 128 steps on 128 distinct real code
pairs from a prepared set of 512, and runs two groups of two actual token-GRPO
completions. Three separate short held-out programs are checked with identical
greedy generation budgets. This verifies useful pretrained training is possible
on CPU, but is not a 50K-completion code-LLM experiment. Its results must be
reported separately from the primary edit-policy system comparison.

The supplied original resume claims of 12% and 5% are goals, not fixed acceptance
criteria or values to insert into reports. The observed figures are saved from
the actual before/after measurements.

## Combined system development evaluation

After the three-program pretrained pilot showed a correct edit from the SFT
model, a hybrid planner was added. It uses the bounded GRPO planner plus at most
one greedy full-program generation for sources at most 300 characters, within
the same three-proposal budget. The limit bounds CPU cost; the policy does not
inspect problem ids or target programs. All 24 programs are evaluated, including
ones where the model is not invoked. The deployed fallback uses the SFT adapter;
the four-completion token-GRPO pilot returned the same final programs as SFT and
does not establish an additional language-model quality gain.

This architecture choice uses feedback from the three-program pilot, so the
24-program result is a development evaluation on held-out training problems,
not a completely untouched final test. Fresh confirmation establishes local
repeatability of the selected edits, not statistical generalization. Identical
selected before/after source files are assigned runtime ratio 1 in comparisons,
so separately timed copies of the same code cannot masquerade as a model gain.
