# Implementation and scope

The graph is `baseline → plan → compile → evaluate → refine`. It first verifies
an unmodified source, then proposes transformations, invokes compiler/execution
REST tools and consumes feedback. Compilation or output failure cannot replace
the best program. Passing candidates require a measured runtime reduction of
at least 5% and a positive paired bootstrap lower bound. A final recheck, and a
separate frozen-output confirmation run, revert nonrepeatable apparent gains.

## Policies

The bounded planner is a PyTorch MLP with eleven source/feedback features, a
64-unit hidden layer and seven actions: preserve, fast I/O, replace streamed
endl, combine those changes, remove Linux-inapplicable system("pause"), combine
pause removal with fast I/O, or combine all edits. Mixed C/C++ I/O is left alone
by the fast-I/O edit. Pause deletion substitutes a no-op statement so if/else
structure remains intact. Regexes alone do not establish equivalence; every
candidate still goes through compilation and expected-output tests.

Supervised action labels are inferred from transformations present in real PIE
human targets. GRPO never receives target code. Legacy four-action checkpoints
load with their original ten-feature weights and original action logits intact;
new feature weights are zero and new action logits suppressed. This makes the
old/new comparison faithful to the actual original trained planner.

HFLM loads a real pretrained Qwen model through Transformers. CPU LoRA uses
bfloat16, rank eight, q_proj/v_proj and zero dropout. Adapters are trainable on
reload and save independently of the pinned base. SFT learns full program
responses from actual human faster programs. Response-token scoring starts at
the last prompt logit; prompts and padding never contribute to the loss.
ByteLM is retained for lightweight integration tests and is not the deployed
pretrained model. HF integration is checked against a tiny local GPT-2 and the
real Qwen SFT/GRPO runs.

The hybrid planner uses the bounded GRPO policy and one greedy SFT-model fallback
for short sources within the same total proposal budget. It makes no lookup of
problem ids, hidden human targets or precomputed faster submissions. The CPU
length gate limits generation cost; long sources still use the bounded planner.

## GRPO

Within each group the advantage is `(reward-mean)/max(population_std,1e-6)`.
The action/token ratio is `exp(current_logp-old_logp)`. The loss uses the clipped
PPO surrogate plus `beta*(exp(ref_logp-current_logp)-(ref_logp-current_logp)-1)`.
Old/reference log-probabilities are detached; the reference is frozen. There is
no critic. Token losses average over each response and then over the group,
including EOS and excluding prompt/padding. Sampling and scoring disable dropout.
Equal-reward groups produce zero relative advantage; they are reported honestly.

Compiler or output failure receives -1. Correct candidates receive the clipped
measured fractional reduction, and unchanged programs receive 0. The action
trainer reuses measurements only for the exact task and exact candidate hash;
every reuse is logged. The 50,004 rollouts span 63 valid train programs and 151
distinct measured candidates. They are not 50,004 independent timing trials.
The pretrained token-GRPO pilot has four completions, one informative group and
a measured adapter parameter change; it does not establish a GRPO-only quality
gain. SFT and token-GRPO selected identical programs in that three-program pilot.

## Timing and correctness

Compilation uses the same g++ C++17 -O2 -pipe flags for every baseline and
candidate. Expected outputs are matched as whitespace-separated exact tokens,
without undocumented floating tolerances or multiple-valid-output checkers.
Every task case is checked before timing and during every timed execution.

A native parent forks and execs the resource-limited executable, measures
CLOCK_MONOTONIC until wait4 completes, and records wall and child CPU time. This
includes C++ startup and any shell child invoked by the source, while excluding
Python polling overhead. Baseline/candidate order is randomized in paired trials.
Medians aggregate total time across task cases. Bootstrap resampling of paired
ratios supplies a local interval; it is not a hidden-test generalization bound.
Dataset gem5 timings are never substituted for local timings.

Failed original baselines are recorded and excluded symmetrically from matched
comparisons. Unchanged valid programs contribute speedup 1 to the geometric
mean. Comparisons also normalize identical selected source files to runtime
ratio 1, so host noise between identical programs cannot imply model improvement.
Fresh confirmation performs no new search. The resulting pilot figures measure
this system revision; see EXPERIMENT_PROTOCOL.md for the development-selection
limits and the distinction between system improvement and GRPO attribution.

## Execution

The worker applies CPU, address-space, output-file-size, core-dump and descriptor
limits, bounds captured output and kills process groups on exit or timeout. Its
launcher cache key incorporates the launcher source hash. Inputs accept artifact
hashes rather than user-supplied executable paths. Resource limits are not an OS
security sandbox. The optional non-root Docker worker adds container limits,
internal networking and a read-only root; its build was not run here.
