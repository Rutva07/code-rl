# CodeRL: Agentic AI for Code Optimization — Updated Project

Python · LangGraph · PyTorch · C++ · GRPO · REST APIs

The updated system now shows a repeatable improvement on the real PIE benchmark
pilot: **0 → 3 correct optimizations among 22 valid programs**, and
**8.00% lower geometric-mean runtime** after fresh 21-trial confirmation.
Every selected program passed every supplied usable case. This folder contains
the implementation, real data, trained policies and LoRA adapters, actual logs,
optimized C++ files, verification results and reproduction commands.

## Confirmed results

| Measurement | Old system | Updated combined system |
|---|---:|---:|
| Requested test programs | 24 | 24 |
| Original programs passing baseline checks | 22 | 22 |
| Correct optimizations | 0 | 3 |
| Correct-optimization rate | 0.00% | 13.64% |
| Geometric-mean speedup over all valid programs | 1.0000× | 1.0870× |
| Geometric-mean runtime reduction | 0.00% | 8.00% |
| Output correctness of selected programs | 100% | 100% |
| Original proposal budget | 3 | 3 |
| Fresh confirmation trials | 21 | 21 |

The optimization-rate gain is **13.64 percentage points**; relative percentage
improvement is undefined because the old rate was zero. Two original programs
failed the expanded exact-output baseline check and are excluded from both
systems. The 22 valid programs have 2,171 usable cases, out of 2,379 across the
24 requested programs. Unchanged programs count as speedup 1 in the aggregate.

| Confirmed program | Runtime reduction | Main change |
|---|---:|---|
| p03796:s009177254 | 48.69% | Pretrained model generates a stdio-based factorial rewrite |
| p03697:s018012495 | 37.43% | Learned removal of a Windows pause shell call on Linux |
| p02898:s043860229 | 50.26% | Pretrained model generates a stdio-based rewrite |

Raw paired samples, final sources and source diffs are in
`outputs/v2/confirmation/`. The comparison is a **small development pilot on
held-out training problems**, not a broad unseen-benchmark or GRPO-only result.
The combined planner was chosen after a three-program model pilot. Fresh
confirmation demonstrates local repeatability, not hidden-input equivalence.
See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for all selection
rules and the distinction between development evaluation and final testing.

## What changed

The bounded policy now learns seven edit actions from real human source/target
pairs before GRPO, including removal of Linux-inapplicable system("pause") calls.
Training covers 64 distinct train problems, with 63 passing exact-output checks,
rather than the original 23. V2 uses the authors' merged benchmark cases and
excludes all held-out problem ids from supervised and GRPO training.

A real pinned Qwen2.5-Coder-0.5B-Instruct is warm-started with full-program LoRA
SFT. The LangGraph planner can fall back to that model for short sources, within
the same proposal budget. Every generated program goes through the compiler,
expected-output tests, paired timing and final recheck. No faster target program
is supplied to the planner or GRPO actor.

A native CLOCK_MONOTONIC fork/exec/wait4 timer removes Python polling overhead.
Both the old and updated systems are rerun with this timer and the same g++
C++17 -O2 -pipe flags. The reported gain therefore comes from changed C++ output,
not from switching timers or substituting dataset reference runtimes.

## Actually executed

| Run | Observed |
|---|---:|
| Passing software verification tests | 20 |
| Real source/target pairs used for action warm start | 64 |
| Action-policy GRPO sampled rollouts | 50,004 |
| Distinct compiler-scored action candidates | 151 |
| Valid action-GRPO training programs | 63 |
| Informative action-GRPO groups | 783 |
| Real pretrained LoRA SFT steps / distinct pairs visited | 128 / 128 |
| Prepared full-program SFT pairs | 512 |
| Actual pretrained token-GRPO completions | 4 |
| Informative pretrained token-GRPO groups | 1 |

The 50,004 count is on-policy **bounded edit-action rollouts**, with exact-task,
exact-candidate reward reuse logged. It is not 50,004 unique programs or a
50K-completion pretrained language-model run. Base weights are approximately
1 GB and are downloaded at the pinned revision; both trained LoRA adapters are
included in this folder.

The full 24-program ablation returned one correct optimization from the
supervised bounded planner and one from the GRPO bounded planner, with identical
selected source files. The three-program pretrained pilot likewise selected
identical final programs from SFT and token GRPO. **Neither ablation establishes
additional quality improvement from GRPO alone.** The confirmed 8.00% figure is
for the revised combined system versus the old system. The original supplied
12%/5% resume figures should not be copied as though they were this experiment's
measured values.

## Real data and files

PIE programs and human faster submissions derive from IBM Project CodeNet.
The bundled release is pinned
[stablegradients/pie-gem5-pairs](https://huggingface.co/datasets/stablegradients/pie-gem5-pairs),
with 58,863 train, 2,116 validation and 878 test pairs. V2 uses the original
[authors' merged archive](https://github.com/LearningOpt/pie), containing public
CodeNet cases and existing AlphaCode-generated benchmark cases. This project
synthesized no replacement benchmark programs, target code or expected outputs.
Attribution, model license and hashes accompany the data.

| Path | Contents |
|---|---|
| src/coderl/agent.py | LangGraph planning, compilation, measurement and refinement |
| src/coderl/api.py, client.py | Actual compiler/execution JSON REST tools |
| src/coderl/worker.py, launcher.cpp | Resource limits, correctness and native paired timing |
| src/coderl/edits.py, train.py, grpo.py | Neural edit policy, human-pair warm start and GRPO |
| src/coderl/llm.py, train_lm.py | Pretrained generation, hybrid planner, LoRA SFT and token GRPO |
| data/raw/, data/v2/, data/merged_tests/ | Full real pair releases, fixed subsets and original test cases |
| outputs/v2/confirmation/ | Primary confirmed result, all selected C++ files and raw timings |
| outputs/v2/hybrid/ | Full combined-system graph histories |
| outputs/v2/grpo/, edit_sft/ | Actual action training, logs and checkpoints |
| outputs/v2/qwen_sft/, pretrained/ | Real model training, adapters and model ablations |
| outputs/test_results.txt, artifact_verification.json | Executed verification and data/rollout audit |
| docs/LEGACY_RESULTS.md, outputs/verified/ | Preserved original zero-improvement run |

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python scripts/verify_artifacts.py
```

See [docs/RUNNING.md](docs/RUNNING.md) for complete training, evaluation,
confirmation and individual optimization commands. The main edit experiment can
run without downloading base model weights. The pretrained fallback uses the
included adapters plus the pinned model downloader. Linux/POSIX and g++ are
required. Optional Docker configuration is supplied, but its build was not run
because Docker was unavailable here.
