# CodeRL: Agentic AI for Code Optimization

Python · LangGraph · PyTorch · C++ · GRPO · REST APIs

A complete C++ optimization project: plan a transformation, invoke compiler and
execution tools through JSON REST APIs, inspect feedback, refine the next attempt,
and preserve a verified best program. The project includes real data, reproducible
training/evaluation, checkpoints, and executed test reports.

## Files

| File | Purpose |
|---|---|
| `src/coderl/agent.py` | LangGraph planning, compiler, evaluation, and refinement loop |
| `src/coderl/api.py`, `client.py` | FastAPI compiler/execution REST tools and HTTP client |
| `src/coderl/worker.py`, `launcher.cpp` | C++17 compilation, limited execution, output checks, paired timing |
| `src/coderl/edits.py` | PyTorch neural edit planner and bounded C++ I/O transformations |
| `src/coderl/grpo.py`, `train.py` | Actual group-relative policy training on compiler-scored actions |
| `src/coderl/llm.py`, `train_lm.py` | Full-program causal-LM generation, supervised warm start, token-level GRPO |
| `src/coderl/data.py` | Pinned real-data downloads, public-test matching, split isolation |
| `src/coderl/evaluate.py` | Matched before/after evaluation and final-program rechecks |
| `src/coderl/cli.py` | Run the API, training, evaluation, or an individual optimization |
| `scripts/run_experiment.py` | Start a real HTTP worker and execute the experiment |
| `scripts/verify_artifacts.py` | Verify data hashes and rollout accounting |
| `tests/` | Compiler, REST, graph, GRPO, token-model, and regression verification |
| `data/` | Full real C++ pair files, prepared programs, real public tests, provenance |
| `outputs/` | Actual rollouts, measurements, checkpoints, C++ outputs, and test reports |
| `Dockerfile`, `compose.yaml` | Optional containerized worker configuration |

## Dataset

Programs and human faster edits come from **PIE**, based on **IBM Project CodeNet**.
The bundled pair release is [stablegradients/pie-gem5-pairs](https://huggingface.co/datasets/stablegradients/pie-gem5-pairs).
Expected outputs and input cases are downloaded from the [CodeNet public-test mirror](https://huggingface.co/datasets/rootacess/pie-perf-testcases).
No experimental programs, target edits, or expected outputs were synthesized.

| Split | Full real pairs bundled | Prepared programs | Public cases |
|---|---:|---:|---:|
| Train | 58,863 | 24 | 88 |
| Validation | 2,116 | 6 | 16 |
| Test | 878 | 12 | 49 |

Both downloads are pinned to specific revisions. `data/manifest.json` records
file hashes and selection rules. Prepared splits have disjoint problem ids and
source hashes. GRPO and evaluation never give the human target code to the actor.
One training source failed the exact-output baseline checker and was excluded;
its different valid-looking output order is recorded in `invalid_baselines.json`.
Public examples are correctness evidence, not a proof of equivalence on every input.

## Executed results

These are results from this project run, not results copied from a paper.

| Measurement | Observed |
|---|---:|
| Passing verification tests | 15 |
| GRPO groups / group size | 12,501 / 4 |
| Sampled single-step edit-policy rollouts | 50,004 |
| Training programs passing exact-output baseline checks | 23 |
| Distinct compiler-scored candidate evaluations | 56 |
| Groups with nonconstant rewards | 510 |
| CPU edit-policy training time | 91.67 seconds |
| Held-out programs / real public cases | 12 / 49 |
| Correctness of final selected outputs | 100% |
| Verified correct optimizations before / after GRPO | 0 / 0 |
| Correct-optimization rate gain | 0.0 percentage points |
| Runtime reduction versus initial policy | 0.0% |

**The supplied +12% correct-optimization and 5% runtime claims were not established
by this run.** Every held-out program was ultimately preserved. The 50,004 count
is actual on-policy sampled edit rollouts across 23 prompts, with exact-candidate
reward reuse recorded in every log; it does not mean 50,004 distinct programs or
independent timing experiments. The executed CPU experiment trains a neural edit
planner. It is not a 50K-rollout training run of a pretrained code language model.

The separate byte-GRU generator completed four supervised warm-start steps and
eight actual compiler-scored token-GRPO completions. All eight failed compilation,
so that run had zero informative reward groups and establishes integration only.
The Transformers adapter was verified using a locally initialized tiny GPT-2.
Pretrained code-model training is implemented but was not run in this CPU environment.

Benchmarks use the same `g++ -std=c++17 -O2 -pipe`, randomized paired executions,
one warm-up, and 11 evaluation trials. Candidates must pass every public case,
exceed a 5% runtime-reduction threshold, and have a positive bootstrap lower bound.
A final recheck reverts candidates that lose their gain. Native wall times include
process startup and are noisy on small public cases. Dataset gem5 reference times
are not used as this project's measurements.

## Run

See [docs/RUNNING.md](docs/RUNNING.md) for setup and complete commands, and
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for policy scope and GRPO details.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
pytest -q
python scripts/run_experiment.py --steps 12501 --include-lm-smoke
```

The worker requires Linux/POSIX and `g++`. Large pretrained weights are not bundled.
Docker was unavailable here; its supplied configuration is not an executed check.
Data attribution and third-party terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
