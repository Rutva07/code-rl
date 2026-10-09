---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code-optimization
- code-performance
- gem5
- pie
pretty_name: PIE gem5-timed code optimization (src,tgt pairs)
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# PIE gem5-timed code optimization (src,tgt pairs)

C++ program-optimization data derived from the [PIE](https://pie4perf.com/) dataset
(["Learning Performance-Improving Code Edits"](https://arxiv.org/abs/2302.07867)),
**re-timed end-to-end with gem5** (x86 Skylake, syscall-emulation mode) at
**per-test-case granularity**. One row per official (source, target) program pair.

This dataset is **reward-agnostic**: it ships the full per-test-case reference timings and
case manifests so a downstream RL / eval pipeline decides at runtime how many cases to use
(`K`) and which reward transform to apply. Nothing is baked in.

> **For an automated agent reading this card:** every column is documented in
> [§ Field dictionary](#field-dictionary) below, including its type, encoding, and which
> splits populate it. Start with [§ How to compute the reward](#how-to-compute-the-reward).

## TL;DR units & encoding
- **Ticks are gem5 picoseconds.** `seconds = ticks * 1e-12`.
- Columns named `*_per_tc_ticks`, `*_tc2time`, `*_case_ids`, `*_excluded` are **JSON strings**
  (a dict `{case_idx: value}` or a list `[case_idx, ...]`) — `json.loads` them.
  Case indices are integers `0 .. n_cases_available-1`.
- `gem5_*` / `our_*` / `*_case_ids` / `n_*` / `flag_*` columns are produced by **this re-timing**.
  `pie_*` columns are **carried verbatim** from the official PIE release.

## How to compute the reward
The reward is a **speedup ratio** over a common set of test cases, robust to which cases you
sample:
```python
import json, random
src = json.loads(row["gem5_src_per_tc_ticks"])     # {idx: ticks}
tgt = json.loads(row["gem5_tgt_per_tc_ticks"])      # ("oracle" variant in the bysrc dataset)
usable = json.loads(row["usable_case_ids"])         # correctness-kept ∩ both-timed ∩ ticks<=wall
K = 5                                                # runtime choice; -1 / len(usable) = use all
cases = usable if K < 0 else random.sample(usable, min(K, len(usable)))
speedup = sum(src[str(c)] for c in cases) / sum(tgt[str(c)] for c in cases)
```
`usable_case_ids` is already the correct, fully-filtered set — **do not** re-derive correctness
or apply the wall yourself. Always clamp `K` to `len(usable)`; never pad, skip, or zero-reward a
pair that has fewer than `K` usable cases. K=5–10 reproduces the full ~100-case aggregate within
~1–2% (per-case time is near-uniform within a program).

`gem5_fastest_id` / `gem5_fastest_per_tc_ticks` give the **fastest surviving program of the
problem** (chosen among programs present in this dataset, ranked over the per-problem common
usable basis) as an oracle ceiling; `our_fastest_speedup` is src-vs-fastest over `usable_case_ids`.

## Provenance & validation
- Timing engine: **gem5.fast**, X86 SE mode, with fork-from-live startup amortization (one
  startup simulation, COW-forked per case; per-case ticks equal a fresh run, validated r=1.0).
- Validated against PIE's published numbers: per-test-case r=1.0 on the test split (aligned on
  PIE's released `tests` indices); speedup log-correlation ~0.97–0.98, median ratio ~0.99.
- **Caveat — pin your toolchain.** The gem5 startup floor depends on the compiler/glibc/link
  mode (≈13M ticks static ↔ ≈195M ticks dynamic). These reference ticks were produced with one
  fixed toolchain; if you re-time rollouts, use the *same* toolchain or regenerate references,
  otherwise speedups are not comparable.

## Filtering (rows are pre-filtered; bad records are DROPPED, not flagged)
- **Compile-fail** programs dropped (`g++ -O2 -std=gnu++17`).
- **Bad-apple** programs dropped: a submission failing > max(5, 5% of cases) is treated as
  wrong/mislabeled (line-wise + 1e-3 float comparator).
- **gem5 hard-excludes** (startup_fail / all-cases-hang-or-crash) dropped.
- **Case kept** iff every surviving good program of the problem passes it (correctness intersection).
- **Strict 2-min wall**: cases above 1.4e10 ticks dropped from `usable`.
- **Pair criterion**: drop a pair if |src.fails ∪ tgt.fails| > max(5, 5%).
- 3 degenerate problems (empty input+output: p00000, p02068, p00754) dropped.
- Retained NON-dropping flags: `flag_low_headroom`, `flag_single_case`, `flag_empty_kept`.

## Splits & why some columns are null
| split | rows | notes |
|---|---|---|
| train | 58863 | PIE released reward/difficulty annotations but no benchmark `tests` here. |
| validation | 2116 | same as train. |
| test | 878 | PIE released benchmark `tests` indices + per-tc reference times here. |

PIE ships **different columns for test vs train/val**, so each row only fills the `pie_*`
columns that apply to its split; the rest are null **by design** (the schema is the union across
splits). Specifically: `pie_*_code_*`, `pie_fastest_code*`, `pie_*_tc2time`, `pie_n_tests`,
`pie_test_*`, `pie_tests_case_ids` are **test-only**; `pie_speedup`, `pie_*_verdict`,
`pie_*_reward_updated*`, `pie_fastest_agg_runtime_updated` are **train/val-only**. None of the
`gem5_*` / `our_*` / `*_case_ids` / `n_*` columns are ever null.

## Field dictionary
Availability: **all** = populated in every split; **test-only** / **train/val-only** = null elsewhere.

### Identity

| field | splits | meaning |
|---|---|---|
| `problem_id` | all | PIE/online-judge problem id. Groups every submission for one problem; the unit the correctness-intersection and `tests` benchmark are defined over. |
| `src_id` | all | Submission id of the SOURCE (slower, to-be-optimized) program. |
| `tgt_id` | all | Submission id of the TARGET (faster) program in this pair. [pairs only] |
| `split` | all | Dataset split: train / validation / test. |
| `language` | all | Programming language. Always `cpp` (C++). |

### Code

| field | splits | meaning |
|---|---|---|
| `src_code` | all | Full C++ source of the source program. |
| `tgt_code` | all | Full C++ source of the target program. [pairs only] |

### Our gem5 per-test-case timing

| field | splits | meaning |
|---|---|---|
| `gem5_src_per_tc_ticks` | all | JSON {case_idx: ticks}. gem5 picoseconds PER TEST CASE for the SRC program. Seconds = ticks * 1e-12. Holds EVERY case gem5 timed (incl. cases later discarded by correctness/wall filters) — select cases via `usable_case_ids`. |
| `gem5_tgt_per_tc_ticks` | all | JSON {case_idx: ticks} for the TGT program (same convention). [pairs only] |
| `gem5_fastest_id` | all | Submission id of the FASTEST surviving program for this problem, chosen among programs that appear as a src/tgt in THIS dataset, ranked by summed gem5 ticks over the per-problem common usable case basis. The practical oracle ceiling. |
| `gem5_fastest_per_tc_ticks` | all | JSON {case_idx: ticks} for the FASTEST program of this problem (see `gem5_fastest_id`); full per-case gem5 timing, same convention. |
| `gem5_fastest_usable_case_ids` | all | Usable case set for the fastest program = kept ∩ fastest-timed ∩ wall. |
| `gem5_fastest_sum_ticks_usable` | all | Sum of FASTEST-program ticks over `usable_case_ids`. Denominator of `our_fastest_speedup`. |
| `gem5_src_excluded` | all | JSON list of case indices gem5 could NOT time for the SRC program (hang / crash / over compute budget during the sweep). |
| `gem5_tgt_excluded` | all | JSON list of case indices gem5 could NOT time for the TGT program. [pairs only] |
| `gem5_src_status` | all | gem5 sweep status for the src program: `ok` (all cases timed) or `forkloop_partial` (some cases excluded). Hard-fail statuses (startup_fail / native_all_excluded) are already dropped. |
| `gem5_tgt_status` | all | gem5 sweep status for the tgt program. [pairs only] |
| `gem5_src_sum_ticks_all` | all | Sum of SRC ticks over `common_timed_case_ids` (cases timed on both sides). Numerator of `our_speedup_all`. |
| `gem5_tgt_sum_ticks_all` | all | Sum of TGT ticks over `common_timed_case_ids`. Denominator of `our_speedup_all`. [pairs only] |

### Case-id sets (JSON int lists) + counts

| field | splits | meaning |
|---|---|---|
| `common_timed_case_ids` | all | Case indices gem5 timed on BOTH src and tgt. [pairs only] |
| `correctness_kept_case_ids` | all | Case indices passed by EVERY good (non-bad-apple, compile-ok) program of the problem = the correctness intersection. Problem-level, identical across rows of a problem. |
| `usable_case_ids` | all | THE case set to use for the reward = correctness-kept ∩ both-sides-timed ∩ ticks≤1.4e10 (pairs) / kept ∩ src-timed ∩ wall (bysrc). Falls back to all-timed when the kept set is empty (single-case problems). |
| `n_common_timed` | all | len(common_timed_case_ids). [pairs only] |
| `n_kept` | all | len(correctness_kept_case_ids). |
| `n_usable` | all | len(usable_case_ids). The effective case count behind the speedup. |
| `n_cases_available` | all | Total test cases that exist for the problem in the merged test-case set. |
| `src_n_fail` | all | # cases the SRC program failed in the native correctness screen (line-wise + 1e-3 float comparator). 0 = passed all. |
| `tgt_n_fail` | all | # cases the TGT program failed in the native correctness screen. [pairs only] |
| `pair_n_fail` | all | Size of (src.fails ∪ tgt.fails) — # cases failed by at least one side of the pair. [pairs only] |

### Derived speedups + flags

| field | splits | meaning |
|---|---|---|
| `our_speedup_usable` | all | **Reference speedup** = sum(src ticks) / sum(tgt ticks) over `usable_case_ids`. Case-set-invariant; this is what training should reproduce. |
| `our_speedup_all` | all | sum(src)/sum(tgt) over `common_timed_case_ids` (no correctness/wall filtering). Provenance/debug. |
| `our_fastest_speedup` | all | sum(src ticks)/sum(fastest ticks) over `usable_case_ids`. Speedup of the src against the FASTEST program of the problem = the practical oracle ceiling for this source. |
| `ceiling_speedup` | all | Mean per-case src ticks / startup floor (192M ticks). Theoretical max speedup attainable for this src given the gem5 startup floor. Drives `flag_low_headroom`. |
| `flag_low_headroom` | all | True if ceiling_speedup < 1.5 — little room to optimize (kept, not dropped). |
| `flag_single_case` | all | True if the problem has only one test case. |
| `flag_empty_kept` | all | True if the correctness intersection was empty (single-case false-alarm problems; usable then falls back to all-timed). |

### Carried verbatim from PIE (`pie_*`)

| field | splits | meaning |
|---|---|---|
| `pie_src_agg_runtime` | all | PIE's published aggregate wall-clock runtime (seconds) of the src program (their gem5 measurement). |
| `pie_tgt_agg_runtime` | all | PIE's published aggregate runtime (s) of the tgt program. [pairs only] |
| `pie_speedup` | train/val-only | PIE's published src/tgt speedup. [train/val only] |
| `pie_fastest_agg_runtime` | all | PIE's published aggregate runtime (s) of the problem's fastest solution. |
| `pie_src_verdict` | train/val-only | Online-judge verdict for the src (e.g. Accepted). [train/val only] |
| `pie_tgt_verdict` | train/val-only | Online-judge verdict for the tgt. [pairs, train/val only] |
| `pie_fastest_agg_runtime_updated` | train/val-only | PIE's re-measured fastest-solution agg runtime. [train/val only] |
| `pie_src_reward_updated` | train/val-only | PIE's annotated reward for the src. [train/val only] |
| `pie_target_reward_updated` | train/val-only | PIE's annotated reward for the tgt. [pairs, train/val only] |
| `pie_src_reward_updated_pct_bin` | train/val-only | PIE's difficulty/percentile bin for the src reward. [train/val only] |
| `pie_target_reward_updated_pct_bin` | train/val-only | PIE's difficulty/percentile bin for the tgt reward. [pairs, train/val only] |
| `pie_n_tests` | test-only | # benchmark test cases PIE used for this problem. [test only] |
| `pie_test_agg_runtime` | test-only | PIE's agg runtime (s) over its benchmark `tests` cases. [test only] |
| `pie_test_accuracy` | test-only | PIE's reported accuracy over the benchmark cases. [test only] |
| `pie_test_compilation` | test-only | PIE's reported compilation success over the benchmark. [test only] |
| `pie_src_code_accuracy` | test-only | PIE's src accuracy over the benchmark cases. [test only] |
| `pie_tgt_code_accuracy` | test-only | PIE's tgt accuracy. [pairs, test only] |
| `pie_src_code_compilation` | test-only | PIE's src compilation flag. [test only] |
| `pie_tgt_code_compilation` | test-only | PIE's tgt compilation flag. [pairs, test only] |
| `pie_src_code_runtime` | test-only | PIE's src agg runtime (s) on the benchmark. [pairs, test only] |
| `pie_tgt_code_runtime` | test-only | PIE's tgt agg runtime (s) on the benchmark. [pairs, test only] |
| `pie_fastest_code` | test-only | Full C++ source of the fastest solution PIE identified for the problem. [test only] |
| `pie_fastest_code_len` | test-only | Token/char length of pie_fastest_code as PIE recorded it. [test only] |
| `pie_fastest_code_runtime` | test-only | PIE's agg runtime (s) of pie_fastest_code. [test only] |
| `pie_fastest_code_accuracy` | test-only | PIE's accuracy of pie_fastest_code. [pairs, test only] |
| `pie_fastest_code_compilation` | test-only | PIE's compilation flag of pie_fastest_code. [pairs, test only] |
| `pie_src_code_tc2time` | test-only | PIE's OWN per-test-case times for the src, JSON {idx: seconds}. [test only] |
| `pie_tgt_code_tc2time` | test-only | PIE's OWN per-test-case times for the tgt, JSON {idx: seconds}. [pairs, test only] |
| `pie_fastest_code_tc2time` | test-only | PIE's OWN per-test-case times for the fastest solution, JSON {idx: seconds}. [test only] |
| `pie_tests_case_ids` | test-only | JSON list of the benchmark case indices PIE released for this test problem (subset of 0..n_cases_available-1). [test only] |

## Companion dataset
This is one of a pair built from the same timing:
- **pie-gem5-pairs** — one row per official (src, tgt) pair.
- **pie-gem5-bysrc** — one row per unique (problem, src); best surviving target kept as oracle.

Built 2026-06-12. Timing: gem5.fast with fork-from-live startup amortization.
