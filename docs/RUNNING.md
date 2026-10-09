# Run the updated CodeRL project

Use Linux/POSIX, Python 3.11 or later and g++ with C++17. The real code pairs,
prepared splits, benchmark cases, policies, LoRA adapters and executed reports
are bundled. The approximately 1 GB pretrained base is pinned and downloaded
separately; its weights are not needed for the bounded edit-policy experiment.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python scripts/verify_artifacts.py
```

`requirements-tested.txt` records the actual versions used. The project uses
local JSON REST tools at `/compile`, `/run`, `/compare` and `/health`, and
LangGraph invokes those tools through a real HTTP client. API documentation is
at http://127.0.0.1:8000/docs when the worker is running.

## Reproduce the system experiment

Use a new output directory to preserve the bundled results. These scripts start
and stop their own localhost REST workers. Run timing experiments sequentially.

```bash
python scripts/run_v2.py --output outputs/my_v2
python scripts/download_model.py --bind-adapters outputs/v2/qwen_sft/model outputs/v2/pretrained/grpo/model
python scripts/run_hybrid.py --checkpoint outputs/my_v2/grpo/policy.pt --output outputs/my_v2/hybrid
python scripts/confirm_outputs.py --input outputs/my_v2/hybrid --before outputs/my_v2/before --output outputs/my_v2/confirmation
```

The first script warm-starts a seven-action policy on 64 real train pairs, runs
50,004 compiler-scored GRPO action rollouts, and compares the old, supervised
and GRPO planners on all 24 fixed test programs. The hybrid script adds a greedy
pretrained full-program fallback for sources at most 300 characters within the
same three-proposal budget. The confirmation script retests the frozen outputs
with 21 fresh trials and reverts gains that fail to repeat. No new candidates
are generated during confirmation.

The model downloader pins Qwen2.5-Coder-0.5B-Instruct to the exact revision used
here. `--bind-adapters` points the included adapters at that downloaded local
base; it changes configuration only, not the trained adapter tensors. Without
rebinding, the portable adapter configuration downloads the same pinned base
through Hugging Face automatically when first loaded.

## Reproduce pretrained training

The actual CPU run uses bfloat16, four threads, rank-eight LoRA on q_proj/v_proj,
128 supervised steps and four token-GRPO completions. It is a useful small
pretrained-model run, not 50K code-LLM completions.

```bash
coderl sft --model models/qwen-coder --lora --dtype bfloat16 --threads 4 --tasks data/v2/sft.jsonl --steps 128 --max-prompt-tokens 1024 --output outputs/my_qwen_sft
python scripts/run_pretrained.py --base models/qwen-coder --sft outputs/my_qwen_sft/model --output outputs/my_pretrained
python scripts/run_hybrid.py --model outputs/my_qwen_sft/model --output outputs/my_hybrid
```

`run_pretrained.py` runs actual token-level GRPO against compiler rewards and
compares the base, SFT and token-GRPO models on three fixed short held-out
programs. Source/target SFT pairs are real PIE pairs, deduplicated and capped
by problem. Only 128 of the prepared 512 pairs are visited in the bundled run.
Increase steps and context budgets for a larger experiment with suitable compute.
The frozen reference copy requires memory for both actor and reference.

## Optimize one real program

Start the worker in one terminal:

```bash
coderl serve
```

In another terminal:

```bash
coderl optimize --tasks data/v2/test.jsonl --task-id p03796:s009177254 --checkpoint outputs/v2/grpo/policy.pt --model outputs/v2/qwen_sft/model --hybrid --attempts 3 --output outputs/one_program.json
```

Omit `--model --hybrid` to use just the bounded policy. Use `--model` alone for
unrestricted full-program model proposals. The graph passes compilation errors,
failed outputs and timing feedback into subsequent proposals, and preserves a
verified best program. Final selected benchmark programs are also saved as C++.

## Rebuild data

The prepared data is already included. For provenance or a clean re-preparation:

```bash
pip install gdown
gdown 1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr -O merged_tests.tar.gz
python scripts/prepare_v2.py --archive merged_tests.tar.gz
```

The v2 subset selector reads the bundled pinned parquet releases. The archive
is the original authors' merged CodeNet public and AlphaCode-generated test
archive, with its hash in `data/v2/protocol.json`. Preparation never synthesizes
replacement inputs or expected outputs. V2 held-out problems are excluded from
all v2 supervised and GRPO training. The real full pair counts are 58,863 train,
2,116 validation and 878 test.

## Reports and earlier experiment

- `outputs/v2/confirmation/`: final confirmed before/after programs, paired
  timings, summaries and comparison.
- `outputs/v2/hybrid/`: all 24 development evaluations and agent histories.
- `outputs/v2/grpo/`: 50,004 sampled actions, cached measurement accounting and
  initial/final policies.
- `outputs/v2/edit_sft/`: real-pair action labels and supervised checkpoint.
- `outputs/v2/qwen_sft/`: real pretrained SFT losses and trained adapter.
- `outputs/v2/pretrained/`: actual token-GRPO rollouts, adapter changes and the
  three-program model comparison.
- `outputs/test_results.txt`, `test_results.xml`: executed regression suite.
- `outputs/artifact_verification.json`: real-data hash and rollout audit.
- `outputs/verified/`, `docs/LEGACY_RESULTS.md`: original zero-improvement run.

`python scripts/run_experiment.py` remains the earlier experiment workflow, now
using the current worker and action schema. It does not reproduce the old timer
bit for bit. Optional Docker configuration is supplied; Docker was not available
in this environment and its build is not claimed as an executed verification.
