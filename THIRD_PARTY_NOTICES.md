# Data attribution

Programs are actual competitive-programming C++ submissions from the PIE dataset,
which derives from IBM Project CodeNet. This project did not synthesize benchmark
programs, targets, or expected outputs. V2 also uses the authors' merged archive,
which contains public CodeNet cases and AlphaCode-generated dataset cases.
These are existing benchmark cases, not newly synthesized here.
Unit-test fixtures in `tests/` are handcrafted
software tests and are not part of the experimental dataset.

- Original PIE: https://github.com/madaan/pie-perf
- Updated authors' repository: https://github.com/LearningOpt/pie
- Original paper: https://arxiv.org/abs/2302.07867
- IBM Project CodeNet: https://github.com/IBM/Project_CodeNet
- Downloaded pair release: https://huggingface.co/datasets/stablegradients/pie-gem5-pairs
  (revision `8bbdbe1fb21aa0c04bca171421a6c5d8513f6644`). Its dataset card declares MIT.
  A complete copy of the card accompanies the data at `data/raw/pie_card.md`.
- Public test mirror: https://huggingface.co/datasets/rootacess/pie-perf-testcases
  (revision `f47671d14853791fcee46bc29a4e2b47fa0ebeb6`). The mirror does not publish a
  dataset card/license declaration. Preserve original CodeNet/online-judge terms;
  the project's MIT license does not apply to the mirrored programs or tests.

The downloader pins both revisions. `data/manifest.json` records the SHA-256 of
each downloaded file. Human target edits are used only for SFT; they are never
given to the GRPO actor, planner, or evaluation graph.

## V2 benchmark and pretrained model

- Original authors' merged tests: https://github.com/LearningOpt/pie . Download
  identifier `1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr`; archive SHA-256
  `02f21225ba963ec9de89b3a5e42f2e9647d8333b717a02cf5bd8b4d088b8309c`.
  The authors describe these as merged public CodeNet and generated AlphaCode
  tests. Original dataset/online-judge terms still apply.
- Base model: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct , pinned to
  `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`. The Apache-2.0 LICENSE and model
  card are preserved in `data/model_provenance/`. Base weights are downloaded
  by `scripts/download_model.py`; trained LoRA adapters are bundled.
- Derived v2 files are hashed in `data/v2/manifest.json`. Sampling rules and
  archive provenance are in `data/v2/protocol.json`.

## Citations

Madaan et al. *Learning Performance-Improving Code Edits*, arXiv:2302.07867.

Puri et al. *CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity
of Coding Tasks*, arXiv:2105.12655.

Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
Language Models*, arXiv:2402.03300 (GRPO).
