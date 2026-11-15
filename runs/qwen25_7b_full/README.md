# Qwen2.5-7B-Instruct full run

This directory contains the immutable 96-episode model run and its separately versioned Version 2 action-ordering regrade.

- 16 benchmark cases × 6 prompt/tool configurations = 96 episodes.
- Model: Qwen2.5-7B-Instruct via Ollama.
- Sampling: greedy decoding, temperature 0.0, seed 42.
- Original files: evaluation_results.csv, evaluation_traces.json, metrics_summary.json.
- Regraded files: evaluation_results_regraded_v2.csv, evaluation_traces_regraded_v2.json, metrics_summary_regraded_v2.json.

The v2 outputs are regenerated from the original traces by scripts/regrade_v2.py. Do not overwrite the original files when regrading. Run python scripts/verify_artifacts.py to verify that the committed v2 artifacts reproduce byte-for-byte.

The experiment is exploratory and single-model/single-seed. The six prompt/tool configurations are not a fully crossed factorial design.
