---
title: Proof Is Not Truth
emoji: "🔎"
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Proof Is Not Truth — Trace Explorer

Explore the recorded 96-episode benchmark run. Filter by prompt/tool configuration and task split, inspect per-episode evaluator outputs, and replay the agent's recorded tool actions.

**This is a static replay application.** It does not run inference, execute visitor-provided code, or expose the benchmark's local model runner.

The run is exploratory: Qwen2.5-7B-Instruct, greedy decoding, one seed (42), 16 cases and six prompt/tool configurations. The configurations are not a fully crossed factorial design, so comparisons are descriptive rather than clean causal estimates.

See the GitHub repository for the complete source, tests, raw traces, v2 regraded artifacts and reproduction instructions:

https://github.com/Vedant2100/proof-is-not-truth
