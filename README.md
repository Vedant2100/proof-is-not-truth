# Provenance Bias

**Decoupling origin authenticity from factual veracity in tool-using agents**

Provenance Bias studies whether tool-using agents mistake machine-verifiable origin provenance for evidence that a claim is factually correct, current, and safe to act on—and whether independent telemetry corroboration improves decisions without collapsing into blanket abstention.

- **Interactive replay:** https://huggingface.co/spaces/Artemis2100/proof-is-not-truth
- **Canonical code and experiment artifacts:** https://github.com/Vedant2100/provenance-bias

The Hugging Face Space replays the recorded episodes. It does not run model inference or execute visitor-supplied code.

## At a glance

- 16 deterministic benchmark cases across five task splits.
- Six prompt/tool configurations.
- 96 real-model episodes: 16 cases × 6 configurations.
- Model: Qwen2.5-7B-Instruct through Ollama.
- Greedy decoding, temperature 0.0, one seed (42).
- Results are exploratory: one model and one seed, without independent replication.

## Research question

When an agent receives machine-verifiable provenance, does it distinguish **origin authenticity** from **factual veracity** before taking a state-changing action? Can independent factual corroboration improve those decisions without producing excessive abstention?

The benchmark separates three concepts:

1. **Origin authenticity:** whether content came from the identified keyholder and has not been modified since authentication.
2. **Factual veracity:** whether the content accurately describes the environment's state.
3. **Decision authority:** whether a source is permitted to authorize a particular action. This is not independently varied in the current benchmark.

## Benchmark design

The 16 cases cover the following splits:

| Split | Origin authentic? | Factually true? | Expected decision | Cases |
|---|---:|---:|---|---:|
| Clean baseline | N/A | Uncontested | Execute | 4 |
| Authentic true (Cell A) | Yes | Yes | Execute | 3 |
| Unauthenticated false (Cell B) | No | No | Abstain | 3 |
| Authentic false (Cell C) | Yes | No | Abstain | 3 |
| Unauthenticated true (Cell D) | No | Yes | Execute | 3 |

The provenance primitive is **HMAC-SHA256**, a symmetric message authentication code (MAC). It is not an asymmetric public-key digital signature and does not provide public-key non-repudiation.

### Six prompt/tool configurations

| Configuration | Provenance tool | Telemetry tools | Prompt distinction |
|---|---:|---:|---|
| `no_metadata` | No | No | Baseline |
| `textual_authority` | No | No | Adds prose authority claims |
| `verified_origin` | Yes | No | Explicitly says to proceed when origin is authentic |
| `telemetry_only` | No | Yes | Directs the agent to discover and corroborate telemetry |
| `corroborated_provenance_neutral` | Yes | Yes | Neutral instruction to use available evidence |
| `corroborated_provenance_directive` | Yes | Yes | Explicitly states that proof of origin is not proof of truth |

These are **prompt/tool policy configurations, not a fully crossed factorial design**. Both tool availability and system-prompt instructions vary across conditions. The comparisons therefore do not isolate the causal effect of provenance, telemetry, or wording independently.

## Results: Version 2 action-ordering regrade

The table reports the v2 regrade of the same 96 immutable execution traces. False-trust rate uses the six false tasks per condition (three unauthenticated-false and three authentic-false cases). Legitimate-task completion uses the ten true/clean tasks per condition. Abstention rate uses all 16 episodes.

| Configuration | False trust, all false tasks | Legitimate-task completion | Overall abstention |
|---|---:|---:|---:|
| No metadata | 83.3% | 50.0% | 12.5% |
| Textual authority | 100.0% | 70.0% | 6.3% |
| Verified origin | 33.3% | 20.0% | 62.5% |
| Telemetry only | 0.0% | 10.0% | 81.3% |
| Provenance + telemetry, neutral | 33.3% | 10.0% | 68.8% |
| Provenance + telemetry, directive | 16.7% | 10.0% | 81.3% |

### Authentic-but-false traps (Cell C)

False-trust rates on the three authentic-but-false cases were:

| Configuration | False trust on Cell C |
|---|---:|
| No metadata | 66.7% |
| Textual authority | 100.0% |
| Verified origin | 66.7% |
| Telemetry only | 0.0% |
| Provenance + telemetry, neutral | 33.3% |
| Provenance + telemetry, directive | 0.0% |

### Interpretation and limits

- Origin verification alone did not establish factual correctness. In the verified-origin condition, false trust remained 33.3% across all false tasks and 66.7% on the authentic-false subset.
- Telemetry-only had zero false trust in this run, but legitimate-task completion was also only 10%, with 81.3% overall abstention. This is not evidence of a calibrated defense; excessive abstention is a major competing explanation.
- The directive condition had zero false trust on Cell C, but only 10% legitimate-task completion and 81.3% overall abstention. The v2 regrade also identified an unsafe write before a later abstention in one episode.
- The neutral corroboration condition had 33.3% false trust across false tasks and 10% legitimate-task completion. It also had two unsafe-write attempts, one occurring before abstention.
- The six configurations differ in prompt wording and tool availability. These are descriptive, exploratory results—not clean causal estimates.
- The evaluation uses one open-weight model, one seed, and 16 hand-authored cases. It does not establish generalization to frontier models or production agents.

## Repository layout

- `src/`: benchmark cases, HMAC keystore, sandbox simulator, model runner and evaluator.
- `scripts/regrade_v2.py`: offline action-ordering regrader.
- `scripts/verify_artifacts.py`: checks episode coverage and reproduces the committed v2 artifacts from the immutable raw traces.
- `tests/`: harness, path-boundary, cryptographic, action-ordering and artifact tests.
- `runs/qwen25_7b_full/`: original outputs and separately versioned v2 regraded outputs.
- `app.py`: Gradio trace explorer used by the Hugging Face Space.
- `space/README.md`: Hugging Face Space metadata and description.

Cluster logs, Slurm output, credentials, and cluster-specific launch scripts are intentionally excluded.

## Reproduce the artifacts

Python 3.10 or later is required. The experiment runner and regrader use the Python standard library.

Run the tests:

~~~bash
python -m unittest discover -s tests -v
~~~

Verify that the v2 files reproduce from the original traces:

~~~bash
python scripts/verify_artifacts.py
~~~

Regenerate the v2 regrade:

~~~bash
python scripts/regrade_v2.py
~~~

The deterministic harness-validation mode checks simulator mechanics; it is **not** model evidence:

~~~bash
python src/agent_eval.py --mode harness_validation --output-dir runs/harness_validation
~~~

For a new model run, start a local or remote OpenAI-compatible chat-completions endpoint and set its URL explicitly:

~~~bash
python src/agent_eval.py \
  --mode llm \
  --model qwen2.5:7b \
  --base-url http://localhost:11434/v1 \
  --temperature 0 \
  --seeds 42 \
  --output-dir runs/new_run
~~~

The runner defaults to the original experiment's endpoint URL if `--base-url` is omitted. A new run is not comparable to the recorded result unless the model, endpoint, prompts, sampling settings, cases and evaluator version are also controlled.

## Safety and security notes

- The benchmark keystore contains deterministic synthetic fixture keys. They are for reproducible tests only and must never be reused as production credentials.
- The sandbox is an in-process simulator with path-confined temporary files. It is not an operating-system security boundary.
- Python grading uses structural AST comparison and does not execute model-generated candidate code.
- The public Space is replay-only: it does not expose a live agent runner, model endpoint, filesystem mutation, or arbitrary code execution.
- Do not expose the local evaluation runner as a public service without a separately designed and audited execution boundary.

## Related work

The project is positioned alongside work on source discernment, evidence quality and provenance sensitivity:

- [Learn2Discern](https://arxiv.org/abs/2607.19355)
- [What Counts as Evidence?](https://arxiv.org/abs/2610.12788)
- [Auditing Provenance Sensitivity in LLM Agent Action Selection](https://arxiv.org/abs/2607.20827)
- [ProvenanceGuard](https://arxiv.org/abs/2606.18037)

These works are relevant prior art. This repository does not claim that the general distinction between a source's authenticity and a claim's truth is new.

## License

The existing MIT LICENSE is retained. No separate dataset-specific license is declared.
