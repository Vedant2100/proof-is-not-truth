from __future__ import annotations
import json
import sys
from pathlib import Path
import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cases import get_all_cases

RUNS = ROOT / "runs" / "qwen25_7b_full"
RESULTS = pd.read_csv(RUNS / "evaluation_results_regraded_v2.csv")
TRACES = json.loads((RUNS / "evaluation_traces_regraded_v2.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((RUNS / "metrics_summary_regraded_v2.json").read_text(encoding="utf-8"))
CASE_BY_ID = {case.case_id: case for case in get_all_cases()}
CONDITIONS = [
    "no_metadata", "textual_authority", "verified_origin", "telemetry_only",
    "corroborated_provenance_neutral", "corroborated_provenance_directive",
]
SPLITS = ["clean_baseline", "authentic_true", "unauth_false", "authentic_false_trap", "unauth_true"]
DISPLAY_COLUMNS = [
    "case_id", "split", "condition", "correct_decision", "false_trust", "abstained",
    "safe_abstention", "unsafe_write_attempted", "unsafe_write_before_abstain",
    "post_action_abstention", "tool_call_count",
]
TRACE_BY_KEY = {f"{x['case_id']} :: {x['condition']}": x for x in TRACES}


def make_metrics_frame() -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        metrics = SUMMARY["by_condition"][condition]
        trap = SUMMARY["trap_analysis"][condition]
        rows.append({
            "condition": condition,
            "false_trust_rate": metrics["false_trust_rate"],
            "legitimate_task_completion": metrics["legitimate_task_completion"],
            "abstention_rate": metrics["abstention_rate"],
            "cell_c_false_trust": trap["false_trust_under_trap"],
            "unsafe_writes_attempted": metrics["unsafe_writes_attempted"],
            "episodes": metrics["total_episodes"],
        })
    return pd.DataFrame(rows)


METRICS_FRAME = make_metrics_frame()


def filtered_results(condition: str, split: str) -> pd.DataFrame:
    frame = RESULTS.copy()
    if condition != "All":
        frame = frame[frame["condition"] == condition]
    if split != "All":
        frame = frame[frame["split"] == split]
    return frame[DISPLAY_COLUMNS].reset_index(drop=True)


def render_episode(key: str | None):
    if not key or key not in TRACE_BY_KEY:
        return "No episode selected for the current filters.", {}, []
    item = TRACE_BY_KEY[key]
    case = CASE_BY_ID[item["case_id"]]
    details = (
        f"### {case.case_id}\n\n"
        f"**Condition:** {item['condition']}  \n"
        f"**Split:** {case.split}  \n"
        f"**Expected decision:** {case.expected_decision}\n\n"
        f"**Task:** {case.user_request}"
    )
    return details, item["grade"], item["trace"]


def update_view(condition: str, split: str):
    selected_items = [
        item for item in TRACES
        if (condition == "All" or item["condition"] == condition)
        and (split == "All" or item["split"] == split)
    ]
    choices = [f"{x['case_id']} :: {x['condition']}" for x in selected_items]
    selected = choices[0] if choices else None
    details, grade, trace = render_episode(selected)
    return filtered_results(condition, split), gr.update(choices=choices, value=selected), details, grade, trace


initial_choices = list(TRACE_BY_KEY)
initial_selection = initial_choices[0] if initial_choices else None
initial_details, initial_grade, initial_trace = render_episode(initial_selection)

with gr.Blocks(title="Proof Is Not Truth — Trace Explorer") as demo:
    gr.Markdown(
        "# Proof Is Not Truth\n"
        "### Interactive replay of the 96-episode Qwen2.5-7B-Instruct evaluation\n\n"
        "**Replay only:** this Space displays recorded actions and evaluator outputs. "
        "It does not call a model, execute submitted code, or mutate a filesystem.\n\n"
        "The run uses one model, one greedy seed (42), and 16 cases across six prompt/tool "
        "configurations. Treat the results as exploratory, not as a replicated or causal study."
    )
    with gr.Row():
        gr.Number(value=96, label="Recorded episodes", interactive=False)
        gr.Number(value=16, label="Benchmark cases", interactive=False)
        gr.Number(value=6, label="Prompt/tool configurations", interactive=False)
    gr.Markdown("## Condition-level results")
    gr.Markdown(
        "False trust uses six false tasks per condition; legitimate-task completion uses ten "
        "clean/true tasks; abstention uses all 16 episodes."
    )
    gr.Dataframe(value=METRICS_FRAME, interactive=False, wrap=True, label="Version 2 metrics")
    with gr.Row():
        gr.BarPlot(
            value=METRICS_FRAME[["condition", "false_trust_rate"]],
            x="condition", y="false_trust_rate",
            title="False trust across all false tasks",
        )
        gr.BarPlot(
            value=METRICS_FRAME[["condition", "legitimate_task_completion"]],
            x="condition", y="legitimate_task_completion",
            title="Legitimate-task completion",
        )
    gr.Markdown("## Inspect individual episodes")
    with gr.Row():
        condition_filter = gr.Dropdown(choices=["All"] + CONDITIONS, value="All", label="Condition")
        split_filter = gr.Dropdown(choices=["All"] + SPLITS, value="All", label="Task split")
    episode_select = gr.Dropdown(choices=initial_choices, value=initial_selection, label="Episode")
    results_table = gr.Dataframe(
        value=filtered_results("All", "All"), interactive=False, wrap=True,
        max_height=420, label="Filtered episode results",
    )
    episode_details = gr.Markdown(value=initial_details)
    with gr.Row():
        grade_json = gr.JSON(value=initial_grade, label="Version 2 evaluator output")
        trace_json = gr.JSON(value=initial_trace, label="Recorded agent trace")
    condition_filter.change(
        update_view, inputs=[condition_filter, split_filter],
        outputs=[results_table, episode_select, episode_details, grade_json, trace_json],
    )
    split_filter.change(
        update_view, inputs=[condition_filter, split_filter],
        outputs=[results_table, episode_select, episode_details, grade_json, trace_json],
    )
    episode_select.change(
        render_episode, inputs=episode_select,
        outputs=[episode_details, grade_json, trace_json],
    )
    gr.Markdown(
        "## Methodological caveats\n"
        "- Conditions vary both tool access and system-prompt policy; they are not a fully crossed factorial design.\n"
        "- HMAC-SHA256 is a symmetric MAC, not an asymmetric digital signature.\n"
        "- Zero false trust must be read alongside legitimate-task completion and abstention.\n"
        "- One model and one seed do not establish generalization."
    )

if __name__ == "__main__":
    demo.launch()
