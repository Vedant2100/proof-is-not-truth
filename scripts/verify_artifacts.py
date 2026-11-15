"""Standalone integrity checks for the committed experiment artifacts."""
from __future__ import annotations
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, SCRIPTS = ROOT / "src", ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from cases import get_all_cases
from regrade_v2 import regrade_traces

RUNS = ROOT / "runs" / "qwen25_7b_full"
RAW_TRACES = RUNS / "evaluation_traces.json"
V2_CSV = RUNS / "evaluation_results_regraded_v2.csv"
V2_TRACES = RUNS / "evaluation_traces_regraded_v2.json"
V2_SUMMARY = RUNS / "metrics_summary_regraded_v2.json"


def verify() -> None:
    cases = get_all_cases()
    raw = json.loads(RAW_TRACES.read_text(encoding="utf-8"))
    v2 = json.loads(V2_TRACES.read_text(encoding="utf-8"))
    with V2_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_conditions = {
        "no_metadata", "textual_authority", "verified_origin", "telemetry_only",
        "corroborated_provenance_neutral", "corroborated_provenance_directive",
    }
    case_ids = {case.case_id for case in cases}
    raw_keys = {(x["case_id"], x["condition"], x.get("seed", 42)) for x in raw}
    v2_keys = {(x["case_id"], x["condition"], x.get("seed", 42)) for x in v2}
    csv_keys = {(x["case_id"], x["condition"], int(x["seed"])) for x in rows}

    assert len(cases) == 16, f"Expected 16 cases, got {len(cases)}"
    assert len(raw) == len(v2) == len(rows) == 96, "Expected 96 raw, v2 and CSV episodes"
    assert {x["condition"] for x in raw} == expected_conditions
    assert len(raw_keys) == 96, "Raw traces contain duplicate case/condition/seed keys"
    assert raw_keys == v2_keys == csv_keys, "Raw, v2 trace and CSV coverage differs"
    assert {x["case_id"] for x in raw} == case_ids, "Raw case coverage differs from cases.py"

    with tempfile.TemporaryDirectory(prefix="pnt_verify_") as temp:
        regrade_traces(RAW_TRACES, Path(temp))
        for filename in (V2_CSV.name, V2_TRACES.name, V2_SUMMARY.name):
            assert (Path(temp) / filename).read_bytes() == (RUNS / filename).read_bytes(), (
                f"Committed artifact does not reproduce: {filename}"
            )
    summary = json.loads(V2_SUMMARY.read_text(encoding="utf-8"))
    assert set(summary["by_condition"]) == expected_conditions
    assert all(x["total_episodes"] == 16 for x in summary["by_condition"].values())
    print("PASS: 16 cases × 6 conditions = 96 unique episodes.")
    print("PASS: v2 CSV, traces and metrics reproduce byte-for-byte from immutable raw traces.")


if __name__ == "__main__":
    verify()
