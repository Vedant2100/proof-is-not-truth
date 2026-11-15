"""Tests for the public, versioned benchmark result files."""
from __future__ import annotations
import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from cases import get_all_cases
RUNS = ROOT / "runs" / "qwen25_7b_full"


class TestPublishedArtifacts(unittest.TestCase):
    def test_raw_run_has_complete_unique_coverage(self):
        traces = json.loads((RUNS / "evaluation_traces.json").read_text(encoding="utf-8"))
        expected = {
            (case.case_id, condition, 42)
            for case in get_all_cases()
            for condition in (
                "no_metadata", "textual_authority", "verified_origin", "telemetry_only",
                "corroborated_provenance_neutral", "corroborated_provenance_directive",
            )
        }
        actual = {(x["case_id"], x["condition"], x.get("seed", 42)) for x in traces}
        self.assertEqual(len(traces), 96)
        self.assertEqual(actual, expected)

    def test_v2_results_have_expected_denominators(self):
        with (RUNS / "evaluation_results_regraded_v2.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads((RUNS / "metrics_summary_regraded_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 96)
        self.assertEqual(set(summary["by_condition"]), {
            "no_metadata", "textual_authority", "verified_origin", "telemetry_only",
            "corroborated_provenance_neutral", "corroborated_provenance_directive",
        })
        for metrics in summary["by_condition"].values():
            self.assertEqual(metrics["total_episodes"], 16)


if __name__ == "__main__":
    unittest.main()
