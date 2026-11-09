# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Regression tests for export_tritonbench.py target naming.

Plain-python unittest (run: `python3 export_tritonbench_test.py`). Guards the
fix for repeated ops in targets.yaml silently overwriting each other's case
dirs (fp8_gemm_rowwise fast_accum vs prefill shared target dir
`tritonbench/fp8_gemm_rowwise/`, leaving stale manifest entries).
"""

import unittest

try:
    from triton.benchmarks.tools.export_tritonbench import (
        entry_target,
        latency_for,
        parse_run_latencies,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    from export_tritonbench import (  # pyrefly: ignore [missing-import]
        entry_target,
        latency_for,
        parse_run_latencies,
    )


class EntryTargetTest(unittest.TestCase):
    def test_unique_op_keeps_plain_name(self) -> None:
        entry = {"op": "softmax", "args": "--op softmax --logging-group softmax"}
        self.assertEqual(entry_target(entry, 0, {"softmax": 1}), "softmax")

    def test_repeated_op_disambiguates_by_logging_group(self) -> None:
        counts = {"fp8_gemm_rowwise": 2}
        fast = {
            "op": "fp8_gemm_rowwise",
            "args": "--op fp8_gemm_rowwise --logging-group fp8_gemm_rowwise_fast_accum",
        }
        prefill = {
            "op": "fp8_gemm_rowwise",
            "args": "--op fp8_gemm_rowwise --logging-group fp8_gemm_rowwise_fast_accum_prefill",
        }
        self.assertEqual(entry_target(fast, 0, counts), "fp8_gemm_rowwise_fast_accum")
        self.assertEqual(
            entry_target(prefill, 1, counts),
            "fp8_gemm_rowwise_fast_accum_prefill",
        )

    def test_repeated_op_without_logging_group_falls_back_to_index(self) -> None:
        counts = {"gemm": 2}
        self.assertEqual(entry_target({"op": "gemm"}, 3, counts), "gemm__e3")
        self.assertEqual(entry_target({"op": "gemm"}, 4, counts), "gemm__e4")


SAMPLE_LOG = """\
some warmup noise
          (M, N, K)    hw_roofline      _triton-latency    _triton-tflops
-------------------  -------------  -------------------  ----------------
  (128, 1280, 1024)           2612    0.004361 (±1.48%)         76.9391
  (128, 1280, 3584)           2612    0.008094 (±0.30%)        145.097
trailing noise
"""


class ParseRunLatenciesTest(unittest.TestCase):
    def test_parses_single_column(self) -> None:
        table = parse_run_latencies(SAMPLE_LOG)
        self.assertEqual(list(table), ["_triton"])
        self.assertEqual(table["_triton"], [0.004361, 0.008094])

    def test_no_table_yields_empty(self) -> None:
        self.assertEqual(parse_run_latencies("no results here\n"), {})

    def test_latency_for_matches_fn_and_row(self) -> None:
        table = parse_run_latencies(SAMPLE_LOG)
        self.assertEqual(latency_for(table, "_triton", "1"), 0.008094)
        self.assertIsNone(latency_for(table, "_triton", "9"))
        # Single-column fallback when the fn name differs.
        self.assertEqual(latency_for(table, "other_fn", "0"), 0.004361)
        self.assertIsNone(latency_for({}, "_triton", "0"))


if __name__ == "__main__":
    unittest.main()
