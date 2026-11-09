# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for candidates.py taxonomy resolution and validation (plain python3)."""

import os
import tempfile
import unittest

try:
    from triton.benchmarks.tools.candidates import (
        check,
        coverage,
        label_entry,
        load_taxonomy,
        resolve_masking,
        resolve_strategy,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    from candidates import (  # pyrefly: ignore [missing-import]
        check,
        coverage,
        label_entry,
        load_taxonomy,
        resolve_masking,
        resolve_strategy,
    )

_TAXONOMY_YAML = """\
version: 1
arch:
  gfx950:
    tlx_ops:
      - op: mm
        strategies:
          - name: stream_k
            kernel_patterns: [_iw_streamk_kernel]
            notes: "t"
          - name: inter_wave
            kernel_patterns: [_iw_]
            notes: "t"
          - name: register_resident
            kernel_patterns: [_register_kernel_impl]
            notes: "t"
        masking: [even, k_tail, mn_tail]
      - op: grouped_gemm
        default_strategy: quadrant_pipeline
        strategies:
          - name: quadrant_pipeline
            kernel_patterns: [_grouped_gemm_tile]
            notes: "t"
      - op: hstu_attn_dev
        default_strategy: ragged_fwd
        default_applies_to: fwd
        strategies:
          - name: ragged_fwd
            kernel_patterns: []
            notes: "t"
          - name: scheduled_bwd
            kernel_patterns: [schedule_bwd]
            notes: "t"
"""

_TARGETS_YAML = """\
version: 1
arch:
  gfx950:
    tlx_ops:
      - op: mm
        status: ready
      - op: grouped_gemm
        status: ready
      - op: hstu_attn_dev
        status: ready
"""


def _write_tmp(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class StrategyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = load_taxonomy(_write_tmp(_TAXONOMY_YAML))

    def test_torchtlx_resolves_by_op(self) -> None:
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "mm_torchtlx", "inductor_0"),
            "inductor",
        )

    def test_first_pattern_wins(self) -> None:
        # _iw_streamk_kernel matches both stream_k and inter_wave (_iw_).
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "mm", "_iw_streamk_kernel"),
            "stream_k",
        )
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "mm", "_iw_a16w16_8wave"),
            "inter_wave",
        )

    def test_launch_is_fallback_haystack(self) -> None:
        self.assertEqual(
            resolve_strategy(
                self.taxonomy, "gfx950", "mm", None, launch="_register_kernel_impl"
            ),
            "register_resident",
        )

    def test_default_applies_to_any(self) -> None:
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "grouped_gemm", "mystery"),
            "quadrant_pipeline",
        )

    def test_default_applies_to_mode(self) -> None:
        self.assertEqual(
            resolve_strategy(
                self.taxonomy, "gfx950", "hstu_attn_dev", "mystery", bwd=False
            ),
            "ragged_fwd",
        )
        self.assertEqual(
            resolve_strategy(
                self.taxonomy, "gfx950", "hstu_attn_dev", "mystery", bwd=True
            ),
            "unknown",
        )
        self.assertEqual(
            resolve_strategy(
                self.taxonomy,
                "gfx950",
                "hstu_attn_dev",
                "_tlx_gfx950_hstu_fa_schedule_bwd_kernel",
                bwd=True,
            ),
            "scheduled_bwd",
        )

    def test_unknown_op_and_kernel(self) -> None:
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "nope", "whatever"), "unknown"
        )
        self.assertEqual(
            resolve_strategy(self.taxonomy, "gfx950", "mm", "whatever"), "unknown"
        )


class MaskingTest(unittest.TestCase):
    _CFG = "triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8)"

    def test_gemm_even(self) -> None:
        spec = {"M": 256, "N": 512, "K": 128}
        self.assertEqual(resolve_masking("mm", spec, self._CFG), "even")

    def test_gemm_k_tail(self) -> None:
        spec = {"M": 256, "N": 512, "K": 100}
        self.assertEqual(resolve_masking("mm", spec, self._CFG), "k_tail")

    def test_gemm_mn_tail(self) -> None:
        spec = {"M": 200, "N": 512, "K": 128}
        self.assertEqual(resolve_masking("mm", spec, self._CFG), "mn_tail")

    def test_gemm_entry_format_and_unknown(self) -> None:
        spec = {"entry": ["256", "512", "128"]}
        self.assertEqual(resolve_masking("addmm", spec, self._CFG), "even")
        self.assertEqual(resolve_masking("mm", {"M": 1}, None), "unknown")
        self.assertEqual(
            resolve_masking("mm", {"M": 256, "N": 512, "K": 128}, "no blocks here"),
            "unknown",
        )

    def test_grouped_raggedness(self) -> None:
        self.assertEqual(
            resolve_masking("grouped_gemm", {"groups": [[8, 8, 8], [8, 8, 8]]}),
            "uniform",
        )
        self.assertEqual(
            resolve_masking("grouped_gemm", {"groups": [[8, 8, 8], [4, 8, 8]]}),
            "ragged",
        )
        self.assertEqual(
            resolve_masking("grouped_gemm_mxfp8", {"split_sizes": [128, 128]}),
            "uniform",
        )
        self.assertEqual(
            resolve_masking("grouped_gemm_mxfp8", {"split_sizes": [128, 64]}),
            "ragged",
        )
        self.assertEqual(resolve_masking("grouped_gemm", {}), "unknown")

    def test_attention_causal(self) -> None:
        self.assertEqual(resolve_masking("flash_attn", {"causal": True}), "causal")
        self.assertEqual(resolve_masking("flash_attn", {"causal": False}), "noncausal")
        self.assertEqual(
            resolve_masking("flash_attn_varlen", {"causal": True}), "packed_causal"
        )
        self.assertEqual(
            resolve_masking("flash_attn_varlen_backward", {"causal": False}),
            "packed_noncausal",
        )
        self.assertEqual(resolve_masking("flash_attn", {}), "unknown")

    def test_not_applicable(self) -> None:
        self.assertEqual(resolve_masking("kda_paged_prefill", {}), "n/a")


class LabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = load_taxonomy(_write_tmp(_TAXONOMY_YAML))

    def test_label_entry(self) -> None:
        entry = {
            "target": "mm",
            "kernel_name": "_register_kernel_impl",
            "input_spec": {"M": 256, "N": 512, "K": 128},
            "extra": {
                "launch": "_register_kernel_impl",
                "winner_config": MaskingTest._CFG,
            },
        }
        self.assertEqual(
            label_entry(self.taxonomy, "gfx950", entry),
            {"strategy": "register_resident", "masking": "even"},
        )


class CheckTest(unittest.TestCase):
    def test_valid_taxonomy(self) -> None:
        problems = check(_write_tmp(_TAXONOMY_YAML), _write_tmp(_TARGETS_YAML))
        self.assertEqual(problems, [])

    def test_rejects_bad_taxonomy(self) -> None:
        bad = _TAXONOMY_YAML.replace("version: 1", "version: 2")
        problems = check(_write_tmp(bad), _write_tmp(_TARGETS_YAML))
        self.assertTrue(problems)
        missing_op = _TARGETS_YAML + "      - op: ghost\n        status: ready\n"
        problems = check(_write_tmp(_TAXONOMY_YAML), _write_tmp(missing_op))
        self.assertTrue(any("ghost" in p for p in problems))


class CoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = load_taxonomy(_write_tmp(_TAXONOMY_YAML))

    def _entry(self, op, strategy, masking):
        return {
            "arch": "gfx950",
            "target": op,
            "extra": {"strategy": strategy, "masking": masking},
        }

    def test_coverage_counts_and_missing(self) -> None:
        entries = [
            self._entry("mm", "register_resident", "even"),
            self._entry("mm", "register_resident", "k_tail"),
            self._entry("mm", "unknown", "unknown"),
        ]
        report = coverage(self.taxonomy, entries)
        rep = report["gfx950"]["mm"]
        self.assertEqual(rep["strategies"]["register_resident"], 2)
        self.assertEqual(rep["strategies"]["unknown"], 1)
        self.assertIn("stream_k", rep["missing_strategies"])
        self.assertIn("mn_tail", rep["missing_masking"])
        self.assertNotIn("even", rep["missing_masking"])


if __name__ == "__main__":
    unittest.main()
