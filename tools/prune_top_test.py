# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for prune_top.py grouping and filtering (plain python3)."""

import unittest

try:
    from triton.benchmarks.tools.prune_top import (
        candidate_key,
        cap_shapes,
        config_stem,
        filter_top_configs,
        mode_key,
        shape_key,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    from prune_top import (  # pyrefly: ignore [missing-import]
        candidate_key,
        cap_shapes,
        config_stem,
        filter_top_configs,
        mode_key,
        shape_key,
    )


def artifact(source, target, case, kernel="k", spec=None, latency=None, extra=None):
    return {
        "arch": "gfx950",
        "source": source,
        "target": target,
        "case": case,
        "kernel_name": kernel,
        "input_spec": spec or {},
        "compile": {},
        "extra": extra or {},
        "latency_ms": latency,
    }


class StemTest(unittest.TestCase):
    def test_strips_dedup_suffix(self) -> None:
        self.assertEqual(config_stem("addmm_512_register__k12"), "addmm_512_register")

    def test_plain_case_unchanged(self) -> None:
        self.assertEqual(config_stem("mm_fwd_1_2_3"), "mm_fwd_1_2_3")


class TopConfigsTest(unittest.TestCase):
    def test_keeps_fastest_five(self) -> None:
        arts = [
            artifact(
                "tlx_tutorials", "t", f"c__k{i}" if i else "c", latency=float(10 - i)
            )
            for i in range(8)
        ]
        kept, dropped = filter_top_configs(arts, 5)
        self.assertEqual(len(kept), 5)
        self.assertEqual(len(dropped), 3)
        self.assertEqual(
            sorted(a["latency_ms"] for a in kept), [3.0, 4.0, 5.0, 6.0, 7.0]
        )

    def test_small_groups_untouched(self) -> None:
        arts = [artifact("s", "t", f"c__k{i}", latency=1.0) for i in range(3)]
        kept, dropped = filter_top_configs(arts, 5)
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_unmeasured_sorts_last(self) -> None:
        arts = [artifact("s", "t", f"c__k{i}", latency=None) for i in range(6)]
        arts[0]["latency_ms"] = 1.0
        kept, dropped = filter_top_configs(arts, 5)
        self.assertIn(arts[0], kept)
        self.assertEqual(len(dropped), 1)

    def test_kernels_group_independently(self) -> None:
        arts = [
            artifact("s", "t", "c", kernel="fwd", latency=9.0),
            artifact("s", "t", "c__k1", kernel="bwd", latency=8.0),
            artifact("s", "t", "c__k2", kernel="bwd", latency=7.0),
            artifact("s", "t", "c__k3", kernel="bwd", latency=6.0),
            artifact("s", "t", "c__k4", kernel="bwd", latency=5.0),
        ]
        kept, dropped = filter_top_configs(arts, 3)
        self.assertIn(arts[0], kept)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["latency_ms"], 8.0)


class ShapeCapTest(unittest.TestCase):
    def _shape(self, idx, latency, **spec):
        base = {"M": idx, "N": 8, "K": 8, "dtype": "fp16"}
        base.update(spec)
        return artifact("tlx_ops", "mm", f"mm_{idx}", spec=base, latency=latency)

    def test_keeps_longest_and_median(self) -> None:
        arts = [self._shape(i, float(i)) for i in range(1, 5)]
        kept, dropped = cap_shapes(arts, 2)
        self.assertEqual({a["case"] for a in kept}, {"mm_4", "mm_2"})
        self.assertEqual(len(dropped), 2)

    def test_keeps_longest_and_true_median(self) -> None:
        arts = [self._shape(i, float(i)) for i in range(1, 6)]
        kept, dropped = cap_shapes(arts, 2)
        self.assertEqual({a["case"] for a in kept}, {"mm_5", "mm_3"})
        self.assertEqual(len(dropped), 3)

    def test_modes_capped_separately(self) -> None:
        arts = [self._shape(i, float(i), causal=False) for i in range(1, 4)]
        arts += [self._shape(i, float(i), causal=True) for i in range(1, 4)]
        kept, _ = cap_shapes(arts, 2)
        by_mode = {}
        for a in kept:
            by_mode.setdefault(a["input_spec"]["causal"], []).append(a["case"])
        self.assertEqual(len(by_mode[False]), 2)
        self.assertEqual(len(by_mode[True]), 2)

    def test_unmeasured_shapes_exempt(self) -> None:
        arts = [self._shape(i, float(i)) for i in range(1, 4)]
        arts.append(self._shape(9, None))
        kept, dropped = cap_shapes(arts, 2)
        self.assertIn("mm_9", {a["case"] for a in kept})
        self.assertEqual(len(kept), 3)

    def test_shape_latency_sums_kernels(self) -> None:
        # Shape 1: one slow kernel (10ms). Shape 2: two 6ms kernels = 12ms.
        a1 = self._shape(1, 10.0)
        b1 = self._shape(2, 6.0)
        b2 = self._shape(2, 6.0)
        b2["kernel_name"] = "helper"
        c1 = self._shape(3, 1.0)
        kept, _ = cap_shapes([a1, b1, b2, c1], 2)
        cases = {a["case"] for a in kept}
        self.assertIn("mm_2", cases)
        self.assertIn("mm_1", cases)
        self.assertNotIn("mm_3", cases)

    def test_few_shapes_untouched(self) -> None:
        arts = [self._shape(i, float(i)) for i in range(1, 3)]
        kept, dropped = cap_shapes(arts, 2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_strategies_capped_separately(self) -> None:
        arts = []
        for strategy in ("register_resident", "persistent"):
            for i in range(1, 4):
                a = self._shape(i, float(i))
                a["case"] = f"{strategy}_{i}"
                a["extra"] = {"strategy": strategy, "masking": "even"}
                arts.append(a)
        kept, _ = cap_shapes(arts, 2)
        by_strategy = {}
        for a in kept:
            by_strategy.setdefault(a["extra"]["strategy"], []).append(a["case"])
        self.assertEqual(len(by_strategy["register_resident"]), 2)
        self.assertEqual(len(by_strategy["persistent"]), 2)

    def test_masking_capped_separately(self) -> None:
        arts = []
        for masking in ("even", "k_tail"):
            for i in range(1, 4):
                a = self._shape(i, float(i))
                a["case"] = f"{masking}_{i}"
                a["extra"] = {"strategy": "register_resident", "masking": masking}
                arts.append(a)
        kept, _ = cap_shapes(arts, 2)
        by_masking = {}
        for a in kept:
            by_masking.setdefault(a["extra"]["masking"], []).append(a["case"])
        self.assertEqual(len(by_masking["even"]), 2)
        self.assertEqual(len(by_masking["k_tail"]), 2)


class KeyTest(unittest.TestCase):
    def test_mode_key_defaults(self) -> None:
        self.assertEqual(mode_key(artifact("s", "t", "c")), (False, False))

    def test_shape_key_ignores_modes(self) -> None:
        a = artifact("s", "t", "c", spec={"M": 1, "causal": True, "bwd": False})
        b = artifact("s", "t", "c", spec={"M": 1, "causal": False, "bwd": False})
        self.assertEqual(shape_key(a), shape_key(b))

    def test_candidate_key_defaults(self) -> None:
        self.assertEqual(candidate_key(artifact("s", "t", "c")), (None, None))
        a = artifact("s", "t", "c", extra={"strategy": "x", "masking": "y"})
        self.assertEqual(candidate_key(a), ("x", "y"))


if __name__ == "__main__":
    unittest.main()
