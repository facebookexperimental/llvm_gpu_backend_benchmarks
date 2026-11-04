# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for collect_metadata.py (plain python3)."""

import json
import os
import tempfile
import unittest

try:
    from triton.benchmarks.tools.collect_metadata import (
        case_record,
        collect,
        group_key,
        kernel_record,
        main,
        problem_flops,
    )
except ImportError:
    # Plain python3 (no buck packaging): the module lives next to this test.
    # pyrefly: ignore [missing-import]
    from collect_metadata import (
        case_record,
        collect,
        group_key,
        kernel_record,
        main,
        problem_flops,
    )

_LLIR = """\
; ModuleID = 't_kernel'
source_filename = "t_kernel"
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @my_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %n, float %s, ptr addrspace(1) %g, ptr addrspace(1) %p) {
  ret void
}
"""


def _launch():
    return {
        "grid": [2, 1, 1],
        "block": {"num_warps": 4},
        "device": "cuda",
        "args": [
            {
                "name": "a",
                "kind": "tensor",
                "shape": [4],
                "dtype": "float32",
                "stride": [1],
                "role": "in",
            },
            {
                "name": "b",
                "kind": "tensor",
                "shape": [4],
                "dtype": "float32",
                "stride": [1],
                "role": "out",
            },
            {"name": "n", "kind": "int", "value": 8},
            {"name": "s", "kind": "float", "value": 1.5},
        ],
    }


def _write_case(root, source, target, case, llir=_LLIR, launch="dflt", spec=None):
    cdir = os.path.join(root, "gfx950", source, target, case)
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "kernel.llir"), "w") as f:
        f.write(llir)
    compile_info = {}
    if launch == "dflt":
        compile_info["launch"] = _launch()
    elif launch is not None:
        compile_info["launch"] = launch
    if spec is not None:
        compile_info["input_spec"] = spec
    with open(os.path.join(cdir, "compile.json"), "w") as f:
        json.dump(compile_info, f)
    return cdir


_MM_SPEC = {"M": 4, "N": 8, "K": 16}
_FLASH_SPEC = {
    "batch": 2,
    "heads": 4,
    "context": 64,
    "head_dim": 32,
    "causal": True,
    "bwd": False,
}


class FlopsTest(unittest.TestCase):
    def test_gemm(self) -> None:
        flops, formula = problem_flops("tlx_ops", "mm", _MM_SPEC)
        self.assertEqual((flops, formula), (1024, "gemm"))

    def test_gemm_batch(self) -> None:
        spec = {"B": 4, "M": 512, "N": 512, "K": 512}
        flops, formula = problem_flops("tlx_tutorials", "amd_bmm", spec)
        self.assertEqual((flops, formula), (2 * 4 * 512**3, "gemm"))

    def test_flash_alias_keys(self) -> None:
        spec = {"B": 2, "H": 8, "S": 1024, "D": 128, "causal": True}
        flops, formula = problem_flops("tlx_tutorials", "amd_fa_cluster", spec)
        self.assertEqual(
            (flops, formula), (4 * 2 * 8 * 1024**2 * 128 // 2, "flash-attn")
        )

    def test_flash_causal(self) -> None:
        flops, formula = problem_flops("tlx_ops", "flash_attn", _FLASH_SPEC)
        self.assertEqual((flops, formula), (2097152, "flash-attn"))

    def test_flash_bwd(self) -> None:
        spec = dict(_FLASH_SPEC, causal=False, bwd=True)
        flops, _formula = problem_flops("tlx_ops", "flash_attn", spec)
        self.assertEqual(flops, 4194304 * 5 // 2)

    def test_varlen(self) -> None:
        spec = {"heads": 2, "head_dim": 16, "lengths": [4, 8]}
        flops, formula = problem_flops("tlx_ops", "flash_attn_varlen", spec)
        self.assertEqual((flops, formula), (4 * 2 * 16 * (16 + 64), "varlen-attn"))

    def test_grouped(self) -> None:
        spec = {"groups": [[2, 4, 8], [1, 4, 8]]}
        flops, formula = problem_flops("tlx_ops", "grouped_gemm", spec)
        self.assertEqual((flops, formula), (2 * 64 + 2 * 32, "grouped-gemm"))

    def test_mxfp8_grouped(self) -> None:
        spec = {"n": 4, "k": 8, "split_sizes": [0, 2]}
        flops, formula = problem_flops("tlx_ops", "grouped_gemm_mxfp8", spec)
        self.assertEqual((flops, formula), (2 * 2 * 4 * 8, "mxfp8-grouped"))

    def test_unknown(self) -> None:
        self.assertEqual(
            problem_flops("tritonbench", "opaque", {"args": "--op x"}), (None, None)
        )
        self.assertEqual(problem_flops("tlx_ops", "mm", {"M": 4}), (None, None))
        self.assertEqual(problem_flops("tlx_ops", "mm", None), (None, None))


class GroupTest(unittest.TestCase):
    def test_group_key(self) -> None:
        self.assertEqual(group_key("case1"), "case1")
        self.assertEqual(group_key("case1__k1"), "case1")
        self.assertEqual(group_key("case1__k12"), "case1")

    def test_case_record_flops_once(self) -> None:
        kernels = [
            {
                "dir": "c",
                "status": "ok",
                "input_bytes": 10,
                "output_bytes": 4,
                "input_spec": _MM_SPEC,
            },
            {
                "dir": "c__k1",
                "status": "ok",
                "input_bytes": 10,
                "output_bytes": 4,
                "input_spec": _MM_SPEC,
            },
        ]
        rec = case_record("tlx_ops", "mm", "c", kernels)
        self.assertEqual(rec["flops"], 1024)
        self.assertEqual(rec["total_bytes"], 28)
        self.assertAlmostEqual(rec["intensity_flop_per_byte"], 1024 / 28)
        self.assertNotIn("input_spec", kernels[0])

    def test_case_record_zero_flops_intensity(self) -> None:
        kernels = [
            {
                "dir": "c",
                "status": "ok",
                "input_bytes": 10,
                "output_bytes": 4,
                "input_spec": {"M": 0, "N": 8, "K": 16},
            }
        ]
        rec = case_record("tlx_ops", "mm", "c", kernels)
        self.assertEqual(rec["flops"], 0)
        self.assertIsNotNone(rec["intensity_flop_per_byte"])
        self.assertEqual(rec["intensity_flop_per_byte"], 0.0)


class KernelRecordTest(unittest.TestCase):
    def test_bad_launch_details_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = _write_case(
                tmp,
                "tlx_ops",
                "mm",
                "m1",
                launch={"block": {}, "args": []},
                spec=_MM_SPEC,
            )
            rec = kernel_record(
                cdir,
                lambda llir: ("k", [("ptr", 64)]),
                lambda sig, launch=None: (sig, 0),
                lambda launch, sig: [],
            )
        self.assertEqual(rec["status"], "bad-launch")
        self.assertEqual(rec["input_spec"], _MM_SPEC)


class CollectTest(unittest.TestCase):
    def _tree(self, tmp: str) -> str:
        _write_case(tmp, "tlx_ops", "mm", "m1", spec=_MM_SPEC)
        _write_case(tmp, "tlx_ops", "mm", "m1__k1", spec=_MM_SPEC)
        _write_case(tmp, "tlx_ops", "flash_attn", "f1", spec=_FLASH_SPEC)
        _write_case(tmp, "tritonbench", "tb", "t1", spec={"args": "--op x"})
        _write_case(tmp, "tlx_ops", "mm", "nolaunch", launch=None, spec=_MM_SPEC)
        bad = _launch()
        bad["args"][0]["dtype"] = "weird"
        _write_case(tmp, "tlx_ops", "mm", "baddtype", launch=bad, spec=_MM_SPEC)
        _write_case(tmp, "tlx_ops", "mm", "badllir", llir="not ir", spec=_MM_SPEC)
        return tmp

    def test_collect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            doc = collect(self._tree(tmp), "gfx950")
        self.assertEqual(doc["version"], 1)
        self.assertEqual(doc["arch"], "gfx950")
        by_case = {(c["source"], c["target"], c["case"]): c for c in doc["cases"]}
        mm = by_case[("tlx_ops", "mm", "m1")]
        self.assertEqual(len(mm["kernels"]), 2)
        self.assertEqual(mm["flops"], 1024)
        self.assertEqual(mm["total_bytes"], 80)
        self.assertAlmostEqual(mm["intensity_flop_per_byte"], 1024 / 80)
        k0 = mm["kernels"][0]
        self.assertEqual(k0["status"], "ok")
        self.assertEqual(k0["grid"], [2, 1, 1])
        self.assertEqual(k0["ctas"], 2)
        self.assertEqual(k0["threads"], 256)
        self.assertEqual(k0["input_bytes"], 24)
        self.assertEqual(k0["output_bytes"], 16)
        fa = by_case[("tlx_ops", "flash_attn", "f1")]
        self.assertEqual(fa["flops"], 2097152)
        tb = by_case[("tritonbench", "tb", "t1")]
        self.assertIsNone(tb["flops"])
        self.assertEqual(tb["total_bytes"], 40)
        nl = by_case[("tlx_ops", "mm", "nolaunch")]
        self.assertEqual(nl["kernels"][0]["status"], "missing-launch")
        self.assertIsNone(nl["total_bytes"])
        bd = by_case[("tlx_ops", "mm", "baddtype")]
        self.assertIsNone(bd["total_bytes"])
        self.assertIsNone(bd["intensity_flop_per_byte"])
        bl = by_case[("tlx_ops", "mm", "badllir")]
        self.assertEqual(bl["kernels"][0]["status"], "bad-llir")
        summary = doc["summary"]
        self.assertEqual(
            (summary["cases"], summary["kernels"], summary["replay_ready"]),
            (6, 7, 5),
        )
        self.assertEqual(summary["flops_known"], 5)

    def test_main_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            self.assertEqual(main(["--arch", "gfx950", "--out", root]), 0)
            path = os.path.join(root, "gfx950", "metadata.json")
            with open(path) as f:
                doc = json.load(f)
        self.assertEqual(doc["summary"]["cases"], 6)


if __name__ == "__main__":
    unittest.main()
