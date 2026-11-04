# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Remote-GPU smoke test: replay the probe fixtures on a real MI350.

Runs on RE workers (``python_unittest_remote_gpu``, amd-gpu/MI350) or a
local MI350X host; skips anywhere without a GPU::

    buck2 test @fbcode//mode/opt-amd-gpu \
        fbcode//triton/benchmarks/tools:launch_re_test

Uses the prebuilt probe HSACO (no triton import on the worker; needs
torch + HIP only) and checks exactness against reproduced RNG inputs.
"""

import json
import os
import sys
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

try:
    from llvm_benchmarks_tools.launch_hip import _hip, launch_case
except ImportError:
    _pkg = types.ModuleType("llvm_benchmarks_tools")
    _pkg.__path__ = [os.path.dirname(os.path.abspath(__file__))]
    sys.modules["llvm_benchmarks_tools"] = _pkg
    from llvm_benchmarks_tools.launch_hip import _hip, launch_case

HERE = os.path.dirname(os.path.abspath(__file__))


def _cdir() -> str:
    cdir = os.path.join(HERE, "re_probe")
    if not os.path.isdir(cdir):
        raise AssertionError(
            f"re_probe fixtures missing at {cdir}; "
            f"contents of {HERE}: {sorted(os.listdir(HERE))}"
        )
    return cdir


class ReProbeTest(unittest.TestCase):
    def test_probe_add_on_gpu(self) -> None:
        if torch is None:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA/HIP device")
        hip = _hip()
        cdir = _cdir()
        with open(os.path.join(cdir, "compile.json")) as f:
            launch = json.load(f)["launch"]
        res = launch_case(
            cdir,
            "gfx950",
            torch,
            hip,
            0,
            False,
            hsaco_path=os.path.join(cdir, "kernel.hsaco"),
            collect_outputs=True,
        )
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["finite"])
        self.assertGreater(res["median_ms"], 0)
        self.assertEqual(res["grid"], launch["grid"])
        n = next(r["value"] for r in launch["args"] if r["name"] == "n")
        torch.manual_seed(0)
        if hasattr(torch.cuda, "manual_seed_all"):
            torch.cuda.manual_seed_all(0)
        a = torch.randn((n,), dtype=torch.float32, device="cuda")
        b = torch.randn((n,), dtype=torch.float32, device="cuda")
        expected = (a + b).cpu()
        got = res["outputs"]["c_ptr"]
        max_diff = float((got - expected).abs().max())
        self.assertTrue(
            torch.allclose(got, expected, rtol=1e-5, atol=1e-6),
            f"c != a+b (max diff {max_diff})",
        )


if __name__ == "__main__":
    unittest.main()
