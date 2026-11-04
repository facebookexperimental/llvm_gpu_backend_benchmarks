# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Launcher-vs-Triton performance parity on gfx950.

Runs a simple vec-add Triton kernel on GPU, replays it through the raw-HIP
launcher (``launch_hip.launch_case``), and checks:

1. Correctness: launcher outputs match the Triton/torch reference and are
   finite.
2. GPU performance sanity: achieved memory bandwidth from the launcher
   timing clears a conservative floor (guards against silently broken
   codegen that still produces finite outputs).
3. Parity: the launcher median is no slower than the base-Triton median
   beyond tolerance (similar/consistent, or faster).

Two tests share the assertion helper:

- ``test_live_launcher_parity``: full loop on a large vector (capture via
  ``CaptureSession``, fresh HSACO via ``llir_build``, replay, compare).
  Needs the AMD codegen .so; skips without it.
- ``test_fixture_launcher_parity``: replays the checked-in ``re_probe/``
  fixtures (prebuilt HSACO, no codegen needed) against a live Triton
  baseline at the same shape. Covers RE workers.

Runs on RE MI350 workers (amd-gpu/MI350) or locally on any gfx950 host::

    buck2 test @fbcode//mode/opt-amd-gpu \\
        fbcode//triton/benchmarks/tools:launch_parity_test
    # force local execution on this GPU host (e.g. gpu98):
    buck2 test @fbcode//mode/opt-amd-gpu -c fbcode.re_gpu_tests=False \\
        fbcode//triton/benchmarks/tools:launch_parity_test
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

try:
    import torch
except ImportError:
    torch = None

try:
    from triton.benchmarks.tools.capture import CaptureSession, write_case
    from triton.benchmarks.tools.launch_hip import _hip, launch_case, REPS, SEED, WARMUP
except ImportError:
    # Plain python3: sibling files in the same directory.
    # pyrefly: ignore [missing-import]
    from capture import CaptureSession, write_case

    # pyrefly: ignore [missing-import]
    from launch_hip import _hip, launch_case, REPS, SEED, WARMUP

HERE = os.path.dirname(os.path.abspath(__file__))

N_LIVE = 1 << 24
BLOCK = 1024
NUM_WARPS = 4
# Launcher median may exceed the Triton median by this fraction before the
# test fails (shared-RE timing noise; both sides use the same WARMUP/REPS).
PARITY_TOL = 0.35
# Floor for the live (large-N, bandwidth-bound) replay. MI350 vec-add clears
# multiple TB/s; this only catches pathologically slow launches.
MIN_GBPS = 500.0


if triton is not None and tl is not None:

    @triton.jit
    def _add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


def _mean_ms(fire) -> float:
    """Mean ms per rep over REPS launches after WARMUP warmups."""
    assert torch is not None, "torch required"
    for _ in range(WARMUP):
        fire()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fire()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / REPS


def _gbps(numel: int, median_ms: float) -> float:
    """Achieved bandwidth for fp32 vec-add (2 reads + 1 write)."""
    return (3 * numel * 4) / (median_ms / 1e3) / 1e9


def _seeded_inputs(numel: int) -> tuple:
    assert torch is not None, "torch required"
    torch.manual_seed(SEED)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(SEED)
    x = torch.randn((numel,), dtype=torch.float32, device="cuda")
    y = torch.randn((numel,), dtype=torch.float32, device="cuda")
    return x, y


class ParityTest(unittest.TestCase):
    def _assert_parity(
        self,
        triton_ms: float,
        res: dict,
        expected,
        out_name: str,
        numel: int,
        min_gbps: float | None = MIN_GBPS,
    ) -> None:
        assert torch is not None, "torch required"
        self.assertEqual(res["status"], "ok", f"launcher problems: {res}")
        self.assertTrue(res["finite"])
        launcher_ms = res["median_ms"]
        self.assertGreater(triton_ms, 0)
        self.assertGreater(launcher_ms, 0)
        got = res["outputs"][out_name]
        max_diff = float((got.cpu() - expected.cpu()).abs().max())
        self.assertTrue(
            torch.allclose(got.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6),
            f"launcher output != reference (max diff {max_diff})",
        )
        if min_gbps is not None:
            self.assertGreaterEqual(
                _gbps(numel, launcher_ms),
                min_gbps,
                f"launcher bandwidth {_gbps(numel, launcher_ms):.1f} GB/s "
                f"below floor {min_gbps} GB/s",
            )
        ratio = launcher_ms / triton_ms
        print(
            f"[parity] n={numel} triton={triton_ms:.4f}ms "
            f"launcher={launcher_ms:.4f}ms ratio={ratio:.3f} "
            f"bw={_gbps(numel, launcher_ms):.1f}GB/s max_diff={max_diff:.2e}",
            flush=True,
        )
        self.assertLessEqual(
            ratio,
            1.0 + PARITY_TOL,
            f"launcher {launcher_ms:.4f}ms vs triton {triton_ms:.4f}ms "
            f"(ratio {ratio:.3f} > {1.0 + PARITY_TOL})",
        )

    def _live_baseline(self, x, y, out_bench, grid) -> float:
        def _fire() -> None:
            _add_kernel[grid](
                x, y, out_bench, x.numel(), BLOCK=BLOCK, num_warps=NUM_WARPS
            )

        return min(_mean_ms(_fire) for _ in range(3))

    def test_live_launcher_parity(self) -> None:
        if triton is None or tl is None:
            self.skipTest("triton unavailable")
        if torch is None or not torch.cuda.is_available():
            self.skipTest("no CUDA/HIP device")
        try:
            hip = _hip()
        except RuntimeError as e:
            self.skipTest(f"HIP runtime unavailable: {e}")

        n = N_LIVE
        grid = (triton.cdiv(n, BLOCK),)
        torch.manual_seed(SEED)
        x = torch.randn((n,), dtype=torch.float32, device="cuda")
        y = torch.randn((n,), dtype=torch.float32, device="cuda")
        out = torch.empty((n,), dtype=torch.float32, device="cuda")

        with CaptureSession() as sess:
            _add_kernel[grid](x, y, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(out.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-6),
            "triton baseline itself is wrong",
        )
        self.assertTrue(sess.kernels, "capture recorded no kernels")
        _label, kernel = sess.kernels[-1]
        launch = sess.launches.get(id(kernel))
        self.assertIsNotNone(launch, "capture recorded no launch")
        llir = (getattr(kernel, "asm", None) or {}).get("llir")
        if not llir:
            self.skipTest("captured kernel has no llir")
        metadata = getattr(kernel, "metadata", None) or {}
        target = getattr(metadata, "target", None) or {}
        arch = str(getattr(target, "arch", "gfx950"))

        with tempfile.TemporaryDirectory() as tmp:
            write_case(
                tmp,
                kernel,
                target="vec_add",
                case=f"n{n}",
                arch=arch,
                source="parity",
                launch=launch,
            )
            out_bench = torch.empty((n,), dtype=torch.float32, device="cuda")
            triton_ms = self._live_baseline(x, y, out_bench, grid)

            try:
                from triton.benchmarks.tools.llir_build import build_hsaco
            except ImportError:
                try:
                    # Plain python3: sibling file in the same directory.
                    # pyrefly: ignore [missing-import]
                    from llir_build import build_hsaco
                except ImportError as e:
                    self.skipTest(f"llir_build unavailable: {e!r}")
            try:
                _opts, hsaco = build_hsaco(llir, arch)
            except Exception as e:  # noqa: BLE001 - codegen env, skip with reason
                self.skipTest(f"HSACO build unavailable: {e!r}")
            hsaco_path = os.path.join(tmp, "kernel.hsaco")
            with open(hsaco_path, "wb") as f:
                f.write(hsaco)

            res = launch_case(
                tmp,
                arch,
                torch,
                hip,
                SEED,
                False,
                hsaco_path=hsaco_path,
                collect_outputs=True,
            )
            x0, y0 = _seeded_inputs(n)
            self._assert_parity(
                triton_ms=triton_ms,
                res=res,
                expected=(x0 + y0),
                out_name="out_ptr",
                numel=n,
            )

    def test_fixture_launcher_parity(self) -> None:
        if triton is None or tl is None:
            self.skipTest("triton unavailable")
        if torch is None or not torch.cuda.is_available():
            self.skipTest("no CUDA/HIP device")
        cdir = os.path.join(HERE, "re_probe")
        if not os.path.isdir(cdir):
            self.skipTest("re_probe fixtures missing")
        try:
            hip = _hip()
        except RuntimeError as e:
            self.skipTest(f"HIP runtime unavailable: {e}")

        with open(os.path.join(cdir, "compile.json")) as f:
            launch = json.load(f)["launch"]
        n = next(r["value"] for r in launch["args"] if r["name"] == "n")
        grid = (triton.cdiv(n, BLOCK),)
        self.assertEqual([int(v) for v in launch["grid"]], [grid[0], 1, 1])

        torch.manual_seed(SEED)
        x = torch.randn((n,), dtype=torch.float32, device="cuda")
        y = torch.randn((n,), dtype=torch.float32, device="cuda")
        out = torch.empty((n,), dtype=torch.float32, device="cuda")
        _add_kernel[grid](x, y, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(out.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-6),
            "triton baseline itself is wrong",
        )
        out_bench = torch.empty((n,), dtype=torch.float32, device="cuda")
        triton_ms = self._live_baseline(x, y, out_bench, grid)

        res = launch_case(
            cdir,
            "gfx950",
            torch,
            hip,
            SEED,
            False,
            hsaco_path=os.path.join(cdir, "kernel.hsaco"),
            collect_outputs=True,
        )
        x0, y0 = _seeded_inputs(n)
        # Tiny kernel: launch overhead dominates, so only check correctness
        # and parity ratio, not absolute bandwidth.
        self._assert_parity(
            triton_ms=triton_ms,
            res=res,
            expected=(x0 + y0),
            out_name="c_ptr",
            numel=n,
            min_gbps=None,
        )


if __name__ == "__main__":
    unittest.main()
