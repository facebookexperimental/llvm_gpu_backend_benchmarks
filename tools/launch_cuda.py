# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CUDA runtime launcher for exported LLVM-stage kernels (NVIDIA).

Mirror of ``launch_hip`` for sm100: for every case dir with a kernel.llir
plus a compile.json ``launch`` record (see capture.record_launch), this
builds a launchable CUBIN from the saved LLIR (default backend options),
replays the recorded launch on GPU via raw CUDA driver calls, and reports
median ms plus output finiteness::

    buck2 run fbcode//triton/benchmarks/tools:launch_cuda -- \\
        --arch sm100 --out /data/llvm_export_20250101 [--match mm] [--limit 5]

Requires a GPU host and torch (same as the exporters). Cases without
launch metadata report ``missing-launch``; multi-CTA/cluster launches
report ``unsupported`` (via the shared launch validation).
Top-level imports are stdlib only so the pure helpers stay plain-python
testable; torch/triton load lazily inside the GPU path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys

try:
    from triton.benchmarks.tools.launch_hip import (
        _outputs_finite,
        _resolve_arg_value,
        _scratch_ptrs,
        _timed_ms,
        alloc_tensors,
        make_arg_ptrs,
        parse_kernel_signature,
        SEED,
        strip_scratch,
        validate_launch,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    # pyrefly: ignore [missing-import]
    from launch_hip import (
        _outputs_finite,
        _resolve_arg_value,
        _scratch_ptrs,
        _timed_ms,
        alloc_tensors,
        make_arg_ptrs,
        parse_kernel_signature,
        SEED,
        strip_scratch,
        validate_launch,
    )

_WARP_SIZE = 32


def _cuda():
    """Load libcuda with the module-launch entry points (GPU hosts)."""
    last = None
    for soname in ("libcuda.so", "libcuda.so.1"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError as e:
            last = e
    else:
        raise RuntimeError(f"could not load CUDA driver ({last})")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.cuModuleLoadData.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.restype = ctypes.c_int
    _check(lib, lib.cuInit(0), "cuInit")
    return lib


def _check(lib, status: int, what: str) -> None:
    if status != 0:
        try:
            msg_ptr = ctypes.c_char_p()
            lib.cuGetErrorString(status, ctypes.byref(msg_ptr))
            msg = (
                msg_ptr.value.decode("utf-8", "replace")
                if msg_ptr.value
                else f"code {status}"
            )
        except Exception:
            msg = f"code {status}"
        raise RuntimeError(f"{what} failed: {msg}")


def launch_case(
    cdir: str,
    arch: str,
    torch,
    cuda,
    seed: int,
    write_cubin: bool,
    cubin_path: str | None = None,
    collect_outputs: bool = False,
) -> dict:
    """Build CUBIN from kernel.llir and replay the recorded launch on GPU.

    Pass ``cubin_path`` to replay a prebuilt object instead of compiling
    (skips the ``llir_build`` backend import entirely). Pass
    ``collect_outputs`` to also return CPU copies of the output tensors
    under ``res["outputs"]`` (test-only; the CLI never sets this).
    """
    # Establish the CUDA primary context before any driver module calls.
    torch.cuda.current_stream()
    with open(os.path.join(cdir, "kernel.llir")) as f:
        llir = f.read()
    with open(os.path.join(cdir, "compile.json")) as f:
        compile_info = json.load(f)
    name, sig = parse_kernel_signature(llir)
    launch = compile_info.get("launch")
    problems = validate_launch(launch, sig)
    if problems:
        return {"status": problems[0], "problems": problems}
    assert launch is not None
    _user_sig, n_scratch = strip_scratch(sig, launch)
    if cubin_path is not None:
        with open(cubin_path, "rb") as f:
            cubin = f.read()
    else:
        try:
            from triton.benchmarks.tools.llir_build import build_cubin
        except ImportError:
            # Plain python3: sibling file in the same directory.
            # pyrefly: ignore [missing-import]
            from llir_build import build_cubin

        _opts_label, cubin = build_cubin(llir, arch)
        if write_cubin:
            with open(os.path.join(cdir, "kernel.cubin"), "wb") as f:
                f.write(cubin)
    module = ctypes.c_void_p()
    _check(cuda, cuda.cuModuleLoadData(ctypes.byref(module), cubin), "module load")
    try:
        func = ctypes.c_void_p()
        _check(
            cuda,
            cuda.cuModuleGetFunction(ctypes.byref(func), module, name.encode()),
            "get function",
        )
        tensors = alloc_tensors(torch, launch["args"], launch.get("device"), seed)
        values = []
        rec: dict = {}
        try:
            for rec in launch["args"]:
                values.append(_resolve_arg_value(rec, tensors))
        except ValueError:
            return {"status": "unsupported-arg", "arg": rec.get("name")}
        values += _scratch_ptrs(torch, launch, n_scratch)
        buffers, arg_array = make_arg_ptrs(values, sig)
        gx, gy, gz = (int(v) for v in launch["grid"])
        block = launch["block"]
        threads = int(block["num_warps"]) * _WARP_SIZE
        shared = int(block.get("shared") or 0)
        stream = torch.cuda.current_stream().cuda_stream

        def _fire():
            _check(
                cuda,
                cuda.cuLaunchKernel(
                    func,
                    gx,
                    gy,
                    gz,
                    threads,
                    1,
                    1,
                    shared,
                    ctypes.c_void_p(stream),
                    arg_array,
                    None,
                ),
                "launch",
            )

        ms = _timed_ms(torch, _fire)
        torch.cuda.synchronize()
        finite = _outputs_finite(torch, launch, tensors)
        res = {
            "status": "ok",
            "median_ms": ms,
            "finite": finite,
            "grid": launch["grid"],
            "seed": seed,
        }
        if collect_outputs:
            res["outputs"] = {
                rec["name"]: tensors[rec["name"]].cpu()
                for rec in launch["args"]
                if rec.get("kind") == "tensor" and rec.get("role") in ("out", "unknown")
            }
        del buffers
        return res
    finally:
        try:
            if hasattr(cuda, "cuModuleUnload"):
                cuda.cuModuleUnload(module)
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay exported kernels via CUDA.")
    parser.add_argument("--arch", required=True, choices=["sm100"])
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument("--limit", type=int, default=0, help="launch at most N cases")
    parser.add_argument(
        "--match", default="", help="only launch cases whose path contains this"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="input RNG seed")
    parser.add_argument(
        "--write-cubin",
        action="store_true",
        help="save built kernel.cubin next to kernel.llir",
    )
    parser.add_argument(
        "--cubin",
        default=None,
        help="replay this prebuilt CUBIN instead of compiling kernel.llir",
    )
    args = parser.parse_args(argv)

    import torch

    try:
        from triton.benchmarks.tools.llir_build import iter_cases
    except ImportError:
        # Plain python3: sibling file in the same directory.
        # pyrefly: ignore [missing-import]
        from llir_build import iter_cases

    cuda = _cuda()
    cases = [
        c
        for c in iter_cases(args.out, args.arch)
        if not args.match or args.match in "/".join(c[:3])
    ]
    if args.limit:
        cases = cases[: args.limit]
    print(f"[launch] {len(cases)} cases", flush=True)
    results: dict[str, dict] = {}
    for source, target, case, cdir in cases:
        name = f"{source}/{target}/{case}"
        try:
            res = launch_case(
                cdir,
                args.arch,
                torch,
                cuda,
                args.seed,
                args.write_cubin,
                cubin_path=args.cubin,
            )
        except Exception as e:  # noqa: BLE001 - record and continue
            res = {"status": "error", "error": repr(e)[:300]}
        results[name] = res
        ms = res.get("median_ms")
        # pyrefly: ignore[bad-argument-type]
        extra = f" {ms:.4f}ms finite={res.get('finite')}" if ms is not None else ""
        print(f"[launch] {res['status']} {name}{extra}", flush=True)
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(json.dumps({"cases": len(results), "ok": ok, "results": results}))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
