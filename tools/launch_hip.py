# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""HIP runtime launcher for exported LLVM-stage kernels.

For every case dir with a kernel.llir plus a compile.json ``launch`` record
(see capture.record_launch), this builds a launchable HSACO from the saved
LLIR (default backend options), replays the recorded launch on GPU via raw
HIP module calls, and reports median ms plus output finiteness::

    buck2 run @fbcode//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:launch_hip -- \\
        --arch gfx950 --out /data/llvm_export_20250101 [--match mm] [--limit 5]

Requires a GPU host, torch, and TRITON_AMD_CODEGEN_PATH (same as the
exporters). Cases without launch metadata (older exports) report
``missing-launch``; multi-CTA/cluster launches report ``unsupported``.
Top-level imports are stdlib only so the pure helpers stay plain-python
testable; torch/triton load lazily inside the GPU path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from typing import Any

WARMUP = 10
REPS = 30
SEED = 0

_SIG_RE = re.compile(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)\(")
_WARP_SIZE = 64


def parse_kernel_signature(llir: str) -> tuple[str, list[tuple[str, int]]]:
    """Parse the amdgpu_kernel/function name plus [(kind, bits)] params.

    Kinds: ``ptr`` (any addrspace), ``int`` (iN), ``float`` (half/bfloat/
    float/double). Raises ValueError when no kernel or an unknown type.
    """
    m = _SIG_RE.search(llir)
    if not m:
        raise ValueError("no amdgpu_kernel definition")
    name = m.group(1)
    params = _kernel_param_text(llir, m.end())
    return name, [_classify_param(p) for p in _split_top_level(params)]


def _kernel_param_text(llir: str, start: int) -> str:
    """Param text up to the matching close paren (addrspace(1) nests)."""
    depth, i = 1, start
    while i < len(llir) and depth:
        if llir[i] == "(":
            depth += 1
        elif llir[i] == ")":
            depth -= 1
        i += 1
    if depth:
        raise ValueError("unterminated kernel signature")
    return llir[start : i - 1].strip()


def _classify_param(param: str) -> tuple[str, int]:
    """Map one param's leading LLVM type to (kind, bits)."""
    text = param.strip()
    m = re.match(r"(ptr|i\d+|half|bfloat|float|double)\b", text)
    if not m:
        raise ValueError(f"unsupported param {text!r}")
    ty = m.group(1)
    if ty == "ptr":
        return ("ptr", 64)
    if ty.startswith("i"):
        return ("int", int(ty[1:]))
    if ty in ("half", "bfloat"):
        return ("float", 16)
    if ty == "float":
        return ("float", 32)
    return ("float", 64)


def _split_top_level(params: str) -> list[str]:
    """Split signature params on top-level commas (attrs nest parens)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(params):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(params[start:i])
            start = i + 1
    tail = params[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def strip_scratch(
    sig: list[tuple[str, int]], launch: dict | None = None
) -> tuple[list[tuple[str, int]], int]:
    """Split a kernel signature into (user_params, n_scratch).

    Triton appends two trailing pointer params (global_scratch,
    profile_scratch); the replay passes NULL or an allocated buffer for
    each. Signatures without that trailer (hand-written LLIR) strip to
    (sig, 0).

    With a launch record, the trailer must be confirmed by it: recorded
    user args covering the whole signature mean the trailing pointers
    are user args, not scratch. Without a record, fall back to the
    trailing-pointer heuristic.
    """
    if len(sig) < 2 or sig[-1][0] != "ptr" or sig[-2][0] != "ptr":
        return sig, 0
    if launch is None:
        return sig[:-2], 2
    args = launch.get("args") or []
    if len(args) == len(sig):
        return sig, 0
    block = launch.get("block") or {}
    if (
        len(args) == len(sig) - 2
        or "global_scratch_size" in block
        or "profile_scratch_size" in block
    ):
        return sig[:-2], 2
    return sig, 0


def validate_launch(launch: dict | None, sig: list[tuple[str, int]]) -> list[str]:
    """Check a launch record against the parsed signature. Returns problems."""
    if not launch:
        return ["missing-launch"]
    problems = []
    grid = launch.get("grid") or []
    if len(grid) != 3 or not all(isinstance(v, int) and v >= 0 for v in grid):
        problems.append("bad-grid")
    block = launch.get("block") or {}
    if not isinstance(block.get("num_warps"), int):
        problems.append("missing-num_warps")
    if block.get("num_ctas", 1) != 1:
        problems.append("unsupported: num_ctas != 1")
    if block.get("cluster_dims") is not None:
        problems.append("unsupported: cluster_dims set")
    for key in ("global_scratch_size", "profile_scratch_size"):
        size = block.get(key, 0)
        if not isinstance(size, int) or size < 0:
            problems.append(f"bad-{key}")
    user_sig, _n_scratch = strip_scratch(sig, launch)
    args = launch.get("args") or []
    if len(args) != len(user_sig):
        problems.append(f"arg-count {len(args)} != signature {len(user_sig)}")
        return problems
    for rec, (kind, _bits) in zip(args, user_sig):
        want = "ptr" if rec.get("kind") == "tensor" else "scalar"
        got = "ptr" if kind == "ptr" else "scalar"
        if want != got:
            problems.append(f"arg {rec.get('name')}: {want} != {got}")
    return problems


def make_arg_ptrs(
    arg_values: list, sig: list[tuple[str, int]]
) -> tuple[list, ctypes.Array]:
    """Build ctypes storage plus the void* array for hipModuleLaunchKernel.

    Returns (buffers, ptr_array); callers must keep buffers alive through
    the launch. Tensor values expose ``data_ptr()`` (raw ints also pass as
    addresses, for NULL scratch); scalars are ints, floats, or bools.
    Raises ValueError on kind/width mismatch.
    """
    if len(arg_values) != len(sig):
        raise ValueError(f"arg-count {len(arg_values)} != signature {len(sig)}")
    buffers: list = []
    ptrs = []
    for value, (kind, bits) in zip(arg_values, sig):
        if kind == "ptr":
            addr = value.data_ptr() if hasattr(value, "data_ptr") else int(value)
            buf = ctypes.c_uint64(addr)
        elif kind == "int":
            buf = _pack_int(value, bits)
        elif kind == "float":
            buf = _pack_float(value, bits)
        else:
            raise ValueError(f"unsupported kind {kind!r}")
        buffers.append(buf)
        ptrs.append(ctypes.cast(ctypes.byref(buf), ctypes.c_void_p))
    return buffers, (ctypes.c_void_p * len(ptrs))(*ptrs)


def _pack_int(value, bits: int):
    """ctypes cell for an iN kernel arg (exact bit pattern)."""
    if bits <= 0 or bits > 64:
        raise ValueError(f"unsupported int width {bits}")
    ival = int(value) & ((1 << bits) - 1)
    if bits <= 8:
        return ctypes.c_uint8(ival)
    if bits <= 16:
        return ctypes.c_uint16(ival)
    if bits <= 32:
        return ctypes.c_uint32(ival)
    return ctypes.c_uint64(ival)


def _pack_float(value, bits: int):
    """ctypes cell for a floating kernel arg (no f16 scalars)."""
    fval = float(value)
    if bits == 16:
        raise ValueError("f16 scalar args unsupported")
    if bits == 32:
        return ctypes.c_float(fval)
    if bits == 64:
        return ctypes.c_double(fval)
    raise ValueError(f"unsupported float width {bits}")


def _hip():
    """Load libamdhip64 with the module-launch entry points (GPU hosts)."""
    last = None
    for soname in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            lib = ctypes.CDLL(soname)
            break
        except OSError as e:
            last = e
    else:
        raise RuntimeError(f"could not load HIP runtime ({last})")
    lib.hipModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.hipModuleLaunchKernel.restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib


def _check(lib, status: int, what: str) -> None:
    if status != 0:
        try:
            msg = lib.hipGetErrorString(status).decode("utf-8", "replace")
        except Exception:
            msg = f"code {status}"
        raise RuntimeError(f"{what} failed: {msg}")


_DTYPE_CTORS = {
    "float16": "half",
    "bfloat16": "bfloat16",
    "float32": "float",
    "float64": "double",
    "int32": "int",
    "int64": "long",
    "bool": "bool",
    "uint8": "uint8",
}


def alloc_tensors(torch, records: list[dict], device: str, seed: int) -> dict[str, Any]:
    """Allocate recorded tensor args: randn inputs, zeros outputs (seeded)."""
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)
    out: dict[str, object] = {}
    for rec in records:
        if rec.get("kind") != "tensor":
            continue
        ctor = _DTYPE_CTORS.get(rec.get("dtype", ""))
        if ctor is None:
            raise ValueError(f"unsupported dtype {rec.get('dtype')!r}")
        shape = tuple(rec.get("shape", []))
        make = torch.zeros if rec.get("role") == "out" else torch.randn
        out[rec["name"]] = make(
            shape, dtype=getattr(torch, ctor), device=device or "cuda"
        )
    return out


def _resolve_arg_value(rec: dict, tensors: dict):
    """Replay value for one recorded arg (raises ValueError when unusable)."""
    kind = rec.get("kind")
    if kind == "tensor":
        return tensors[rec["name"]]
    if kind in ("int", "bool"):
        return int(rec["value"])
    if kind == "float":
        return float(rec["value"])
    raise ValueError(f"unsupported arg {rec.get('name')}")


def _timed_ms(torch, fire) -> float:
    """Mean ms per rep over REPS timed launches after WARMUP warmups."""
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
    return start.elapsed_time(end) / REPS


def _outputs_finite(torch, launch: dict, tensors: dict) -> bool:
    """True when every replayed output tensor holds only finite values."""
    for rec in launch["args"]:
        if rec.get("kind") == "tensor" and rec.get("role") in ("out", "unknown"):
            if not bool(torch.isfinite(tensors[rec["name"]]).all()):
                return False
    return True


def _scratch_ptrs(torch, launch: dict, n_scratch: int) -> list:
    """ABI scratch args: allocated zero buffers, or NULL (0) when size is 0.

    Callers must keep the returned tensors alive through the launch.
    """
    if not n_scratch:
        return []
    block = launch.get("block") or {}
    device = launch.get("device") or "cuda"
    ptrs = []
    for key in ("global_scratch_size", "profile_scratch_size")[:n_scratch]:
        size = int(block.get(key) or 0)
        if size > 0:
            ptrs.append(torch.zeros((size,), dtype=torch.int8, device=device))
        else:
            ptrs.append(0)
    return ptrs


def launch_case(
    cdir: str,
    arch: str,
    torch,
    hip,
    seed: int,
    write_hsaco: bool,
    hsaco_path: str | None = None,
    collect_outputs: bool = False,
) -> dict:
    """Build HSACO from kernel.llir and replay the recorded launch on GPU.

    Pass ``hsaco_path`` to replay a prebuilt object instead of compiling
    (skips the ``llir_build`` backend import entirely). Pass
    ``collect_outputs`` to also return CPU copies of the output tensors
    under ``res["outputs"]`` (test-only; the CLI never sets this).
    """
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
    if hsaco_path is not None:
        with open(hsaco_path, "rb") as f:
            hsaco = f.read()
    else:
        try:
            from triton.benchmarks.tools.llir_build import build_hsaco
        except ImportError:
            # Plain python3: sibling file in the same directory.
            # pyrefly: ignore [missing-import]
            from llir_build import build_hsaco

        _opts_label, hsaco = build_hsaco(llir, arch)
        if write_hsaco:
            with open(os.path.join(cdir, "kernel.hsaco"), "wb") as f:
                f.write(hsaco)
    module = ctypes.c_void_p()
    _check(hip, hip.hipModuleLoadData(ctypes.byref(module), hsaco), "module load")
    try:
        func = ctypes.c_void_p()
        _check(
            hip,
            hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode()),
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
                hip,
                hip.hipModuleLaunchKernel(
                    func,
                    gx,
                    gy,
                    gz,
                    threads,
                    1,
                    1,
                    shared,
                    ctypes.c_void_p(stream),
                    ctypes.cast(arg_array, ctypes.c_void_p),
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
            if hasattr(hip, "hipModuleUnload"):
                hip.hipModuleUnload(module)
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay exported kernels via HIP.")
    parser.add_argument("--arch", required=True, choices=["gfx950", "gfx942"])
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument("--limit", type=int, default=0, help="launch at most N cases")
    parser.add_argument(
        "--match", default="", help="only launch cases whose path contains this"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="input RNG seed")
    parser.add_argument(
        "--write-hsaco",
        action="store_true",
        help="save built kernel.hsaco next to kernel.llir",
    )
    parser.add_argument(
        "--hsaco",
        default=None,
        help="replay this prebuilt HSACO instead of compiling kernel.llir",
    )
    args = parser.parse_args(argv)

    import torch

    try:
        from triton.benchmarks.tools.llir_build import iter_cases
    except ImportError:
        # Plain python3: sibling file in the same directory.
        # pyrefly: ignore [missing-import]
        from llir_build import iter_cases

    hip = _hip()
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
                hip,
                args.seed,
                args.write_hsaco,
                hsaco_path=args.hsaco,
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
