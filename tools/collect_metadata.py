# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Collect benchmark-ready metadata for the pruned corpus.

Walks ``<out>/<arch>`` case dirs (grouping multi-kernel ``__kN`` siblings)
and writes ``metadata.json``: per-kernel launch readiness plus grid, block,
and per-arg byte summaries, with per-case problem FLOPs (approximate
formulas from each case's ``input_spec``) and arithmetic intensity.

CPU-only; needs no GPU or triton import::

    python3 collect_metadata.py --arch gfx950 --out /data/llvm_export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

VERSION = 1
CASE_SUFFIX_RE = re.compile(r"__k\d+$")
WAVEFRONT = 64  # AMD wavefront size (mirrors launch_hip)


def _siblings():
    """Sibling helpers (buck layout, mirrored for plain python3)."""
    try:
        from triton.benchmarks.tools.capture import iter_cases
        from triton.benchmarks.tools.launch_hip import (
            parse_kernel_signature,
            strip_scratch,
            validate_launch,
        )
    except ImportError:
        from capture import iter_cases  # pyrefly: ignore [missing-import]
        from launch_hip import (  # pyrefly: ignore [missing-import]
            parse_kernel_signature,
            strip_scratch,
            validate_launch,
        )
    return iter_cases, parse_kernel_signature, strip_scratch, validate_launch


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2fnuz": 1,
    "float16": 2,
    "bfloat16": 2,
    "half": 2,
    "bf16": 2,
    "fp16": 2,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "fp32": 4,
    "float64": 8,
    "int64": 8,
    "uint64": 8,
    "fp64": 8,
}


def _num(value):
    """Non-negative int from a spec value, else None (bools rejected)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _gemm_flops(spec: dict):
    """2*B*M*N*K dense matmul FLOPs (B defaults to 1)."""
    m, n, k = _num(spec.get("M")), _num(spec.get("N")), _num(spec.get("K"))
    if m is None or n is None or k is None:
        return None
    b = _num(spec.get("B"))
    return 2 * (b if b is not None else 1) * m * n * k


def _first(spec: dict, *keys):
    """First present numeric value under alias keys, else None."""
    for key in keys:
        if key in spec:
            return _num(spec[key])
    return None


def _flash_flops(spec: dict):
    """Attention FLOPs: 4*B*H*T^2*D fwd (half when causal, 2.5x when bwd).

    Accepts the tlx_ops key names and the tutorial B/H/S/D aliases.
    """
    b = _first(spec, "batch", "B")
    h = _first(spec, "heads", "H")
    t = _first(spec, "context", "S")
    d = _first(spec, "head_dim", "D")
    if b is None or h is None or t is None or d is None:
        return None
    flops = 4 * b * h * t * t * d
    if spec.get("causal"):
        flops //= 2
    if spec.get("bwd"):
        flops = flops * 5 // 2
    return flops


def _varlen_flops(spec: dict):
    """Packed-varlen attention FLOPs over the per-sequence lengths."""
    h, d = _num(spec.get("heads")), _num(spec.get("head_dim"))
    lengths = spec.get("lengths")
    if h is None or d is None or not isinstance(lengths, list) or not lengths:
        return None
    total = 0
    for length in lengths:
        ell = _num(length)
        if ell is None:
            return None
        total += ell * ell
    flops = 4 * h * d * total
    if spec.get("causal"):
        flops //= 2
    return flops


def _grouped_flops(spec: dict):
    """Sum of 2*prod(group) over MoE [M, N, K] groups (order-free)."""
    groups = spec.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    total = 0
    for group in groups:
        if not isinstance(group, list) or len(group) != 3:
            return None
        dims = [_num(v) for v in group]
        m, n, k = dims
        if m is None or n is None or k is None:
            return None
        total += 2 * m * n * k
    return total


def _mxfp8_grouped_flops(spec: dict):
    """2*N*K*sum(splits) over the (possibly empty) row splits."""
    n, k = _num(spec.get("n")), _num(spec.get("k"))
    splits = spec.get("split_sizes")
    if n is None or k is None or not isinstance(splits, list) or not splits:
        return None
    total = 0
    for split in splits:
        m = _num(split)
        if m is None:
            return None
        total += 2 * m * n * k
    return total


def problem_flops(source: str, target: str, spec: dict | None):
    """Problem FLOPs from a case input_spec. Returns (flops, formula)."""
    if not isinstance(spec, dict):
        return None, None
    if "lengths" in spec:
        flops = _varlen_flops(spec)
        return (flops, "varlen-attn") if flops is not None else (None, None)
    if "groups" in spec:
        flops = _grouped_flops(spec)
        return (flops, "grouped-gemm") if flops is not None else (None, None)
    if "split_sizes" in spec:
        flops = _mxfp8_grouped_flops(spec)
        return (flops, "mxfp8-grouped") if flops is not None else (None, None)
    flops = _flash_flops(spec)
    if flops is not None:
        return flops, "flash-attn"
    flops = _gemm_flops(spec)
    if flops is not None:
        return flops, "gemm"
    return None, None


def _arg_nbytes(rec: dict, sig_entry) -> int | None:
    """Byte size of one recorded arg (tensor dims or scalar width)."""
    if rec.get("kind") == "tensor":
        shape = rec.get("shape") or []
        try:
            elems = 1
            for dim in shape:
                elems *= int(dim)
        except (TypeError, ValueError):
            return None
        size = _DTYPE_BYTES.get(str(rec.get("dtype", "")))
        return elems * size if size is not None else None
    _kind, bits = sig_entry
    return bits // 8


def _launch_details(launch: dict, sig, strip_scratch) -> dict:
    """Grid/block/arg summaries plus input/output bytes for a valid launch."""
    user_sig, _n_scratch = strip_scratch(sig, launch)
    block = launch.get("block") or {}
    grid = launch["grid"]
    ctas = 1
    for dim in grid:
        ctas *= dim
    args = []
    in_bytes = out_bytes = 0
    exact = True
    for rec, entry in zip(launch["args"], user_sig):
        nbytes = _arg_nbytes(rec, entry)
        args.append({"name": rec.get("name"), "nbytes": nbytes})
        if nbytes is None:
            exact = False
        elif rec.get("role") == "out":
            out_bytes += nbytes
        else:
            in_bytes += nbytes
    return {
        "grid": grid,
        "ctas": ctas,
        "threads": int(block["num_warps"]) * WAVEFRONT,
        "shared": block.get("shared"),
        "args": args,
        "input_bytes": in_bytes if exact else None,
        "output_bytes": out_bytes if exact else None,
    }


def kernel_record(cdir: str, parse, strip_scratch, validate) -> dict:
    """One kernel's metadata: launch readiness plus summaries when valid."""
    rec: dict = {"dir": os.path.basename(cdir)}
    try:
        with open(os.path.join(cdir, "compile.json")) as f:
            compile_info = json.load(f)
    except (OSError, ValueError):
        compile_info = {}
    if not isinstance(compile_info, dict):
        compile_info = {}
    rec["input_spec"] = compile_info.get("input_spec")
    launch = compile_info.get("launch")
    try:
        with open(os.path.join(cdir, "kernel.llir")) as f:
            llir = f.read()
        name, sig = parse(llir)
    except (OSError, ValueError):
        rec["status"] = "bad-llir"
        return rec
    rec["name"] = name
    problems = validate(launch, sig)
    if problems:
        rec["status"] = problems[0]
        rec["problems"] = problems
        return rec
    if launch is None:
        rec["status"] = "missing-launch"
        return rec
    try:
        details = _launch_details(launch, sig, strip_scratch)
    except (KeyError, TypeError, ValueError) as e:
        rec["status"] = "bad-launch"
        rec["problems"] = [f"bad-launch: {e!r}"]
        return rec
    rec["status"] = "ok"
    rec.update(details)
    return rec


def group_key(case: str) -> str:
    """Case group for a case dir (multi-kernel __kN siblings grouped)."""
    return CASE_SUFFIX_RE.sub("", case)


def case_record(source: str, target: str, group: str, kernels: list) -> dict:
    """One case group's metadata: kernels plus FLOPs and intensity."""
    flops = formula = None
    for kernel in kernels:
        spec = kernel.pop("input_spec", None)
        if flops is None and spec is not None:
            flops, formula = problem_flops(source, target, spec)
    total_in = total_out = 0
    exact = True
    for kernel in kernels:
        for key in ("input_bytes", "output_bytes"):
            value = kernel.get(key)
            if value is None:
                exact = False
            elif key == "input_bytes":
                total_in += value
            else:
                total_out += value
    total = total_in + total_out if exact else None
    intensity = flops / total if flops is not None and total else None
    return {
        "source": source,
        "target": target,
        "case": group,
        "kernels": kernels,
        "flops": flops,
        "flops_from": formula,
        "total_bytes": total,
        "intensity_flop_per_byte": intensity,
    }


def collect(out_root: str, arch: str) -> dict:
    """Collect per-case metadata for one arch. Returns the metadata doc."""
    iter_cases, parse, strip_scratch, validate = _siblings()
    groups: dict[tuple[str, str, str], list] = {}
    for source, target, case, cdir in iter_cases(out_root, arch):
        rec = kernel_record(cdir, parse, strip_scratch, validate)
        groups.setdefault((source, target, group_key(case)), []).append(rec)
    cases = [
        case_record(source, target, group, sorted(kernels, key=lambda r: r["dir"]))
        for (source, target, group), kernels in sorted(groups.items())
    ]
    kernels = [k for case in cases for k in case["kernels"]]
    return {
        "version": VERSION,
        "arch": arch,
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "kernels": len(kernels),
            "replay_ready": sum(1 for k in kernels if k["status"] == "ok"),
            "flops_known": sum(1 for c in cases if c["flops"] is not None),
            "bytes_known": sum(1 for c in cases if c["total_bytes"] is not None),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect corpus metadata.")
    parser.add_argument("--arch", required=True, choices=["gfx950", "gfx942"])
    parser.add_argument("--out", required=True, help="export root")
    args = parser.parse_args(argv)
    doc = collect(args.out, args.arch)
    path = os.path.join(args.out, args.arch, "metadata.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    summary = doc["summary"]
    print(
        f"[collect] {path}: cases={summary['cases']} "
        f"kernels={summary['kernels']} "
        f"replay_ready={summary['replay_ready']} "
        f"flops_known={summary['flops_known']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
