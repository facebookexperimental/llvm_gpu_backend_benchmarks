# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Verify exported kernel.llir files are exactly the IR consumable by LLVM.

For every case dir under <out>/<arch> with a kernel.llir, this re-feeds the
saved LLVM IR text through the production Triton-Beta AMD backend stage
(`HIPBackend.make_amdgcn`, the same `triton_amdgpu_compile` entry the
exporters' compilations used) and compares the result with the saved
kernel.amdgcn. An exact (or comment-normalized) match proves the .llir is
byte-for-byte the input LLVM consumed.

Run on the export host (CPU-only, no GPU kernels launched)::

    buck2 run @//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:verify_llir -- \\
        --arch gfx950 --out /data/llvm_export_20250101

Requires TRITON_AMD_CODEGEN_PATH (same as the exporters).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

import torch  # noqa: F401 - imported for triton backend discovery at runtime
from llvm_benchmarks_tools.llir_build import iter_cases, KERNEL_RE, round_trip


def structural_check(text: str) -> list[str]:
    """Well-formed LLVM IR module for amdgcn; returns problem strings."""
    problems = []
    if "; ModuleID" not in text[:500] and "source_filename" not in text[:500]:
        problems.append("missing LLVM module header")
    if 'target triple = "amdgcn-amd-amdhsa"' not in text:
        problems.append("missing amdgcn triple")
    if "target datalayout" not in text:
        problems.append("missing datalayout")
    if not KERNEL_RE.search(text):
        problems.append("no amdgpu_kernel definition")
    for marker in ("tt.func", "#ttg", "ttg.", "// -----//"):
        if marker in text:
            problems.append(f"contains MLIR marker {marker!r}")
    return problems


def normalize_amdgcn(text: str) -> str:
    """Drop comments/blank lines for comparison (code lines only)."""
    out = []
    for line in text.splitlines():
        line = line.split(";")[0].rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out) + "\n"


_DEFAULT_LABEL = "disable_packed_fp32_ops=False enable_fp_fusion=True"


def _match_candidate(candidates, ref, ref_norm):
    """First exact match wins, else first comment-normalized match."""
    fallback = None
    for opts, regen in candidates:
        if regen == ref:
            return ("exact", opts)
        if fallback is None and normalize_amdgcn(regen) == ref_norm:
            fallback = ("normalized", opts)
    return fallback


def _check_case(source, target, case, cdir, arch, dump_regen) -> str:
    """Verify one case dir. Returns a status counter name; prints details."""
    name = f"{source}/{target}/{case}"
    llir = open(os.path.join(cdir, "kernel.llir")).read()
    problems = structural_check(llir)
    if problems:
        print(f"[verify] STRUCT {name}: {problems}")
        return "structural_fail"
    ref_path = os.path.join(cdir, "kernel.amdgcn")
    if not os.path.isfile(ref_path):
        print(f"[verify] SKIP {name}: no kernel.amdgcn")
        return "skip"
    try:
        candidates = round_trip(llir, arch)
    except Exception as e:  # noqa: BLE001 - record and continue
        print(f"[verify] ROUNDTRIP-FAIL {name}: {e}")
        return "roundtrip_fail"
    ref = open(ref_path).read()
    matched = _match_candidate(candidates, ref, normalize_amdgcn(ref))
    if matched is None:
        # Fall back to the full grid (TLX per-config pressure flags).
        try:
            candidates = round_trip(llir, arch, full_grid=True)
        except Exception as e:  # noqa: BLE001 - record and continue
            print(f"[verify] ROUNDTRIP-FAIL {name}: {e}")
            return "roundtrip_fail"
        matched = _match_candidate(candidates, ref, normalize_amdgcn(ref))
    if matched is not None:
        if matched[1] != _DEFAULT_LABEL:
            print(f"[verify] MATCH {name} [{matched[0]} {matched[1]}]")
        return matched[0]
    best = min(
        candidates,
        key=lambda c: len(
            list(difflib.unified_diff(ref.splitlines(), c[1].splitlines()))
        ),
    )
    n_diff = len(
        list(difflib.unified_diff(ref.splitlines(), best[1].splitlines(), lineterm=""))
    )
    print(f"[verify] DIVERGED {name} [closest={best[0]}]: {n_diff} diff lines")
    if dump_regen:
        dump = os.path.join(dump_regen, source, target, case)
        os.makedirs(dump, exist_ok=True)
        for opts, regen in candidates:
            safe = opts.replace("=", "-").replace(" ", "_")
            with open(os.path.join(dump, f"regen.{safe}.s"), "w") as f:
                f.write(regen)
    return "diverged"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify exported LLIR via LLVM.")
    parser.add_argument("--arch", required=True, choices=["gfx950"])
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument("--limit", type=int, default=0, help="check at most N cases")
    parser.add_argument(
        "--match", default="", help="only check cases whose path contains this"
    )
    parser.add_argument(
        "--dump-regen",
        default="",
        help="write regenerated amdgcn for diverged cases under this dir",
    )
    parser.add_argument(
        "--skip-round-trip",
        action="store_true",
        help="structural checks only (no LLVM compilation)",
    )
    args = parser.parse_args(argv)

    cases = list(iter_cases(args.out, args.arch))
    if args.match:
        cases = [c for c in cases if args.match in "/".join(c[:3])]
    if args.limit:
        cases = cases[: args.limit]
    print(f"[verify] {len(cases)} cases with kernel.llir", flush=True)
    counts = {
        "exact": 0,
        "normalized": 0,
        "structural_fail": 0,
        "roundtrip_fail": 0,
        "diverged": 0,
        "skip": 0,
    }
    for i, (source, target, case, cdir) in enumerate(cases):
        if args.skip_round_trip:
            llir = open(os.path.join(cdir, "kernel.llir")).read()
            problems = structural_check(llir)
            if problems:
                counts["structural_fail"] += 1
                print(f"[verify] STRUCT {source}/{target}/{case}: {problems}")
            continue
        status = _check_case(source, target, case, cdir, args.arch, args.dump_regen)
        if status in counts:
            counts[status] += 1
        if (i + 1) % 50 == 0:
            print(f"[verify] ... {i + 1}/{len(cases)}", flush=True)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "exact": counts["exact"],
                "normalized_match": counts["normalized"],
                "structural_fail": counts["structural_fail"],
                "roundtrip_fail": counts["roundtrip_fail"],
                "diverged": counts["diverged"],
                "skip": counts["skip"],
            }
        )
    )
    failed = counts["structural_fail"] + counts["roundtrip_fail"] + counts["diverged"]
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
