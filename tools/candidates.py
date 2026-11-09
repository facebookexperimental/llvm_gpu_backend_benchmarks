# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Candidate taxonomy: (strategy x capability) tracking for tlx_ops.

``candidates.yaml`` lists, per (arch, op), the major execution strategies
(kernel variants of importance) and the masking capability values the export
must include. This module resolves those labels per case -- strategy from the
compiled kernel name, masking from the input spec plus the winning config --
so exporters can record them and coverage can be checked::

    python3 candidates.py --check
    python3 candidates.py --coverage --manifest /data/llvm_export/gfx950/manifest.json

Stdlib only (no torch/triton/GPU): the YAML subset parses via
``discover.load_yaml_subset``, so plain ``python3`` runs work too.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

CANDIDATES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "candidates.yaml"
)
VERSION = 1
UNKNOWN = "unknown"

# Ops whose masking derives from GEMM dims vs winner BLOCK sizes.
_GEMM_OPS = frozenset({"mm", "addmm"})
# Ops whose masking derives from raggedness of the group list.
_GROUPED_OPS = frozenset({"grouped_gemm", "grouped_gemm_mxfp8"})
# Ops whose masking derives from the causal flag.
_CAUSAL_OPS = frozenset({"flash_attn", "flash_attn_mxfp8", "hstu_attn_dev"})


def _yaml_loader():
    """discover.load_yaml_subset, resolvable under buck and plain python3."""
    try:
        from triton.benchmarks.tools.discover import load_yaml_subset
    except ImportError:
        # Plain python3: sibling file in the same directory.
        from discover import load_yaml_subset
    return load_yaml_subset


def load_taxonomy(path: str = CANDIDATES_FILE) -> dict:
    """Load and version-check candidates.yaml (stdlib only)."""
    spec = _yaml_loader()(path)
    if spec.get("version") != VERSION:
        raise ValueError(f"{path}: expected version {VERSION}")
    return spec


def _op_entry(taxonomy: dict, arch: str, op: str) -> dict | None:
    ops = taxonomy.get("arch", {}).get(arch, {}).get("tlx_ops", []) or []
    for entry in ops:
        if entry.get("op") == op:
            return entry
    return None


def resolve_strategy(
    taxonomy: dict,
    arch: str,
    op: str,
    kernel_name: str | None,
    launch: str | None = None,
    bwd: bool = False,
) -> str:
    """Resolve the execution strategy for one captured kernel.

    Torchtlx ops resolve by op (inductor generates the kernels). Otherwise
    a mode-specific default wins outright, then first kernel-name pattern
    hit in taxonomy order, then the arch-agnostic default, else unknown.
    """
    if op.endswith("_torchtlx"):
        return "inductor"
    entry = _op_entry(taxonomy, arch, op)
    if entry is None:
        return UNKNOWN
    default = entry.get("default_strategy")
    applies = entry.get("default_applies_to", "any")
    if default and applies in ("fwd", "bwd") and (applies == "bwd") == bool(bwd):
        return default
    haystack = f"{kernel_name or ''} {launch or ''}"
    for strategy in entry.get("strategies", []) or []:
        patterns = strategy.get("kernel_patterns", []) or []
        if any(p and p in haystack for p in patterns):
            return strategy["name"]
    if default and applies == "any":
        return default
    return UNKNOWN


def _block_size(config_repr: str | None, key: str) -> int | None:
    if not config_repr:
        return None
    m = re.search(rf"['\"]{key}['\"]\s*:\s*(\d+)", config_repr)
    return int(m.group(1)) if m else None


def _gemm_dims(spec: dict) -> tuple | None:
    if all(k in spec for k in ("M", "N", "K")):
        return (spec["M"], spec["N"], spec["K"])
    entry = spec.get("entry")
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        try:
            return (int(entry[0]), int(entry[1]), int(entry[2]))
        except (TypeError, ValueError):
            return None
    return None


def _gemm_masking(spec: dict, winner_config: str | None) -> str:
    """Masking from GEMM dims vs the winning BLOCK sizes."""
    dims = _gemm_dims(spec)
    blocks = tuple(
        _block_size(winner_config, k) for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K")
    )
    if dims is None or any(v is None for v in dims + blocks):
        return UNKNOWN
    (m, n, k), (bm, bn, bk) = dims, blocks
    if m % bm == 0 and n % bn == 0 and k % bk == 0:
        return "even"
    if k % bk != 0:
        return "k_tail"
    return "mn_tail"


def _grouped_masking(spec: dict) -> str:
    """Masking from group-list raggedness (grouped GEMM families)."""
    groups = spec.get("groups")
    if groups is None and "split_sizes" in spec:
        groups = [(s,) for s in spec["split_sizes"]]
    if not groups:
        return UNKNOWN

    def _key(g):
        return tuple(g) if isinstance(g, (list, tuple)) else g

    return "uniform" if len({_key(g) for g in groups}) == 1 else "ragged"


def resolve_masking(
    op: str, input_spec: dict | None, winner_config: str | None = None
) -> str:
    """Resolve the masking capability for one case.

    GEMM ops compare dims against the winning BLOCK sizes; grouped ops read
    raggedness off the group list; attention ops read the causal flag.
    Returns ``n/a`` where masking does not apply, ``unknown`` when the
    inputs needed to decide are missing.
    """
    spec = input_spec or {}
    if op in _GEMM_OPS:
        return _gemm_masking(spec, winner_config)
    if op in _GROUPED_OPS:
        return _grouped_masking(spec)
    if op in ("flash_attn_varlen", "flash_attn_varlen_backward"):
        if "causal" not in spec:
            return UNKNOWN
        return "packed_causal" if spec["causal"] else "packed_noncausal"
    if op in _CAUSAL_OPS:
        if "causal" not in spec:
            return UNKNOWN
        return "causal" if spec["causal"] else "noncausal"
    return "n/a"


def label_entry(taxonomy: dict, arch: str, entry: dict) -> dict:
    """Resolve (strategy, masking) labels for one manifest entry."""
    op = entry.get("target", "")
    spec = entry.get("input_spec") or {}
    extra = entry.get("extra") or {}
    return {
        "strategy": resolve_strategy(
            taxonomy,
            arch,
            op,
            entry.get("kernel_name"),
            extra.get("launch"),
            bool(spec.get("bwd", False)),
        ),
        "masking": resolve_masking(op, spec, extra.get("winner_config")),
    }


def _is_aux(strategy: dict) -> bool:
    return str(strategy.get("auxiliary", "")).lower() == "true"


def _check_op_entry(path: str, arch: str, entry: dict) -> list[str]:
    """Validate one taxonomy op entry (strategies, default, masking)."""
    problems: list[str] = []
    op = entry.get("op", "")
    names: set[str] = set()
    for strategy in entry.get("strategies", []) or []:
        name = strategy.get("name", "")
        if name in names:
            problems.append(f"{path}: duplicate {arch}/{op} strategy {name}")
        names.add(name)
        if not strategy.get("notes"):
            problems.append(f"{path}: {arch}/{op}/{name} needs notes")
        if not isinstance(strategy.get("kernel_patterns", []), list):
            problems.append(
                f"{path}: {arch}/{op}/{name} kernel_patterns must be a list"
            )
    default = entry.get("default_strategy")
    if default is not None:
        if default not in names:
            problems.append(f"{path}: {arch}/{op} default {default} not listed")
        applies = entry.get("default_applies_to", "any")
        if applies not in ("fwd", "bwd", "any"):
            problems.append(
                f"{path}: {arch}/{op} default_applies_to must be fwd/bwd/any"
            )
    if "masking" in entry and not entry["masking"]:
        problems.append(f"{path}: {arch}/{op} masking must be non-empty")
    return problems


def check(taxonomy_path: str, targets_path: str) -> list[str]:
    """Validate candidates.yaml structure plus ops mirrored with targets.yaml."""
    problems: list[str] = []
    try:
        taxonomy = load_taxonomy(taxonomy_path)
    except (OSError, ValueError) as e:
        return [str(e)]
    targets = _yaml_loader()(targets_path)
    for arch, arch_spec in (taxonomy.get("arch", {}) or {}).items():
        if arch != "gfx950":
            problems.append(f"{taxonomy_path}: unknown arch {arch}")
            continue
        entries = arch_spec.get("tlx_ops", []) or []
        seen_ops: set[str] = set()
        for entry in entries:
            op = entry.get("op", "")
            if op in seen_ops:
                problems.append(f"{taxonomy_path}: duplicate {arch} op {op}")
            seen_ops.add(op)
            problems.extend(_check_op_entry(taxonomy_path, arch, entry))
        target_ops = {
            e.get("op", "")
            for e in targets.get("arch", {}).get(arch, {}).get("tlx_ops", []) or []
        }
        for op in sorted(seen_ops - target_ops):
            problems.append(f"{taxonomy_path}: {arch}/{op} has no targets.yaml entry")
        for op in sorted(target_ops - seen_ops):
            problems.append(f"{taxonomy_path}: {arch}/{op} target lacks taxonomy")
    return problems


def coverage(taxonomy: dict, entries: list[dict]) -> dict:
    """Per-(arch, op) strategy/masking coverage over manifest entries.

    Returns ``{arch: {op: {"strategies": {name: count}, "masking": ...,
    "missing_strategies": [...], "missing_masking": [...]}}}``. Only major
    (non-auxiliary) strategies and taxonomy-listed masking values count as
    missing; auxiliaries and unknowns are reported, never required.
    """
    out: dict = {}
    for arch, arch_spec in (taxonomy.get("arch", {}) or {}).items():
        out.setdefault(arch, {})
        for entry in arch_spec.get("tlx_ops", []) or []:
            op = entry.get("op", "")
            majors = [
                s["name"]
                for s in entry.get("strategies", []) or []
                if s.get("name") and not _is_aux(s)
            ]
            want_masking = list(entry.get("masking", []) or [])
            got_strategy: dict[str, int] = {}
            got_masking: dict[str, int] = {}
            for case in entries:
                if case.get("arch") != arch or case.get("target") != op:
                    continue
                extra = case.get("extra") or {}
                strategy = extra.get("strategy", UNKNOWN)
                masking = extra.get("masking", UNKNOWN)
                if not strategy:
                    strategy = UNKNOWN
                if not masking:
                    masking = UNKNOWN
                got_strategy[strategy] = got_strategy.get(strategy, 0) + 1
                if want_masking:
                    got_masking[masking] = got_masking.get(masking, 0) + 1
            out[arch][op] = {
                "strategies": got_strategy,
                "masking": got_masking,
                "missing_strategies": [m for m in majors if not got_strategy.get(m)],
                "missing_masking": [m for m in want_masking if not got_masking.get(m)],
            }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="tlx_ops candidate taxonomy.")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--candidates", default=os.path.join(here, "candidates.yaml"))
    parser.add_argument(
        "--targets", default=os.path.join(here, "targets.yaml"), help="for --check"
    )
    parser.add_argument("--check", action="store_true", help="validate the taxonomy")
    parser.add_argument(
        "--coverage", action="store_true", help="report manifest coverage"
    )
    parser.add_argument("--manifest", default=None, help="manifest.json for --coverage")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.candidates, args.targets)
        for p in problems:
            print(p)
        print("candidates.yaml OK" if not problems else f"{len(problems)} problems")
        return 1 if problems else 0
    if args.coverage:
        if not args.manifest:
            parser.error("--coverage needs --manifest")
        taxonomy = load_taxonomy(args.candidates)
        with open(args.manifest) as f:
            manifest = json.load(f)
        report = coverage(taxonomy, manifest.get("artifacts", []))
        missing = False
        for arch in sorted(report):
            for op in sorted(report[arch]):
                rep = report[arch][op]
                print(
                    f"{arch}/{op}: strategies={rep['strategies']} masking={rep['masking']}"
                )
                for m in rep["missing_strategies"] + rep["missing_masking"]:
                    print(f"  MISSING {m}")
                    missing = True
        return 1 if missing else 0
    parser.error("need --check or --coverage")
    return 2


if __name__ == "__main__":
    sys.exit(main())
