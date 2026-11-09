# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Prune an LLVM-stage export to top configs and top shapes by latency.

Two filters, applied in order:

1. Top configs: cases sharing a (stem, kernel) group (``__k<N>`` configs
   of one kernel from an autotune sweep) keep the ``--top-configs``
   lowest ``latency_ms`` (default 3 per the design). Distinct kernels
   of one case group independently, so slow kernels are never dropped
   as if they were slow configs.
2. Shape cap: within each (source, target, mode, strategy, masking)
   group, shapes are ranked by total winner latency (sum over kernels of
   each kernel's fastest config); the longest-latency shape is kept plus
   the ``--top-shapes - 1`` shapes nearest the group median (default:
   longest + median, the two shapes of varying computational intensity).
   Mode is (causal, backward) from ``input_spec``; strategy/masking are
   the ``extra.*`` candidate labels, so every candidate survives pruning.

Shapes with any unmeasured case are exempt from the cap (kept). Dropped case
dirs are deleted and the manifest rewritten. Idempotent.

Run after the export stages (``run_export.sh`` runs it as stage 6)::

    python3 prune_top.py --arch gfx950 --out /data/llvm_export_20250101
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

MODE_KEYS = ("causal", "bwd")

# input_spec keys that never identify a shape (modes + export metadata).
NON_SHAPE_KEYS = frozenset(
    {"causal", "bwd", "space", "op", "args", "input_loader", "subops"}
)


def config_stem(case: str) -> str:
    """Strip the ``__k<N>`` dedup suffix; the stem identifies one sweep."""
    return re.sub(r"__k\d+$", "", case)


def mode_key(artifact: dict) -> tuple:
    spec = artifact.get("input_spec") or {}
    return tuple(bool(spec.get(k)) for k in MODE_KEYS)


def candidate_key(artifact: dict) -> tuple:
    """(strategy, masking) labels the exporter recorded in ``extra``."""
    extra = artifact.get("extra") or {}
    return (extra.get("strategy"), extra.get("masking"))


def shape_key(artifact: dict):
    """Identity of the shape a case benchmarks."""
    if artifact.get("source") == "tritonbench":
        compile = artifact.get("compile") or {}
        return (
            compile.get("tritonbench_benchmark"),
            compile.get("tritonbench_fn"),
            compile.get("tritonbench_input_id"),
        )
    spec = artifact.get("input_spec") or {}
    return tuple(
        sorted((k, str(v)) for k, v in spec.items() if k not in NON_SHAPE_KEYS)
    )


def _latency(artifact: dict):
    return artifact.get("latency_ms")


def filter_top_configs(artifacts: list[dict], top_n: int):
    """Keep the ``top_n`` fastest cases per kernel. Returns (kept, dropped)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for a in artifacts:
        groups[
            (
                a.get("source"),
                a.get("target"),
                config_stem(a.get("case", "")),
                a.get("kernel_name"),
            )
        ].append(a)
    kept, dropped = [], []
    for members in groups.values():
        if len(members) <= top_n:
            kept.extend(members)
            continue
        ranked = sorted(members, key=lambda a: (_latency(a) is None, _latency(a)))
        kept.extend(ranked[:top_n])
        dropped.extend(ranked[top_n:])
    return kept, dropped


def cap_shapes(artifacts: list[dict], top_n: int):
    """Keep the longest- and median-latency shapes per candidate group.

    Groups are (source, target, mode, strategy, masking). A shape's latency
    is the sum over its kernels of each kernel's fastest config. Shapes rank
    slowest-first; rank 0 (longest) is always kept plus the ``top_n - 1``
    shapes nearest the group median (ties toward slower), so the default
    ``top_n=2`` keeps longest + median. Shapes with any unmeasured case are
    exempt (kept). Returns (kept, dropped).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for a in artifacts:
        key = (a.get("source"), a.get("target"), mode_key(a), *candidate_key(a))
        groups[key].append(a)
    kept, dropped = [], []
    for members in groups.values():
        shapes: dict[tuple, list[dict]] = defaultdict(list)
        for a in members:
            shapes[shape_key(a)].append(a)
        if len(shapes) <= top_n:
            kept.extend(members)
            continue
        scored = []
        for key, cases in shapes.items():
            by_kernel: dict[str | None, list[dict]] = defaultdict(list)
            for a in cases:
                by_kernel[a.get("kernel_name")].append(a)
            total, measurable = 0.0, True
            for kernel_cases in by_kernel.values():
                latencies = [_latency(a) for a in kernel_cases]
                if any(v is None for v in latencies):
                    measurable = False
                    break
                total += min(v for v in latencies if v is not None)
            scored.append((key, cases, measurable, total))
        exempt = [cases for _, cases, measurable, _ in scored if not measurable]
        ranked = sorted(
            [
                (key, cases, total)
                for key, cases, measurable, total in scored
                if measurable
            ],
            key=lambda t: t[2],
            reverse=True,
        )
        median = len(ranked) // 2
        # Slowest rank first, then outward from the median (ties: slower wins).
        near_median = sorted(range(len(ranked)), key=lambda i: (abs(i - median), i))
        keep_idx = {0} | set(near_median[: max(top_n - 1, 0)])
        for cases in exempt:
            kept.extend(cases)
        for i, (_, cases, _) in enumerate(ranked):
            if i in keep_idx:
                kept.extend(cases)
            else:
                dropped.extend(cases)
    return kept, dropped


def case_dir_for(out_root: str, artifact: dict) -> str:
    try:
        from triton.benchmarks.tools.capture import case_dir
    except ImportError:
        # Plain python3: capture.py is a sibling file.
        from capture import case_dir  # pyrefly: ignore [missing-import]

    return case_dir(
        out_root,
        artifact["arch"],
        artifact["source"],
        artifact["target"],
        artifact["case"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prune export to top latencies.")
    parser.add_argument("--arch", required=True, choices=["gfx950"])
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument("--top-configs", type=int, default=3)
    parser.add_argument("--top-shapes", type=int, default=2)
    parser.add_argument(
        "--dry-run", action="store_true", help="report without deleting"
    )
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.out, args.arch, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    artifacts = manifest.get("artifacts", [])
    kept, dropped_configs = filter_top_configs(artifacts, args.top_configs)
    kept, dropped_shapes = cap_shapes(kept, args.top_shapes)
    kept_ids = {id(a) for a in kept}
    dropped = [a for a in artifacts if id(a) not in kept_ids]
    print(
        f"[prune] artifacts={len(artifacts)} kept={len(kept)} "
        f"dropped_configs={len(dropped_configs)} dropped_shapes={len(dropped_shapes)}"
    )
    for a in sorted(dropped, key=lambda a: (a["source"], a["target"], a["case"])):
        print(f"[prune] DROP {a['source']}/{a['target']}/{a['case']}")
    if args.dry_run:
        return 0
    for a in dropped:
        shutil.rmtree(case_dir_for(args.out, a), ignore_errors=True)
    manifest["artifacts"] = kept
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[prune] rewrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
