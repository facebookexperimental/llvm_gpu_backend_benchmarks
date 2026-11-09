# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Export LLVM-stage inputs for tritonbench operators on AMD hardware.

Host-side driver (no torch/triton import): shells out to the Triton-Beta
tritonbench binary once per targets.yaml entry and converts its ``--dump-ir``
output into the hardware-sorted export layout::

    buck2 run fbcode//triton/benchmarks/tools:export_tritonbench -- \
        --arch gfx950 --out /tmp/llvm_export

Must run ON the target MI350X GPU host (gfx950):
``buck2 run`` executes locally. Multi-config kernels export the autotuner
winner: tritonbench benchmarks all candidates and ``--dump-ir`` captures the
selected launch. Subops that never compile Triton kernels (aten/eager/mkl
baselines) yield no ``.llir`` files and are recorded as filtered -- no
hand-maintained subop allowlist needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

try:
    from triton.benchmarks.tools.capture import (
        append_run_manifest,
        case_dir,
        MANIFEST_VERSION,
        sha256_text,
    )
except ImportError:
    # Plain python3: capture.py is a sibling file.
    from capture import (  # pyrefly: ignore [missing-import]
        append_run_manifest,
        case_dir,
        MANIFEST_VERSION,
        sha256_text,
    )
try:
    from triton.benchmarks.tools.discover import load_yaml_subset
except ImportError:
    # Plain python3: sibling file in the same directory.
    from discover import load_yaml_subset  # pyrefly: ignore [missing-import]

TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.yaml")

STAGE_EXTS = ("llir", "ttgir", "ttir", "amdgcn")
# tritonbench dump_ir names: {kernel}_k{kid}_input_x{input_id}.{ir} (and
# {kernel}_k{kid}_x{input_id}.sass for cubin, which never appears on AMD).
IR_RE = re.compile(
    r"(?P<kernel>.+)_k(?P<kid>\d+)_input_x(?P<input_id>\d+)\.(?P<stage>llir|ttgir|ttir|amdgcn)$"
)


def entry_target(entry: dict, idx: int, op_counts: dict[str, int]) -> str:
    """Target dir/name for one targets.yaml entry.

    Entries usually map 1:1 to ops, but an op may appear in several entries
    (e.g. fp8_gemm_rowwise fast_accum + prefill). Those would write the same
    case dirs and silently overwrite each other, so disambiguate repeats by
    their --logging-group (falling back to the entry index).
    """
    op = entry["op"]
    if op_counts.get(op, 0) < 2:
        return op
    m = re.search(r"--logging-group\s+(\S+)", entry.get("args") or "")
    if m:
        return m.group(1)
    return f"{op}__e{idx}"


def parse_run_latencies(log_text: str) -> dict[str, list[float | None]]:
    """Parse per-input latencies (ms) from a run_beta results table.

    Returns {subop: [ms per table row]}. Rows follow input order, so row i is
    input x{i}. Cells that do not parse yield None instead of failing.
    """
    lines = log_text.splitlines()
    header_idx, columns = -1, []
    for i, line in enumerate(lines):
        if "-latency" in line and re.match(r"\s*\(.*\)\s+\S", line):
            rest = re.sub(r"^\s*\(.*?\)\s*", "", line)
            columns = rest.split()
            header_idx = i
            break
    if header_idx < 0:
        return {}
    lat_cols = [
        (name[: -len("-latency")], idx)
        for idx, name in enumerate(columns)
        if name.endswith("-latency")
    ]
    table: dict[str, list[float | None]] = {name: [] for name, _ in lat_cols}
    for line in lines[header_idx + 1 :]:
        if not re.match(r"\s*\(.*?\)\s+\S", line):
            if line.strip() and not line.strip().startswith("-"):
                break
            continue
        rest = re.sub(r"^\s*\(.*?\)\s*", "", line)
        toks = [t for t in rest.split() if not t.startswith("(±")]
        for name, idx in lat_cols:
            try:
                table[name].append(float(toks[idx]))
            except (IndexError, ValueError):
                table[name].append(None)
    return table


def latency_for(latencies, fn_name: str, input_id: str) -> float | None:
    """Look up one case's latency from a parsed run table (row = input)."""
    if not latencies:
        return None
    col = latencies.get(fn_name)
    if col is None and len(latencies) == 1:
        col = next(iter(latencies.values()))
    if col is None:
        return None
    try:
        return col[int(input_id)]
    except (IndexError, ValueError):
        return None


def build_command(entry: dict, dump_dir: str, mode: str, buck_mode: str) -> list[str]:
    cmd = [
        "buck2",
        "--client-metadata",
        "id=llvm_benchmarks_tools",
        "run",
        buck_mode,
        "fbcode//pytorch/tritonbench:run_beta",
        "--",
    ]
    if entry.get("args"):
        cmd.extend(entry["args"].split())
    else:
        cmd.extend(["--op", entry["op"], "--mode", entry.get("mode", "fwd")])
        if entry.get("input_loader"):
            cmd.extend(["--input-loader", entry["input_loader"]])
        subops = entry.get("subops")
        if isinstance(subops, list):
            cmd.extend(["--only", ",".join(subops)])
    cmd.extend(["--dump-ir", dump_dir])
    if mode == "opt-amd-gpu":
        pass
    return cmd


def collect_cases(
    dump_dir: str,
    out_root: str,
    arch: str,
    op: str,
    entry: dict,
    target: str,
    latencies: dict[str, list[float | None]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Move dump-ir output into case dirs. Returns (entries, filtered_fn_names)."""
    entries: list[dict] = []
    filtered: list[str] = []
    # Layout: <dump>/<benchmark>/<fn>/<files>. A (benchmark, fn) dir with no
    # .llir means the subop compiled no Triton kernels (non-triton baseline).
    for bench_dir in sorted(glob.glob(os.path.join(dump_dir, "*"))):
        if not os.path.isdir(bench_dir):
            continue
        benchmark = os.path.basename(bench_dir)
        for fn_dir in sorted(glob.glob(os.path.join(bench_dir, "*"))):
            if not os.path.isdir(fn_dir):
                continue
            fn_name = os.path.basename(fn_dir)
            llirs = sorted(glob.glob(os.path.join(fn_dir, "*.llir")))
            if not llirs:
                filtered.append(f"{benchmark}/{fn_name}")
                continue
            for llir_path in llirs:
                m = IR_RE.match(os.path.basename(llir_path))
                kernel = (
                    m.group("kernel")
                    if m
                    else os.path.splitext(os.path.basename(llir_path))[0]
                )
                input_id = m.group("input_id") if m else "0"
                kid = m.group("kid") if m else "0"
                case = f"{benchmark}__{fn_name}__{kernel}_x{input_id}_k{kid}"
                out_dir = case_dir(out_root, arch, "tritonbench", target, case)
                latency_ms = latency_for(latencies, fn_name, input_id)
                os.makedirs(out_dir, exist_ok=True)
                files: dict[str, dict] = {}
                for stage in STAGE_EXTS:
                    src = os.path.splitext(llir_path)[0] + f".{stage}"
                    if os.path.exists(src):
                        with open(src) as f:
                            text = f.read()
                        with open(os.path.join(out_dir, f"kernel.{stage}"), "w") as f:
                            f.write(text)
                        files[stage] = {
                            "file": f"kernel.{stage}",
                            "sha256": sha256_text(text),
                        }
                with open(llir_path) as f:
                    llir_text = f.read()
                entry_json = {
                    "manifest_version": MANIFEST_VERSION,
                    "arch": arch,
                    "source": "tritonbench",
                    "target": target,
                    "case": case,
                    "kernel_name": kernel,
                    "latency_ms": latency_ms,
                    "latency_kind": (
                        "tritonbench_log" if latency_ms is not None else None
                    ),
                    "files": files,
                    "compile": {
                        "stages_present": sorted(files.keys()),
                        "tritonbench_benchmark": benchmark,
                        "tritonbench_fn": fn_name,
                        "tritonbench_input_id": input_id,
                    },
                    "input_spec": {
                        "op": op,
                        "args": entry.get("args"),
                        "input_loader": entry.get("input_loader"),
                        "subops": entry.get("subops"),
                    },
                    "extra": {"llir_sha256": sha256_text(llir_text)},
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(os.path.join(out_dir, "compile.json"), "w") as f:
                    json.dump(entry_json, f, indent=2, sort_keys=True)
                    f.write("\n")
                entries.append(entry_json)
    return entries, filtered


def export_entry(
    op_idx: int,
    entry: dict,
    arch: str,
    out_root: str,
    buck_mode: str,
    timeout_s: int,
    keep_dump: bool,
    target: str,
) -> tuple[list[dict], list[dict]]:
    op = entry["op"]
    dump_dir = tempfile.mkdtemp(prefix=f"tb_dump_{arch}_{op}_{op_idx}_")
    log_path = os.path.join(out_root, arch, "tritonbench", f"run_{op}_{op_idx}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = build_command(entry, dump_dir, arch, buck_mode)
    print(f"[export] {' '.join(cmd)}", flush=True)
    try:
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s
            )
    except subprocess.TimeoutExpired:
        return [], [
            {
                "target": target,
                "case": "*",
                "error": f"timeout after {timeout_s}s",
                "log": log_path,
            }
        ]
    if proc.returncode != 0:
        return [], [
            {
                "target": target,
                "case": "*",
                "error": f"run_beta exited {proc.returncode}",
                "log": log_path,
            }
        ]
    try:
        with open(log_path) as f:
            latencies = parse_run_latencies(f.read())
    except OSError:
        latencies = {}
    if not latencies:
        print(f"[export] {op}: no latency table in {log_path}")
    entries, filtered = collect_cases(
        dump_dir, out_root, arch, op, entry, target, latencies
    )
    failures = []
    if not entries:
        failures.append(
            {
                "target": target,
                "case": "*",
                "error": "no .llir captured (no triton subops ran?)",
                "log": log_path,
                "filtered": filtered,
            }
        )
    else:
        print(
            f"[export] {op}: {len(entries)} cases, filtered non-triton: {filtered or 'none'}"
        )
    if not keep_dump:
        shutil.rmtree(dump_dir, ignore_errors=True)
    return entries, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export tritonbench LLVM-stage inputs."
    )
    parser.add_argument("--arch", required=True, choices=["gfx950"])
    parser.add_argument(
        "--out",
        required=True,
        help="output root (arch-sorted layout is created under it)",
    )
    parser.add_argument("--targets", default=TARGETS_FILE)
    parser.add_argument("--op", default=None, help="only export this op")
    parser.add_argument("--buck-mode", default="@//mode/opt-amd-gpu")
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument(
        "--keep-dump", action="store_true", help="keep raw --dump-ir dirs"
    )
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running"
    )
    args = parser.parse_args(argv)

    spec = load_yaml_subset(args.targets)
    entries = spec["arch"][args.arch].get("tritonbench", []) or []
    # Counted over the full list (not the --op filter) so target dirs are
    # stable no matter which subset is run.
    op_counts: dict[str, int] = {}
    for entry in entries:
        op_counts[entry["op"]] = op_counts.get(entry["op"], 0) + 1
    all_entries, all_failures = [], []
    for idx, entry in enumerate(entries):
        if args.op and entry["op"] != args.op:
            continue
        if entry.get("status") == "provisional" and not args.include_provisional:
            print(
                f"[export] skip provisional {entry['op']} (pass --include-provisional)"
            )
            continue
        target = entry_target(entry, idx, op_counts)
        if args.dry_run:
            print(" ".join(build_command(entry, "<dump>", args.arch, args.buck_mode)))
            continue
        e, f = export_entry(
            idx,
            entry,
            args.arch,
            args.out,
            args.buck_mode,
            args.timeout_s,
            args.keep_dump,
            target,
        )
        status = entry.get("status", "ready")
        for x in f:
            x.setdefault("status", status)
        all_entries.extend(e)
        all_failures.extend(f)
    if not args.dry_run:
        path = append_run_manifest(
            args.out, args.arch, "tritonbench", all_entries, all_failures
        )
        print(
            f"[export] {len(all_entries)} cases, {len(all_failures)} failures -> {path}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
