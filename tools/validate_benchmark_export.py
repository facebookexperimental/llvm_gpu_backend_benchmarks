# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Validate exported benchmark corpus integrity using only the standard library.

Require complete, nonempty manifest coverage, matching architecture and case
identities, agreement with compile.json, and hashes matching artifact contents.
Reject missing kernels, ready-export failures, location remnants, and LLVM
modules missing required structure. This does not parse LLVM IR; verify_llir
provides deeper checks.

Optional changed-file input checks benchmark scope and kernel/config pairing::

    fbpython validate_benchmark_export.py --arch gfx942 --out ..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

BENCHMARKS_PREFIX = "fbcode/triton/benchmarks"

_LOC_DEF_RE = re.compile(r"^\s*#loc\d*\s*=")
_LOC_CALL_RE = re.compile(r" loc\(")
_DBG_ATTACH_RE = re.compile(r"!dbg\s+!\d+")
_DI_DEF_RE = re.compile(r"^!\d+\s*=\s*(distinct\s+)?!DI")
_ASM_LOC_RE = re.compile(r"^\s*\.(loc|file)\b")
_ASM_DEBUG_RE = re.compile(r"^\s*\.(cfi|Ltmp|debug)")
_ASM_SKIP_RE = re.compile(r"^\s*\.(Lfunc_begin|Lline_table)")

_KERNEL_RE = re.compile(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)")
_PTX_KERNEL_RE = re.compile(r"define ptx_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)")

# Assembled from pieces so this hand-written file does not carry the
# token and get classified as generated itself.
_GENERATED_TOKEN = "@" + "generated"
_KERNEL_FILES = (
    "kernel.ttir",
    "kernel.ttgir",
    "kernel.llir",
    "kernel.amdgcn",
    "kernel.ptx",
)


def _repo_rel(path: str) -> str:
    """Normalize a path to repo-relative form for scope checks."""
    path = path.replace(os.sep, "/")
    idx = path.find(BENCHMARKS_PREFIX)
    return path[idx:] if idx >= 0 else path


def check_benchmark_only(files: list[str], arch: str, scope: str = "") -> list[str]:
    """Return error strings for files outside the allowed benchmark subtree."""
    allowed = f"{BENCHMARKS_PREFIX}/{arch}/"
    if scope:
        allowed += scope.strip("/") + "/"
    problems = []
    for f in files:
        rel = _repo_rel(f)
        if not rel.startswith(allowed):
            problems.append(f"out of scope: {f} (must be under {allowed})")
    return problems


def _mlir_loc_problems(filename: str, text: str) -> list[str]:
    """MLIR loc definitions and attachments in ttir/ttgir."""
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        if _LOC_DEF_RE.match(line):
            problems.append(f"{filename}:{i}: MLIR #loc definition")
        elif _LOC_CALL_RE.search(line):
            problems.append(f"{filename}:{i}: MLIR loc() attachment")
    return problems


def _llir_loc_problems(filename: str, text: str) -> list[str]:
    """LLVM debug attachments and metadata in llir."""
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        if _DBG_ATTACH_RE.search(line):
            problems.append(f"{filename}:{i}: LLVM !dbg attachment")
        elif (
            _DI_DEF_RE.match(line)
            or line.startswith("!llvm.dbg.cu")
            or "@llvm.dbg." in line
        ):
            problems.append(f"{filename}:{i}: LLVM debug metadata")
    return problems


def _amdgcn_loc_problems(filename: str, text: str) -> list[str]:
    """Assembler loc/debug directives and sections in amdgcn."""
    problems = []
    in_debug = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(".section"):
            in_debug = ".debug" in stripped
            if in_debug:
                problems.append(f"{filename}:{i}: .debug section")
            continue
        if stripped.startswith(".text"):
            in_debug = False
        if (
            in_debug
            or _ASM_LOC_RE.match(line)
            or _ASM_DEBUG_RE.match(line)
            or _ASM_SKIP_RE.match(line)
        ):
            problems.append(f"{filename}:{i}: assembler loc/debug line")
    return problems


def find_generated_problems(filename: str, text: str) -> list[str]:
    """Return a problem when an exported kernel file lacks the marker."""
    if filename in _KERNEL_FILES and _GENERATED_TOKEN not in text:
        return [f"{filename}: missing generated-file marker"]
    return []


def find_loc_problems(filename: str, text: str) -> list[str]:
    """Return loc/debug remnants in one artifact (empty when clean)."""
    if filename in ("kernel.ttir", "kernel.ttgir"):
        return _mlir_loc_problems(filename, text)
    if filename == "kernel.llir":
        return _llir_loc_problems(filename, text)
    if filename in ("kernel.amdgcn", "kernel.ptx"):
        return _amdgcn_loc_problems(filename, text)
    return []


def check_llir_structure(text: str, arch: str = "gfx950") -> list[str]:
    """Well-formed LLVM IR module for the arch; returns problem strings."""
    problems = []
    if "; ModuleID" not in text[:500] and "source_filename" not in text[:500]:
        problems.append("missing LLVM module header")
    if arch.startswith("sm"):
        # NVPTX LLIR from the MLIR LLVM dialect carries no target-triple
        # line; the ptx_kernel definitions are the arch marker.
        if not _PTX_KERNEL_RE.search(text):
            problems.append("no ptx_kernel definition")
    else:
        if not re.search(r'target triple = "amdgpu[\d.]*-amd-amdhsa"', text) and (
            'target triple = "amdgcn-amd-amdhsa"' not in text
        ):
            problems.append("missing amdgcn triple")
        if not _KERNEL_RE.search(text):
            problems.append("no amdgpu_kernel definition")
    if "target datalayout" not in text:
        problems.append("missing datalayout")
    for marker in ("tt.func", "#ttg", "ttg.", "// -----//"):
        if marker in text:
            problems.append(f"contains MLIR marker {marker!r}")
    return problems


def check_hashes(cdir: str) -> list[str]:
    """Verify compile.json sha256 entries match the files on disk."""
    problems = []
    path = os.path.join(cdir, "compile.json")
    if not os.path.isfile(path):
        return ["missing compile.json"]
    try:
        info = _load_json(path)
    except (OSError, ValueError) as error:
        return [f"invalid compile.json: {error}"]
    files = info.get("files")
    if not isinstance(files, dict) or "llir" not in files:
        return ["compile.json must record the llir file and its hash"]
    for stage, entry in files.items():
        if not isinstance(entry, dict):
            problems.append(f"invalid file record for {stage}")
            continue
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or os.path.basename(filename) != filename
        ):
            problems.append(f"invalid filename for {stage}: {filename!r}")
            continue
        fpath = os.path.join(cdir, filename)
        if not os.path.isfile(fpath):
            problems.append(f"{entry.get('file')}: listed in compile.json but missing")
            continue
        with open(fpath, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != entry.get("sha256"):
            problems.append(f"{entry.get('file')}: sha256 mismatch (stage {stage})")
    return problems


def _load_json(path: str) -> dict:
    with open(path) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _manifest_entry(entry: dict, cases: dict, arch: str) -> tuple[tuple, list[str]]:
    key = tuple(entry.get(field) for field in ("source", "target", "case"))
    if not all(isinstance(part, str) and part for part in key):
        return (), ["manifest artifact has an invalid source, target, or case"]
    key = tuple(str(part) for part in key)
    name = "/".join(key)
    if key not in cases:
        return key, [f"{name}: manifest case is missing from the output"]
    info = cases[key]
    problems = []
    if entry.get("arch", arch) != arch or info.get("arch", arch) != arch:
        problems.append(f"{name}: architecture mismatch")
    if not entry.get("files") or entry.get("files") != info.get("files"):
        problems.append(f"{name}: manifest file records disagree with compile.json")
    return key, problems


def _manifest_coverage(
    artifacts: list, cases: dict, arch: str, source: str, target: str
) -> list[str]:
    problems = []
    seen = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            problems.append("invalid manifest artifact")
            continue
        if source and entry.get("source") != source:
            continue
        if target and entry.get("target") != target:
            continue
        key, errors = _manifest_entry(entry, cases, arch)
        problems.extend(errors)
        if key in seen:
            problems.append(f"duplicate manifest artifact: {key}")
        if key:
            seen.add(key)
    for key in cases.keys() - seen:
        problems.append(f"output case is absent from manifest: {'/'.join(key)}")
    if not cases:
        problems.append("output contains no kernel cases")
    return problems


def _manifest_problems(
    out_root: str, arch: str, cases: dict, source: str, target: str
) -> list[str]:
    try:
        manifest = _load_json(os.path.join(out_root, arch, "manifest.json"))
    except (OSError, ValueError) as error:
        return [f"invalid manifest.json: {error}"]
    problems = []
    if manifest.get("arch") != arch:
        problems.append(
            "manifest architecture does not match the requested architecture"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return problems + ["manifest must contain exported artifacts"]
    problems.extend(_manifest_coverage(artifacts, cases, arch, source, target))
    for failure in manifest.get("failures", []):
        if failure.get("status") not in ("provisional", "skipped"):
            problems.append(f"ready target failed: {failure.get('target')}")
    return problems


def _iter_case_dirs(out_root: str, arch: str, source: str, target: str):
    """Yield (source, target, case, casedir) under <out>/<arch>[/source[/target]]."""
    walk_root = os.path.join(out_root, arch)
    if not os.path.isdir(walk_root):
        return
    for dirpath, _dirnames, filenames in os.walk(walk_root):
        if not (
            "compile.json" in filenames
            or any(name.startswith("kernel.") for name in filenames)
        ):
            continue
        rel = os.path.relpath(dirpath, os.path.join(out_root, arch))
        parts = rel.split(os.sep)
        if len(parts) != 3:
            continue
        if source and parts[0] != source:
            continue
        if target and parts[1] != target:
            continue
        yield parts[0], parts[1], parts[2], dirpath


def validate_tree(out_root: str, arch: str, source: str = "", target: str = "") -> dict:
    """Run checks 2+3 over an export tree. Returns a JSON-serializable report."""
    report: dict = {"cases": 0, "ok": 0, "failures": {}}
    cases = {}
    for s, t, case, cdir in _iter_case_dirs(out_root, arch, source, target):
        name = f"{s}/{t}/{case}"
        problems = []
        for filename in _KERNEL_FILES:
            fpath = os.path.join(cdir, filename)
            if not os.path.isfile(fpath):
                continue
            with open(fpath) as f:
                text = f.read()
            problems.extend(find_loc_problems(filename, text))
            problems.extend(find_generated_problems(filename, text))
        llir_path = os.path.join(cdir, "kernel.llir")
        if os.path.isfile(llir_path):
            with open(llir_path) as f:
                problems.extend(check_llir_structure(f.read(), arch))
        problems.extend(check_hashes(cdir))
        if not os.path.isfile(llir_path):
            problems.append("missing kernel.llir")
        try:
            info = _load_json(os.path.join(cdir, "compile.json"))
        except (OSError, ValueError):
            info = {}
        key = (info.get("source", s), info.get("target", t), info.get("case", case))
        if not all(isinstance(part, str) and part for part in key):
            problems.append("invalid compile.json case identity")
        elif key in cases:
            problems.append("duplicate output case identity")
        else:
            cases[key] = info
        report["cases"] += 1
        if problems:
            report["failures"][name] = problems
        else:
            report["ok"] += 1
    manifest_problems = _manifest_problems(out_root, arch, cases, source, target)
    if manifest_problems:
        report["failures"]["manifest.json"] = manifest_problems
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a benchmark export.")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["gfx950", "gfx942", "gfx1250", "sm90", "sm100"],
    )
    parser.add_argument("--out", required=True, help="export root (parent of <arch>)")
    parser.add_argument(
        "--source", default="", help="limit to one source (e.g. tlx_ops)"
    )
    parser.add_argument("--target", default="", help="limit to one op (e.g. mm)")
    parser.add_argument(
        "--changed-files",
        default="",
        help="file with repo-relative changed paths (one per line) for the benchmark-only gate",
    )
    args = parser.parse_args(argv)

    failed = False
    if args.changed_files:
        with open(args.changed_files) as f:
            files = [line.strip() for line in f if line.strip()]
        scope = f"{args.source}/{args.target}".strip("/") if args.source else ""
        scope_problems = check_benchmark_only(files, args.arch, scope)
        if scope_problems:
            print(f"[validate] BENCHMARK-ONLY FAIL ({len(scope_problems)} files):")
            for p in scope_problems[:20]:
                print(f"  {p}")
            failed = True
        else:
            print(f"[validate] benchmark-only OK ({len(files)} files)")
        # Kernel + config pairing: each touched case dir needs both.
        dirs: dict[str, set[str]] = {}
        for f in files:
            rel = _repo_rel(f)
            parts = rel.split("/")
            if len(parts) >= 6:
                key = "/".join(parts[3:6])
                dirs.setdefault(key, set()).add(parts[-1])
        for key in sorted(dirs):
            names = dirs[key]
            has_kernel = any(n.startswith("kernel.") for n in names)
            has_config = "compile.json" in names
            status = "OK" if has_kernel and has_config else "INCOMPLETE"
            if status != "OK":
                failed = True
            print(f"[validate] {status} {key}: {sorted(names)}")

    report = validate_tree(args.out, args.arch, args.source, args.target)
    n_failed = len(report["failures"])
    print(
        f"[validate] cases={report['cases']} ok={report['ok']} failed={n_failed}",
        flush=True,
    )
    for name in sorted(report["failures"])[:20]:
        print(f"[validate] FAIL {name}:")
        for p in report["failures"][name][:10]:
            print(f"    {p}")
    return 1 if (failed or n_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
