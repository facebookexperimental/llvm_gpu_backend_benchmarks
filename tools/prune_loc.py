# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Prune all location (loc/debug) info from exported benchmark artifacts.

Exported ``kernel.ttir``/``kernel.ttgir`` carry MLIR ``#loc`` definitions
with absolute buck-out paths, ``kernel.llir`` carries ``!dbg``/``!DI*``
metadata, and ``kernel.amdgcn`` carries ``.file``/``.loc`` directives. None
of it affects codegen (the LLIR round-trip consumes instructions, not debug
attachments), but it embeds host paths, so every regeneration diffs every
file. Stripping it makes artifacts host-independent and smaller.

Run after the export stages (``generate_search_space.sh`` stage 6)::

    python3 prune_loc.py --arch gfx950 --out /data/llvm_export_20250101

Stdlib only (no torch/triton/GPU): plain ``python3`` runs work. Idempotent;
``--dry-run`` reports without writing. File hashes in ``compile.json`` and
``manifest.json`` are refreshed for every rewritten file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

STAGE_FILES = {
    "ttir": "kernel.ttir",
    "ttgir": "kernel.ttgir",
    "llir": "kernel.llir",
    "amdgcn": "kernel.amdgcn",
}

_LOC_DEF_RE = re.compile(r"^\s*#loc\d*\s*=")
_DBG_ATTACH_RE = re.compile(r",?\s*!dbg\s+!\d+")
_DI_DEF_RE = re.compile(r"^!\d+\s*=\s*(distinct\s+)?!DI")
_ASM_LOC_RE = re.compile(r"^\s*\.(loc|file)\b")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_loc_calls(line: str) -> str:
    """Remove every `` loc(...)`` attachment (balanced parens) from a line."""
    out: list[str] = []
    i = 0
    while True:
        j = line.find(" loc(", i)
        if j < 0:
            out.append(line[i:])
            return "".join(out)
        out.append(line[i:j])
        depth, k, in_str = 0, j + 1, False
        while k < len(line):
            c = line[k]
            if in_str:
                in_str = c != '"'
            elif c == '"':
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1


def strip_mlir(text: str) -> str:
    """Strip MLIR loc definitions and attachments (ttir/ttgir)."""
    lines = []
    for line in text.splitlines():
        if _LOC_DEF_RE.match(line):
            continue
        lines.append(_strip_loc_calls(line).rstrip())
    return "\n".join(lines) + "\n"


_META_DEF_RE = re.compile(r"^!(\d+)\s*=")
_META_REF_RE = re.compile(r"!(\d+)")


def strip_llir(text: str) -> str:
    """Strip LLVM ``!dbg`` attachments and ``!DI*`` metadata (llir)."""
    kept: list[str] = []
    defs: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("!llvm.dbg.cu") or "@llvm.dbg." in line:
            continue
        m = _META_DEF_RE.match(line)
        if m:
            defs[m.group(1)] = line
        else:
            kept.append(_DBG_ATTACH_RE.sub("", line).rstrip())
    dropped = {n for n, d in defs.items() if _DI_DEF_RE.match(d)}
    changed = True
    while changed:
        changed = False
        for num, definition in defs.items():
            if num not in dropped and set(_META_REF_RE.findall(definition)) & dropped:
                dropped.add(num)
                changed = True
    kept.extend(d for n, d in defs.items() if n not in dropped)
    return "\n".join(kept) + "\n"


_ASM_DEBUG_RE = re.compile(r"^\s*\.(cfi|Ltmp|debug)")


def strip_amdgcn(text: str) -> str:
    """Strip assembler debug directives and sections (amdgcn)."""
    lines: list[str] = []
    in_debug = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".section"):
            in_debug = ".debug" in stripped
            if in_debug:
                continue
        elif stripped.startswith(".text"):
            in_debug = False
        if in_debug or _ASM_LOC_RE.match(line) or _ASM_DEBUG_RE.match(line):
            continue
        if stripped.startswith((".Lfunc_begin", ".Lline_table")):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def strip_text(filename: str, text: str) -> str:
    """Strip loc info for one artifact filename (unchanged if unknown)."""
    if filename in ("kernel.ttir", "kernel.ttgir"):
        return strip_mlir(text)
    if filename == "kernel.llir":
        return strip_llir(text)
    if filename == "kernel.amdgcn":
        return strip_amdgcn(text)
    return text


def _iter_case_dirs(out_root: str, arch: str):
    """Yield (source, target, case, casedir) under <out>/<arch>."""
    base = os.path.join(out_root, arch)
    if not os.path.isdir(base):
        return
    for source in sorted(os.listdir(base)):
        sdir = os.path.join(base, source)
        if not os.path.isdir(sdir):
            continue
        for target in sorted(os.listdir(sdir)):
            tdir = os.path.join(sdir, target)
            if not os.path.isdir(tdir):
                continue
            for case in sorted(os.listdir(tdir)):
                cdir = os.path.join(tdir, case)
                if os.path.isdir(cdir):
                    yield source, target, case, cdir


def _prune_case(cdir: str, dry_run: bool) -> tuple[int, int, dict]:
    """Strip one case dir. Returns (files_changed, bytes_saved, new_hashes)."""
    changed, saved, hashes = 0, 0, {}
    for stage, filename in STAGE_FILES.items():
        path = os.path.join(cdir, filename)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            old = f.read()
        new = strip_text(filename, old)
        hashes[stage] = _sha256_text(new)
        if new == old:
            continue
        changed += 1
        saved += len(old.encode("utf-8")) - len(new.encode("utf-8"))
        if not dry_run:
            with open(path, "w") as f:
                f.write(new)
    return changed, saved, hashes


def _refresh_hashes(cdir: str, hashes: dict, dry_run: bool) -> dict:
    """Sync compile.json file hashes. Returns the updated files mapping."""
    path = os.path.join(cdir, "compile.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        info = json.load(f)
    files = info.get("files") or {}
    for stage, digest in hashes.items():
        if stage in files:
            files[stage]["sha256"] = digest
    if not dry_run:
        with open(path, "w") as f:
            json.dump(info, f, indent=2, sort_keys=True)
            f.write("\n")
    return files


def prune_tree(out_root: str, arch: str, dry_run: bool = False) -> dict:
    """Prune loc info under <out>/<arch> and refresh all recorded hashes."""
    stats = {"cases": 0, "files_changed": 0, "bytes_saved": 0}
    updated: dict[tuple, dict] = {}
    for source, target, case, cdir in _iter_case_dirs(out_root, arch):
        stats["cases"] += 1
        changed, saved, hashes = _prune_case(cdir, dry_run)
        stats["files_changed"] += changed
        stats["bytes_saved"] += saved
        updated[(source, target, case)] = _refresh_hashes(cdir, hashes, dry_run)
    manifest_path = os.path.join(out_root, arch, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest.get("artifacts", []):
            key = (entry.get("source"), entry.get("target"), entry.get("case"))
            if key in updated and updated[key]:
                entry["files"] = updated[key]
        if not dry_run:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prune loc info from exports.")
    parser.add_argument(
        "--arch", required=True, choices=["gfx950", "gfx942", "gfx1250"]
    )
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)
    stats = prune_tree(args.out, args.arch, dry_run=args.dry_run)
    print(
        f"[prune-loc] cases={stats['cases']} "
        f"files_changed={stats['files_changed']} "
        f"bytes_saved={stats['bytes_saved']}" + (" (dry-run)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
