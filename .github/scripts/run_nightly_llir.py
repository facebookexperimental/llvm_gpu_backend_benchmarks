#!/usr/bin/env python3
"""Compile every committed kernel.llir with the installed LLVM (llc).

Nightly helper for .github/workflows/nightly-llvm.yml. CPU-only and
stdlib-only so it runs on stock GitHub-hosted runners: for each
``kernel.llir`` in the repo it invokes ``llc -mcpu=<arch>`` and records
whether AMDGCN code generation succeeded, plus a hash of the emitted
assembly so codegen drift between LLVM builds is visible in the
results file.

The arch is read from ``compile.json`` next to the kernel when present
(``arch`` field), else from the first ``gfxNNN`` path component, else
``gfx950``. Set ``LLVM_BIN`` to the directory containing ``llc`` when
it is not on PATH.

Writes ``nightly-results.json`` and ``nightly-summary.md`` in the
current directory. Exit status is 1 when any kernel fails to compile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ARCH_RE = re.compile(r"gfx\d+[a-z]?")


def find_llc() -> str:
    llvm_bin = os.environ.get("LLVM_BIN", "")
    candidates = [os.path.join(llvm_bin, "llc")] if llvm_bin else []
    candidates.append("llc")
    candidates += [f"llc-{v}" for v in range(30, 14, -1)]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit("llc not found (set LLVM_BIN to the LLVM bin directory)")


def arch_for(case_dir: Path) -> str:
    compile_json = case_dir / "compile.json"
    if compile_json.exists():
        try:
            arch = json.loads(compile_json.read_text()).get("arch", "")
            if ARCH_RE.fullmatch(str(arch)):
                return str(arch)
        except (ValueError, OSError):
            pass
    for part in case_dir.parts:
        if ARCH_RE.fullmatch(part):
            return part
    return "gfx950"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    llc = find_llc()
    version_lines = subprocess.run(
        [llc, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    version = next(
        (line.strip() for line in version_lines if "LLVM version" in line),
        version_lines[0].strip(),
    )
    kernels = sorted(root.rglob("kernel.llir"))
    results = []
    failures = 0
    for llir in kernels:
        arch = arch_for(llir.parent)
        proc = subprocess.run(
            [llc, "-mcpu=" + arch, "-o", "-", str(llir)],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        failures += 0 if ok else 1
        results.append(
            {
                "case": str(llir.parent.relative_to(root)),
                "arch": arch,
                "ok": ok,
                "asm_sha256": (
                    hashlib.sha256(proc.stdout.encode()).hexdigest() if ok else None
                ),
                "error": None if ok else proc.stderr.strip().splitlines()[-1:],
            }
        )
    payload = {
        "llvm": version,
        "llc": llc,
        "total": len(results),
        "failures": failures,
        "results": results,
    }
    Path("nightly-results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Nightly LLVM LLIR results",
        "",
        f"LLVM: `{version}`",
        f"Kernels: {len(results)} total, {failures} failed",
        "",
        "| Case | Arch | Result | ASM sha256 |",
        "|---|---|---|---|",
    ]
    for row in results:
        mark = "pass" if row["ok"] else "FAIL"
        digest = (row["asm_sha256"] or "-")[:12]
        lines.append(f"| {row['case']} | {row['arch']} | {mark} | {digest} |")
    Path("nightly-summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(results)} kernels, {failures} failures, {version}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
