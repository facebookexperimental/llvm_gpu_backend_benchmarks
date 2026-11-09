# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Scheduled (cron) entrypoint: dispatch per-arch exports to GPU hosts.

Reads ``hosts.yaml``; each arch with a configured host runs remotely over
ssh (non-interactive ``BatchMode``, so cron never hangs on a password
prompt). Arches without a host run host-side validation (``discover
--check``) and are reported pending instead of failing silently.

Exit status: 0 when every arch exported; 1 when a remote export failed
or host-side validation failed; 2 when any arch is pending for lack of
a host.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ARCHES = ("gfx950",)
VERSION = 1
EXPORT_SUBDIR = os.path.join(
    "fbcode", "scripts", "njriasan", "triton-tlx-llvm-benchmark-export"
)
DEFAULT_HOSTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hosts.yaml")


def _yaml_loader():
    """discover.load_yaml_subset, resolvable under buck and plain python3."""
    try:
        from triton.benchmarks.tools.discover import load_yaml_subset
    except ImportError:
        # Plain python3: sibling file in the same directory.
        from discover import load_yaml_subset
    return load_yaml_subset


def load_hosts(path: str = DEFAULT_HOSTS) -> dict:
    """Load and version-check hosts.yaml (stdlib only)."""
    spec = _yaml_loader()(path)
    if spec.get("version") != VERSION:
        raise ValueError(f"{path}: expected version {VERSION}")
    if spec.get("method", "ssh") != "ssh":
        raise ValueError(f"{path}: only method=ssh is supported")
    return spec


def arch_entry(hosts: dict, arch: str) -> dict:
    """Per-arch host entry (empty when unconfigured)."""
    entry = hosts.get(arch) or {}
    return entry if isinstance(entry, dict) else {}


def remote_command(fbsource: str, arch: str, out: str, provisional: bool) -> str:
    """Remote shell: refresh the checkout, then run the export stages."""
    workdir = os.path.join(fbsource, EXPORT_SUBDIR)
    cmd = (
        f"cd {shlex.quote(workdir)} && sl pull >/dev/null 2>&1 && "
        f"sl goto remote/master >/dev/null 2>&1 && "
        f"./run_export.sh --arch {arch} --out {shlex.quote(out)}"
    )
    if provisional:
        cmd += " --include-provisional"
    return cmd


def ssh_command(host: str, user: str | None, remote_cmd: str) -> list[str]:
    """Non-interactive ssh invocation (cron-safe: never prompts)."""
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        target,
        remote_cmd,
    ]


def validate_hostside(runner) -> bool:
    """Host-side validation for arches without a GPU host."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "discover.py")
    proc = runner([sys.executable, script, "--check"])
    return proc.returncode == 0


def run_arch(arch, entry, out, provisional, runner) -> tuple[str, str]:
    """Run one arch: dispatch, or validate + report pending. Returns (status, detail)."""
    host = entry.get("host")
    if not host:
        ok = validate_hostside(runner)
        if not ok:
            return "failed", "host-side discover --check failed"
        return "pending", "no host configured (fill in hosts.yaml)"
    if not entry.get("fbsource"):
        return "failed", "host set without fbsource checkout path"
    remote_out = entry.get("out") or out
    cmd = ssh_command(
        host,
        entry.get("user"),
        remote_command(entry.get("fbsource") or "", arch, remote_out, provisional),
    )
    proc = runner(cmd)
    if proc.returncode == 0:
        return "exported", f"{host}:{remote_out}"
    return "failed", f"ssh exit {proc.returncode}"


def main(argv=None, _runner=None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch scheduled exports.")
    parser.add_argument("--out", required=True, help="export root")
    parser.add_argument(
        "--arch", action="append", choices=list(ARCHES), help="repeatable"
    )
    parser.add_argument("--hosts", default=DEFAULT_HOSTS, help="hosts.yaml path")
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print actions without running"
    )
    args = parser.parse_args(argv)
    runner = _runner or subprocess.run
    try:
        hosts = load_hosts(args.hosts)
    except (OSError, ValueError) as e:
        print(f"[sched] ERROR: {e}", flush=True)
        return 1
    arches = args.arch or list(ARCHES)
    if args.dry_run:
        for arch in arches:
            entry = arch_entry(hosts, arch)
            if entry.get("host"):
                print(f"[sched] would dispatch {arch} to {entry['host']}")
            else:
                print(f"[sched] would validate {arch} host-side (pending)")
        return 0
    statuses = {}
    for arch in arches:
        status, detail = run_arch(
            arch, arch_entry(hosts, arch), args.out, args.include_provisional, runner
        )
        statuses[arch] = status
        print(f"[sched] {status} {arch} ({detail})", flush=True)
    if any(s == "failed" for s in statuses.values()):
        return 1
    if any(s == "pending" for s in statuses.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
