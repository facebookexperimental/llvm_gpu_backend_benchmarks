# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for run_scheduled.py (cron dispatcher; fake runner, no ssh)."""

import os
import tempfile
import unittest

try:
    from triton.benchmarks.tools.run_scheduled import (
        arch_entry,
        load_hosts,
        main,
        remote_command,
        run_arch,
        ssh_command,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    from run_scheduled import (  # pyrefly: ignore [missing-import]
        arch_entry,
        load_hosts,
        main,
        remote_command,
        run_arch,
        ssh_command,
    )


class _Proc:
    __slots__ = ("returncode",)

    def __init__(self, returncode):
        self.returncode = returncode


class _Runner:
    """Fake subprocess.run: records argv, returns scripted exit codes."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def __call__(self, cmd, **_kwargs):
        self.calls.append(cmd)
        return _Proc(self.returncode)


HOSTS_TEMPLATE = """\
version: 1
method: ssh

gfx950:
  host: {h950}
  user: null
  fbsource: {f950}
  out: null
"""


def _write_hosts(body):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(body)
    tmp.close()
    return tmp.name


class RunScheduledTest(unittest.TestCase):
    def test_load_hosts_valid(self):
        path = _write_hosts(
            HOSTS_TEMPLATE.format(
                h950='"mi350x-dev-01"',
                f950='"/data/u/fbsource"',
            )
        )
        try:
            hosts = load_hosts(path)
            self.assertEqual(hosts["version"], 1)
            self.assertEqual(hosts["gfx950"]["host"], "mi350x-dev-01")
            self.assertIsNone(hosts["gfx950"]["user"])
        finally:
            os.unlink(path)

    def test_load_hosts_bad_version(self):
        path = _write_hosts("version: 2\nmethod: ssh\n")
        try:
            with self.assertRaises(ValueError):
                load_hosts(path)
        finally:
            os.unlink(path)

    def test_arch_entry_missing(self):
        self.assertEqual(arch_entry({}, "gfx950"), {})
        self.assertEqual(arch_entry({"gfx950": None}, "gfx950"), {})

    def test_remote_command(self):
        cmd = remote_command("/data/u/fbsource", "gfx950", "/tmp/out", False)
        self.assertIn("run_export.sh --arch gfx950 --out /tmp/out", cmd)
        self.assertIn("sl pull", cmd)
        self.assertNotIn("--include-provisional", cmd)
        prov = remote_command("/data/u/fbsource", "gfx950", "/tmp/out", True)
        self.assertIn("--include-provisional", prov)

    def test_ssh_command_batch_mode(self):
        cmd = ssh_command("h", "u", "echo hi")
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("BatchMode=yes", cmd)
        self.assertIn("u@h", cmd)
        self.assertEqual(ssh_command("h", None, "echo hi")[-2], "h")

    def test_run_arch_pending(self):
        runner = _Runner(returncode=0)
        status, _ = run_arch("gfx950", {}, "/tmp/out", False, runner)
        self.assertEqual(status, "pending")
        # host-side validation runs discover --check, never ssh.
        self.assertEqual(len(runner.calls), 1)
        self.assertNotIn("ssh", runner.calls[0][0])
        self.assertIn("--check", runner.calls[0])

    def test_run_arch_pending_validation_failed(self):
        runner = _Runner(returncode=1)
        status, _ = run_arch("gfx950", {}, "/tmp/out", False, runner)
        self.assertEqual(status, "failed")

    def test_run_arch_host_without_fbsource(self):
        runner = _Runner()
        status, _ = run_arch(
            "gfx950", {"host": "h", "fbsource": None}, "/tmp/out", False, runner
        )
        self.assertEqual(status, "failed")
        self.assertEqual(runner.calls, [])

    def test_run_arch_dispatch_ok(self):
        runner = _Runner(returncode=0)
        entry = {"host": "h", "user": None, "fbsource": "/fb", "out": None}
        status, detail = run_arch("gfx950", entry, "/tmp/out", False, runner)
        self.assertEqual(status, "exported")
        self.assertIn("h", detail)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][0], "ssh")

    def test_run_arch_dispatch_failed(self):
        runner = _Runner(returncode=255)
        entry = {"host": "h", "user": None, "fbsource": "/fb", "out": None}
        status, _ = run_arch("gfx950", entry, "/tmp/out", False, runner)
        self.assertEqual(status, "failed")

    def test_main_all_pending_exit_2(self):
        path = _write_hosts(HOSTS_TEMPLATE.format(h950="null", f950="null"))
        try:
            runner = _Runner(returncode=0)
            rc = main(
                ["--out", "/tmp/out", "--hosts", path, "--arch", "gfx950"],
                _runner=runner,
            )
            self.assertEqual(rc, 2)
        finally:
            os.unlink(path)

    def test_main_dispatch_exit_0(self):
        path = _write_hosts(
            HOSTS_TEMPLATE.format(
                h950='"h"',
                f950='"/fb"',
            )
        )
        try:
            runner = _Runner(returncode=0)
            rc = main(
                ["--out", "/tmp/out", "--hosts", path, "--arch", "gfx950"],
                _runner=runner,
            )
            self.assertEqual(rc, 0)
        finally:
            os.unlink(path)

    def test_main_dry_run(self):
        path = _write_hosts(
            HOSTS_TEMPLATE.format(
                h950='"h"',
                f950='"/fb"',
            )
        )
        try:
            runner = _Runner(returncode=0)
            rc = main(
                [
                    "--out",
                    "/tmp/out",
                    "--hosts",
                    path,
                    "--arch",
                    "gfx950",
                    "--dry-run",
                ],
                _runner=runner,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(runner.calls, [])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
