# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for validate_benchmark_export.py (plain python3)."""

import hashlib
import json
import os
import tempfile
import unittest

try:
    from triton.benchmarks.tools.validate_benchmark_export import (
        check_benchmark_only,
        check_hashes,
        check_llir_structure,
        find_generated_problems,
        find_loc_problems,
        main,
        validate_tree,
    )
except ImportError:
    # Plain python3 (no buck packaging): the module lives next to this test.
    # pyrefly: ignore [missing-import]
    from validate_benchmark_export import (
        check_benchmark_only,
        check_hashes,
        check_llir_structure,
        find_generated_problems,
        find_loc_problems,
        main,
        validate_tree,
    )

_TOKEN = "@" + "generated"

_CLEAN_LLIR = (
    f"; {_TOKEN} by triton.benchmarks.tools.capture; do not edit.\n"
    + """\
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32"
define amdgpu_kernel void @k(ptr %a) {
  %v = load i32, ptr %a
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
"""
)

_DIRTY_LLIR = _CLEAN_LLIR + '!1 = !DIFile(filename: "gfx942.py", directory: "/tmp")\n'

_CLEAN_TTIR = f"""\
// {_TOKEN} by triton.benchmarks.tools.capture; do not edit.
module {{
  tt.func public @k(%a: !tt.ptr<i64>) {{
    %c = arith.constant 128 : i32
    tt.return
  }}
}}
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_case(root: str, llir: str = _CLEAN_LLIR) -> str:
    cdir = os.path.join(root, "gfx942", "tlx_ops", "mm", "case1")
    os.makedirs(cdir)
    files = {
        "kernel.ttir": _CLEAN_TTIR,
        "kernel.ttgir": _CLEAN_TTIR,
        "kernel.llir": llir,
        "kernel.amdgcn": f"; {_TOKEN} by triton.benchmarks.tools.capture\n\t.text\n\ts_mov_b32 s0, 0\n",
    }
    compile_info = {
        "arch": "gfx942",
        "source": "tlx_ops",
        "target": "mm",
        "case": "case1",
        "files": {
            stage: {"file": name, "sha256": _sha(text)}
            for stage, (name, text) in zip(
                ("ttir", "ttgir", "llir", "amdgcn"), files.items()
            )
        },
    }
    for name, text in files.items():
        with open(os.path.join(cdir, name), "w") as f:
            f.write(text)
    with open(os.path.join(cdir, "compile.json"), "w") as f:
        json.dump(compile_info, f)
    manifest = {
        "arch": "gfx942",
        "artifacts": [compile_info],
        "failures": [],
    }
    with open(os.path.join(root, "gfx942", "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return cdir


class GateTest(unittest.TestCase):
    def test_benchmark_only(self) -> None:
        ok = ["fbcode/triton/benchmarks/gfx942/tlx_ops/mm/case1/kernel.llir"]
        self.assertEqual(check_benchmark_only(ok, "gfx942", "tlx_ops/mm"), [])
        bad = ok + ["fbcode/triton/beta/foo.py"]
        self.assertEqual(len(check_benchmark_only(bad, "gfx942", "tlx_ops/mm")), 1)
        wrong_arch = ["fbcode/triton/benchmarks/gfx950/tlx_ops/mm/case1/kernel.llir"]
        self.assertEqual(len(check_benchmark_only(wrong_arch, "gfx942")), 1)

    def test_loc_detection(self) -> None:
        self.assertEqual(find_loc_problems("kernel.ttir", _CLEAN_TTIR), [])
        dirty_mlir = _CLEAN_TTIR + "#loc = loc(/tmp/x.py:1:1)\n"
        self.assertTrue(find_loc_problems("kernel.ttir", dirty_mlir))
        self.assertEqual(find_loc_problems("kernel.llir", _CLEAN_LLIR), [])
        self.assertTrue(find_loc_problems("kernel.llir", _DIRTY_LLIR))
        self.assertEqual(
            find_loc_problems("kernel.amdgcn", "\t.text\n\ts_mov_b32 s0, 0\n"), []
        )
        dirty_asm = '\t.file\t1 "/tmp/x.py"\n\t.text\n'
        self.assertTrue(find_loc_problems("kernel.amdgcn", dirty_asm))

    def test_llir_structure(self) -> None:
        self.assertEqual(check_llir_structure(_CLEAN_LLIR), [])
        self.assertTrue(check_llir_structure("not ir at all"))

    def test_generated_marker_detection(self) -> None:
        self.assertEqual(find_generated_problems("kernel.llir", _CLEAN_LLIR), [])
        self.assertTrue(find_generated_problems("kernel.llir", "; ModuleID = 'x'\n"))
        self.assertEqual(find_generated_problems("compile.json", "{}"), [])

    def test_unmarked_kernel_fails_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_case(tmp, llir=_CLEAN_LLIR.replace(_TOKEN, "plain"))
            report = validate_tree(tmp, "gfx942", "tlx_ops", "mm")
            self.assertEqual(report["ok"], 0)
            self.assertTrue(
                any(
                    "missing generated" in problem
                    for problem in report["failures"]["tlx_ops/mm/case1"]
                )
            )

    def test_tree_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = _write_case(tmp)
            self.assertEqual(check_hashes(cdir), [])
            report = validate_tree(tmp, "gfx942", "tlx_ops", "mm")
            self.assertEqual(report["cases"], 1)
            self.assertEqual(report["ok"], 1)
            self.assertEqual(report["failures"], {})
            # Corrupt one file: hash gate must fire.
            with open(os.path.join(cdir, "kernel.llir"), "a") as f:
                f.write("#loc = loc(/tmp/x.py:1:1)\n")
            report = validate_tree(tmp, "gfx942", "tlx_ops", "mm")
            self.assertEqual(report["ok"], 0)
            self.assertIn("tlx_ops/mm/case1", report["failures"])
            self.assertEqual(main(["--arch", "gfx942", "--out", tmp]), 1)


class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.cdir = _write_case(self.root)
        self.path = os.path.join(self.root, "gfx942", "manifest.json")

    def read_manifest(self):
        with open(self.path) as stream:
            return json.load(stream)

    def write_manifest(self, manifest):
        with open(self.path, "w") as stream:
            json.dump(manifest, stream)

    def test_stale_manifest_hash_fails(self) -> None:
        manifest = self.read_manifest()
        manifest["artifacts"][0]["files"]["llir"]["sha256"] = "0" * 64
        self.write_manifest(manifest)
        self.assertEqual(check_hashes(self.cdir), [])
        report = validate_tree(self.root, "gfx942")
        self.assertIn("manifest.json", report["failures"])
        self.assertIn("disagree", report["failures"]["manifest.json"][0])
        self.assertEqual(main(["--arch", "gfx942", "--out", self.root]), 1)

    def test_missing_manifest_fails(self) -> None:
        os.unlink(self.path)
        self.assertIn("manifest.json", validate_tree(self.root, "gfx942")["failures"])

    def test_empty_corpus_fails(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(main(["--arch", "gfx942", "--out", empty]), 1)

    def test_wrong_architecture_fails(self) -> None:
        manifest = self.read_manifest()
        manifest["arch"] = "gfx950"
        self.write_manifest(manifest)
        self.assertIn("manifest.json", validate_tree(self.root, "gfx942")["failures"])

    def test_missing_kernel_fails(self) -> None:
        os.unlink(os.path.join(self.cdir, "kernel.llir"))
        report = validate_tree(self.root, "gfx942")
        self.assertIn("tlx_ops/mm/case1", report["failures"])

    def test_duplicate_manifest_case_fails(self) -> None:
        manifest = self.read_manifest()
        manifest["artifacts"].append(manifest["artifacts"][0])
        self.write_manifest(manifest)
        report = validate_tree(self.root, "gfx942")
        self.assertTrue(
            any("duplicate" in issue for issue in report["failures"]["manifest.json"])
        )

    def test_orphan_output_case_fails(self) -> None:
        manifest = self.read_manifest()
        manifest["artifacts"][0]["case"] = "other_case"
        self.write_manifest(manifest)
        report = validate_tree(self.root, "gfx942")
        self.assertTrue(
            any(
                "absent from manifest" in issue
                for issue in report["failures"]["manifest.json"]
            )
        )

    def test_hash_record_cannot_escape_case_directory(self) -> None:
        path = os.path.join(self.cdir, "compile.json")
        with open(path) as stream:
            info = json.load(stream)
        info["files"]["llir"]["file"] = "../kernel.llir"
        with open(path, "w") as stream:
            json.dump(info, stream)
        self.assertTrue(
            any("invalid filename" in issue for issue in check_hashes(self.cdir))
        )
