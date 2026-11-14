# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for prune_loc.py (plain python3)."""

import hashlib
import json
import os
import tempfile
import unittest

try:
    from triton.benchmarks.tools.prune_loc import (
        main,
        prune_tree,
        strip_amdgcn,
        strip_llir,
        strip_mlir,
    )
except ImportError:
    # Plain python3 (no buck packaging): the module lives next to this test.
    # pyrefly: ignore [missing-import]
    from prune_loc import main, prune_tree, strip_amdgcn, strip_llir, strip_mlir

_MLIR = """\
#loc = loc("/data/buck-out/hash/link-tree/kernels/mm/gfx950.py":424:1)
#loc1 = loc(unknown)
#loc2 = loc(callsite(#loc at #loc1))
module {
  tt.func public @k(%a: !tt.ptr<i64> loc("a"(#loc))) {
    %c = arith.constant 128 : i32 loc(#loc)
    tt.return loc(#loc2)
  }
}
"""

_LLIR = """\
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k(ptr %a) !dbg !10 {
  %v = load i32, ptr %a, !dbg !11
  ret void, !dbg !11
}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "gfx950.py", directory: "/data/buck-out/hash/mm")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1, size: 64)
!8 = !{null, !3, !3}
!9 = distinct !DISubprogram(name: "k", type: !8)
!10 = distinct !DISubprogram(name: "k", file: !1)
!11 = !DILocation(line: 424, column: 1, scope: !10)
"""

_AMDGCN = """\
\t.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
\t.text
\t.file\t1 "/data/buck-out/hash/mm/gfx950.py"
k:
.Lfunc_begin0:
\t.cfi_startproc
\ts_mov_b32 s0, 0
\t.loc\t1 424 0 prologue_end
.Ltmp0:
\ts_endpgm
\t.section\t.debug_line,"",@progbits
.Lline_table_start0:
\t.byte\t1
"""


class StripTest(unittest.TestCase):
    def test_mlir(self) -> None:
        out = strip_mlir(_MLIR)
        self.assertNotIn("loc(", out)
        self.assertIn("tt.func public @k(%a: !tt.ptr<i64>)", out)
        self.assertIn("%c = arith.constant 128 : i32", out)
        self.assertEqual(strip_mlir(out), out)

    def test_llir(self) -> None:
        out = strip_llir(_LLIR)
        self.assertNotIn("!dbg", out)
        self.assertNotIn("!DI", out)
        self.assertNotIn("!3", out)
        self.assertNotIn("!8", out)
        self.assertNotIn("/data/buck-out", out)
        self.assertIn("define amdgpu_kernel void @k(ptr %a) {", out)
        self.assertIn("%v = load i32, ptr %a", out)
        self.assertIn("!llvm.module.flags = !{!0}", out)
        self.assertEqual(strip_llir(out), out)

    def test_amdgcn(self) -> None:
        out = strip_amdgcn(_AMDGCN)
        self.assertNotIn(".loc", out)
        self.assertNotIn(".file", out)
        self.assertNotIn(".cfi", out)
        self.assertNotIn(".Ltmp", out)
        self.assertNotIn(".debug", out)
        self.assertNotIn("/data/buck-out", out)
        self.assertIn("s_mov_b32 s0, 0", out)
        self.assertEqual(strip_amdgcn(out), out)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PruneTreeTest(unittest.TestCase):
    def test_prune_updates_hashes_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cdir = os.path.join(tmp, "gfx950", "tlx_ops", "mm", "case1")
            os.makedirs(cdir)
            files = {
                "ttir": ("kernel.ttir", _MLIR),
                "llir": ("kernel.llir", _LLIR),
                "amdgcn": ("kernel.amdgcn", _AMDGCN),
            }
            compile_info = {
                "files": {
                    stage: {"file": name, "sha256": _sha(text)}
                    for stage, (name, text) in files.items()
                }
            }
            for _stage, (name, text) in files.items():
                with open(os.path.join(cdir, name), "w") as f:
                    f.write(text)
            with open(os.path.join(cdir, "compile.json"), "w") as f:
                json.dump(compile_info, f)
            manifest = {
                "arch": "gfx950",
                "artifacts": [
                    {
                        "source": "tlx_ops",
                        "target": "mm",
                        "case": "case1",
                        "files": dict(compile_info["files"]),
                    }
                ],
                "failures": [],
            }
            with open(os.path.join(tmp, "gfx950", "manifest.json"), "w") as f:
                json.dump(manifest, f)
            stats = prune_tree(tmp, "gfx950")
            self.assertEqual(stats["cases"], 1)
            self.assertGreater(stats["bytes_saved"], 0)
            with open(os.path.join(cdir, "compile.json")) as f:
                updated = json.load(f)
            for stage, (name, _text) in files.items():
                with open(os.path.join(cdir, name)) as f:
                    self.assertEqual(updated["files"][stage]["sha256"], _sha(f.read()))
            with open(os.path.join(tmp, "gfx950", "manifest.json")) as f:
                self.assertEqual(
                    json.load(f)["artifacts"][0]["files"], updated["files"]
                )
            self.assertEqual(prune_tree(tmp, "gfx950")["bytes_saved"], 0)
            self.assertEqual(main(["--arch", "gfx950", "--out", tmp]), 0)


if __name__ == "__main__":
    unittest.main()
