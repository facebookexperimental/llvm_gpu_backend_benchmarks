# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for launch_hip.py (pure helpers + fake-HIP replay) and capture recording."""

import ctypes
import json
import os
import sys
import tempfile
import types
import unittest

try:
    from triton.benchmarks.tools.capture import (
        bind_launch_args,
        eval_grid,
        record_launch,
        write_case,
    )
    from triton.benchmarks.tools.launch_hip import (
        launch_case,
        make_arg_ptrs,
        parse_kernel_signature,
        strip_scratch,
        validate_launch,
    )
except ImportError:
    # Plain python3: sibling files in the same directory.
    from capture import bind_launch_args, eval_grid, record_launch, write_case
    from launch_hip import (
        launch_case,
        make_arg_ptrs,
        parse_kernel_signature,
        strip_scratch,
        validate_launch,
    )

_LLIR = """\
; ModuleID = 't_kernel'
source_filename = "t_kernel"
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @my_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %n, float %scale) {
  ret void
}
"""

_LLIR_SCRATCH = _LLIR.replace(
    "float %scale)", "float %scale, ptr addrspace(1) %g, ptr addrspace(1) %p)"
)


class SignatureTest(unittest.TestCase):
    def test_parse(self) -> None:
        name, sig = parse_kernel_signature(_LLIR)
        self.assertEqual(name, "my_kernel")
        self.assertEqual(sig, [("ptr", 64), ("ptr", 64), ("int", 32), ("float", 32)])

    def test_int_widths(self) -> None:
        llir = _LLIR.replace("i32 %n", "i64 %n")
        _, sig = parse_kernel_signature(llir)
        self.assertIn(("int", 64), sig)

    def test_attributes_ignored(self) -> None:
        llir = (
            "define amdgpu_kernel void @_register_kernel_impl("
            "ptr addrspace(1) inreg nofree readonly captures(none) %0, "
            "i32 signext %1) local_unnamed_addr #0 !dbg !4 {"
        )
        name, sig = parse_kernel_signature(llir)
        self.assertEqual(name, "_register_kernel_impl")
        self.assertEqual(sig, [("ptr", 64), ("int", 32)])

    def test_no_kernel(self) -> None:
        with self.assertRaises(ValueError):
            parse_kernel_signature("define void @f() {}")


class ValidateTest(unittest.TestCase):
    def _launch(self, **over):
        base = {
            "grid": [4, 1, 1],
            "block": {"num_warps": 4, "num_ctas": 1, "shared": 0},
            "args": [
                {"name": "a", "kind": "tensor"},
                {"name": "b", "kind": "tensor"},
                {"name": "n", "kind": "int", "value": 8},
                {"name": "s", "kind": "float", "value": 1.5},
            ],
        }
        base.update(over)
        return base

    def _sig(self):
        return [("ptr", 64), ("ptr", 64), ("int", 32), ("float", 32)]

    def test_ok(self) -> None:
        self.assertEqual(validate_launch(self._launch(), self._sig()), [])

    def test_missing_launch(self) -> None:
        self.assertEqual(validate_launch(None, self._sig()), ["missing-launch"])

    def test_bad_grid_and_block(self) -> None:
        problems = validate_launch(self._launch(grid=[4, -1]), self._sig())
        self.assertIn("bad-grid", problems)
        problems = validate_launch(self._launch(block={}), self._sig())
        self.assertIn("missing-num_warps", problems)

    def test_unsupported_ctas_and_cluster(self) -> None:
        problems = validate_launch(
            self._launch(block={"num_warps": 4, "num_ctas": 2}), self._sig()
        )
        self.assertTrue(any("num_ctas" in p for p in problems))
        problems = validate_launch(
            self._launch(block={"num_warps": 4, "cluster_dims": [2, 1, 1]}),
            self._sig(),
        )
        self.assertTrue(any("cluster" in p for p in problems))

    def test_arg_mismatch(self) -> None:
        problems = validate_launch(self._launch(args=[]), self._sig())
        self.assertTrue(any("arg-count" in p for p in problems))
        bad = self._launch()
        bad["args"][0] = {"name": "a", "kind": "int", "value": 1}
        problems = validate_launch(bad, self._sig())
        self.assertTrue(any("ptr" in p for p in problems))

    def test_scratch_trailer_ok(self) -> None:
        sig = self._sig() + [("ptr", 64), ("ptr", 64)]
        self.assertEqual(validate_launch(self._launch(), sig), [])

    def test_bad_scratch_size(self) -> None:
        bad = self._launch(block={"num_warps": 4, "global_scratch_size": -8})
        problems = validate_launch(bad, self._sig())
        self.assertIn("bad-global_scratch_size", problems)

    def test_trailing_user_ptrs_not_scratch(self) -> None:
        sig = [("int", 32), ("ptr", 64), ("ptr", 64)]
        launch = self._launch(
            args=[
                {"name": "n", "kind": "int", "value": 8},
                {"name": "a", "kind": "tensor"},
                {"name": "b", "kind": "tensor"},
            ]
        )
        self.assertEqual(validate_launch(launch, sig), [])
        user, n = strip_scratch(sig, launch)
        self.assertEqual((user, n), (sig, 0))

    def test_strip_scratch(self) -> None:
        user, n = strip_scratch(self._sig() + [("ptr", 64), ("ptr", 64)])
        self.assertEqual((user, n), (self._sig(), 2))
        self.assertEqual(strip_scratch(self._sig()), (self._sig(), 0))
        self.assertEqual(strip_scratch([("ptr", 64)]), ([("ptr", 64)], 0))


class FakePtr:
    def __init__(self, addr: int):
        self._addr = addr

    def data_ptr(self) -> int:
        return self._addr


class ArgPtrsTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        sig = [("ptr", 64), ("int", 32), ("float", 32), ("int", 64)]
        bufs, arr = make_arg_ptrs([FakePtr(0x1000), 7, 2.5, 2**40], sig)
        self.assertEqual(len(bufs), 4)
        self.assertEqual(
            ctypes.cast(arr[0], ctypes.POINTER(ctypes.c_uint64)).contents.value, 0x1000
        )
        self.assertEqual(
            ctypes.cast(arr[1], ctypes.POINTER(ctypes.c_int32)).contents.value, 7
        )
        self.assertAlmostEqual(
            ctypes.cast(arr[2], ctypes.POINTER(ctypes.c_float)).contents.value, 2.5
        )
        self.assertEqual(
            ctypes.cast(arr[3], ctypes.POINTER(ctypes.c_int64)).contents.value, 2**40
        )

    def test_unsigned_int_packing(self) -> None:
        bufs, _arr = make_arg_ptrs([2**31, -1], [("int", 32), ("int", 32)])
        self.assertEqual(bufs[0].value, 2**31)
        self.assertEqual(bufs[1].value, 2**32 - 1)
        bufs, _arr = make_arg_ptrs([2**63], [("int", 64)])
        self.assertEqual(bufs[0].value, 2**63)

    def test_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            make_arg_ptrs([1], [("int", 32), ("int", 32)])

    def test_null_ptr_from_int(self) -> None:
        _bufs, arr = make_arg_ptrs([0], [("ptr", 64)])
        self.assertEqual(
            ctypes.cast(arr[0], ctypes.POINTER(ctypes.c_uint64)).contents.value, 0
        )


class _FakeSlice:
    def __init__(self, data):
        self._data = list(data)

    def cpu(self):
        return self

    def tolist(self):
        return list(self._data)


class _FakeFlat:
    def __init__(self, data):
        self._data = data

    @property
    def shape(self):
        return (len(self._data),)

    def __getitem__(self, idx):
        return _FakeSlice(self._data[idx])

    def cpu(self):
        return self

    def tolist(self):
        return list(self._data)


class _FakeTensor:
    def __init__(self, data, shape=(8,), dtype="float32", device="cuda:0"):
        self._data = list(data)
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def stride(self):
        return (1,) * len(self._shape)

    @property
    def device(self):
        return self._device

    def data_ptr(self):
        return 0x1000

    def element_size(self):
        return 4

    def numel(self):
        return len(self._data)

    def detach(self):
        return self

    def reshape(self, *args):
        return _FakeFlat(self._data)

    def cpu(self):
        return self

    def tolist(self):
        return list(self._data)


class _FakeParam:
    __slots__ = ("is_constexpr",)

    def __init__(self, is_constexpr: bool):
        self.is_constexpr = is_constexpr


class _FakeFn:
    def __init__(self, arg_names, constexprs):
        self.arg_names = arg_names
        self.fn = types.SimpleNamespace(params=[_FakeParam(c) for c in constexprs])


class BindTest(unittest.TestCase):
    def test_binds_names_and_constexprs(self) -> None:
        jit_fn = _FakeFn(["a", "n", "BLOCK"], [False, False, True])
        bound = bind_launch_args(jit_fn, ["A", 8], {"BLOCK": 128, "grid": (1,)})
        self.assertEqual(
            [(n, v, c) for n, v, c in bound],
            [("a", "A", False), ("n", 8, False), ("BLOCK", 128, True)],
        )

    def test_fallback_without_names(self) -> None:
        bound = bind_launch_args(object(), ["A"], {"grid": (2,), "k": 3})
        self.assertEqual(bound, [("arg0", "A", False), ("k", 3, False)])


class GridTest(unittest.TestCase):
    def test_tuple_and_list(self) -> None:
        self.assertEqual(eval_grid((4, 2), {}), (4, 2, 1))
        self.assertEqual(eval_grid([1, 1, 1], {}), (1, 1, 1))

    def test_callable_with_meta(self) -> None:
        self.assertEqual(eval_grid(lambda m: (m["N"] // 2,), {"N": 8}), (4, 1, 1))

    def test_undecidable(self) -> None:
        self.assertIsNone(eval_grid(lambda m: (m["X"],), {}))
        self.assertIsNone(eval_grid("nope", {}))


class RecordTest(unittest.TestCase):
    def test_roles_and_scalars(self) -> None:
        out = _FakeTensor([0.0] * 8)
        inp = _FakeTensor([1.0] * 8)
        jit_fn = _FakeFn(["o", "i", "n"], [False, False, False])
        # Simulate pre-hashes taken before the run, then mutate the output.
        try:
            from triton.benchmarks.tools.capture import _tensor_sample_hash
        except ImportError:
            # Plain python3: sibling file in the same directory.
            from capture import _tensor_sample_hash

        pre = {"o": _tensor_sample_hash(out), "i": _tensor_sample_hash(inp)}
        out._data = [2.0] * 8
        kernel = types.SimpleNamespace(
            metadata=types.SimpleNamespace(num_warps=4, num_ctas=1, shared=0)
        )
        desc = record_launch(
            jit_fn, [out, inp, 8], {"grid": (4, 1, 1)}, kernel, pre_hashes=pre
        )
        self.assertEqual(desc["grid"], [4, 1, 1])
        self.assertEqual(desc["block"]["num_warps"], 4)
        roles = {a["name"]: a["role"] for a in desc["args"] if a["kind"] == "tensor"}
        self.assertEqual(roles, {"o": "out", "i": "in"})
        scalar = [a for a in desc["args"] if a["name"] == "n"][0]
        self.assertEqual((scalar["kind"], scalar["value"]), ("int", 8))

    def test_constexprs_excluded(self) -> None:
        jit_fn = _FakeFn(["a", "BLOCK"], [False, True])
        kernel = types.SimpleNamespace(metadata=types.SimpleNamespace(num_warps=1))
        desc = record_launch(
            jit_fn, [_FakeTensor([1.0])], {"BLOCK": 32, "grid": (1,)}, kernel
        )
        names = [a["name"] for a in desc["args"]]
        self.assertEqual(names, ["a"])


class WriteCaseTest(unittest.TestCase):
    def test_launch_recorded(self) -> None:
        kernel = types.SimpleNamespace(asm={}, name="k")
        with tempfile.TemporaryDirectory() as tmp:
            launch = {"grid": [2, 1, 1], "args": []}
            entry = write_case(
                tmp,
                kernel,
                target="t",
                case="c",
                arch="gfx950",
                source="s",
                launch=launch,
            )
            with open(os.path.join(tmp, "compile.json")) as f:
                saved = json.load(f)
        self.assertEqual(saved["launch"], launch)
        self.assertEqual(entry["launch"], launch)

    def test_launch_defaults_none(self) -> None:
        kernel = types.SimpleNamespace(asm={}, name="k")
        with tempfile.TemporaryDirectory() as tmp:
            entry = write_case(
                tmp, kernel, target="t", case="c", arch="gfx950", source="s"
            )
        self.assertIsNone(entry["launch"])


class _ReplayTensor:
    __slots__ = ("_ptr", "shape")

    def __init__(self, ptr: int, shape):
        self._ptr = ptr
        self.shape = shape

    def data_ptr(self) -> int:
        return self._ptr

    def cpu(self):
        return self


class _FakeEvent:
    def __init__(self, enable_timing: bool = True, ms: float = 3.0):
        assert enable_timing
        self._ms = ms

    def record(self) -> None:
        pass

    def elapsed_time(self, _other) -> float:
        return self._ms


class _FakeCuda:
    def __init__(self):
        self.syncs = 0

    def manual_seed_all(self, _seed: int) -> None:
        pass

    def synchronize(self) -> None:
        self.syncs += 1

    def current_stream(self):
        return types.SimpleNamespace(cuda_stream=0)

    def __getattr__(self, name: str):
        if name == "Event":
            return _FakeEvent
        raise AttributeError(name)


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()
        self._ptr = 0x1000
        self.zeros_shapes: list = []

    def manual_seed(self, _seed: int) -> None:
        pass

    def _alloc(self, shape, **_kwargs) -> _ReplayTensor:
        self._ptr += 0x100
        return _ReplayTensor(self._ptr, shape)

    def zeros(self, shape, **kwargs) -> _ReplayTensor:
        self.zeros_shapes.append(tuple(shape))
        return self._alloc(shape, **kwargs)

    def randn(self, shape, **kwargs) -> _ReplayTensor:
        return self._alloc(shape, **kwargs)

    def isfinite(self, _t):
        return types.SimpleNamespace(all=lambda: True)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return name


class _FakeHip:
    """HIP entry points via __getattr__ (avoids camelCase defs); all succeed."""

    def __init__(self):
        self.launches = 0
        self.unloads = 0

    def __getattr__(self, name: str):
        if name == "hipModuleLaunchKernel":

            def _launch(*args) -> int:
                assert len(args) == 11
                self.launches += 1
                return 0

            return _launch
        if name == "hipModuleUnload":

            def _unload(_module) -> int:
                self.unloads += 1
                return 0

            return _unload
        if name == "hipGetErrorString":

            def _err(code: int) -> bytes:
                return f"code {code}".encode()

            return _err
        if name.startswith("hipModule"):

            def _ok(*_args) -> int:
                return 0

            return _ok
        raise AttributeError(name)


class LaunchCaseTest(unittest.TestCase):
    def _write_case(self, tmp: str, args: list, llir: str, block: dict) -> None:
        with open(os.path.join(tmp, "kernel.llir"), "w") as f:
            f.write(llir)
        compile_info = {
            "launch": {
                "grid": [2, 1, 1],
                "block": block,
                "device": "cuda",
                "args": args,
            }
        }
        with open(os.path.join(tmp, "compile.json"), "w") as f:
            json.dump(compile_info, f)

    def _args(self) -> list:
        return [
            {
                "name": "a",
                "kind": "tensor",
                "shape": [4],
                "dtype": "float32",
                "role": "in",
            },
            {
                "name": "b",
                "kind": "tensor",
                "shape": [4],
                "dtype": "float32",
                "role": "out",
            },
            {"name": "n", "kind": "int", "value": 16},
            {"name": "scale", "kind": "float", "value": 2.5},
        ]

    def _replay(
        self,
        args: list,
        prebuilt: bytes | None = None,
        llir: str = _LLIR,
        block: dict | None = None,
        collect: bool = False,
    ):
        built = []

        def _fake_build(llir: str, arch: str):
            built.append((llir, arch))
            self.assertIn("my_kernel", llir)
            self.assertEqual(arch, "gfx950")
            return ("fake-opts", b"HSACO")

        fake = types.ModuleType("triton.benchmarks.tools.llir_build")
        fake.build_hsaco = _fake_build  # type: ignore[attr-defined]
        keys = ("triton.benchmarks.tools.llir_build", "llir_build")
        reals = {k: sys.modules.get(k) for k in keys}
        for k in keys:
            sys.modules[k] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self._write_case(tmp, args, llir, block or {"num_warps": 4})
                hsaco_path = None
                if prebuilt is not None:
                    hsaco_path = os.path.join(tmp, "prebuilt.hsaco")
                    with open(hsaco_path, "wb") as f:
                        f.write(prebuilt)
                torch, hip = _FakeTorch(), _FakeHip()
                res = launch_case(
                    tmp, "gfx950", torch, hip, 0, False, hsaco_path, collect
                )
        finally:
            for k in keys:
                if reals[k] is None:
                    del sys.modules[k]
                else:
                    sys.modules[k] = reals[k]
        return res, torch, hip, built

    def test_replay_ok(self) -> None:
        res, _torch, hip, built = self._replay(self._args())
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["median_ms"], 3.0 / 30)
        self.assertTrue(res["finite"])
        self.assertEqual(res["seed"], 0)
        self.assertEqual(res["grid"], [2, 1, 1])
        self.assertEqual(hip.launches, 10 + 30)
        self.assertEqual(hip.unloads, 1)
        self.assertEqual(len(built), 1)

    def test_replay_prebuilt_hsaco(self) -> None:
        res, _torch, hip, built = self._replay(self._args(), prebuilt=b"HSACO")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(hip.launches, 10 + 30)
        self.assertEqual(built, [])

    def test_replay_unsupported_arg(self) -> None:
        args = self._args()
        args[-1] = {"name": "scale", "kind": "other", "repr": "f32?"}
        res, _torch, hip, _built = self._replay(args)
        self.assertEqual(res["status"], "unsupported-arg")
        self.assertEqual(res["arg"], "scale")
        self.assertEqual(hip.launches, 0)

    def test_replay_scratch(self) -> None:
        block = {
            "num_warps": 4,
            "global_scratch_size": 64,
            "profile_scratch_size": 0,
        }
        res, torch, hip, _built = self._replay(
            self._args(), prebuilt=b"HSACO", llir=_LLIR_SCRATCH, block=block
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(hip.launches, 10 + 30)
        self.assertIn((64,), torch.zeros_shapes)

    def test_collect_outputs(self) -> None:
        res, _torch, _hip, _built = self._replay(self._args(), collect=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(set(res["outputs"]), {"b"})
        res, _torch, _hip, _built = self._replay(self._args())
        self.assertNotIn("outputs", res)


if __name__ == "__main__":
    unittest.main()
