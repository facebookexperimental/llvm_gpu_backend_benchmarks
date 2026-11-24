# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the inductor-template capture hook in capture.py."""

import sys
import types
import unittest

try:
    from triton.benchmarks.tools.capture import CaptureSession
except ImportError:
    # Plain python3 (no buck packaging): the module lives next to this test.
    # pyrefly: ignore [missing-import]
    from capture import CaptureSession


class _FakeTensor:
    def __init__(self) -> None:
        self.shape = (4, 8)
        self.dtype = "float16"
        self.device = "cuda:0"
        self.data_ptr = 0
        self.element_size = 2
        self.numel = 32

    def stride(self):
        return [8, 1]

    def detach(self):
        return self

    def reshape(self, *_args):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [0.0]

    def __getitem__(self, _key):
        return self


class _FakeParam:
    __slots__ = ("is_constexpr",)

    def __init__(self, is_constexpr: bool) -> None:
        self.is_constexpr = is_constexpr


class _FakeJitFn:
    arg_names = ["arg_A", "arg_B", "out_ptr0"]
    params = [_FakeParam(False)] * 3


class _FakeKernel:
    def __init__(self) -> None:
        self.name = "triton_tem_fused_mm_0"
        self.metadata = None
        self.asm = {"llir": "define void @k() {}"}


def _launcher_with_kernel(kernel):
    scope: dict = {}

    def _define() -> None:
        exec("def launcher(*args, **kwargs):\n    return 'launched'\n", scope)

    _define()
    scope["bin"] = kernel
    return scope["launcher"]


class _FakeAutotuner:
    """Stand-in for torch._inductor.runtime.triton_heuristics.CachingAutotuner."""

    def __init__(self, fn, launchers, inductor_meta, fail=False) -> None:
        self.fn = fn
        self.launchers = launchers
        self.inductor_meta = inductor_meta
        self.fail = fail

    def run(self, *args, stream=None, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        return "launched"


_HEURISTICS_PATH = "torch._inductor.runtime.triton_heuristics"


def _stub_heuristics(cls):
    module = types.ModuleType(_HEURISTICS_PATH)
    module.CachingAutotuner = cls
    parents = ["torch", "torch._inductor", "torch._inductor.runtime"]
    stubs = {}
    for name in parents:
        parent = types.ModuleType(name)
        parent.__path__ = []  # type: ignore[attr-defined]
        stubs[name] = parent
    return stubs, module


class CaptureAutotunerTest(unittest.TestCase):
    def _enter(self, autotuner):
        stubs, module = _stub_heuristics(type(autotuner))
        saved = {}
        for name, stub in {**stubs, _HEURISTICS_PATH: module}.items():
            saved[name] = sys.modules.get(name, None)
            sys.modules[name] = stub
        self.addCleanup(self._restore_modules, saved)
        session = CaptureSession()
        session.__enter__()
        self.addCleanup(session.__exit__, None, None, None)
        return session

    @staticmethod
    def _restore_modules(saved) -> None:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def _autotuner(self, fail=False):
        kernel = _FakeKernel()
        launcher = _launcher_with_kernel(kernel)
        return _FakeAutotuner(
            _FakeJitFn(),
            [launcher],
            {"extra_launcher_args": ["_grid_0", "_grid_1", "_grid_2"]},
            fail=fail,
        ), kernel

    def test_template_launch_is_captured(self) -> None:
        autotuner, kernel = self._autotuner()
        session = self._enter(autotuner)
        a, b, out = _FakeTensor(), _FakeTensor(), _FakeTensor()
        result = autotuner.run(a, b, out, 1024, 1, 1, stream=0)
        self.assertEqual(result, "launched")
        self.assertEqual(session.kernels, [("triton_tem_fused_mm_0", kernel)])
        launch = session.launches[id(kernel)]
        self.assertEqual(launch["grid"], [1024, 1, 1])
        self.assertEqual(
            [arg["name"] for arg in launch["args"]],
            ["arg_A", "arg_B", "out_ptr0"],
        )

    def test_template_launch_failure_records_nothing(self) -> None:
        autotuner, kernel = self._autotuner(fail=True)
        session = self._enter(autotuner)
        with self.assertRaises(RuntimeError):
            autotuner.run(
                _FakeTensor(), _FakeTensor(), _FakeTensor(), 1, 1, 1, stream=0
            )
        self.assertEqual(session.kernels, [])
        self.assertEqual(session.launches, {})

    def test_template_launch_without_kernel_records_nothing(self) -> None:
        kernel = _FakeKernel()
        scope: dict = {}
        exec("def launcher(*args, **kwargs):\n    return 'launched'\n", scope)
        autotuner = _FakeAutotuner(
            _FakeJitFn(),
            [scope["launcher"]],
            {"extra_launcher_args": ["_grid_0", "_grid_1", "_grid_2"]},
        )
        session = self._enter(autotuner)
        autotuner.run(_FakeTensor(), _FakeTensor(), _FakeTensor(), 1, 1, 1, stream=0)
        self.assertEqual(session.kernels, [])
        self.assertNotIn(id(kernel), session.launches)

    def test_repeated_template_launch_records_once(self) -> None:
        autotuner, kernel = self._autotuner()
        session = self._enter(autotuner)
        args = (_FakeTensor(), _FakeTensor(), _FakeTensor())
        autotuner.run(*args, 512, 1, 1, stream=0)
        autotuner.run(*args, 256, 1, 1, stream=0)
        self.assertEqual(session.kernels, [("triton_tem_fused_mm_0", kernel)])
        self.assertEqual(session.launches[id(kernel)]["grid"], [256, 1, 1])
