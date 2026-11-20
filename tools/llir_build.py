# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""LLVM IR -> backend assembly -> binary via the production Triton-Beta backends.

Shared by ``verify_llir`` (round-trip checking) and the runtime replay
launchers (``launch_hip`` on AMD, ``launch_cuda`` on NVIDIA). GPU-host
only: imports triton plus, on AMD, the standalone codegen .so
(``TRITON_AMD_CODEGEN_PATH``). Plain-python tooling must not import this at
module level; import it lazily inside GPU-only functions.
"""

from __future__ import annotations

import itertools
import os
import re

try:
    from triton.benchmarks.tools.capture import (  # noqa: F401 - re-export
        iter_cases as iter_cases,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    # pyrefly: ignore [missing-import]
    from capture import iter_cases as iter_cases  # noqa: F401 - re-export
from triton.backends.amd.compiler import HIPBackend, HIPOptions


KERNEL_RE = re.compile(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)")
PTX_KERNEL_RE = re.compile(r"define ptx_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)")

PRESSURE_FLAGS = (
    "reverse_local_assignment",
    "sink_insts_to_avoid_spills",
    "regclass_priority_trumps_globalness",
    "disable_unclustered_high_rp_reschedule",
)


def _option_grid(full: bool):
    """Yield HIPOptions kwargs dicts. Full grid adds TLX's pressure flags.

    Only ``enable_fp_fusion`` is still an ``HIPOptions`` field; the legacy
    packed-fp32 and LLVM pressure flags are kept in the yielded dicts for
    label provenance and filtered out before constructing ``HIPOptions``.
    """
    for fp_fusion in (True, False):
        base = {
            "enable_fp_fusion": fp_fusion,
        }
        if not full:
            yield base
            continue
        for bits in itertools.product((False, True), repeat=4):
            if not any(bits):
                yield dict(base)
                continue  # base grid already covered all-False
            yield dict(base, **dict(zip(PRESSURE_FLAGS, bits)))


def round_trip(llir: str, arch: str, full_grid: bool = False) -> list[tuple[str, str]]:
    """Compile saved llir via the production backend under each option combo.

    Backend options affecting llir->amdgcn (fp fusion) are not recorded in
    compile.json, so compile the grid (defaults first) and let the caller
    accept whichever combo fits. Returns [(opts, amdgcn), ...] for combos
    that compiled.
    """
    results = []
    errors = []
    for kwargs in _option_grid(full_grid):
        label = " ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        hip_kwargs = {k: v for k, v in kwargs.items() if k in ("enable_fp_fusion",)}
        try:
            options = HIPOptions(arch=arch, **hip_kwargs)  # type: ignore
            out = HIPBackend.make_amdgcn(llir, {}, options)
        except Exception as e:  # noqa: BLE001 - option grid, report below
            errors.append(f"{label}: {e}")
            continue
        results.append((label, out))
    if not results:
        raise RuntimeError(f"all option combos failed ({errors})")
    return results


_NVPTX_WARM = False


def _warm_nvptx() -> None:
    """Compile one trivial kernel so the NVPTX LLVM target is initialized.

    In this Triton build, calling ``translate_to_asm`` before any MLIR
    compile segfaults; a single compile-only warmup (no launch) registers
    the target for the rest of the process.
    """
    global _NVPTX_WARM
    if _NVPTX_WARM:
        return
    _NVPTX_WARM = True
    try:
        import triton
        import triton.language as tl
        from triton.compiler import ASTSource, compile as _triton_compile
        from triton.compiler.compiler import GPUTarget

        @triton.jit
        def _warm_kernel(x_ptr, n_elements: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * 1024 + tl.arange(0, 1024)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(x_ptr + offs, x + 1, mask=mask)

        # Vary the specialization per process so the compile cache cannot
        # skip the MLIR compile that performs the registration.
        src = ASTSource(
            _warm_kernel,
            {"x_ptr": "*fp32", "n_elements": "constexpr"},
            constexprs={"n_elements": 1024 * (1 + os.getpid() % 251)},
        )
        _triton_compile(src, target=GPUTarget("cuda", 100, 32))
    except Exception:
        pass


def _cuda_backend(arch: str):
    """(CUDABackend, capability) for an NVIDIA arch like ``sm100``."""
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler.compiler import GPUTarget

    # Touching the active driver registers the NVPTX LLVM target; without
    # it, a CPU-only process segfaults inside translate_to_asm.
    try:
        from triton.runtime.driver import driver

        driver.active.get_current_target()
    except Exception:
        pass
    capability = int(arch.removeprefix("sm"))
    return CUDABackend(GPUTarget("cuda", capability, 32)), capability


def round_trip_ptx(llir: str, arch: str) -> list[tuple[str, str]]:
    """Compile saved llir to PTX under each fp-fusion option combo.

    Mirrors ``round_trip`` for the NVIDIA backend: backend options affecting
    llir->ptx are not recorded in compile.json, so compile the small grid
    (defaults first) and let the caller accept whichever combo fits.
    Returns [(opts_label, ptx), ...] for combos that compiled.
    """
    _warm_nvptx()
    backend, capability = _cuda_backend(arch)
    results, errors = [], []
    for fp_fusion in (True, False):
        label = f"enable_fp_fusion={fp_fusion}"
        try:
            options = backend.parse_options({"enable_fp_fusion": fp_fusion})
            out = backend.make_ptx(llir, {}, options, capability)
        except Exception as e:  # noqa: BLE001 - option grid, report below
            errors.append(f"{label}: {e}")
            continue
        results.append((label, out))
    if not results:
        raise RuntimeError(f"all option combos failed ({errors})")
    return results


def build_cubin(llir: str, arch: str) -> tuple[str, bytes]:
    """Build a launchable CUBIN from saved llir. Returns (opts_label, cubin).

    Uses the first (default-options) combo that compiles; launch replay does
    not need the exact original codegen options, only a valid object.
    """
    backend, capability = _cuda_backend(arch)
    label, ptx = round_trip_ptx(llir, arch)[0]
    options = backend.parse_options({})
    return label, backend.make_cubin(ptx, {}, options, capability)


def build_hsaco(llir: str, arch: str) -> tuple[str, bytes]:
    """Build a launchable HSACO from saved llir. Returns (opts_label, hsaco).

    Uses the first (default-options) combo that compiles; launch replay does
    not need the exact original codegen options, only a valid object.
    """
    label, amdgcn = round_trip(llir, arch)[0]
    # NOTE: assemble via HIPBackend.make_hsaco (the backend's ctypes wrapper
    # around the standalone codegen .so), not the raw
    # ``amd.assemble_amdgcn`` binding: libtriton's statically linked LLVM has
    # no AMDGPU target registered, so the raw binding fails standalone with a
    # target lookup error.
    return label, HIPBackend.make_hsaco(amdgcn, {}, HIPOptions(arch=arch))
