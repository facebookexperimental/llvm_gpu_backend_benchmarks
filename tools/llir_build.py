# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""LLVM IR -> AMDGCN -> HSACO via the production Triton-Beta AMD backend.

Shared by ``verify_llir`` (round-trip checking) and ``launch_hip`` (runtime
replay). GPU-host only: imports triton plus the standalone AMD codegen .so
(``TRITON_AMD_CODEGEN_PATH``). Plain-python tooling must not import this at
module level; import it lazily inside GPU-only functions.
"""

from __future__ import annotations

import itertools
import re

try:
    from triton.benchmarks.tools.capture import (  # noqa: F401 - re-export
        iter_cases as iter_cases,
    )
except ImportError:
    # Plain python3: sibling file in the same directory.
    from capture import iter_cases as iter_cases  # noqa: F401 - re-export
from triton.backends.amd.compiler import HIPBackend, HIPOptions


KERNEL_RE = re.compile(r"define amdgpu_kernel void @([a-zA-Z_][a-zA-Z0-9_]*)")

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
