# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Shared LLVM-stage capture for the Triton-Beta export workstream.

Import surface (GPU drivers only; importing this module never imports torch
or triton, so host-side tooling can reuse the naming/manifest helpers)::

    from capture import CaptureSession, write_case, run_manifest_entry

How capture works: every kernel compiled through Triton Beta ends up as a
``CompiledKernel`` with an ``asm`` dict. The LLVM stage is
``HIPBackend.make_amdgcn``, whose input is exactly ``asm["llir"]`` -- that
string is the artifact we export, plus ``ttgir`` (translator input), ``ttir``
and ``amdgcn`` (reference output) for provenance. Two hooks cover every
launch path:

- ``JITFunction.run`` (covers ``@triton.jit`` launches and autotune winners:
  the autotuner benchmarks candidates internally and the winning launch is
  what ``run`` returns, so the winner's ``llir`` is what we capture).
- ``triton.compiler.compiler.compile`` (covers inductor-generated and other
  direct compiles, e.g. the torchtlx path).

Best-config selection for multi-config kernels is therefore the runtime
winner by default. ``config_filter`` (a ``re`` pattern matched against each
candidate config's repr) can narrow the candidate set first; it is applied
by wrapping ``triton.autotune``'s config list at capture time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

MANIFEST_VERSION = 1

# Stage preference order: llir is the deliverable, the rest is provenance.
STAGES = ("llir", "ttgir", "ttir", "amdgcn")


def case_dir(out_root: str, arch: str, source: str, target: str, case: str) -> str:
    """Hardware-sorted output directory for one compiled kernel case."""

    def safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

    return os.path.join(out_root, arch, source, safe(target), safe(case))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def triton_provenance() -> dict:
    """Best-effort Triton Beta build provenance (never raises)."""
    prov: dict = {}
    try:
        import triton

        prov["triton_version"] = getattr(triton, "__version__", "unknown")
        try:
            from triton.fb_version_info import get_version_info  # type: ignore

            prov["fb_version_info"] = _jsonable(get_version_info())
        except Exception:
            pass
        try:
            import triton.backends.compiler as _c  # noqa: F401
        except Exception:
            pass
    except Exception as e:
        prov["triton_import_error"] = repr(e)
    for key in ("TRITON_CACHE_DIR", "TRITON_PRINT_AUTOTUNING"):
        if os.environ.get(key):
            prov[key] = os.environ[key]
    return prov


def probe_arch_id() -> str:
    """Best-effort arch identifiers for device 0 (never raises).

    Some hosts report a generic marketing name (e.g. 'AMD Radeon Graphics')
    from both rocm-smi and torch, so name matching alone cannot confirm the
    hardware. This collects arch-level identifiers instead: torch's
    gcnArchName and the Triton driver's target arch.
    """
    parts: list[str] = []
    try:
        import torch

        parts.append(str(torch.cuda.get_device_properties(0).gcnArchName))
    except Exception:
        pass
    try:
        # NOTE: `import triton.runtime.driver as drv` binds the DriverConfig
        # *instance* (the package shadows the submodule), which the type
        # checker cannot see; import the instance from the module instead.
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        parts.append(str(getattr(target, "arch", target)))
    except Exception as e:
        parts.append(f"target-probe-error:{e!r}")
    return " ".join(parts)


def timed_launch(torch, fn) -> float | None:
    """Time one relaunch of an already-captured case (never raises).

    Caches are hot (the case just ran under capture), so this measures a
    single launch without benchmarking overhead. Noisy for tiny kernels, but
    shape rankings span orders of magnitude.
    """
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))
    except Exception:
        return None


def case_latencies(session, torch, run, timeable: bool = True) -> dict:
    """Map each captured kernel to ``(latency_ms, latency_kind)``.

    Autotuned configs carry their sweep median; anything else shares one timed
    relaunch of the case (exact for single-kernel cases). Non-timeable cases
    (e.g. script executions with side effects) keep whatever the sweep
    recorded, if anything.
    """
    out: dict = {}
    missing = []
    for _, kernel in session.kernels:
        hit = session.timings.get(id(kernel))
        if hit is not None:
            out[id(kernel)] = (hit[1], "autotune_median")
        else:
            missing.append(kernel)
    if missing and timeable:
        ms = timed_launch(torch, run)
        if ms is not None:
            for kernel in missing:
                out[id(kernel)] = (ms, "single_launch")
    return out


def describe_compiled(kernel) -> dict:
    """Extract backend/arch/options/config from a CompiledKernel (never raises)."""
    info: dict = {"name": getattr(kernel, "name", "unknown")}
    try:
        metadata = getattr(kernel, "metadata", None)
        if metadata is not None:
            for attr in (
                "target",
                "num_warps",
                "num_ctas",
                "num_stages",
                "cluster_dims",
            ):
                try:
                    info[attr] = _jsonable(getattr(metadata, attr))
                except Exception:
                    pass
            try:
                info["target_arch"] = metadata.target.arch
            except Exception:
                pass
            try:
                info["backend"] = type(metadata.target.backend).__name__
            except Exception:
                pass
        asm = getattr(kernel, "asm", {}) or {}
        info["stages_present"] = sorted(asm.keys())
    except Exception as e:
        info["describe_error"] = repr(e)
    return info


def _is_tensor(value) -> bool:
    """Duck-typed torch.Tensor check (no torch import)."""
    return all(
        hasattr(value, attr)
        for attr in ("shape", "dtype", "stride", "data_ptr", "element_size", "numel")
    )


def _tensor_sample_hash(tensor) -> str | None:
    """Short hash over head+tail samples (cheap role probe, never raises)."""
    try:
        flat = tensor.detach().reshape(-1)
        n = flat.shape[0]
        head = flat[:8192].cpu().tolist()
        tail = flat[-8192:].cpu().tolist() if n > 8192 else head
        return hashlib.sha256(repr((head, tail)).encode("utf-8")).hexdigest()[:32]
    except Exception:
        return None


def _bind_by_names(jit_fn, args, kwargs, names, params) -> list:
    """Bind positional/kwarg values to kernel arg names (empty when none)."""
    bound = []
    pos = list(args)
    for i, name in enumerate(names):
        if i < len(pos):
            value = pos[i]
        elif name in kwargs:
            value = kwargs[name]
        else:
            continue
        is_const = False
        if i < len(params):
            try:
                is_const = bool(params[i].is_constexpr)
            except Exception:
                pass
        bound.append((name, value, is_const))
    return bound


def bind_launch_args(jit_fn, args, kwargs) -> list:
    """Bind run args to [(name, value, is_constexpr)], skipping grid/warmup."""
    try:
        names = list(getattr(jit_fn, "arg_names", []) or [])
        # JITFunction carries resolved Parameter objects on `.params`
        # (with `.is_constexpr`); `.fn.params` is the raw-function fallback
        # used by the unit-test fakes.
        params = getattr(jit_fn, "params", None) or []
        if not params:
            params = getattr(getattr(jit_fn, "fn", None), "params", None) or []
        if names:
            bound = _bind_by_names(jit_fn, args, kwargs or {}, names, params)
            if bound:
                return bound
    except Exception:
        pass
    bound = [(f"arg{i}", v, False) for i, v in enumerate(args)]
    for k, v in (kwargs or {}).items():
        if k in ("grid", "warmup"):
            continue
        bound.append((k, v, False))
    return bound


def eval_grid(grid, meta: dict | None) -> tuple | None:
    """Evaluated (x, y, z) grid tuple, or None when not decidable."""
    try:
        if callable(grid):
            grid = grid(meta or {})
        vals = tuple(int(v) for v in grid)
        if len(vals) > 3 or not all(v >= 0 for v in vals):
            return None
        return vals + (1,) * (3 - len(vals))
    except Exception:
        return None


def _launch_block(compiled) -> dict:
    """Block/cluster/shared/scratch launch properties from kernel metadata.

    Best effort; the scratch sizes size the two trailing ABI params
    (global_scratch, profile_scratch) the replay appends after user args.
    """
    info: dict = {}
    for attr in (
        "num_warps",
        "num_ctas",
        "cluster_dims",
        "shared",
        "global_scratch_size",
        "profile_scratch_size",
    ):
        try:
            md = getattr(compiled, "metadata", None)
            v = getattr(md, attr) if md is not None else getattr(compiled, attr)
            info[attr] = _jsonable(v)
        except Exception:
            pass
    return info


def _record_grid(desc: dict, bound: list, kwargs) -> None:
    """Fill grid/grid_repr from the run kwargs plus constexpr meta values."""
    try:
        meta = {
            name: value
            for name, value, is_const in bound
            if is_const and isinstance(value, (bool, int, float))
        }
        grid = (kwargs or {}).get("grid")
        desc["grid_repr"] = repr(grid)[:200]
        grid_vals = eval_grid(grid, meta)
        desc["grid"] = list(grid_vals) if grid_vals else None
    except Exception:
        pass


def _record_tensor_arg(value, rec: dict, pre) -> None:
    """Fill a tensor arg record (shape/dtype/stride plus an in/out role)."""
    rec["kind"] = "tensor"
    rec["shape"] = [int(d) for d in value.shape]
    rec["dtype"] = str(value.dtype).split(".")[-1]
    rec["stride"] = [int(s) for s in value.stride()]
    post = _tensor_sample_hash(value)
    if pre is None or post is None:
        rec["role"] = "unknown"
    else:
        rec["role"] = "in" if pre == post else "out"


def _record_scalar_arg(value, rec: dict) -> None:
    """Fill a non-tensor arg record (scalars record values, else a repr)."""
    if isinstance(value, bool):
        rec["kind"] = "bool"
        rec["value"] = value
    elif isinstance(value, int):
        rec["kind"] = "int"
        rec["value"] = value
    elif isinstance(value, float):
        rec["kind"] = "float"
        rec["value"] = value
    else:
        rec["kind"] = "other"
        rec["repr"] = repr(value)[:200]


def record_launch(jit_fn, args, kwargs, compiled, pre_hashes=None) -> dict:
    """Launch-desc record for HIP replay (never raises; Nones when undecidable).

    Tensors record shape/dtype/stride plus an in/out role from pre/post
    sample hashes (skipped during autotune benching so timing stays clean);
    constexprs are excluded (baked into the binary); scalars record values.
    """
    desc: dict = {
        "grid": None,
        "grid_repr": None,
        "block": {},
        "device": None,
        "args": [],
    }
    try:
        bound = bind_launch_args(jit_fn, args, kwargs)
    except Exception:
        return desc
    _record_grid(desc, bound, kwargs)
    try:
        desc["block"] = _launch_block(compiled)
    except Exception:
        pass
    pre_hashes = pre_hashes or {}
    for name, value, is_const in bound:
        if not is_const:
            desc["args"].append(
                _record_one_arg(desc, name, value, pre_hashes.get(name))
            )
    return desc


def _record_one_arg(desc: dict, name, value, pre) -> dict:
    """One launch arg record; also fills desc device from the first tensor."""
    rec: dict = {"name": str(name)}
    try:
        if _is_tensor(value):
            _record_tensor_arg(value, rec, pre)
            if desc["device"] is None:
                try:
                    desc["device"] = str(value.device)
                except Exception:
                    pass
        else:
            _record_scalar_arg(value, rec)
    except Exception:
        rec["kind"] = "other"
        rec["repr"] = "unrecordable"
    return rec


def _snapshot_pre_hashes(jit_fn, args, kwargs) -> dict:
    """Pre-run sample hashes for tensor args, keyed by arg name."""
    try:
        return {
            name: _tensor_sample_hash(value)
            for name, value, is_const in bind_launch_args(jit_fn, args, kwargs)
            if not is_const and _is_tensor(value)
        }
    except Exception:
        return {}


def write_case(
    out_dir: str,
    kernel,
    *,
    target: str,
    case: str,
    arch: str,
    source: str,
    input_spec: dict | None = None,
    extra: dict | None = None,
    latency_ms: float | None = None,
    latency_kind: str | None = None,
    launch: dict | None = None,
) -> dict:
    """Write one captured kernel's artifacts + compile.json. Returns the manifest entry."""
    os.makedirs(out_dir, exist_ok=True)
    asm = getattr(kernel, "asm", {}) or {}
    files: dict[str, dict] = {}
    for stage in STAGES:
        if stage in asm and isinstance(asm[stage], str):
            fname = f"kernel.{stage}"
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(asm[stage])
            files[stage] = {"file": fname, "sha256": sha256_text(asm[stage])}
    desc = describe_compiled(kernel)
    entry = {
        "manifest_version": MANIFEST_VERSION,
        "arch": arch,
        "source": source,
        "target": target,
        "case": case,
        "kernel_name": desc.get("name"),
        "target_arch": desc.get("target_arch"),
        "latency_ms": latency_ms,
        "latency_kind": latency_kind,
        "launch": launch,
        "files": files,
        "compile": desc,
        "input_spec": input_spec or {},
        "extra": extra or {},
        "provenance": triton_provenance(),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "compile.json"), "w") as f:
        json.dump(entry, f, indent=2, sort_keys=True)
        f.write("\n")
    return entry


def _run_wrapper(session, original_run):
    """JITFunction.run wrapper capturing compiles plus launch metadata."""

    def run_and_capture(jit_fn, *args, **kwargs):
        pre_hashes = (
            {} if session._bench_active else _snapshot_pre_hashes(jit_fn, args, kwargs)
        )
        compiled = original_run(jit_fn, *args, **kwargs)
        # Cache-hit relaunches return the same object (manual benchmark
        # loops without CUDA graphs relaunch thousands of times); capture
        # distinct compiles only.
        if id(compiled) not in session._seen:
            session._seen.add(id(compiled))
            try:
                fn_name = getattr(jit_fn, "__name__", type(jit_fn).__name__)
            except Exception:
                fn_name = "unknown"
            session.kernels.append((fn_name, compiled))
        # Launch metadata: latest launch per kernel wins (bench launches
        # share grid/args with the winner; the winner's final run sets
        # roles outside the timed region).
        try:
            session.launches[id(compiled)] = record_launch(
                jit_fn, args, kwargs, compiled, pre_hashes=pre_hashes
            )
        except Exception:
            pass
        return compiled

    return run_and_capture


def _compile_wrapper(session, original_compile):
    """triton.compile wrapper capturing kernels not already seen via run()."""

    def compile_and_capture(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)
        # Avoid double-counting kernels already seen via run().
        if id(compiled) not in session._seen:
            session._seen.add(id(compiled))
            session.kernels.append(("triton.compile", compiled))
        return compiled

    return compile_and_capture


class CaptureSession:
    """Context manager installing JITFunction.run + triton.compile hooks.

    ``session.kernels`` accumulates distinct (label, CompiledKernel) pairs
    (cache-hit relaunches return an already-seen object and are skipped).
    ``label`` identifies the launch site; the driver assigns target/case
    names when writing.
    """

    def __init__(self, config_filter: str | None = None):
        self.config_filter = config_filter
        self.kernels: list[tuple[str, object]] = []
        self._seen: set[int] = set()
        self._patches: list[tuple[object, str, object]] = []
        self._config_stats: dict[str, int] = {
            "candidates_seen": 0,
            "candidates_kept": 0,
        }
        # id(CompiledKernel) -> (config repr, median ms) for autotuned configs.
        self.timings: dict[int, tuple[str, float]] = {}
        # id(CompiledKernel) -> launch-desc record for HIP replay (latest wins).
        self.launches: dict[int, dict] = {}
        # True inside Autotuner._bench: skip role hashing so timing stays clean.
        self._bench_active: bool = False

    def _patch(self, obj, name: str, wrapper):
        original = getattr(obj, name)
        setattr(obj, name, wrapper)
        self._patches.append((obj, name, original))
        return original

    def __enter__(self) -> "CaptureSession":
        from unittest import mock  # noqa: F401  (documents the technique)

        from triton.runtime.jit import JITFunction

        self._patch(JITFunction, "run", _run_wrapper(self, JITFunction.run))

        try:
            import triton.compiler.compiler as compiler_mod

            if hasattr(compiler_mod, "compile"):
                self._patch(
                    compiler_mod,
                    "compile",
                    _compile_wrapper(self, compiler_mod.compile),
                )
        except Exception:
            pass

        if self.config_filter:
            self._install_config_filter()
        self._install_bench_hook()
        return self

    def _install_bench_hook(self):
        """Record per-config autotune timings keyed by compiled kernel.

        ``Autotuner._bench`` returns ``[median, p20, p80]`` ms and records the
        config's CompiledKernel on ``_last_compiled_kernel`` -- the only point
        where a config's timing and its compiled artifact meet.
        """
        session = self
        try:
            import triton.runtime.autotuner as autotuner_mod

            if not hasattr(autotuner_mod.Autotuner, "_bench"):
                return
            original_bench = autotuner_mod.Autotuner._bench

            def bench_and_record(autotuner, *args, config=None, **meta):
                session._bench_active = True
                try:
                    timing = original_bench(autotuner, *args, config=config, **meta)
                finally:
                    session._bench_active = False
                try:
                    kernel = getattr(autotuner, "_last_compiled_kernel", None)
                    ms = timing[0] if isinstance(timing, (list, tuple)) else timing
                    if kernel is not None:
                        session.timings[id(kernel)] = (repr(config), float(ms))
                except Exception:
                    pass
                return timing

            self._patch(autotuner_mod.Autotuner, "_bench", bench_and_record)
        except Exception:
            pass

    def _install_config_filter(self):
        """Narrow autotune candidate lists by regex before selection."""
        import triton.runtime.jit as jit_mod

        pattern = re.compile(self.config_filter)
        session = self
        original_autotune = jit_mod.autotune

        def filtered_autotune(configs, *args, **kwargs):
            kept = [c for c in configs if pattern.search(repr(c))]
            session._config_stats["candidates_seen"] += len(configs)
            session._config_stats["candidates_kept"] += len(kept)
            return original_autotune(kept or configs, *args, **kwargs)

        # Only patch when used as @autotune(configs, ...) decorator factory.
        try:
            self._patch(jit_mod, "autotune", filtered_autotune)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        for obj, name, original in reversed(self._patches):
            try:
                setattr(obj, name, original)
            except Exception:
                pass
        self._patches = []
        return False

    def config_selection_report(self) -> dict:
        return dict(self._config_stats)


def iter_cases(root: str, arch: str):
    """Yield (source, target, case, casedir) for case dirs holding kernel.llir."""
    base = os.path.join(root, arch)
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
                llir = os.path.join(cdir, "kernel.llir")
                if os.path.isfile(llir):
                    yield source, target, case, cdir


def write_run_manifest(
    out_root: str, arch: str, entries: list[dict], failures: list[dict]
) -> str:
    """Write the per-arch run manifest. Returns its path."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "arch": arch,
        "triton_channel": "beta",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": triton_provenance(),
        "artifacts": entries,
        "failures": failures,
    }
    path = os.path.join(out_root, arch, "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def append_run_manifest(
    out_root: str, arch: str, source: str, entries: list[dict], failures: list[dict]
) -> str:
    """Merge one source's results into the per-arch run manifest.

    Drivers run sequentially (tlx_ops, tlx_tutorials, tritonbench), each
    appending its section; re-running a source replaces that source's section.
    Returns the manifest path.
    """
    path = os.path.join(out_root, arch, "manifest.json")
    manifest: dict | None = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = None
    if not manifest:
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "arch": arch,
            "triton_channel": "beta",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": triton_provenance(),
            "artifacts": [],
            "failures": [],
        }
    manifest["artifacts"] = [
        a for a in manifest.get("artifacts", []) if a.get("source") != source
    ]
    manifest["failures"] = [
        x for x in manifest.get("failures", []) if x.get("source") != source
    ]
    for e in entries:
        e = dict(e)
        e["source"] = source
        manifest["artifacts"].append(e)
    for x in failures:
        x = dict(x)
        x["source"] = source
        manifest["failures"].append(x)
    manifest["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
