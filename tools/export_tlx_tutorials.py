# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Export LLVM-stage inputs for TLX tutorial kernels on AMD hardware.

Runs on the target MI350X GPU (gfx950) under
``ovr_config//triton:beta``::

    buck2 run @//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:export_tlx_tutorials -- \
        --arch gfx950 --out /tmp/llvm_export

Tutorials ship as buck *resources*, not importable modules, so this driver
locates the tutorials directory (``--tutorials-dir``, else the
``triton.language.extra.tlx`` package's ``tutorials`` entry, else a
``triton.language.extra.tlx.tutorials`` import) and sys.path-inserts it, then
imports tutorial modules by bare name -- the pattern the packaging macro
documents. Each targets.yaml entry runs in one of three modes:

- ``launcher: <fn>`` -- import the module, build inputs for the declared
  family, call the launcher under capture.
- ``launcher: registry`` / ``get_kernel`` / ``auto`` -- fan out over a
  module's kernel variants.
- ``script: <relpath>`` -- execute the file as ``__main__`` under capture.

No correctness checking: inputs must be valid (shapes/dtypes/devices) but
outputs are never compared. Multi-config kernels export the autotuner
winner's ``llir``; ``--config-filter`` narrows candidates first.
"""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
import traceback

from triton.benchmarks.tools.capture import (
    append_run_manifest,
    CaptureSession,
    case_dir,
    case_latencies,
    probe_arch_id,
    write_case,
)
from triton.benchmarks.tools.discover import load_yaml_subset

TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.yaml")


def find_tutorials_dir(explicit: str | None) -> str:
    if explicit:
        if not os.path.isdir(explicit):
            raise RuntimeError(f"--tutorials-dir not a directory: {explicit}")
        return explicit
    # Inplace buck builds and pip installs keep tutorials next to the package.
    try:
        import triton.language.extra.tlx as tlx_pkg

        for base in getattr(tlx_pkg, "__path__", []):
            cand = os.path.join(base, "tutorials")
            if os.path.isdir(cand):
                return cand
    except Exception:
        pass
    try:
        tut_pkg = importlib.import_module("triton.language.extra.tlx.tutorials")

        for base in getattr(tut_pkg, "__path__", []):
            if os.path.isdir(base):
                return base
    except Exception:
        pass
    raise RuntimeError(
        "cannot locate TLX tutorials: pass --tutorials-dir "
        "(third-party/triton/beta/triton/third_party/tlx/tutorials)"
    )


def _torch_dtype(torch, name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _check_arch(torch, arch: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: run on the target AMD GPU host")
    name = torch.cuda.get_device_name(0)
    want = "MI350"
    if want not in name:
        # Generic marketing names (e.g. 'AMD Radeon Graphics') carry no model
        # info; confirm via arch-level probes instead.
        arch_id = probe_arch_id()
        if arch not in arch_id:
            raise RuntimeError(
                f"arch={arch} needs {want} but device 0 is {name!r}"
                f" (arch probes: {arch_id!r})"
            )
        name = f"{name} [{arch_id}]"
    print(f"[export] device={name}")


def _write_captures(
    session, out_root, arch, target, case, input_spec, latencies, label_detail=""
):
    entries = []
    for kid, (label, kernel) in enumerate(session.kernels):
        case_name = case if kid == 0 else f"{case}__k{kid}"
        out_dir = case_dir(out_root, arch, "tlx_tutorials", target, case_name)
        latency_ms, latency_kind = latencies.get(id(kernel), (None, None))
        entries.append(
            write_case(
                out_dir,
                kernel,
                target=target,
                case=case_name,
                arch=arch,
                source="tlx_tutorials",
                input_spec=input_spec,
                extra={
                    "launch": label,
                    "detail": label_detail,
                    "selection": session.config_selection_report(),
                },
                latency_ms=latency_ms,
                latency_kind=latency_kind,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Input families (shapes/dtypes follow the tutorial tests; values are random).
# ---------------------------------------------------------------------------


def _gemm_inputs(torch, shapes, dtype):
    out = []
    for M, N, K in shapes:
        torch.manual_seed(0)
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b = torch.randn((K, N), device="cuda", dtype=dtype)
        out.append(((M, N, K), (a, b)))
    return out


def run_launcher(torch, session_mod, launcher: str, inputs: dict, target: str):
    """Run one launcher target. Yields (case_suffix, thunk, input_spec)."""
    family = inputs.get("family")
    if not isinstance(family, str):
        raise ValueError(f"unknown input family: {family}")
    builder = _FAMILY_BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"unknown input family: {family}")
    yield from builder(torch, session_mod, launcher, inputs)


def _family_gemm(torch, session_mod, launcher, inputs):
    dtype = _torch_dtype(torch, inputs.get("dtype", "float16"))
    fn = getattr(session_mod, launcher)
    for (M, N, K), (a, b) in _gemm_inputs(torch, inputs["shapes"], dtype):
        yield (
            f"gemm_{M}x{N}x{K}",
            lambda fn=fn, a=a, b=b: fn(a, b),
            {"M": M, "N": N, "K": K},
        )


def _family_addmm(torch, session_mod, launcher, inputs):
    dtype = _torch_dtype(torch, inputs.get("dtype", "float16"))
    fn = getattr(session_mod, launcher)
    for (M, N, K), (a, b) in _gemm_inputs(torch, inputs["shapes"], dtype):
        bias = torch.randn((N,), device="cuda", dtype=dtype)
        paths = ["default"]
        if inputs.get("paths") == "auto" and hasattr(session_mod, "available_paths"):
            try:
                paths = list(session_mod.available_paths(bias, a, b)) or paths
            except Exception:
                pass
        for path in paths:
            kw = {} if path == "default" else {"path": path}
            spec = {"M": M, "N": N, "K": K, "path": path, "b_layout": "row_major"}
            yield (
                f"addmm_{M}x{N}x{K}_{path}",
                _addmm_thunk(fn, bias, a, b, kw, spec),
                spec,
            )


def _addmm_thunk(fn, bias, a, b, kw, spec):
    """Launch an addmm case, retrying with column-major B if required.

    Most launchers take row-major B, but some (e.g. amd_addmm_gfx950)
    require column-major; the ValueError raises before any compile, so the
    retry is cheap. Records the layout that ran in the case spec.
    """

    def run():
        try:
            return fn(bias, a, b, **kw)
        except ValueError as e:
            if "column-major" not in str(e):
                raise
            spec["b_layout"] = "col_major"
            return fn(bias, a, b.t().contiguous().t(), **kw)

    return run


def _family_bmm(torch, session_mod, launcher, inputs):
    dtype = _torch_dtype(torch, inputs.get("dtype", "float16"))
    fn = getattr(session_mod, launcher)
    for shape in inputs["shapes"]:
        B, M, N, K = shape
        torch.manual_seed(0)
        if inputs.get("input_fn") and hasattr(session_mod, inputs["input_fn"]):
            a, b = getattr(session_mod, inputs["input_fn"])(
                B, M, N, K, "cuda", dtype=dtype
            )
        else:
            a = torch.randn((B, M, K), device="cuda", dtype=dtype)
            b = torch.randn((B, K, N), device="cuda", dtype=dtype)
        yield (
            f"bmm_{B}x{M}x{N}x{K}",
            lambda fn=fn, a=a, b=b: fn(a, b),
            {"B": B, "M": M, "N": N, "K": K},
        )


def _family_attn(torch, session_mod, launcher, inputs):
    dtype = _torch_dtype(torch, inputs.get("dtype", "float16"))
    fns = _attn_variants(session_mod, launcher, inputs)
    seqlens = (
        inputs["seqlen"] if isinstance(inputs["seqlen"], list) else [inputs["seqlen"]]
    )
    for name, fn in fns:
        for S in seqlens:
            for causal in (False, True):
                torch.manual_seed(0)
                B, H, D = inputs["batch"], inputs["heads"], inputs["head_dim"]
                q = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
                k = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
                v = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
                scale = D**-0.5
                yield (
                    f"{name}_b{B}h{H}s{S}d{D}_{'causal' if causal else 'ncausal'}",
                    lambda fn=fn, q=q, k=k, v=v, scale=scale, causal=causal: fn(
                        q, k, v, scale, causal
                    ),
                    {"B": B, "H": H, "S": S, "D": D, "causal": causal},
                )


def _family_addmm_glu(torch, session_mod, launcher, inputs):
    import inspect

    dtype = _torch_dtype(torch, inputs.get("dtype", "float16"))
    registry = session_mod.KERNEL_REGISTRY
    M = getattr(session_mod, "M", 1024)
    N = getattr(session_mod, "N", 1024)
    # Runners index their best-config tables by K; mirror the test's parametrize.
    for kernel_name in inputs["kernels"]:
        for K in inputs.get("K_list", [256, 512, 1024]):
            torch.manual_seed(0)
            a = torch.randn((M, K), device="cuda", dtype=dtype)
            b = torch.randn((K, N), device="cuda", dtype=dtype)
            bias = torch.randn((N,), device="cuda", dtype=dtype)
            y = torch.randn((M, N), device="cuda", dtype=dtype)
            entry = registry[kernel_name]
            runner = entry[0] if isinstance(entry, (tuple, list)) else entry
            sig = inspect.signature(runner) if callable(runner) else None
            yield (
                f"glu_{kernel_name}_k{K}",
                lambda runner=runner, a=a, b=b, bias=bias, y=y, sig=sig: _call_glu(
                    runner, sig, a, b, bias, y
                ),
                {"kernel": kernel_name, "M": M, "N": N, "K": K},
            )


def _family_pa_decode(torch, session_mod, launcher, inputs):
    import math

    from triton.language.extra.tlx.ops.amd_pa_decode import pa_decode_tlx

    build = session_mod.build_inputs
    num_kv_heads, group, head_dim, page_size = 2, 4, 128, 16
    num_q_heads = num_kv_heads * group
    ctx_lens = [40, 71]
    sm_scale = 1.0 / math.sqrt(head_dim)
    for query_length, num_splits, layout in [(1, 4, "5d")]:
        q, kc, vc, cl, bt = build(
            len(ctx_lens),
            ctx_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            query_length=query_length,
            device="cuda",
            cache_layout=layout,
        )
        out = torch.empty_like(q)
        yield (
            f"pa_decode_ql{query_length}_sp{num_splits}_{layout}",
            lambda out=out,
            q=q,
            kc=kc,
            vc=vc,
            cl=cl,
            bt=bt,
            sm_scale=sm_scale,
            query_length=query_length,
            num_splits=num_splits: pa_decode_tlx(
                out,
                q,
                kc,
                vc,
                cl,
                bt,
                sm_scale,
                query_length=query_length,
                num_splits=num_splits,
            ),
            {
                "query_length": query_length,
                "num_splits": num_splits,
                "layout": layout,
            },
        )


def _family_gdpa(torch, session_mod, launcher, inputs):
    gen = session_mod.generate_gdpa_data
    fn = getattr(session_mod, launcher)
    B, max_M, H, dff, sparsity = 8, 500, 4, 256, 0.68
    D = H * 64
    data = gen(
        B,
        max_M,
        D,
        H,
        dff,
        sparsity=sparsity,
        dtype=torch.bfloat16,
        device="cuda",
        seed=42,
        seq_len_mode="uniform",
    )
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "NUM_BUFFERS": 2,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "num_stages": 1,
        "num_warps": 4,
    }
    q, k, v, q_offsets = data["q"], data["k"], data["v"], data["q_offsets"]
    yield (
        "gdpa_uniform",
        lambda: fn(q, k, v, q_offsets, dff, qk_scale=1.0, config=config),
        {"B": B, "max_M": max_M, "H": H, "dff": dff, "config": config},
    )


def _family_varlen_bwd(torch, session_mod, launcher, inputs):
    from itertools import accumulate

    torch.manual_seed(549)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(549)
    q_lens, kv_lens = [512, 256], [512, 256]
    total_q, total_kv = sum(q_lens), sum(kv_lens)
    q = torch.randn(
        (total_q, 4, 128), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    k = torch.randn(
        (total_kv, 4, 128), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    v = torch.randn(
        (total_kv, 4, 128), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    do = torch.randn(
        (total_q, 4, 128), dtype=torch.bfloat16, device="cuda", generator=gen
    )
    o = torch.empty_like(q)
    lse = torch.zeros((4, total_q), dtype=torch.float32, device="cuda")
    cu_q = torch.tensor(
        [0, *accumulate(q_lens)],
        device="cuda",
        dtype=torch.int64,
    )
    cu_kv = torch.tensor(
        [0, *accumulate(kv_lens)],
        device="cuda",
        dtype=torch.int64,
    )
    plan = session_mod.prepare_varlen_backward(cu_q, cu_kv)
    fn = getattr(session_mod, launcher)
    yield (
        "varlen_bwd_d128",
        lambda: fn(q, k, v, o, do, lse, plan, 128**-0.5),
        {"q_lens": q_lens, "kv_lens": kv_lens, "heads": 4},
    )


def _family_ikbo(torch, session_mod, launcher, inputs):
    for sub in ("ikbo_fa", "ikbo_lce"):
        try:
            mod = importlib.import_module(f"{session_mod.__name__}.{sub}_triton")
        except ImportError:
            try:
                mod = importlib.import_module(sub)
            except ImportError:
                continue
        create = getattr(mod, "create_inputs", None)
        kernel = getattr(mod, sub, None)
        if create is None or kernel is None:
            continue
        inp = create()
        args = inp if isinstance(inp, tuple) else (inp,)
        yield (
            f"ikbo_{sub}",
            lambda kernel=kernel, args=args: kernel(*args),
            {"sub": sub},
        )


_FAMILY_BUILDERS = {
    "gemm": _family_gemm,
    "addmm": _family_addmm,
    "bmm": _family_bmm,
    "attn": _family_attn,
    "addmm_glu": _family_addmm_glu,
    "pa_decode": _family_pa_decode,
    "gdpa": _family_gdpa,
    "varlen_bwd": _family_varlen_bwd,
    "ikbo": _family_ikbo,
}


def _attn_variants(session_mod, launcher, inputs):
    if launcher == "get_kernel":
        get_kernel = session_mod.get_kernel
        return [(name, get_kernel(name)) for name in inputs.get("kernels", [])]
    fns = [(launcher, getattr(session_mod, launcher))]
    if inputs.get("persistent") and hasattr(session_mod, "persistent_attention"):
        fns.append(("persistent_attention", session_mod.persistent_attention))
    return fns


def _call_glu(runner, sig, a, b, bias, y):
    import inspect

    out = __import__("torch").empty_like(a)
    if sig is None:
        return runner(a, b, bias, y, out)
    # run_kernel_*(a, b, bias, y, out, cfg[, num_cus, ...]): pass what it takes.
    kwargs: dict = {}
    params = list(sig.parameters.values())
    positional = [a, b, bias, y, out]
    args = positional[
        : max(0, len([p for p in params if p.default is inspect.Parameter.empty]))
    ]
    if "cfg" in sig.parameters:
        kwargs["cfg"] = None
    return runner(*args, **kwargs)


def run_script(tutorials_dir: str, module: str, script: str, inputs: dict):
    """Yield (case_suffix, thunk, input_spec) for script-mode targets."""
    base = os.path.join(tutorials_dir, module) if "/" not in script else tutorials_dir
    if module == "gfx9_gemm":
        # Series bench first, then per-version matmul for full coverage.
        path = os.path.join(tutorials_dir, module, script)
        yield (
            "series_bench",
            lambda: _exec_main(path, []),
            {"script": f"{module}/{script}"},
        )
        versions = [f"a16w16.v{i}" for i in range(10)] + ["a16w16_matmul"]
        for ver in versions:
            yield (
                f"version_{ver}",
                lambda ver=ver: _gfx9_version_matmul(tutorials_dir, ver),
                {"version": ver},
            )
        yield (
            "inter_wave",
            lambda: _gfx9_inter_wave(tutorials_dir),
            {"version": "inter_wave"},
        )
        return
    if script == "main":
        path = os.path.join(tutorials_dir, module + ".py")
        if not os.path.exists(path):
            path = os.path.join(tutorials_dir, module, "__main__.py")
        yield "main", lambda: _exec_main(path, []), {"script": module}
        return
    path = os.path.join(base, script)
    yield "script", lambda: _exec_main(path, inputs.get("args", [])), {"script": script}


def _exec_main(path: str, args: list[str]):
    old_argv = sys.argv
    sys.argv = [path, *args]
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = old_argv


def _gfx9_version_matmul(tutorials_dir: str, version: str):
    import torch

    subdir = os.path.join(tutorials_dir, "gfx9_gemm", *version.split("."))
    if subdir not in sys.path:
        sys.path.insert(0, subdir)
    if (
        os.path.isdir(os.path.join(tutorials_dir, "gfx9_gemm"))
        and os.path.join(tutorials_dir, "gfx9_gemm") not in sys.path
    ):
        sys.path.insert(0, os.path.join(tutorials_dir, "gfx9_gemm"))
    try:
        mod = importlib.import_module("matmul_kernel")
    except ImportError:
        mod = importlib.import_module(
            f"gfx9_gemm.{version}.matmul_kernel".replace(
                "a16w16_matmul", "a16w16_matmul"
            )
        )
    torch.manual_seed(0)
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    try:
        return mod.matmul(a, b)
    except TypeError:
        return mod.matmul(a, b, None)


def _gfx9_inter_wave(tutorials_dir: str):
    import torch

    d = os.path.join(tutorials_dir, "gfx9_gemm", "inter_wave", "a16w16")
    if d not in sys.path:
        sys.path.insert(0, d)
    mod = importlib.import_module("matmul_kernel_split_m")
    torch.manual_seed(0)
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    try:
        return mod.matmul(a, b)
    except TypeError:
        return mod.matmul(a, b, None)


def export_target(
    torch, arch, out_root, entry, tutorials_dir, config_filter, dry_run=False
):
    module = entry["module"]
    entries_out, failures = [], []
    if entry.get("requires") == ["multi_gpu"] or "multi_gpu" in (
        entry.get("requires") or []
    ):
        if torch.cuda.device_count() < 2:
            failures.append(
                {
                    "target": module,
                    "case": "*",
                    "error": "skipped: requires multi_gpu (world_size>=2), only "
                    f"{torch.cuda.device_count()} visible",
                }
            )
            return entries_out, failures
    try:
        if "script" in entry:
            cases = list(
                run_script(
                    tutorials_dir,
                    module,
                    entry["script"],
                    entry.get("inputs", {}) or {},
                )
            )
        else:
            mod = importlib.import_module(module.replace("/", "."))
            cases = list(
                run_launcher(
                    torch, mod, entry["launcher"], entry.get("inputs", {}) or {}, module
                )
            )
    except Exception as e:  # noqa: BLE001 - record per-target, keep going
        failures.append(
            {"target": module, "case": "*", "error": f"{type(e).__name__}: {e}"}
        )
        return entries_out, failures
    for case_suffix, thunk, input_spec in cases:
        case = f"{module.replace('/', '_')}_{case_suffix}"
        if dry_run:
            entries_out.append({"target": module, "case": case, "dry_run": True})
            continue
        try:
            with CaptureSession(config_filter=config_filter) as session:
                thunk()
                torch.cuda.synchronize()
            if not session.kernels:
                raise RuntimeError("no compiled kernels captured")
            latencies = case_latencies(
                session, torch, thunk, timeable="script" not in entry
            )
            entries_out.extend(
                _write_captures(
                    session, out_root, arch, module, case, input_spec, latencies
                )
            )
        except Exception as e:  # noqa: BLE001 - record per-case, keep going
            failures.append(
                {"target": module, "case": case, "error": f"{type(e).__name__}: {e}"}
            )
            print(f"[export] FAIL {module} {case}: {e}")
        torch.cuda.empty_cache()
    return entries_out, failures


def _run_entries(torch, entries, tutorials_dir, args):
    all_entries, all_failures = [], []
    for entry in entries:
        if args.module and entry["module"] != args.module:
            continue
        if entry.get("status") == "provisional" and not args.include_provisional:
            print(
                f"[export] skip provisional {entry['module']} (pass --include-provisional)"
            )
            continue
        try:
            e, f = export_target(
                torch,
                args.arch,
                args.out,
                entry,
                tutorials_dir,
                args.config_filter,
                dry_run=args.dry_run,
            )
        except Exception:  # noqa: BLE001 - record per-target, keep going
            all_failures.append(
                {
                    "target": entry["module"],
                    "case": "*",
                    "error": traceback.format_exc()[-2000:],
                }
            )
            continue
        status = entry.get("status", "ready")
        for x in f:
            x.setdefault("status", status)
        all_entries.extend(e)
        all_failures.extend(f)
    return all_entries, all_failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export TLX tutorial LLVM-stage inputs."
    )
    parser.add_argument("--arch", required=True, choices=["gfx950"])
    parser.add_argument(
        "--out",
        required=True,
        help="output root (arch-sorted layout is created under it)",
    )
    parser.add_argument("--targets", default=TARGETS_FILE)
    parser.add_argument("--tutorials-dir", default=None)
    parser.add_argument("--module", default=None, help="only export this module")
    parser.add_argument(
        "--config-filter", default=None, help="re pattern narrowing autotune candidates"
    )
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import torch  # noqa: F401

    tutorials_dir = find_tutorials_dir(args.tutorials_dir)
    if tutorials_dir not in sys.path:
        sys.path.insert(0, tutorials_dir)
    testing_dir = os.path.join(tutorials_dir, "testing")
    if os.path.isdir(testing_dir) and testing_dir not in sys.path:
        sys.path.insert(0, testing_dir)
    print(f"[export] tutorials_dir={tutorials_dir}")
    if not args.dry_run:
        _check_arch(torch, args.arch)
    spec = load_yaml_subset(args.targets)
    if args.config_filter is None:
        args.config_filter = spec.get("config_filter")
    entries = spec["arch"][args.arch].get("tlx_tutorials", []) or []
    all_entries, all_failures = _run_entries(torch, entries, tutorials_dir, args)
    if not args.dry_run:
        path = append_run_manifest(
            args.out, args.arch, "tlx_tutorials", all_entries, all_failures
        )
        print(
            f"[export] {len(all_entries)} cases, {len(all_failures)} failures -> {path}"
        )
    else:
        for e in all_entries:
            print(f"  {e['target']} {e['case']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
