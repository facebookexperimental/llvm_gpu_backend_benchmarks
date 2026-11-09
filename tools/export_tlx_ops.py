# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Export LLVM-stage inputs for tlx.ops targets on AMD hardware.

Runs on the target MI350X GPU (gfx950) under
``ovr_config//triton:beta``::

    buck2 run @//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:export_tlx_ops -- \
        --arch gfx950 --out /tmp/llvm_export

For every ``tlx_ops`` entry in targets.yaml for the requested arch, this
resolves shapes from the op's ``_shapes`` registry, builds inputs exactly
like the op's unit tests, runs the public ``triton.tlx.ops`` wrapper (or the
inductor ``mode="force"`` provider for ``*_torchtlx``) under capture, and
writes one case directory per compiled kernel. Multi-config kernels export
the autotuner winner's ``llir``; ``--config-filter`` narrows candidates first.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

from triton.benchmarks.tools.candidates import label_entry, load_taxonomy
from triton.benchmarks.tools.capture import (
    append_run_manifest,
    CaptureSession,
    case_dir,
    case_latencies,
    describe_compiled,
    probe_arch_id,
    write_case,
)
from triton.benchmarks.tools.discover import load_yaml_subset

TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets.yaml")
CANDIDATES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "candidates.yaml"
)

_TAXONOMY = None


def _taxonomy():
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = load_taxonomy(CANDIDATES_FILE)
    return _TAXONOMY


DTYPE_NAMES = {"fp16": "float16", "bf16": "bfloat16"}

# op -> kernel package holding _shapes.py (or the shapes module directly).
SHAPES_MODULE = {
    "mm": "triton.tlx.ops.kernels.mm._shapes",
    "addmm": "triton.tlx.ops.kernels.addmm._shapes",
    "mm_torchtlx": "triton.tlx.ops.kernels.mm._shapes",
    "addmm_torchtlx": "triton.tlx.ops.kernels.addmm._shapes",
    "bmm_torchtlx": "triton.tlx.ops.kernels.bmm._shapes",
    "grouped_gemm": "triton.tlx.ops.kernels.grouped_gemm._shapes",
    "flash_attn": "triton.tlx.ops.kernels.flash_attn._shapes",
    "flash_attn_mxfp8": "triton.tlx.ops.kernels.flash_attn_mxfp8._shapes",
    "flash_attn_varlen": "triton.tlx.ops.kernels.flash_attn_varlen._shapes",
    "flash_attn_varlen_backward": "triton.tlx.ops.kernels.flash_attn_varlen._shapes",
    "prepare_flash_attn_varlen_backward": "triton.tlx.ops.kernels.flash_attn_varlen._shapes",
    "hstu_attn_dev": "triton.tlx.ops.kernels.hstu_attn._shapes",
    "hstu_cross_attn": "triton.tlx.ops.kernels.hstu_cross_attn._shapes",
    "kda_paged_prefill": "triton.tlx.ops.kernels.kda._prefill_shapes",
    "kda_recurrent_decode": "triton.tlx.ops.kernels.kda._decode_shapes",
}

# op -> static shape entries for ops with no _shapes registry. Each entry
# mirrors the op's unit-test numerics parameters: (split_sizes, N, K).
_STATIC_SHAPES = {
    "grouped_gemm_mxfp8": [
        ((256, 384), 256, 512),
        ((128, 0, 640, 256), 136, 384),
        ((512, 512, 512, 512), 1024, 1024),
        ((128,), 512, 128),
        ((0, 128, 0, 256, 0), 256, 768),
    ],
}

SMOKE_SUBSET_SIZE = 2


def resolve_shapes(op: str, arch: str, suites) -> list:
    """Resolve the shape list for an op following its _shapes registry API."""
    if isinstance(suites, str):
        suites = [suites]
    if op in _STATIC_SHAPES:
        shapes = list(_STATIC_SHAPES[op])
        if suites == ["smoke_subset"]:
            shapes = shapes[:SMOKE_SUBSET_SIZE]
        return shapes
    mod = importlib.import_module(SHAPES_MODULE[op])
    out: list = []
    for suite in suites:
        if suite == "default":
            out.extend(mod.FOCUS.shapes(arch))
        elif suite == "correctness":
            out.extend(mod.CORRECTNESS_SHAPES)
        elif suite == "smoke_subset":
            out.extend(list(mod.FOCUS.shapes(arch))[:SMOKE_SUBSET_SIZE])
        elif suite in ("prefill", "decode"):
            out.extend(mod.FOCUS.shapes(arch))
        else:
            out.extend(mod.FOCUS.shapes(arch, suites=[suite]))
    # Deduplicate, preserving order.
    seen, unique = set(), []
    for shape in out:
        key = tuple(shape) if not hasattr(shape, "_fields") else tuple(shape)
        if key not in seen:
            seen.add(key)
            unique.append(shape)
    return unique


def _torch_dtype(name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _shape_dtype_name(shape) -> str | None:
    if hasattr(shape, "_fields") and "dtype" in shape._fields:
        return shape.dtype
    if isinstance(shape, (tuple, list)) and shape and isinstance(shape[-1], str):
        return shape[-1]
    return None


# ---------------------------------------------------------------------------
# Input builders: (shape_entry, torch) -> (callable, input_spec).
# Each returns a zero-arg callable running the op (fwd, or fwd+bwd) plus a
# JSON-able description of the inputs for compile.json.
# ---------------------------------------------------------------------------


def _build_mm(torch, tlx_mm, shapes_mod, entry, space):
    M, N, K, a_strides, b_strides, dtype_name = tuple(entry)
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(0)
    a = shapes_mod.operand(M, K, a_strides, dtype)
    b = shapes_mod.operand(K, N, b_strides, dtype)

    extra = {} if space is None else {"space": space}

    def run():
        return tlx_mm(a, b, **extra)

    spec = {"M": M, "N": N, "K": K, "dtype": dtype_name, "space": space}
    return run, spec


def _accepts_kwarg(fn, name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _build_addmm(torch, tlx_addmm, shapes_mod, entry, space):
    dtype_name = tuple(entry)[-1]
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(0)
    bias, a, b = shapes_mod.inputs(tuple(entry), dtype)
    out = torch.empty((a.shape[0], b.shape[1]), device="cuda", dtype=dtype)
    extra = {"space": space} if space and _accepts_kwarg(tlx_addmm, "space") else {}

    def run():
        return tlx_addmm(bias, a, b, out=out, **extra)

    spec = {
        "entry": [str(v) for v in tuple(entry)],
        "dtype": dtype_name,
        "space": space,
    }
    return run, spec


def _build_grouped_gemm(torch, tlx_grouped_gemm, shapes_mod, entry, space):
    label, groups = entry.label, entry.shapes
    torch.manual_seed(0)
    a_list, b_list = [], []
    for M, N, K in groups:
        a_list.append(torch.randn((M, K), device="cuda", dtype=torch.float16))
        # Column-major B, matching the unit test's b_storage.T convention.
        b_list.append(torch.randn((N, K), device="cuda", dtype=torch.float16).T)

    def run():
        return tlx_grouped_gemm(a_list, b_list)

    spec = {"label": label, "groups": [list(g) for g in groups]}
    return run, spec


def _build_grouped_gemm_mxfp8(torch, tlx_fn, shapes_mod, entry, space):
    split_sizes, n, k = entry
    gm = sum(split_sizes)
    groups = len(split_sizes)
    scale_k = k // 32
    torch.manual_seed(0)
    x = (torch.randn((gm, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn((groups, n, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    # E8M0 127 == 2**0 scale; matches the unit test's near-unity exponents.
    x_scale = torch.full((gm, scale_k), 127, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    w_scale = torch.full(
        (groups, n, scale_k), 127, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    splits = torch.tensor(list(split_sizes), device="cuda", dtype=torch.int32)

    def run():
        return tlx_fn(x, x_scale, w, w_scale, splits)

    spec = {"split_sizes": list(split_sizes), "n": n, "k": k, "sf_layout": "natural"}
    return run, spec


def _build_flash_attn(torch, tlx_flash_attn, shapes_mod, entry, space, *, bwd):
    batch, heads, context, head_dim, causal, dtype_name = tuple(entry)
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(0)
    shape = (batch, heads, context, head_dim)
    tensors = [
        torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(bwd)
        for _ in range(3)
    ]
    q, k, v = tensors
    scale = None if causal else 0.75 * head_dim**-0.5

    def run():
        out = tlx_flash_attn(
            q,
            k,
            v,
            causal=causal,
            sm_scale=scale,
            **({} if space is None else {"space": space}),
        )
        if bwd:
            out.backward(torch.randn_like(out))
        return out

    spec = {
        "batch": batch,
        "heads": heads,
        "context": context,
        "head_dim": head_dim,
        "causal": causal,
        "dtype": dtype_name,
        "space": space,
        "bwd": bwd,
    }
    return run, spec


def _build_flash_attn_mxfp8(torch, tlx_fn, shapes_mod, entry, space, *, bwd):
    Z, H, N_CTX, HEAD_DIM, causal, dtype_name = tuple(entry)
    torch.manual_seed(20)
    q, k, v = [
        (
            torch.randn((Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            * 0.5
        )
        for _ in range(3)
    ]

    def run():
        return tlx_fn(
            q,
            k,
            v,
            causal=causal,
            sm_scale=0.5,
            **({} if space is None else {"space": space}),
        )

    spec = {
        "batch": Z,
        "heads": H,
        "context": N_CTX,
        "head_dim": HEAD_DIM,
        "causal": causal,
        "dtype": dtype_name,
        "space": space,
    }
    return run, spec


def _build_flash_attn_varlen(torch, tlx_fn, shapes_mod, entry, space):
    # Shape tuples carry trailing kv_min/kv_max_seqlen (0/0 = self-attention);
    # this builder covers self-attention only.
    batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype_name = tuple(entry)[
        :7
    ]
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(20)
    lengths = shapes_mod.seqlens(batch, min_seqlen, max_seqlen)
    q, k, v, cu = shapes_mod.inputs(lengths, heads, head_dim, dtype)

    def run():
        return tlx_fn(q, k, v, cu, cu, max(lengths), max(lengths), causal=causal)

    spec = {
        "lengths": lengths,
        "heads": heads,
        "head_dim": head_dim,
        "causal": causal,
        "dtype": dtype_name,
    }
    return run, spec


def _build_flash_attn_varlen_backward(torch, tlx_fn, shapes_mod, entry, space):
    import triton.tlx.ops as tlx_ops

    batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype_name = tuple(entry)[
        :7
    ]
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(20)
    lengths = shapes_mod.seqlens(batch, min_seqlen, max_seqlen)
    q, k, v, cu = shapes_mod.inputs(lengths, heads, head_dim, dtype)
    # o/lse stand-ins mirror export_tlx_tutorials._family_varlen_bwd: capture
    # only needs their shapes/dtypes, not a real forward result.
    o = torch.empty_like(q)
    do = torch.randn_like(q)
    lse = torch.zeros((heads, sum(lengths)), dtype=torch.float32, device="cuda")
    # Prepared outside capture so run() records only backward kernels.
    plan = tlx_ops.prepare_flash_attn_varlen_backward(cu, cu)

    def run():
        return tlx_fn(q, k, v, o, do, lse, plan, causal=causal)

    spec = {
        "lengths": lengths,
        "heads": heads,
        "head_dim": head_dim,
        "causal": causal,
        "dtype": dtype_name,
        "bwd": True,
    }
    return run, spec


def _build_prepare_flash_attn_varlen_backward(torch, tlx_fn, shapes_mod, entry, space):
    import itertools

    batch, min_seqlen, max_seqlen, heads, head_dim, causal, dtype_name = tuple(entry)[
        :7
    ]
    torch.manual_seed(20)
    lengths = shapes_mod.seqlens(batch, min_seqlen, max_seqlen)
    cu = torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
    )

    def run():
        return tlx_fn(cu, cu)

    spec = {
        "batch": batch,
        "seq_lengths": lengths,
        "total_tokens": sum(lengths),
        "dtype": "int32",
    }
    return run, spec


def _build_hstu(torch, tlx_hstu, shapes_mod, entry, space, *, bwd):
    Z, MAX_SEQ_LEN, H, HEAD_DIM, causal, dtype_name = tuple(entry)
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(0)
    q, k, v, offsets, attn_scale = shapes_mod.inputs(Z, MAX_SEQ_LEN, H, HEAD_DIM, dtype)
    if bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    alpha = 1.0 / HEAD_DIM

    def run():
        out = tlx_hstu(
            q,
            k,
            v,
            offsets,
            MAX_SEQ_LEN,
            attn_scale,
            alpha=alpha,
            causal=causal,
            **({} if space is None else {"space": space}),
        )
        if bwd:
            out.backward(torch.randn_like(out))
        return out

    spec = {
        "Z": Z,
        "max_seq_len": MAX_SEQ_LEN,
        "heads": H,
        "head_dim": HEAD_DIM,
        "causal": causal,
        "dtype": dtype_name,
        "space": space,
        "bwd": bwd,
    }
    return run, spec


def _build_hstu_cross_attn(torch, tlx_fn, shapes_mod, entry, space, *, bwd):
    torch.manual_seed(0)
    kwargs = shapes_mod.inputs(entry, requires_grad=bwd)
    kwargs.pop("lengths_q")
    kwargs.pop("lengths_kv")

    def run():
        out = tlx_fn(
            **kwargs,
            **({} if space is None else {"space": space}),
        )
        if bwd:
            out.backward(torch.randn_like(out))
        return out

    spec = {
        "model": entry.model,
        "draw": entry.draw,
        "batch": entry.batch,
        "heads": entry.heads,
        "head_dim": entry.head_dim,
        "dtype": entry.dtype,
        "space": space,
        "bwd": bwd,
    }
    return run, spec


def _normalized_kda_input(torch, shape, device):
    value = torch.randn(shape, device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(value, dim=-1).to(torch.bfloat16)


def _build_kda_prefill(torch, tlx_fn, shapes_mod, entry, space):
    from itertools import accumulate

    total_tokens, sequences, heads, key_dim, value_dim, dtype_name = tuple(entry)
    device = torch.device("cuda")
    torch.manual_seed(37)
    shape = (1, total_tokens, heads, key_dim)
    q = _normalized_kda_input(torch, shape, device)
    k = _normalized_kda_input(torch, shape, device)
    v = torch.randn(
        1, total_tokens, heads, value_dim, device=device, dtype=torch.bfloat16
    )
    g = -torch.nn.functional.softplus(
        torch.randn(shape, device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, heads, device=device, dtype=torch.float32)
    )
    initial_state = torch.zeros(
        sequences, heads, value_dim, key_dim, device=device, dtype=torch.float32
    )
    base, remainder = divmod(total_tokens, sequences)
    lengths = [base + (i < remainder) for i in range(sequences)]
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device=device, dtype=torch.int64
    )

    def run():
        return tlx_fn(
            q,
            k,
            v,
            g,
            beta,
            scale=1.0,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    spec = {
        "total_tokens": total_tokens,
        "sequences": sequences,
        "heads": heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
    }
    return run, spec


def _build_kda_decode(torch, tlx_fn, shapes_mod, entry, space):
    batch, heads, key_dim, value_dim, dtype_name = tuple(entry)
    device = torch.device("cuda")
    torch.manual_seed(43)
    q_shape = (1, batch, heads, key_dim)
    q = _normalized_kda_input(torch, q_shape, device)
    k = _normalized_kda_input(torch, q_shape, device)
    v = torch.randn(1, batch, heads, value_dim, device=device, dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(
        torch.randn(q_shape, device=device, dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, batch, heads, device=device, dtype=torch.float32)
    )
    state_pool = torch.randn(
        2 * batch, heads, value_dim, key_dim, device=device, dtype=torch.float32
    )
    read_indices = torch.arange(batch, device=device, dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device=device, dtype=torch.int64)

    def run():
        return tlx_fn(
            q,
            k,
            v,
            g,
            beta,
            scale=1.0,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
        )

    spec = {
        "batch": batch,
        "heads": heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
    }
    return run, spec


def _build_torchtlx(torch, provider_fn, shapes_mod, entry, op):
    """Inductor path: impl_for(op, arch) provider with mode="force"."""
    from triton.language.extra.tlx.inductor import tlx_config

    dtype_name = tuple(entry)[-1]
    dtype = _torch_dtype(dtype_name)
    torch.manual_seed(0)
    torch._dynamo.reset()
    args = shapes_mod.inputs(list(entry), dtype)

    def run():
        with tlx_config.patch(use_heuristic_config=True):
            return provider_fn(*args, mode="force")

    spec = {
        "entry": [str(v) for v in tuple(entry)],
        "mode": "force",
        "heuristic_config": True,
    }
    return run, spec


BUILDERS = {
    "mm": _build_mm,
    "addmm": _build_addmm,
    "grouped_gemm": _build_grouped_gemm,
    "grouped_gemm_mxfp8": _build_grouped_gemm_mxfp8,
    "flash_attn": _build_flash_attn,
    "flash_attn_mxfp8": _build_flash_attn_mxfp8,
    "flash_attn_varlen": _build_flash_attn_varlen,
    "flash_attn_varlen_backward": _build_flash_attn_varlen_backward,
    "prepare_flash_attn_varlen_backward": _build_prepare_flash_attn_varlen_backward,
    "hstu_attn_dev": _build_hstu,
    "hstu_cross_attn": _build_hstu_cross_attn,
    "kda_paged_prefill": _build_kda_prefill,
    "kda_recurrent_decode": _build_kda_decode,
}


def _check_arch(arch: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device: run on the target AMD GPU host")
    name = torch.cuda.get_device_name(0)
    want = "MI350"
    if want.lower() not in name.lower().replace(" ", "") and want not in name:
        # Generic marketing names (e.g. 'AMD Radeon Graphics') carry no model
        # info; confirm via arch-level probes instead.
        arch_id = probe_arch_id()
        if arch not in arch_id:
            raise RuntimeError(
                f"arch={arch} needs {want} but device 0 is {name!r}"
                f" (arch probes: {arch_id!r})"
            )
        name = f"{name} [{arch_id}]"
    try:
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        key = getattr(target, "arch", target)
        print(f"[export] device={name} target_arch={key}")
    except Exception as e:
        print(f"[export] device={name} (target probe: {e!r})")


def _resolve_space(want: str | None):
    """Try the requested space, fall back to smoke, then to the default."""
    tried = []
    for space in ([want] if want else []) + ["smoke", None]:
        if space in tried:
            continue
        tried.append(space)
        yield space


def _build_case_run(torch, op, op_fn, shapes_mod, shape, space, mode, builder_kind):
    if builder_kind == "torchtlx":
        return _build_torchtlx(torch, op_fn, shapes_mod, shape, op)
    if op in ("flash_attn", "hstu_attn_dev", "hstu_cross_attn"):
        return BUILDERS[op](torch, op_fn, shapes_mod, shape, space, bwd=(mode == "bwd"))
    if op == "flash_attn_mxfp8":
        return BUILDERS[op](torch, op_fn, shapes_mod, shape, space, bwd=False)
    return BUILDERS[op](torch, op_fn, shapes_mod, shape, space)


def _run_case(
    torch,
    arch,
    out_root,
    op,
    op_fn,
    shapes_mod,
    shape,
    mode,
    case,
    spaces,
    builder_kind,
    config_filter,
    entries_out,
):
    """Run one case across candidate spaces. Returns a failure dict or None."""
    last_error = None
    for try_space in spaces:
        try:
            run, input_spec = _build_case_run(
                torch, op, op_fn, shapes_mod, shape, try_space, mode, builder_kind
            )
            with CaptureSession(config_filter=config_filter) as session:
                run()
                torch.cuda.synchronize()
            if not session.kernels:
                raise RuntimeError("no compiled kernels captured")
            latencies = case_latencies(session, torch, run)
            for kid, (label, kernel) in enumerate(session.kernels):
                case_name = case if kid == 0 else f"{case}__k{kid}"
                out_dir = case_dir(out_root, arch, "tlx_ops", op, case_name)
                latency_ms, latency_kind = latencies.get(id(kernel), (None, None))
                winner_config, _ = session.timings.get(id(kernel), (None, None))
                labels = label_entry(
                    _taxonomy(),
                    arch,
                    {
                        "target": op,
                        "kernel_name": describe_compiled(kernel).get("name"),
                        "input_spec": input_spec,
                        "extra": {
                            "launch": label,
                            "winner_config": winner_config,
                        },
                    },
                )
                entries_out.append(
                    write_case(
                        out_dir,
                        kernel,
                        target=op,
                        case=case_name,
                        arch=arch,
                        source="tlx_ops",
                        input_spec=input_spec,
                        extra={
                            "launch": label,
                            "selection": session.config_selection_report(),
                            "winner_config": winner_config,
                            **labels,
                        },
                        latency_ms=latency_ms,
                        latency_kind=latency_kind,
                        launch=session.launches.get(id(kernel)),
                    )
                )
            return None
        except Exception as e:  # noqa: BLE001 - record per-case, keep going
            last_error = f"{type(e).__name__}: {e}"
    return {"target": op, "case": case, "error": last_error}


def export_op(
    torch, arch, out_root, entry, config_filter, only_mode=None, dry_run=False
):
    import triton.tlx.ops as tlx_ops

    op = entry["op"]
    entries_out, failures = [], []
    shapes_mod = (
        importlib.import_module(SHAPES_MODULE[op]) if op in SHAPES_MODULE else None
    )
    shapes = resolve_shapes(op, arch, entry.get("suites", "default"))
    want_dtypes = set(entry.get("dtypes", []))
    modes = entry.get("modes", ["fwd"])
    if only_mode:
        modes = [only_mode]
    space = entry.get("space")

    if op.endswith("_torchtlx"):
        from triton.tlx.ops._catalog import impl_for

        provider_fn, _spec = impl_for(op, arch=arch)
        builder_kind = "torchtlx"
        op_fn = provider_fn
    else:
        op_fn = getattr(tlx_ops, op)
        builder_kind = "direct"

    forward_only = set()
    if op == "flash_attn" and hasattr(shapes_mod, "FORWARD_ONLY"):
        forward_only = set(shapes_mod.FORWARD_ONLY)

    for shape in shapes:
        dtype_name = _shape_dtype_name(shape)
        if (
            dtype_name
            and want_dtypes
            and DTYPE_NAMES.get(dtype_name, dtype_name) not in want_dtypes
        ):
            continue
        for mode in modes:
            if mode == "bwd" and (
                builder_kind == "torchtlx" or tuple(shape) in forward_only
            ):
                continue
            case = f"{op}_{mode}_" + ("_".join(str(v) for v in tuple(shape)))
            if dry_run:
                entries_out.append({"target": op, "case": case, "dry_run": True})
                continue
            spaces = (
                [space]
                if builder_kind == "torchtlx" or space is None
                else list(_resolve_space(space))
            )
            failure = _run_case(
                torch,
                arch,
                out_root,
                op,
                op_fn,
                shapes_mod,
                shape,
                mode,
                case,
                spaces,
                builder_kind,
                config_filter,
                entries_out,
            )
            if failure is not None:
                failures.append(failure)
                print(f"[export] FAIL {op} {case}: {failure['error']}")
            torch.cuda.empty_cache()
    return entries_out, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export tlx.ops LLVM-stage inputs.")
    parser.add_argument("--arch", required=True, choices=["gfx950"])
    parser.add_argument(
        "--out",
        required=True,
        help="output root (arch-sorted layout is created under it)",
    )
    parser.add_argument("--targets", default=TARGETS_FILE)
    parser.add_argument("--op", default=None, help="only export this op")
    parser.add_argument("--mode", default=None, choices=["fwd", "bwd"])
    parser.add_argument(
        "--config-filter", default=None, help="re pattern narrowing autotune candidates"
    )
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="also run provisional targets",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list cases without running"
    )
    args = parser.parse_args(argv)

    import torch  # noqa: F401

    if not args.dry_run:
        _check_arch(args.arch)
    spec = load_yaml_subset(args.targets)
    if args.config_filter is None:
        args.config_filter = spec.get("config_filter")
    entries = spec["arch"][args.arch].get("tlx_ops", []) or []
    all_entries, all_failures = [], []
    for entry in entries:
        if args.op and entry["op"] != args.op:
            continue
        if entry.get("status") == "provisional" and not args.include_provisional:
            print(
                f"[export] skip provisional {entry['op']} (pass --include-provisional)"
            )
            continue
        try:
            e, f = export_op(
                torch,
                args.arch,
                args.out,
                entry,
                args.config_filter,
                only_mode=args.mode,
                dry_run=args.dry_run,
            )
            status = entry.get("status", "ready")
            for x in f:
                x.setdefault("status", status)
            all_entries.extend(e)
            all_failures.extend(f)
        except Exception:  # noqa: BLE001 - record per-target, keep going
            all_failures.append(
                {
                    "target": entry["op"],
                    "case": "*",
                    "error": traceback.format_exc()[-2000:],
                }
            )
    if not args.dry_run:
        path = append_run_manifest(
            args.out, args.arch, "tlx_ops", all_entries, all_failures
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
