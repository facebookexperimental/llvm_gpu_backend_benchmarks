# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Enumerate export sources and validate targets.yaml against them.

Stdlib only, no GPU required. Run from anywhere inside fbsource::

    python3 discover.py --check
    python3 discover.py --list
    python3 discover.py --regenerate-skeleton

--check (default) fails when a live source contains an in-scope target that
targets.yaml does not mention, or when targets.yaml mentions something the
sources cannot resolve. Wire it into the nightly job so new kernels get
flagged instead of silently skipped.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS_FILE = os.path.join(HERE, "targets.yaml")

ARCHES = ("gfx950",)

# Tutorial modules that discovery must NOT require in targets.yaml, with the
# reason each is out of scope. discover.py --check enforces that this list
# matches reality (every entry must still exist; nothing else may be missing).
TUTORIAL_EXCLUSIONS = {
    # No kernels: reference-case helpers only (0 @triton.jit in file).
    "amd_fa_bwd": "helpers-only, bwd kernel exported via tlx_ops flash_attn",
    # Harness scripts that drive kernel modules covered elsewhere.
    "amd_fa_adaptive_bench": "harness for amd_fa_adaptive",
    "amd_fa_compare": "harness comparing FA variants",
    "accuracy_bench_bwd_autows_vs_tlx": "harness script",
    # Wrong hardware.
    "amd_tdm_gemm_pipelined": "gfx1250-only (is_hip_gfx1250 gate)",
    "amd_fa_tdm_pipelined": "gfx1250-only (TDM path)",
    "amd_grouped_gemm_gfx1250": "gfx1250-only (name-pinned)",
    # NVIDIA-only.
    "blackwell-cross-attention": "nvidia-only",
    "blackwell_fa_clc": "nvidia-only",
    "blackwell_fa_ws_persistent": "nvidia-only",
    "blackwell_fa_ws_pipelined_persistent_mxfp8": "nvidia-only",
    "blackwell_fa_ws_pipelined_persistent": "nvidia-only",
    "blackwell_fa_ws_pipelined": "nvidia-only",
    "blackwell_fa_ws": "nvidia-only",
    "blackwell-gdpa": "nvidia-only",
    "blackwell_gemm_2cta": "nvidia-only",
    "blackwell_gemm_clc": "nvidia-only",
    "blackwell_gemm_pipelined": "nvidia-only",
    "blackwell_gemm_ws_mxfp8": "nvidia-only",
    "blackwell_gemm_ws": "nvidia-only",
    "blackwell-multi-cta-layernorm_test": "nvidia-only",
    "blackwell_multi_cta_layernorm_ws_test": "nvidia-only",
    "blackwell_scaled_mm_ws": "nvidia-only",
    "blackwell-triton-addmm-2cta_test": "nvidia-only",
    "hopper_fa_ws_pipelined_pingpong": "nvidia-only",
    "hopper_gemm_pipelined": "nvidia-only",
    "hopper_gemm_ws": "nvidia-only",
    "hopper-persistent-gemm-ws-cooperative": "nvidia-only",
    "hopper-persistent-gemm-ws-pingpong": "nvidia-only",
    "fused_attention_ws_auto_tma_dp": "nvidia-only",
    "fused_attention_ws_auto_tma": "nvidia-only",
    "fused_attention_ws_device_tma_dp": "nvidia-only",
    "fused_attention_ws_device_tma": "nvidia-only",
    "test_cross_attention_bwd_autows": "nvidia-only",
    "test_self_attention_autows": "nvidia-only",
    "test_self_attention_bwd": "nvidia-only",
    "test_self_attention_tlx_autows": "nvidia-only",
    "vector-add2": "arch-agnostic hello-world, not an AMD kernel target",
}

# mi350_ci.yaml entries with no Triton kernels (non-triton subops only).
TRITONBENCH_GFX950_EXCLUSIONS = {
    "jagged_flash_attention_fwd": "mkl_jfav2 subop only, no Triton kernels",
    "jagged_flash_attention_bwd": "mkl_jfav2 subop only, no Triton kernels",
    "jagged_dense_bmm": "amkl subop only, no Triton kernels",
    "reduction_gemm": "named fused subop removed Jan 2026, persistent is WS/NVIDIA-gated, inductor uncapturable via --dump-ir",
    # GPU-OOM skips: these mi350_ci entries allocate multi-GB inputs and OOM
    # on the shared MI350X host (0 bytes free); no .llir is capturable there.
    "addmm": "GPU OOM on shared host, no .llir captured",
    "flash_attention": "GPU OOM on shared host, no .llir captured",
    "fp8_gemm": "GPU OOM on shared host, no .llir captured",
    "bf16_gemm": "GPU OOM on shared host, no .llir captured",
    "fp32_gemm": "GPU OOM on shared host, no .llir captured",
    "generalized_dot_product_attention": "GPU OOM on shared host, no .llir captured",
}


def find_root(start: str = HERE) -> str:
    """Walk up to the fbsource root (the dir containing third-party/triton)."""
    if os.environ.get("FBSOURCE"):
        return os.environ["FBSOURCE"]
    cur = os.path.abspath(start)
    for _ in range(12):
        if os.path.isdir(os.path.join(cur, "third-party", "triton")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise RuntimeError("could not locate fbsource root (set $FBSOURCE)")


# ---------------------------------------------------------------------------
# Minimal YAML-subset reader (stdlib only; devservers have no PyYAML).
#
# Supports exactly what targets.yaml and mi350_ci.yaml use: nested maps by
# indentation, "- " list items holding maps or scalars, comments, quoted and
# bare scalars, flow lists [...], flow maps {...}, and folded continuation
# lines (a line more indented than its key that starts no new key/item is
# joined to the previous scalar with a space, matching YAML's plain-scalar
# folding for our files).
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    out = []
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            out.append(c)
            if c == quote:
                quote = None
            elif c == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 1
        elif c in ("'", '"'):
            quote = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).rstrip()


def _split_flow(s: str) -> list[str]:
    parts, depth, quote, cur = [], 0, None, []
    for c in s:
        if quote:
            cur.append(c)
            quote = None if c == quote else quote
        elif c in ("'", '"'):
            quote = c
            cur.append(c)
        elif c in "[{":
            depth += 1
            cur.append(c)
        elif c in "]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if cur or not parts:
        parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


def _scalar(text: str):
    text = text.strip()
    if text == "" or text == "~" or text == "null":
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(p) for p in _split_flow(text[1:-1])]
    if text.startswith("{") and text.endswith("}"):
        out = {}
        for part in _split_flow(text[1:-1]):
            k, _, v = part.partition(":")
            out[k.strip()] = _scalar(v)
        return out
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # YAML 1.1 unquoted on/off/yes/no: keep as strings (never used as bools).
    return text


def load_yaml_subset(path: str):
    """Parse our YAML subset into plain dicts/lists/scalars."""
    with open(path) as f:
        raw = [_strip_comment(line.rstrip("\n")) for line in f]
    lines = [(len(line) - len(line.lstrip()), line.strip()) for line in raw]
    lines = [(i, t) for i, t in lines if t]
    root: dict = {}
    # Stack of (indent, container). A `key:` with no value awaits a nested
    # block, decided by peeking at the next line's indent.
    stack: list[tuple[int, object]] = [(-1, root)]

    def container_for(indent: int):
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        return stack[-1][1]

    i = 0
    while i < len(lines):
        indent, text = lines[i]
        parent = container_for(indent)
        if text.startswith("- "):
            i = _parse_list_item(path, lines, i, indent, text, parent, stack)
        elif _KEY_RE.match(text):
            i = _parse_map_key(path, lines, i, indent, text, parent, stack)
        else:
            raise ValueError(f"{path}: cannot parse line: {text!r}")
    return root


_KEY_RE = re.compile(r"^[A-Za-z0-9_./-]+\s*:")
_ITEM_OR_KEY_RE = re.compile(r"^(- |[A-Za-z0-9_./-]+\s*:)")


def _fold_continuations(lines, i: int, indent: int) -> tuple[list[str], int]:
    """Collect deeper-indented plain lines (YAML plain-scalar folding)."""
    parts, j = [], i + 1
    while (
        j < len(lines)
        and lines[j][0] > indent
        and not _ITEM_OR_KEY_RE.match(lines[j][1])
    ):
        parts.append(lines[j][1])
        j += 1
    return parts, j - 1


def _parse_list_item(path, lines, i, indent, text, parent, stack) -> int:
    if not isinstance(parent, list):
        raise ValueError(f"{path}: list item outside a list: {text!r}")
    item = text[2:].strip()
    if item == "" or _KEY_RE.match(item):
        child: dict = {}
        parent.append(child)
        stack.append((indent, child))
        if item:
            # Reprocess the remainder as a mapping line at deeper indent.
            lines[i] = (indent + 2, item)
            return i
        return i + 1
    parent.append(_scalar(item))
    extra, i = _fold_continuations(lines, i, indent)
    for part in extra:
        parent[-1] = f"{parent[-1]} {part}"
    return i + 1


def _parse_map_key(path, lines, i, indent, text, parent, stack) -> int:
    key, _, rest = text.partition(":")
    key, rest = key.strip(), rest.strip()
    if not isinstance(parent, dict):
        raise ValueError(f"{path}: mapping key inside a list: {text!r}")
    if rest == "":
        # Nested block (or empty). Peek: list or map?
        if i + 1 < len(lines) and lines[i + 1][0] > indent:
            child = [] if lines[i + 1][1].startswith("- ") else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = None
        return i + 1
    extra, i = _fold_continuations(lines, i, indent)
    parent[key] = _scalar(" ".join([rest, *extra]))
    return i + 1


# ---------------------------------------------------------------------------
# Source enumeration (all file-local, no imports of torch/triton).
# ---------------------------------------------------------------------------


def catalog_entries(root: str) -> dict[tuple[str, str], dict]:
    """Parse (op, arch) entries out of tlx ops _catalog.py via AST."""
    path = os.path.join(
        root,
        "third-party/triton/beta/triton/third_party/tlx/ops/_catalog.py",
    )
    tree = ast.parse(open(path).read())
    entries: dict[tuple[str, str], dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "OpSpec"):
            continue
        kw = {}
        for k in node.keywords:
            if k.arg in ("op", "arch", "variant", "impl", "supports_backward"):
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except (ValueError, SyntaxError):
                    # Non-literal (e.g. impl=<class ref>): keep a repr so the
                    # entry still counts for drift detection.
                    kw[k.arg] = ast.unparse(k.value)
        if "op" in kw and "arch" in kw:
            entries[(kw["op"], kw["arch"])] = kw
    return entries


def _test_arch_gates(root: str) -> dict[str, set[str]]:
    """Map tutorial name -> arches from test_correctness.py skipifs.

    A test function gated on is_hip_cdna4() or is_hip() maps to gfx950;
    is_hip_gfx1250() maps to none (out of scope).
    """
    path = os.path.join(
        root,
        "third-party/triton/beta/triton/third_party/tlx/tutorials/testing/test_correctness.py",
    )
    src = open(path).read().splitlines()
    gates: dict[str, set[str]] = {}
    pending: set[str] | None = None
    for line in src:
        m = re.search(r"skipif\(not (is_hip_cdna4|is_hip_gfx1250|is_hip)\(\)", line)
        if m:
            kind = m.group(1)
            pending = set() if kind == "is_hip_gfx1250" else {"gfx950"}
            continue
        m = re.match(r"def (test_\w+)\(", line)
        if m and pending is not None:
            gates.setdefault(m.group(1), set()).update(pending)
            pending = None
    return gates


def tutorial_modules(root: str) -> dict[str, dict]:
    """Enumerate tutorial modules with an arch classification each."""
    base = os.path.join(
        root, "third-party/triton/beta/triton/third_party/tlx/tutorials"
    )
    gates = _test_arch_gates(root)
    mods: dict[str, dict] = {}
    for fname in sorted(os.listdir(base)):
        if fname in ("testing", "README.md", "__pycache__") or fname.startswith("."):
            continue
        p = os.path.join(base, fname)
        if os.path.isdir(p):
            mods[fname] = {"path": p, "kind": "group", "arches": set()}
        elif fname.endswith(".py"):
            mods[fname[:-3]] = {"path": p, "kind": "module", "arches": set()}
    for name, info in mods.items():
        info["arches"] = _classify_arch(name, gates)
    for name, arches in _CURATED_ARCHES.items():
        if name in mods:
            mods[name]["arches"] = set(arches)
    return mods


# Groups and a few modules need curated arch info (no direct test-name
# match); keep the overrides explicit and reviewable.
_CURATED_ARCHES = {
    "gfx9_gemm": {"gfx950"},
    "hstu_cross_attn": {"gfx950"},
    "hstu_self_attn": {"gfx950"},
    "ikbo": {"gfx950"},
    "distributed": {"gfx950"},
    "amd_fa_adaptive": {"gfx950"},
    "amd_fa_varlen_bwd": {"gfx950"},
    "amd_bmm_shared_a": {"gfx950"},
    "amd_pa_decode": {"gfx950"},
    "amd_addmm_glu": {"gfx950"},
}


_ARCH_PIN = re.compile(r"gfx\d+")


def _classify_arch(name: str, gates: dict[str, set[str]]) -> set[str]:
    """Filename pins, then test gates referencing the module name.

    A gfxN pin outside ARCHES (MI300X-only or gfx1250-only kernel files)
    is out of scope regardless of test gates; no per-module entry needed.
    """
    m = _ARCH_PIN.search(name)
    if m:
        return {m.group(0)} if m.group(0) in ARCHES else set()
    arches: set[str] = set()
    for test, t_arches in gates.items():
        # test_amd_fa_persistent... -> amd_fa_persistent
        mod_part = test.removeprefix("test_")
        if mod_part == name or mod_part.startswith(name + "_"):
            arches.update(t_arches)
    return arches


def mi350_entries(root: str) -> dict[str, dict]:
    """Parse benchmarks/nightly/mi350_ci.yaml (minus common_args)."""
    path = os.path.join(
        root, "fbcode/pytorch/tritonbench/benchmarks/nightly/mi350_ci.yaml"
    )
    data = load_yaml_subset(path)
    return {k: v for k, v in data.items() if k != "common_args"}


def amd_input_configs(root: str) -> dict[str, dict]:
    """Read metadata blocks of the amd_mi300x/mi350x input-config JSONs."""
    import json

    out = {}
    base = os.path.join(
        root, "fbcode/pytorch/tritonbench/tritonbench/data/input_configs/fb"
    )
    for hw in ("amd_mi300x", "amd_mi350x"):
        d = os.path.join(base, hw)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(d, fname)) as f:
                data = json.load(f)
            out[f"{hw}/{fname}"] = data.get("metadata", {})
    return out


def tritonbench_ops(root: str) -> set[str]:
    """Operator names from the operators/ and operators/fb/ directories."""
    base = os.path.join(root, "fbcode/pytorch/tritonbench/tritonbench/operators")
    names = set()
    for sub in (".", "fb"):
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            if entry.endswith(".py") or entry.startswith(".") or entry == "__pycache__":
                continue
            if os.path.isdir(os.path.join(d, entry)):
                names.add(entry)
    return names


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def check(root: str, targets_path: str = TARGETS_FILE) -> list[str]:
    """Return a list of problems (empty == targets.yaml matches sources)."""
    spec = load_yaml_subset(targets_path)
    if spec.get("version") != 1:
        return [f"{targets_path}: expected version 1"]
    arch_spec = spec.get("arch", {})
    problems: list[str] = []
    problems.extend(_check_ops(root, arch_spec))
    problems.extend(_check_tutorials(root, arch_spec))
    problems.extend(_check_tritonbench(root, arch_spec))
    return problems


def _check_ops(root: str, arch_spec: dict) -> list[str]:
    """Every catalog (op, arch) for our arches must be listed."""
    problems: list[str] = []
    catalog = catalog_entries(root)
    wanted = {(op, arch) for (op, arch) in catalog if arch in ARCHES}
    listed: set[tuple[str, str]] = set()
    for arch in ARCHES:
        for entry in arch_spec.get(arch, {}).get("tlx_ops", []) or []:
            listed.add((entry["op"], arch))
            if (entry["op"], arch) not in catalog:
                problems.append(f"tlx_ops {entry['op']}/{arch}: no catalog entry")
    for op, arch in sorted(wanted - listed):
        problems.append(
            f"tlx_ops {op}/{arch}: in catalog but missing from targets.yaml"
        )
    return problems


def _check_tutorials(root: str, arch_spec: dict) -> list[str]:
    """Every in-scope module must be listed for its arches."""
    problems: list[str] = []
    mods = tutorial_modules(root)
    for name, info in mods.items():
        if name in TUTORIAL_EXCLUSIONS:
            continue
        for arch in sorted(info["arches"]):
            entries = arch_spec.get(arch, {}).get("tlx_tutorials", []) or []
            hit = [
                e
                for e in entries
                if e.get("module") == name or e.get("module", "").startswith(name + "/")
            ]
            if not hit:
                problems.append(
                    f"tlx_tutorials {name}: {arch} missing from targets.yaml ({info['kind']})"
                )
    for name in TUTORIAL_EXCLUSIONS:
        if name not in mods:
            problems.append(f"exclusion {name!r} matches no tutorial module anymore")
    for arch in ARCHES:
        for entry in arch_spec.get(arch, {}).get("tlx_tutorials", []) or []:
            problems.extend(_check_tutorial_entry(entry, arch, mods))
    return problems


def _check_tutorial_entry(entry: dict, arch: str, mods: dict) -> list[str]:
    mod = entry.get("module", "")
    top = mod.split("/")[0]
    if top not in mods:
        return [f"tlx_tutorials {mod}: unknown module"]
    if top in TUTORIAL_EXCLUSIONS:
        return [
            f"tlx_tutorials {mod}: listed but excluded ({TUTORIAL_EXCLUSIONS[top]})"
        ]
    problems = []
    if arch not in mods[top]["arches"] and mods[top]["arches"]:
        problems.append(
            f"tlx_tutorials {mod}: listed for {arch} but sources say {sorted(mods[top]['arches'])}"
        )
    if "launcher" not in entry and "script" not in entry:
        problems.append(f"tlx_tutorials {mod}: needs launcher: or script:")
    return problems


def _check_tritonbench(root: str, arch_spec: dict) -> list[str]:
    """mi350 entries covered/excluded; listed ops exist."""
    problems: list[str] = []
    mi350 = mi350_entries(root)
    listed950 = arch_spec.get("gfx950", {}).get("tritonbench", []) or []
    for name in mi350:
        if name in TRITONBENCH_GFX950_EXCLUSIONS:
            continue
        # Entries are matched by the `mi350_ci.yaml <entry>` tag in notes.
        if not any(name in (e.get("notes") or "") for e in listed950):
            problems.append(
                f"tritonbench gfx950: mi350_ci.yaml entry {name!r} not covered"
            )
    for name in TRITONBENCH_GFX950_EXCLUSIONS:
        if name not in mi350:
            problems.append(
                f"exclusion {name!r} matches no mi350_ci.yaml entry anymore"
            )
    known_ops = tritonbench_ops(root)
    for arch in ARCHES:
        for entry in arch_spec.get(arch, {}).get("tritonbench", []) or []:
            if entry.get("op") not in known_ops:
                problems.append(f"tritonbench {arch}: unknown op {entry.get('op')!r}")
    return problems


def skeleton(root: str) -> str:
    """Emit a provisional targets skeleton from live sources."""
    catalog = catalog_entries(root)
    mods = tutorial_modules(root)
    mi350 = mi350_entries(root)
    lines = ["version: 1", "triton_channel: beta", "arch:"]
    for arch in ARCHES:
        lines.append(f"  {arch}:")
        lines.append("    gpu: MI350X")
        lines.append("    tlx_ops:")
        for op, a in sorted(catalog):
            if a == arch:
                lines.append(f"      - {{op: {op}, status: provisional}}")
        lines.append("    tlx_tutorials:")
        for name in sorted(mods):
            if name in TUTORIAL_EXCLUSIONS:
                continue
            if arch in mods[name]["arches"]:
                lines.append(f"      - {{module: {name}, status: provisional}}")
        lines.append("    tritonbench:")
        for entry in sorted(mi350):
            if entry in TRITONBENCH_GFX950_EXCLUSIONS:
                continue
            lines.append(f"      - {{mi350_entry: {entry}, status: provisional}}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate LLVM-export targets against live sources."
    )
    parser.add_argument(
        "--root", default=None, help="fbsource root (default: auto-detect)"
    )
    parser.add_argument("--targets", default=TARGETS_FILE, help="targets.yaml path")
    parser.add_argument(
        "--check", action="store_true", help="validate targets.yaml (default)"
    )
    parser.add_argument("--list", action="store_true", help="print enumerated sources")
    parser.add_argument(
        "--regenerate-skeleton", action="store_true", help="print discovered skeleton"
    )
    args = parser.parse_args(argv)
    root = args.root or find_root()
    if args.regenerate_skeleton:
        print(skeleton(root), end="")
        return 0
    if args.list:
        catalog = catalog_entries(root)
        print("tlx ops (catalog):")
        for op, arch in sorted(catalog):
            if arch in ARCHES:
                print(f"  {op}/{arch} ({catalog[(op, arch)].get('variant')})")
        print("tutorials:")
        for name, info in sorted(tutorial_modules(root).items()):
            flag = (
                "excluded"
                if name in TUTORIAL_EXCLUSIONS
                else (",".join(sorted(info["arches"])) or "no-arch")
            )
            print(f"  {name}: {flag}")
        print("mi350_ci.yaml:")
        for entry in sorted(mi350_entries(root)):
            flag = "excluded" if entry in TRITONBENCH_GFX950_EXCLUSIONS else "in-scope"
            print(f"  {entry}: {flag}")
        print("amd input configs:")
        for path, meta in sorted(amd_input_configs(root).items()):
            print(f"  {path}: ops={meta.get('tritonbench_ops')}")
        return 0
    problems = check(root, args.targets)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("targets.yaml matches all live sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
