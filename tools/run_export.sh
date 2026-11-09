#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Nightly-style entrypoint for the Triton-Beta LLVM-stage export.
#
# Runs all three sources for one architecture on the target GPU host and
# leaves a hardware-sorted output tree plus a per-arch manifest:
#
#   ./run_export.sh --arch gfx950 --out /data/llvm_export_20250101
#
# Runs on an MI350X host (gfx950). Everything executes through Triton Beta
# (ovr_config//triton:beta, @//mode/opt-amd-gpu).
#
# Exit status: 0 when every ready target exported; 1 when discover --check
# fails (sources drifted from targets.yaml) or any ready target failed.
# Provisional targets are attempted only with --include-provisional and never
# fail the run; their outcome is recorded in the manifest.

set -euo pipefail

# Triton's C-cache fast path compiles without invoking the Python
# JITFunction.run hooks the capture layer (and tritonbench --dump-ir)
# relies on, silently dropping kernels for eligible specializations.
export TRITON_ENABLE_C_CACHE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH=""
OUT=""
INCLUDE_PROVISIONAL=()
BUCK_MODE="@//mode/opt-amd-gpu"

usage() {
    echo "Usage: $0 --arch gfx950 --out DIR [--include-provisional] [--buck-mode MODE]"
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --include-provisional) INCLUDE_PROVISIONAL=(--include-provisional); shift ;;
        --buck-mode) BUCK_MODE="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ "$ARCH" == "gfx950" ]] || usage
[[ -n "$OUT" ]] || usage

EXPECT_GPU="MI350"
EXPECT_GFX="gfx950"
if command -v rocm-smi >/dev/null 2>&1; then
    GPU_NAME="$(rocm-smi --showproductname 2>/dev/null || true)"
    if ! grep -qi "$EXPECT_GPU" <<<"$GPU_NAME" && ! grep -qi "$EXPECT_GFX" <<<"$GPU_NAME"; then
        echo "ERROR: --arch $ARCH needs $EXPECT_GPU; rocm-smi reports:" >&2
        echo "$GPU_NAME" >&2
        exit 1
    fi
else
    echo "WARNING: rocm-smi not found; skipping host GPU check" >&2
fi

echo "==> [1/7] validating targets.yaml against live sources"
python3 "$SCRIPT_DIR/discover.py" --check
python3 "$SCRIPT_DIR/candidates.py" --check

BASE="fbcode//triton/benchmarks/tools"
echo "==> [2/7] building standalone AMD codegen library (Triton Beta)"
if [[ -z "${TRITON_AMD_CODEGEN_PATH:-}" ]]; then
    CODEGEN_SO="$(buck2 --client-metadata id=llvm_benchmarks_tools build "$BUCK_MODE" "fbsource//third-party/triton/beta/triton:triton_amd_codegen" --show-full-output 2>/dev/null | awk '{print $NF}')"
    [[ -f "$CODEGEN_SO" ]] || { echo "ERROR: codegen library not built (got: $CODEGEN_SO)" >&2; exit 1; }
    export TRITON_AMD_CODEGEN_PATH="$CODEGEN_SO"
fi
echo "TRITON_AMD_CODEGEN_PATH=$TRITON_AMD_CODEGEN_PATH"

echo "==> [3/7] exporting tlx_ops ($ARCH)"
buck2 --client-metadata id=llvm_benchmarks_tools run "$BUCK_MODE" "$BASE:export_tlx_ops" -- --arch "$ARCH" --out "$OUT" "${INCLUDE_PROVISIONAL[@]}"

echo "==> [4/7] exporting tlx_tutorials ($ARCH)"
buck2 --client-metadata id=llvm_benchmarks_tools run "$BUCK_MODE" "$BASE:export_tlx_tutorials" -- --arch "$ARCH" --out "$OUT" "${INCLUDE_PROVISIONAL[@]}"

echo "==> [5/7] exporting tritonbench ($ARCH)"
buck2 --client-metadata id=llvm_benchmarks_tools run "$BASE:export_tritonbench" -- --arch "$ARCH" --out "$OUT" --buck-mode "$BUCK_MODE" "${INCLUDE_PROVISIONAL[@]}"

echo "==> [6/7] pruning to top configs/shapes by latency ($ARCH)"
python3 "$SCRIPT_DIR/prune_top.py" --arch "$ARCH" --out "$OUT"

echo "==> [7/7] collecting benchmark metadata ($ARCH)"
python3 "$SCRIPT_DIR/collect_metadata.py" --arch "$ARCH" --out "$OUT"

echo "==> manifest: $OUT/$ARCH/manifest.json"
python3 - "$OUT/$ARCH/manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
arts, fails = m.get("artifacts", []), m.get("failures", [])
hard = [f for f in fails if f.get("status") != "provisional"]
print(f"artifacts={len(arts)} failures={len(fails)} (ready={len(hard)})")
for f in fails:
    print(f"  FAIL [{f.get('source')}/{f.get('status')}] {f.get('target')} {f.get('case')}: {str(f.get('error'))[:200]}")
sys.exit(1 if hard else 0)
EOF
