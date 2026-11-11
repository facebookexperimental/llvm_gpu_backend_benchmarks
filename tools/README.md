# Benchmark tools

Replay exported kernels, verify their LLVM inputs, and collect corpus metadata.
The architecture directories contain the generated kernel files and manifests.

## Verify LLVM inputs

```bash
buck2 run @fbcode//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:verify_llir -- \
  --arch gfx942 --out ..
```

Recompiles each `kernel.llir` through the Triton Beta AMD backend and compares
it with the saved `kernel.amdgcn`. `--skip-round-trip` runs structural checks.

## Replay kernels on GPU

```bash
buck2 run @fbcode//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:launch_hip -- \
  --arch gfx942 --out .. --limit 5
```

The launcher builds HSACO from LLVM IR, allocates the recorded inputs, launches
through HIP, and reports latency and output finiteness. Replay requires valid
launch metadata in `compile.json`.

## Collect metadata

From this directory:

```bash
python3 collect_metadata.py --arch gfx942 --out ..
```

Writes per-case FLOPs, argument sizes, and replay readiness to `metadata.json`.
The tools support `gfx942` and `gfx950`.
