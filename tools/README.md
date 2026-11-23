# Benchmark tools

Replay exported kernels, verify their LLVM inputs, and collect corpus metadata.
The architecture directories contain the generated kernel files and manifests.

## Verify LLVM inputs

```bash
buck2 run @fbcode//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:verify_llir -- \
  --arch gfx942 --out ..
```

Recompiles each `kernel.llir` through the Triton Beta backend (AMD or
NVIDIA, per `--arch`) and compares it with the saved reference output
(`kernel.amdgcn` / `kernel.ptx`). `--skip-round-trip` runs structural checks.

## Replay kernels on GPU

```bash
buck2 run @fbcode//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:launch_hip -- \
  --arch gfx942 --out .. --limit 5
buck2 run fbcode//triton/benchmarks/tools:launch_cuda -- \
  --arch sm100 --out .. --limit 5
```

The launchers build HSACO (HIP) or CUBIN (CUDA) from LLVM IR, allocate the
recorded inputs, launch through the runtime, and report latency and output
finiteness. Replay requires valid launch metadata in `compile.json`.

## Prune location info

```bash
python3 prune_loc.py --arch gfx942 --out ..
```

Strips MLIR `#loc`, LLVM `!dbg`/`!DI*`, and assembler `.file`/`.loc` debug
info from every `kernel.*` artifact and refreshes the recorded file hashes.
Artifacts become host-independent, so regenerations only diff on real IR
changes. `--dry-run` reports the saving without writing.

## Collect metadata

From this directory:

```bash
python3 collect_metadata.py --arch gfx942 --out ..
```

Writes per-case FLOPs, argument sizes, and replay readiness to `metadata.json`.
The tools support `gfx942`, `gfx950`, `gfx1250`, `sm90`, and `sm100`.
Use the CUDA launcher on an H100 for `sm90` kernels.
