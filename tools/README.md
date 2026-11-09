# Tools

Everything needed to regenerate, verify, and replay the `../gfx950/`
corpus, ported from `the original exporter`
under base_module `llvm_benchmarks_tools`. Design: see the exporter `OVERVIEW.md`.

## Regenerate the corpus (IR generation)

On an MI350X (gfx950) host, from this directory:

```bash
./run_export.sh --arch gfx950 --out ..   # writes ../gfx950/{manifest,metadata}.json + case dirs
```

Stages: 1 validate `targets.yaml`/`candidates.yaml` against live sources
(`discover.py --check`, `candidates.py --check`; fails on drift so new
kernels get curated, not skipped) → 2 build standalone AMD codegen
(`TRITON_AMD_CODEGEN_PATH`) → 3-5 export tlx_ops / tlx_tutorials /
tritonbench → 6 prune to top-3 configs + longest/median shapes
(`prune_top.py`) → 7 collect `metadata.json` (`collect_metadata.py`).
Cron: `run_scheduled.py` over `hosts.yaml`. Provisional targets (e.g.
`mm_mxfp8`) run only with `--include-provisional` and never fail the run.

## Verify the corpus

```bash
buck2 run @//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:verify_llir -- --arch gfx950 --out ..
```

Re-feeds each `kernel.llir` through the Beta AMD backend and compares
with the saved `kernel.amdgcn` (exact / comment-normalized / option grid).
`--skip-round-trip` = structural checks only.

## Replay on GPU (launcher)

```bash
buck2 run @//mode/opt-amd-gpu fbcode//triton/benchmarks/tools:launch_hip -- --arch gfx950 --out .. [--match mm] [--limit 5]
```

Builds HSACO from `kernel.llir`, seeded arg allocation, raw HIP launch,
median ms + output finiteness. Cases without `compile.json:launch` report
`missing-launch`; multi-CTA/cluster report `unsupported`.
Smoke test on real MI350 via RE: `:launch_re_test` (`re_probe/` fixtures).

## Tests (plain python3, no GPU)

```bash
python3 launch_hip_test.py        # 31
python3 collect_metadata_test.py  # 15
python3 candidates_test.py        # 17
python3 prune_top_test.py         # 17
python3 export_tritonbench_test.py  # 6
python3 run_scheduled_test.py
python3 discover.py --check && python3 candidates.py --check
```
