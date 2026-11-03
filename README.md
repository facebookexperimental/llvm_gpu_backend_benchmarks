# llvm_gpu_backend_benchmarks

Benchmarks for LLVM GPU backends. This repository collects reproducible
benchmark workloads and harnesses for evaluating code generation, register
allocation, scheduling, and runtime performance across LLVM GPU targets
(e.g. AMDGPU and NVPTX).

## Examples

Benchmark cases are organized by workload. See each subdirectory for the
kernel sources, input shapes, and expected outputs for that case.

## Requirements

llvm_gpu_backend_benchmarks requires or works with:

* Linux
* LLVM toolchain with GPU backend support (AMDGPU / NVPTX)
* Python 3.10+ for the benchmark harness

## Building llvm_gpu_backend_benchmarks

Build instructions depend on the backend under test. Refer to the
documentation in each benchmark subdirectory for the LLVM build flags
and harness invocation for that workload.

## Installing llvm_gpu_backend_benchmarks

Clone the repository and follow the per-benchmark build instructions:

```bash
git clone https://github.com/facebookexperimental/llvm_gpu_backend_benchmarks.git
cd llvm_gpu_backend_benchmarks
```

## How llvm_gpu_backend_benchmarks works

Each benchmark provides a self-contained kernel or IR workload, a driver
that compiles it with the LLVM GPU backend under test, and a checker that
validates correctness and records performance counters. This makes it
possible to compare backend revisions on identical inputs.

## Full documentation

See the benchmark subdirectories and [CONTRIBUTING](CONTRIBUTING.md) for
how to add a new benchmark case.

## Join the llvm_gpu_backend_benchmarks community

* GitHub Issues: https://github.com/facebookexperimental/llvm_gpu_backend_benchmarks/issues

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

llvm_gpu_backend_benchmarks is MIT licensed, as found in the LICENSE file.
