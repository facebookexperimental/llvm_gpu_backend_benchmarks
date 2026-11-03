# Contributing to llvm_gpu_backend_benchmarks
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
This repository contains benchmarks for LLVM GPU backends. The source of
truth is `fbcode/triton/benchmarks/` in fbsource; ShipIt exports every
landed diff to GitHub `main` automatically. GitHub pull requests are
imported to fbsource diffs, reviewed there, and exported on land — do
not merge directly to `main` on GitHub.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* 2 spaces for indentation rather than tabs
* 80 character line length
* Follow the style of the surrounding code and add a copyright header to
  new source files: `Copyright (c) Meta Platforms, Inc. and affiliates.`

## License
By contributing to llvm_gpu_backend_benchmarks, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
