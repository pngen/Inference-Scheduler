# Contributing to Inference Scheduler

Contributions to Inference Scheduler are welcome from individuals and
organizations on the terms of the Apache License 2.0. No Contributor License
Agreement (CLA) is required.

## Development toolchain

* C++20 compiler (Visual Studio 2022 / MSVC on Windows)
* CMake 3.20+
* Ninja (recommended)
* CUDA 12.8+ with a compatible NVIDIA driver (required only for the CUDA
  backend and hardware tests)

## Build

The project configures and builds with CMake. Use the provided helper for a
fully configured toolchain on Windows:

```
scripts/build.ps1 -Configuration Release
```

Or configure directly (on Windows, run from a Visual Studio developer prompt):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Tests

All tests are registered with CTest:

```
ctest --test-dir build --output-on-failure
```

There are no test timeouts; tests run to natural completion.

## Style

* /W4 /WX is required to be clean on MSVC (Release and Debug).
* Prefer the library's Result<T> error model over exceptions for normal
  control flow.
* No global mutable state; inject clocks and policies.
* Keep the public headers under include/inference_scheduler/.

## Before submitting

1. Configure and build Release and Debug with zero warnings.
2. Run the full CTest suite.
3. Run the multiprocess validation proof.
4. Run the CUDA hardware test if the target machine has a CUDA GPU.

## Licensing

By contributing, you agree that your contributions are licensed under the
Apache License 2.0. Avoid adding proprietary content or unrelated attribution.
