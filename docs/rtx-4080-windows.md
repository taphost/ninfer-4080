# RTX 4080 native Windows build

**Status: ACTIVE.** This is the Windows (MSVC + CUDA) build path for the `sm_89` (RTX 4080)
`qwen3.8-27b` GSQ3 artifact. The Linux build path is documented in
[docs/rtx-3090-linux.md](rtx-3090-linux.md).

The engine builds and runs natively on Windows 11. The Windows path is built and validated on
the RTX 4080 (`CMAKE_CUDA_ARCHITECTURES=89`), so it is no longer inherited but untested.

## Toolchain

| Component | Requirement |
|---|---|
| OS | Windows 11 |
| Compiler | MSVC (Visual Studio 2026) via `VsDevCmd.bat` |
| CUDA | 13.3 (`nvcc.exe`, `release 13.3`) |
| Build system | CMake 3.28+ with the Ninja generator |
| Package manager | vcpkg, `x64-windows` triplet |
| Target | `CMAKE_CUDA_ARCHITECTURES=89` (RTX 4080) |

vcpkg installs the project-declared dependencies on first run. The build script verifies the
prerequisites up front and fails fast if any are missing or the CUDA version is wrong.

## Prerequisites

- Visual Studio 2026 with **Desktop development with MSVC** (the C++ workload). This provides
  `cl.exe`, `vswhere.exe`, and `VsDevCmd.bat`.
- CUDA 13.3 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`.
- vcpkg cloned anywhere you like (a `x64-windows` triplet).
- A Developer PowerShell so `cl.exe`, `cmake.exe`, and `ninja.exe` are on `PATH`.

## Build

Configure and build with CMake from a Developer PowerShell, so `cl.exe`, `cmake.exe`, and
`ninja.exe` are on `PATH`. `<vcpkg-root>` is your vcpkg checkout:

```pwsh
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-win --parallel
```

Outputs:

- CLI: `build-win\apps\ninfer.exe`
- Server: `build-win\apps\ninfer-serve.exe`

## Portability fixes

MSVC on x64 has no `unsigned __int128` (hard error C4235), and it lacks the POSIX helpers the
Linux path uses. The Windows build is made correct by the fixes committed in
`dbc83d42` (`fix(4080): windows build portability fixes`, ported from
`roofkid/ninfer-4080` `cb2df08` + `b8c71f0`, credited to lazarusdmx, laurinius, and sergiuszm):

- `src/runtime/portable_u128.h` — new shim. A two-limb `{hi, lo}` emulation using 32-bit limb
  math, selected via `#if defined(__SIZEOF_INT128__)`. GCC/Clang keep the native type; MSVC uses
  the emulation. All operations are `constexpr`. The saturating 128-bit cost math
  (`context_cost.cpp`, `contract/types.h`, `materialization_planner.h`) routes through it.
- `CMakeLists.txt` — `/utf-8` for MSVC. The bundled `fmt`/`spdlog` static-asserts that narrow
  literals are UTF-8 (error C2338 otherwise).
- `src/product/logging/logging.cpp` — `_localtime64_s` for `localtime_r`, `::_isatty`/`::_fileno`
  for `::isatty`.
- `src/product/logging/startup_log.cpp` — `GetConsoleScreenBufferInfo` for the `ioctl`/`TIOCGWINSZ`
  terminal-width probe.
- `src/targets/qwen3.6/impl/runtime/program.h` — explicit (not `= default`) move-assignment bodies
  for the two variant candidate templates. A defaulted member is only emitted in the TU that
  odr-uses it; MSVC left the symbol undefined (LNK2019).
- `apps/perplexity/main.cpp` — `gmtime_s` for `gmtime_r`.

Every change is platform-guarded, so the Linux (GCC/Clang) path stays bit-identical.

## CUDA graph decode fallback

The Windows 4080 build initially crashed at startup:

```
FATAL server failed during startup | CUDA Graph executable update failed:
cudaErrorGraphExecUpdateFailure (update result 5)
```

**Cause.** `update result 5` is `cudaGraphExecUpdateErrorParametersChanged`. The ordinary/MTP
decode family assigns one `topology_class` to every frontier tier, so `cudaGraphExecUpdate` is
called when moving between tiers. Context-size-dependent kernels (envelope `max_visible_keys`,
grid `heads × splits × batch`, `__grid_constant__` TMA descriptors) change parameters at those
boundaries, so the driver rejects the in-place update. DFlash is unaffected because it gives each
tier its own `topology_class`, so it never calls `cudaGraphExecUpdate`.

**Fix.** On a failed update, re-instantiate the executable from the current tier definition
instead of failing. Error 5 means only parameters differ (the topology is identical), and
`instantiate()` calls `reset()` first, so there is no leak or double-free. A failed
`cudaGraphExecUpdate` also leaves a pending `cudaErrorGraphExecUpdateFailure` in the runtime, so
the fallback clears it with `cudaGetLastError()` before re-instantiating; otherwise a later kernel
launch (e.g. the gated-delta-net fold kernel) aborts on the stale status. The fallback is gated by
a runtime flag (`set_graph_update_fallback`, default on); disabling it restores the original
throwing behavior.

**Behavior.** The fallback fires only at tier boundaries where `cudaGraphExecUpdate` fails, so
`CUDA Graph update failed, re-instantiating executable` lines may appear in the startup log (as
they did for the ordinary/MTP decode family) or be absent when every update succeeds in place. It
keeps one resident executable per shared topology (destroy + re-instantiation), which keeps VRAM
low on the 16 GB card; the cost is a re-instantiation at each boundary crossed.

## Tests

The portable `u128` host test (`tests/test_portable_u128.cpp`) compares the emulation against
native `unsigned __int128` and builds under Linux only (see `tests/CMakeLists.txt`). Run it with
`BUILD_TESTING=ON`.
