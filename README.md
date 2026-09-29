# rtip, real-time inference pipeline

A GStreamer-based pipeline that ingests a live video source, runs facial
recognition against a curated gallery, and emits structured metadata.

## Quick start

```sh
nix develop                      # enter the pinned toolchain
cmake --preset dev               # configure  -> build/dev/
cmake --build --preset dev       # build
./build/dev/rtip-hello           # run
```

Expected output:

```
rtip - real-time inference pipeline
environment check
  compiled against : 1.28.6
  running against  : 1.28.6
  plugins found    : 271
smoke pipeline (30 frames)
  ok - hello, world
```

### Without Nix

Everything here is ordinary CMake + pkg-config; nothing requires Nix. On
Debian/Ubuntu:

```sh
sudo apt install build-essential cmake pkg-config \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     gstreamer1.0-plugins-{base,good,bad,libav} gstreamer1.0-vaapi
cmake --preset dev && cmake --build --preset dev && ./build/dev/rtip-hello
```

## Build presets

| Preset    | Build type     | Purpose                                              |
|-----------|----------------|------------------------------------------------------|
| `dev`     | Debug          | Day-to-day development.                               |
| `release` | RelWithDebInfo | Benchmarking, optimised but still profilable.        |
| `asan`    | Debug + ASan   | Proving the "no leaks under sustained load" criterion.|
| `tsan`    | Debug + TSan   | Data races, once the pipeline goes multi-threaded.    |

```sh
cmake --preset asan && cmake --build --preset asan
ASAN_OPTIONS=detect_leaks=1 ./build/asan/rtip-hello
```

## VS Code

Open the folder (or `rtip.code-workspace`). Extensions

- **CMake Tools**
- **clangd**
- **direnv** If you use nix.

---

## Environment

The toolchain is pinned by `flake.lock`. `nix flake metadata` prints the exact
nixpkgs revision; every version below follows from it.

### Verified on this machine

| Component            | Version                             |
|----------------------|-------------------------------------|
| GStreamer            | 1.28.6                              |
| GCC                  | 15.3.0 (`gcc15Stdenv`, pinned)      |
| CMake                | 4.4.2                               |
| GNU Make             | 4.4.1                               |
| OpenVINO             | 2026.4.0 (CPU / GPU / NPU plugins)  |
| ONNX Runtime         | 1.27.1                              |
| OpenCV               | 4.13.0                              |
| libva                | 2.24.1                              |
| intel-media-driver   | 26.2.4 (iHD)                        |

## Continuous integration

| Workflow | What it proves |
|---|---|
| `.github/workflows/ci.yml` | Builds from the pinned flake across the `dev`, `release` and `asan` presets, plus `nix flake check`. |
| `.github/workflows/ubuntu.yml` | Builds from stock apt packages in bare `ubuntu:22.04` / `24.04` / `25.04` containers, with no Nix present. |

`asan` runs by default rather than on demand: the task is judged on not leaking
under sustained load, so a leak should fail the build.

