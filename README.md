# RFD, real-time face detection

A GStreamer-based pipeline that ingests a live video source, runs facial
recognition against a curated gallery, and emits structured metadata.

Detection is YuNet and recognition is SFace. Both executed through OpenCV's DNN
backend.

## Quick start

```sh
nix develop                      # enter the pinned toolchain
cmake --preset dev               # configure  -> build/dev/
cmake --build --preset dev       # build
ctest --preset dev               # every check CI runs

./build/dev/rfd-enroll external/data/gallery -o build/dev/gallery.json
./build/dev/rfd --source external/data/gallery/reagan/reagan.mp4 \
                 --gallery build/dev/gallery.json
```

That last command is the headline: a clip streamed at its own frame rate with
faces recognised, emitting JSON per frame.

### Finding your way around

Nothing here needs a task runner to be discoverable:

```sh
cmake --list-presets            # dev, release, asan, tsan
cmake --build --list-presets
ctest --list-presets            # dev, fast, release, asan
ctest --preset dev -N           # list the 11 tests without running them

./build/dev/rfd --help
./build/dev/rfd-enroll --help
python3 tools/validate_data.py --help
python3 tools/check_pipeline.py --help
```

### Without Nix

Everything here is ordinary CMake + pkg-config; nothing requires Nix. On
Debian/Ubuntu:

```sh
sudo apt install build-essential cmake pkg-config \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     gstreamer1.0-plugins-{base,good,bad,libav} gstreamer1.0-vaapi \
     libopencv-dev nlohmann-json3-dev
cmake --preset dev && cmake --build --preset dev && ./build/dev/rfd-smoke
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
ctest --preset asan
```

## Tests

Everything CI asserts is runnable locally. There is nothing in the workflows that
is not in CTest - both workflows just call `ctest`.

```sh
ctest --preset dev        # all 11 checks (~70 s)
ctest --preset fast       # skips the real-time streaming tests (~20 s)
ctest --preset release    # the same suite optimised; quote throughput from here
ctest --preset asan       # proves "no leaks under sustained load"
ctest --preset dev -R open-set     # one check
```

| Test | What it would catch |
|---|---|
| `smoke` | Toolchain, GStreamer linkage, plugin registry, weights load |
| `model-digests` | A corrupted or substituted weight file |
| `enroll` | A gallery whose identities cannot be separated at any threshold |
| `validate-data` | A greyscale or multi-face still; a clip whose subject is not recognised |
| `pipeline-plugin` | `rfdface` not actually being a GStreamer plugin |
| `pipeline-gst-launch` | Inference not working without our application driving it |
| `pipeline-caps-change` | Resolution or framerate changes mid-stream breaking detection |
| `pipeline-throughput` | A blocking queue dropping frames; inference too slow to sustain the source rate (not asserted under a sanitiser) |
| `pipeline-open-set` | A face absent from the gallery being given a name |
| `pipeline-degradation` | Stalling instead of dropping when overloaded |
| `pipeline-realtime` | Loss of real-time pacing, or frames going unanalysed that inference had time for |

The pipeline checks can also be run on their own, which prints one line per
property rather than CTest's pass/fail summary:

```sh
python3 tools/check_pipeline.py --build-dir build/dev --gallery build/dev/gallery.json
python3 tools/check_pipeline.py --only open-set --only realtime
```

## Models

| File (`external/models/`) | Model | Role | Licence |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet | Detection + 5 landmarks | MIT |
| `face_recognition_sface_2021dec.onnx` | SFace | 128-D embeddings | Apache-2.0 |

```sh
cmake --preset dev -DRFD_MODEL_DIR=/path/to/weights
```

Verify the vendored copies against the digests upstream published:

```sh
(cd external/models && sha256sum -c SHA256SUMS)
```

## Gallery and recognition

The curated faces live in `external/data/gallery/`, one directory per identity
holding colour stills plus a probe clip. See
[external/data/README.md](external/data/README.md).

Build the gallery, then check it:

```sh
./build/dev/rfd-enroll external/data/gallery -o build/dev/gallery.json
python3 tools/validate_data.py --gallery external/data/gallery \
                               --json build/dev/gallery.json
```

`rfd-enroll` detects one face per still, aligns it on YuNet's five landmarks,
embeds it with SFace and averages each identity's embeddings into a unit-length
template. It refuses stills where the subject is ambiguous (no face, or several)
and drops any still that disagrees with its own identity's mean - archive search
really does return a photograph of the wrong man in the right folder.

It prints the between-identity similarity matrix and each identity's worst
internal agreement, because the number that matters is the gap between them:

```
separation margin 0.356  (worst within 0.601 - best between 0.246)
```

## Running the pipeline

```sh
./build/dev/rfd --source external/data/gallery/reagan/reagan.mp4 --gallery gallery.json
./build/dev/rfd --source rtsp://camera/stream --gallery gallery.json
./build/dev/rfd --source /dev/video0 --gallery gallery.json
```

### Architecture

Inference runs **inside a GStreamer element**, not in an application loop:

```
<source> ! videoconvert ! video/x-raw,format=BGR
         ! identity sync=true          # real-time pacing, file sources only
         ! queue leaky=downstream      # drop policy
         ! rfdface gallery=...         # YuNet + SFace, attaches GstRfdFaceMeta
         ! appsink
```

Build a gst plugin:

```sh
GST_PLUGIN_PATH=build/dev/source gst-inspect-1.0 rfdface
GST_PLUGIN_PATH=build/dev/source gst-launch-1.0  filesrc location=external/data/gallery/reagan/reagan.mp4 ! decodebin   ! videoconvert ! video/x-raw,format=BGR ! rfdface gallery=gallery.json ! fakesink
```

### Measured

Real-time, all five clips (each 10.0 s of 29.97 fps video, 512x384):

| clip | wall | analysed | dropped | fps | identified |
|---|---|---|---|---|---|
| bush | 9.98 s | 296 | 4 | 29.66 | 296 |
| carter | 9.98 s | 296 | 4 | 29.65 | 296 |
| ford | 10.00 s | 298 | 3 | 29.79 | 298 |
| nixon | 10.00 s | 296 | 5 | 29.60 | 296 |
| reagan | 10.01 s | 298 | 3 | 29.77 | 298 |

## Environment

The toolchain is pinned by `flake.lock`. `nix flake metadata` prints the exact
nixpkgs revision

### Verified on this machine

| Component            | Version                             |
|----------------------|-------------------------------------|
| GStreamer            | 1.28.6                              |
| GCC                  | 15.3.0 (`gcc15Stdenv`, pinned)      |
| CMake                | 4.4.2                               |
| GNU Make             | 4.4.1                               |
| OpenCV               | 4.13.0 (supplies YuNet/SFace + DNN) |
| nlohmann_json        | 3.12.0                              |
| libva                | 2.24.1                              |
| intel-media-driver   | 26.2.4 (iHD)                        |

## Continuous integration

| Workflow | What it proves |
|---|---|
| `.github/workflows/ci.yml` | Builds from the pinned flake across the `dev`, `release` and `asan` presets, plus `nix flake check`. Verifies the model digests and runs `tools/validate_data.py` over the committed faces. |
| `.github/workflows/ubuntu.yml` | Builds from stock apt packages in bare `ubuntu:22.04` / `24.04` / `25.04` containers, with no Nix present. |

`asan` runs by default rather than on demand: the task is judged on not leaking
under sustained load, so a leak should fail the build.

