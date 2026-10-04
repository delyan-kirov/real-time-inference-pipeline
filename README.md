# RFD, real-time face detection

A GStreamer-based pipeline that ingests a live video source, runs facial
recognition against a curated gallery, and emits structured metadata.

Detection is YuNet and recognition is SFace, both executed through OpenCV's DNN
backend. The weights are vendored in `external/models/`. See
[external/README.md](external/README.md)

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

Expected output:

```
rfd - real-time face detection pipeline
environment check
  compiled against : 1.28.6
  running against  : 1.28.6
  plugins found    : 271
  opencv           : 4.13.0
  yunet            : loaded
  sface            : loaded
smoke pipeline (30 frames)
  ok - smoke test passed
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

`libopencv-dev` supplies `OpenCVConfig.cmake`; the `dnn` and `objdetect` modules
it needs are in the default Debian/Ubuntu build. OpenCV 4.5.4 (the oldest on the
supported images) is the floor for both `FaceDetectorYN` and `FaceRecognizerSF`.

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
| `pipeline-throughput` | Inference too slow to sustain the source rate |
| `pipeline-open-set` | A face absent from the gallery being given a name |
| `pipeline-degradation` | Stalling instead of dropping when overloaded |
| `pipeline-realtime` | Loss of real-time pacing, or frames going unanalysed |

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

### Threshold

`tools/validate_data.py` matches every probe clip twice - once against the full
gallery, once against a gallery with that clip's own subject removed. The second
pass is the open-set test, and doing it leave-one-out measures rejection for every
identity without holding any of them out of the deliverable.

Over the five clips:

```
worst genuine   0.420
best impostor   0.355
usable window   0.355 .. 0.420  (width 0.064)
midpoint        0.388  <- maximises margin both ways
```

So the cosine threshold is **0.388**, not OpenCV's documented 0.363. Upstream's
value does work on this data but sits only 0.008 above the best impostor score -
one blurrier frame from a false accept. The midpoint balances both failure modes.
That number is a property of this gallery and footage, not of SFace; the validator
reprints the window whenever either changes.

The window is only 0.064 wide, which is itself worth knowing: five older white men
is a high inter-class-similarity gallery, so there is less room here than a varied
one would give.

Current result: a face is found in 150/150 sampled frames, rank-1 is 150/150, and
no held-out subject is ever accepted.

### Known limitations

- **No presentation-attack detection.** A printed photograph held up to the camera
  will match. Out of scope here; required before anything like this is deployed.
- **The gallery is demographically narrow** - five older white men. Face
  recognition has well-documented differential error rates across demographics, so
  no accuracy figure here generalises. It also raises inter-class similarity,
  which makes matching harder rather than easier, so the numbers are not flattered
  by it either.
- **Identities are deceased public figures.** GDPR Recital 27 puts the personal
  data of deceased persons outside the Regulation, so no Article 9 basis is needed
  to hold their biometric templates - which would not be true of living subjects,
  however public. Copyright is unaffected by death, so the footage licence is a
  separate question answered in `external/data/`.

## Running the pipeline

```sh
./build/dev/rfd --source external/data/gallery/reagan/reagan.mp4 --gallery gallery.json
./build/dev/rfd --source rtsp://camera/stream --gallery gallery.json
./build/dev/rfd --source /dev/video0 --gallery gallery.json
```

One JSON object per frame on stdout, summary on stderr:

```json
{"frame":0,"pts_ns":4224600000,"t_ns":0,"inference_us":12853,
 "faces":[{"box":[199.2,86.6,119.0,168.0],
           "landmarks":[[218.6,147.3],[269.1,154.2],[226.3,182.4],[215.1,207.5],[260.4,212.6]],
           "detection_score":0.941,"identity":"reagan","similarity":0.737}]}
```

`identity` is `null` for an open-set rejection, with the similarity it did reach,
so a near-miss is visible rather than discarded. Both `pts_ns` and a zero-based
`t_ns` are emitted - our clips start at PTS 4.2 s because they were trimmed with
timestamps preserved, so the raw PTS is right for correlating against the source
file and useless as "time since start".

### Architecture

Inference runs **inside a GStreamer element**, not in an application loop:

```
<source> ! videoconvert ! video/x-raw,format=BGR
         ! identity sync=true          # real-time pacing, file sources only
         ! queue leaky=downstream      # drop policy
         ! rfdface gallery=...        # YuNet + SFace, attaches GstRfdFaceMeta
         ! appsink
```

`rfdface` is a `GstVideoFilter` that transforms in place - it annotates buffers
and never touches pixels - and attaches results as a custom `GstMeta`, so they
cannot desynchronise from the frame that produced them. It builds as a real
plugin, so it works with no code of ours driving it:

```sh
GST_PLUGIN_PATH=build/dev/source gst-inspect-1.0 rfdface
GST_PLUGIN_PATH=build/dev/source gst-launch-1.0   filesrc location=external/data/gallery/reagan/reagan.mp4 ! decodebin   ! videoconvert ! video/x-raw,format=BGR ! rfdface gallery=gallery.json ! fakesink
```

Two placements matter and are easy to get wrong:

- **Pacing goes at the front**, via `identity sync=true`, because that is where a
  live source's own rate would apply. A genuinely live source (`v4l2src`,
  `rtsp://`) paces itself and gets no `identity`. Pacing at a clock-synced
  `appsink` instead makes the sink block `rfdface`, which backs up the queue,
  which then drops ~99% of frames - the queue reads "the sink is pacing correctly"
  as "inference is too slow". That bug cost 297 of 301 frames before it was found.
- **`leaky=downstream` must be off to benchmark.** With dropping on, the rate you
  measure is how fast frames are being thrown away. `--no-drop` makes the queue
  block so throughput means what it says.

### Measured

Real-time, all five clips (each 10.0 s of 29.97 fps video, 512x384):

| clip | wall | analysed | dropped | fps | identified |
|---|---|---|---|---|---|
| bush | 9.98 s | 296 | 4 | 29.66 | 296 |
| carter | 9.98 s | 296 | 4 | 29.65 | 296 |
| ford | 10.00 s | 298 | 3 | 29.79 | 298 |
| nixon | 10.00 s | 296 | 5 | 29.60 | 296 |
| reagan | 10.01 s | 298 | 3 | 29.77 | 298 |

Every dropped frame is in a single startup transient - PTS analysis shows exactly
one discontinuity, between the first and second analysed frame. **Steady state is
lossless at 30 fps.**

Throughput and the bottleneck (`--no-sync --no-drop`, 0 frames dropped):

| stage | median/frame | throughput | share |
|---|---|---|---|
| decode + videoconvert | 1.2 ms | ~833 fps | 10% |
| YuNet detection | 4.8 ms | 189 fps | 40% |
| SFace embed + match | 7.2 ms | - | 60% |
| **full pipeline** | **12.0 ms** | **76 fps** | |

**SFace embedding is the bottleneck**, at 60% of inference time. Decode is
negligible. At 76 fps against a 30 fps source there is 2.5x headroom, so the
cheapest wins would be batching embeddings or skipping re-identification on
tracked faces - not faster decoding.

Models are warmed at startup with one throwaway inference; without it the first
real frame takes ~44 ms against a 12 ms median and costs extra dropped frames.

### Behaviour under stress

| Condition | Result |
|---|---|
| Overload (`--queue 1 --no-sync`) | 287/301 dropped, 14 analysed, **all 14 still correct**. Drops instead of stalling; RSS flat. |
| Resolution + framerate change mid-stream (320x240@30 → 640x480@30 → 1280x720@15) | 3 clean renegotiations, 60/60 frames, 0 errors |
| Open-set: Reagan vs a gallery without Reagan | 298 frames, **0 identified, 298 unknown** |
| ASan + LSan, 3 runs at max throughput | exit 0, no leaks, 301 frames each |

## VS Code

Open the folder (or `rfd.code-workspace`). Extensions:

- **CMake Tools** - reads `CMakePresets.json` directly; no duplicated build config
- **clangd** - IntelliSense from `compile_commands.json`
- **direnv** - if you use nix

Three ways to run things, in increasing specificity:

| Where | Good for |
|---|---|
| **CMake Tools status bar** | Pick a launch target, press Run or Debug. Environment comes from `cmake.debugConfig` in `settings.json`. Best for `rfd-smoke`, which takes no arguments. |
| **Run and Debug (F5)** | `launch.json` has per-target configurations with arguments - `rfd` on a clip, on a webcam, with `rfdface` logging every frame, and `rfd-enroll`. Also "Debug selected CMake target", which follows the status-bar selection. |
| **Run Task** | `tasks.json` mirrors the command-line workflow: build, configure, the two test presets, enroll, validate, the demo, the benchmark, `gst-inspect` and a raw `gst-launch` pipeline. |

Arguments are not set in `cmake.debugConfig` on purpose: it applies to every
target, so arguments meant for `rfd` would also be handed to `rfd-smoke`.
Per-target arguments belong in `launch.json`.

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

