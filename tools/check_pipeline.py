#!/usr/bin/env python3
##
# @file check_pipeline.py
# @brief End-to-end checks on the rfd pipeline.
#
# Where validate_data.py checks the committed *data*, this checks the *pipeline*:
# that it streams at source rate, recognises its subjects, rejects faces it has
# not enrolled, degrades by dropping rather than stalling.
#
# @par Usage
# @code
# tools/check_pipeline.py --build-dir build/dev [--only realtime]
# @endcode
##
"""End-to-end checks on the rfd pipeline.

Where validate_data.py checks the committed *data*, this checks the *pipeline*:
that it streams at source rate, recognises its subjects, rejects faces it has not
enrolled, degrades by dropping rather than stalling

Usage:
    tools/check_pipeline.py --build-dir build/dev [--only realtime]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

## Repository root, derived from this file's location.
REPO = Path(__file__).resolve().parent.parent
## Frame rate of the committed probe clips.
SOURCE_FPS = 29.97
## Below this, real-time pacing has been lost.
MIN_REALTIME_FPS = 25.0
## Above this, something is running unpaced rather than at source rate.
MAX_REALTIME_FPS = 33.0
## Fraction of the analysis rate
MIN_ANALYSED_FRACTION = 0.90
## Frames per clip the subject may go unidentified on.
MAX_UNIDENTIFIED_FRAMES = 4
## Instrumentation costs 2-3x
SANITIZED = os.environ.get("RFD_SANITIZED") == "1"


class Failure(Exception):
    """@brief A check's assertion did not hold; the message explains which."""

    pass


def run(argv: list[str], env: dict[str, str] | None = None, timeout: int = 180):
    """@brief Run a command in the repo root with @p env overlaid.

    @param argv    Command and arguments.
    @param env     Variables added to, not replacing, the current environment.
    @param timeout Seconds before subprocess.TimeoutExpired is raised.
    @return The completed process, with stdout and stderr captured as text.
    """
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(argv, capture_output=True, text=True, env=merged,
                            timeout=timeout, cwd=REPO)
    return result


def summary_field(text: str, label: str, position: int = -1) -> float:
    """@brief Pull a number out of rfd's stderr summary.

    @param text     The captured stderr.
    @param label    Line prefix to look for, e.g. "frames dropped".
    @param position Which number on that line to take; -1 is the last.
    @return The parsed value.
    @exception Failure No line starts with @p label.
    """
    for line in text.splitlines():
        if line.startswith(label):
            numbers = re.findall(r"-?\d+\.?\d*", line)
            if numbers:
                return float(numbers[position])
    raise Failure(f"no '{label}' line in output:\n{text}")


def inference_latency(text: str) -> tuple[float, float, float]:
    """@brief Pull the inference latency distribution out of rfd's summary.

    Not summary_field: "p95" puts a 95 in the middle of that line.

    @param text The captured stderr.
    @return Median, p95 and maximum latency, in milliseconds.
    @exception Failure The summary has no inference line, i.e. nothing was
               analysed at all.
    """
    match = re.search(r"^inference\s+median (\S+) ms, p95 (\S+) ms, max (\S+) ms",
                      text, re.MULTILINE)
    if match is None:
        raise Failure(f"no 'inference' line in output:\n{text}")
    median, p95, maximum = (float(value) for value in match.groups())
    return median, p95, maximum


def gst_env(build: Path, **extra: str) -> dict[str, str]:
    """@brief Environment for driving the stock gst-* tools against our plugin.

    @param build Build directory; its source/ holds libgstrfd.so.
    @param extra Additional variables, e.g. GST_DEBUG.
    @return Variables to overlay on the current environment.
    """
    env = {"GST_PLUGIN_PATH": str(build / "source"), **extra}

    # These tools are not instrumented, so an ASan plugin needs the runtime
    # preloaded. Leak detection goes off with it; rfd itself keeps it on.
    preload = os.environ.get("RFD_ASAN_PRELOAD")
    if preload:
        env["LD_PRELOAD"] = preload
        env["ASAN_OPTIONS"] = "detect_leaks=0"
    return env


def rfd(build: Path, source: Path, gallery: Path, *extra: str):
    """@brief Run the rfd binary over one source.

    @param build   Build directory holding the binaries.
    @param source  Clip, URI or device to analyse.
    @param gallery Gallery JSON to match against.
    @param extra   Additional command-line flags.
    @return The captured stderr, which holds the summary.
    @exception Failure The binary exited non-zero.
    """
    argv = [str(build / "rfd"), "--source", str(source), "--gallery", str(gallery),
            "--quiet", *extra]
    result = run(argv)
    if result.returncode != 0:
        raise Failure(f"rfd exited {result.returncode}\n{result.stderr}")
    return result.stderr


# --- checks ----------------------------------------------------------------

def check_plugin(build: Path, gallery: Path) -> str:
    """@brief rfdface must be discoverable by GStreamer's own tooling.

    @param build   Build directory, which also holds the plugin.
    @param gallery Unused; the signature is uniform across checks.
    @return One line describing what was observed.
    @exception Failure gst-inspect cannot see the element or its properties.
    """
    result = run(["gst-inspect-1.0", "rfdface"], env=gst_env(build))
    if result.returncode != 0:
        raise Failure("gst-inspect-1.0 cannot see rfdface; it is not a real plugin\n"
                      + result.stderr)
    for expected in ("GstVideoFilter", "gallery", "threshold", "max-input-edge"):
        if expected not in result.stdout:
            raise Failure(f"gst-inspect output is missing '{expected}'")
    return "rfdface visible to gst-inspect, derived from GstVideoFilter"


def check_gst_launch(build: Path, gallery: Path) -> str:
    """@brief Inference must run in a pipeline none of our code drives.

    @param build   Build directory, which also holds the plugin.
    @param gallery Gallery JSON to match against.
    @return One line describing what was observed.
    @exception Failure gst-launch failed, or analysed too few frames.
    """
    clip = REPO / "external/data/gallery/reagan/reagan.mp4"
    env = gst_env(build, GST_DEBUG="rfdface:6")
    result = run([
        "gst-launch-1.0", "-q",
        "filesrc", f"location={clip}", "!", "decodebin",
        "!", "videoconvert", "!", "video/x-raw,format=BGR",
        "!", "rfdface", f"gallery={gallery}",
        "!", "fakesink", "sync=false",
    ], env=env)
    if result.returncode != 0:
        raise Failure(f"gst-launch failed\n{result.stderr}")

    frames = len(re.findall(r"face\(s\) in \d+ us", result.stderr))
    if frames < 250:
        raise Failure(f"gst-launch processed only {frames} frames")
    return f"gst-launch ran inference over {frames} frames with no application code"


def check_realtime(build: Path, gallery: Path) -> str:
    """@brief A 10 s clip must take ~10 s, keep up with inference, and stay right.

    Three assertions, separated because only one is about the machine:

    @li @b pacing - decoded frames per second of wall clock, which `identity
        sync=true` holds to the source rate on any hardware.
    @li @b keeping @b up - frames reaching inference, bounded by the latency this
        run measured. A frame is only analysable if inference fits in its 33 ms,
        so on a slow runner the leaky queue dropping is by design.
    @li @b correctness - the subject is identified on every analysed frame bar
        MAX_UNIDENTIFIED_FRAMES.

    @param build   Build directory holding the binaries.
    @param gallery Gallery JSON to match against.
    @return One summary per clip, joined with semicolons.
    @exception Failure Pacing was lost, frames were dropped that inference had
               time for, or the subject went unidentified on more than
               MAX_UNIDENTIFIED_FRAMES frames.
    """
    lines = []
    for clip in sorted((REPO / "external/data/gallery").rglob("*.mp4")):
        identity = clip.parent.name
        out = rfd(build, clip, gallery)

        decoded = summary_field(out, "frames decoded")
        analysed = summary_field(out, "frames analysed")
        identified = summary_field(out, "identified")
        seconds = summary_field(out, "wall clock", 0)

        if analysed <= 0 or seconds <= 0.0:
            raise Failure(f"{identity}: analysed {analysed:.0f} frames in {seconds} s\n{out}")

        # Decoded, not analysed: mixing in what inference kept up with would make
        # this a statement about the machine.
        paced_fps = decoded / seconds
        if not MIN_REALTIME_FPS <= paced_fps <= MAX_REALTIME_FPS:
            raise Failure(f"{identity}: {paced_fps:.1f} fps decoded is not real-time "
                          f"(source is {SOURCE_FPS} fps)")

        # p95 rather than the median: with only a couple of frames buffered it is
        # the slow passes, not the typical one, that decide what gets dropped.
        _, p95_ms, _ = inference_latency(out)
        affordable_fps = min(SOURCE_FPS, 1000.0 / p95_ms) if p95_ms > 0.0 else SOURCE_FPS
        analysed_fps = analysed / seconds
        if analysed_fps < affordable_fps * MIN_ANALYSED_FRACTION:
            raise Failure(f"{identity}: analysed {analysed:.0f} of {decoded:.0f} frames "
                          f"({analysed_fps:.1f} fps), under the {affordable_fps:.1f} fps "
                          f"that {p95_ms:.1f} ms p95 inference affords")

        missed = analysed - identified
        if missed > MAX_UNIDENTIFIED_FRAMES:
            raise Failure(f"{identity}: identified {identified:.0f} of {analysed:.0f} "
                          f"analysed frames, {missed:.0f} missed")
        lines.append(f"{identity} {paced_fps:.1f}fps paced, {analysed:.0f}/{decoded:.0f} "
                     f"analysed at {analysed_fps:.1f}fps, {identified:.0f} identified")
    return "; ".join(lines)


def check_throughput(build: Path, gallery: Path) -> str:
    """@brief With a blocking queue nothing may drop, and throughput means what it says.

    @param build   Build directory holding the binaries.
    @param gallery Gallery JSON to match against.
    @return One line describing what was observed.
    @exception Failure Any drop, a missed subject, or - on an uninstrumented
               build - a rate below the source's.
    """
    clip = REPO / "external/data/gallery/reagan/reagan.mp4"
    out = rfd(build, clip, gallery, "--no-sync", "--no-drop")

    dropped = summary_field(out, "frames dropped")
    analysed = summary_field(out, "frames analysed")
    identified = summary_field(out, "identified")
    fps = summary_field(out, "throughput", 0)

    if dropped != 0:
        raise Failure(f"{dropped:.0f} frames dropped with a non-leaky queue")
    if identified < analysed:
        raise Failure(f"identified {identified:.0f} of {analysed:.0f}")

    # Headroom is only claimed for a build you would ship; under a sanitiser the
    # two assertions above are the point.
    if SANITIZED:
        return (f"{fps:.1f} fps, {analysed:.0f} frames, 0 dropped "
                "(instrumented build, so no headroom asserted)")
    if fps < SOURCE_FPS:
        raise Failure(f"{fps:.1f} fps cannot sustain a {SOURCE_FPS} fps source")
    return f"{fps:.1f} fps, {analysed:.0f} frames, 0 dropped ({fps / SOURCE_FPS:.1f}x headroom)"


def check_open_set(build: Path, gallery: Path) -> str:
    """@brief A subject absent from the gallery must be rejected on every frame.

    Enrols a reduced gallery with one identity removed, then probes it with that
    identity's own clip.

    @param build   Build directory holding the binaries.
    @param gallery Unused; a reduced gallery is enrolled here instead.
    @return One line describing what was observed.
    @exception Failure Any false accept, or no face evaluated at all.
    """
    held_out = "reagan"
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "gallery"
        for directory in sorted((REPO / "external/data/gallery").iterdir()):
            if not directory.is_dir() or directory.name == held_out:
                continue
            target = staged / directory.name
            target.mkdir(parents=True)
            for still in directory.iterdir():
                if still.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    shutil.copy2(still, target / still.name)

        reduced = Path(tmp) / "held_out.json"
        result = run([str(build / "rfd-enroll"), str(staged), "-o", str(reduced)])
        if result.returncode != 0:
            raise Failure(f"enrolling the reduced gallery failed\n{result.stderr}")

        out = rfd(build, REPO / f"external/data/gallery/{held_out}/{held_out}.mp4",
                  reduced, "--no-sync", "--no-drop")
        identified = summary_field(out, "identified")
        unknown = summary_field(out, "unknown")
        if identified != 0:
            raise Failure(f"{identified:.0f} false accepts against a gallery without "
                          f"{held_out}")
        if unknown == 0:
            raise Failure("no faces were evaluated at all")
    return f"{held_out} rejected on all {unknown:.0f} frames by a gallery without him"


def check_degradation(build: Path, gallery: Path) -> str:
    """@brief Overloaded, it must drop frames rather than stall - and stay correct.

    @param build   Build directory holding the binaries.
    @param gallery Gallery JSON to match against.
    @return One line describing what was observed.
    @exception Failure Nothing dropped, results were corrupted, or it stalled.
    """
    clip = REPO / "external/data/gallery/ford/ford_trimmed.mp4"
    started = time.monotonic()
    out = rfd(build, clip, gallery, "--no-sync", "--queue", "1")
    elapsed = time.monotonic() - started

    decoded = summary_field(out, "frames decoded")
    analysed = summary_field(out, "frames analysed")
    identified = summary_field(out, "identified")
    dropped = summary_field(out, "frames dropped")

    if dropped == 0:
        raise Failure("expected drops when the decoder outruns inference")
    if identified < analysed:
        raise Failure(f"dropping corrupted results: {identified:.0f} of {analysed:.0f}")
    # Dropping, not stalling: finishing far slower than real time would mean the
    # queue was blocking instead of leaking.
    if elapsed > 10.0:
        raise Failure(f"took {elapsed:.1f}s - the pipeline stalled instead of dropping")
    return (f"{dropped:.0f}/{decoded:.0f} dropped under overload in {elapsed:.1f}s, "
            f"all {identified:.0f} analysed frames still correct")


def check_caps_change(build: Path, gallery: Path) -> str:
    """@brief Resolution and framerate changes mid-stream must renegotiate cleanly.

    @param build   Build directory, which also holds the plugin.
    @param gallery Gallery JSON to match against.
    @return One line describing what was observed.
    @exception Failure Fewer than three renegotiations, or too few surviving frames.
    """
    env = gst_env(build, GST_DEBUG="rfdface:6")
    result = run([
        "gst-launch-1.0", "-q", "concat", "name=c",
        "!", "videoconvert", "!", "video/x-raw,format=BGR",
        "!", "rfdface", f"gallery={gallery}", "!", "fakesink", "sync=false",
        "videotestsrc", "num-buffers=20", "pattern=ball",
        "!", "video/x-raw,width=320,height=240,framerate=30/1", "!", "c.",
        "videotestsrc", "num-buffers=20", "pattern=ball",
        "!", "video/x-raw,width=640,height=480,framerate=30/1", "!", "c.",
        "videotestsrc", "num-buffers=20", "pattern=ball",
        "!", "video/x-raw,width=1280,height=720,framerate=15/1", "!", "c.",
    ], env=env)
    if result.returncode != 0:
        raise Failure(f"pipeline failed across a caps change\n{result.stderr}")

    negotiations = re.findall(r"negotiated (\d+x\d+) BGR at (\S+) fps", result.stderr)
    if len(negotiations) < 3:
        raise Failure(f"expected 3 renegotiations, saw {len(negotiations)}: {negotiations}")
    frames = len(re.findall(r"face\(s\) in \d+ us", result.stderr))
    if frames < 60:
        raise Failure(f"only {frames} frames survived the caps changes")
    return (f"{len(negotiations)} renegotiations "
            f"({', '.join(f'{w}@{f}' for w, f in negotiations)}), {frames} frames")


## Check name to implementation, as exposed by --only.
CHECKS = {
    "plugin": check_plugin,
    "gst-launch": check_gst_launch,
    "realtime": check_realtime,
    "throughput": check_throughput,
    "open-set": check_open_set,
    "degradation": check_degradation,
    "caps-change": check_caps_change,
}


def main() -> int:
    """@brief Run the selected checks, printing one line each.

    @return 0 when every selected check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/dev")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="gallery.json (default <build-dir>/gallery.json)")
    parser.add_argument("--only", choices=sorted(CHECKS), action="append",
                        help="run only these checks (repeatable)")
    args = parser.parse_args()

    build = args.build_dir.resolve()
    gallery = (args.gallery or build / "gallery.json").resolve()

    if not (build / "rfd").is_file():
        sys.exit(f"error: no rfd binary at {build}; build first")
    if not gallery.is_file():
        sys.exit(f"error: no gallery at {gallery}\n"
                 f"       run: {build}/rfd-enroll external/data/gallery -o {gallery}")

    selected = args.only or sorted(CHECKS)
    failures = 0

    for name in selected:
        try:
            detail = CHECKS[name](build, gallery)
            print(f"  ok   {name}: {detail}")
        except Failure as error:
            print(f"  FAIL {name}: {error}")
            failures += 1
        except subprocess.TimeoutExpired:
            print(f"  FAIL {name}: timed out")
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} of {len(selected)} checks")
        return 1
    print(f"PASSED: {len(selected)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
