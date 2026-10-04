#!/usr/bin/env python3
##
# @file validate_data.py
# @brief Validate the vendored face data in external/data/.
#
# Checks:
#   - every gallery still is decodable, in colour, and contains exactly one face
#     the detector can resolve - the same conditions rfd-enroll imposes;
#   - every probe clip is in colour, within the committed-size budget, and its
#     subject is actually recognised against the gallery;
#   - every clip is ALSO matched against a gallery with its own subject removed,
#     and must be rejected.
#
# This deliberately re-implements detect/align/embed in Python rather than
# calling into rfd_face, so agreement cross-checks the two implementations.
#
# @par Usage
# @code
# tools/validate_data.py                       # uses ./gallery.json
# tools/validate_data.py --json build/g.json --strict
# @endcode
##
"""Validate the vendored face data in external/data/.

Checks:

  * every gallery still is decodable, in colour, and contains exactly one face
    the detector can resolve - the same conditions rfd-enroll imposes;
  * every probe clip is in colour, within the committed-size budget, and its
    subject is actually recognised against the gallery;
  * every clip is ALSO matched against a gallery with its own subject removed,
    and must be rejected. 

This deliberately re-implements detect/align/embed in Python rather than calling
into rfd_face. The gallery templates it matches against were produced by the C++
path, so agreement here cross-checks the two implementations; disagreement means
one of them must be wrong.

Usage:
    tools/validate_data.py                       # uses ./gallery.json
    tools/validate_data.py --json build/g.json --strict

Exit status is non-zero if any check fails, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("error: needs opencv-python and numpy (both are in the nix dev shell)")

## Repository root, derived from this file's location.
REPO = Path(__file__).resolve().parent.parent

# Keep in step with include/detector.hpp and include/embedder.hpp, which hold
# the authoritative values and explain where they came from.

## Longest side of the detector input; larger frames are downscaled.
MAX_INPUT_EDGE = 640
## Minimum cosine similarity to call a face identified.
COSINE_THRESHOLD = 0.388
## Minimum detector confidence to accept a detection.
MIN_DETECTOR_SCORE = 0.9

## Minimum p99 HSV saturation for an image to count as colour.
MIN_COLOUR_SATURATION_P99 = 60

## Fewest usable stills an identity may be enrolled from.
MIN_STILLS_PER_IDENTITY = 3
## Size budget for one committed probe clip.
MAX_CLIP_MEGABYTES = 5.0
## Size budget for everything under external/data.
MAX_TOTAL_DATA_MEGABYTES = 40.0

## Extensions treated as gallery stills.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
## Extensions treated as probe clips.
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".ogv", ".mov"}


@dataclass
class Report:
    """@brief Collected failures and warnings, printed as they arrive."""

    ## Messages that make the run fail.
    failures: list[str] = field(default_factory=list)
    ## Messages that fail the run only under --strict.
    warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        """@brief Record and print a failure.

        @param message What went wrong.
        """
        self.failures.append(message)
        print(f"  FAIL {message}")

    def warn(self, message: str) -> None:
        """@brief Record and print a warning.

        @param message What is suspicious but not fatal.
        """
        self.warnings.append(message)
        print(f"  WARN {message}")


def colour_saturation(image) -> float:
    """@brief Measure how strongly coloured an image is.

    p99 rather than the mean: a colour photograph of a man in a dark suit has a
    low mean saturation but still has strongly coloured pixels somewhere.

    @param image A BGR image.
    @return The 99th percentile of HSV saturation, in 0..255.
    """
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    return float(np.percentile(saturation, 99))


class Pipeline:
    """@brief Detection plus embedding, mirroring the C++ path including downscaling."""

    def __init__(self, models: Path) -> None:
        """@brief Load the vendored YuNet and SFace weights.

        @param models Directory holding the ONNX files.
        """
        yunet = models / "face_detection_yunet_2023mar.onnx"
        sface = models / "face_recognition_sface_2021dec.onnx"
        for path in (yunet, sface):
            if not path.is_file():
                sys.exit(f"error: missing weights: {path}")

        self.detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), MIN_DETECTOR_SCORE, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, frame):
        """@brief Detect faces, in original-frame coordinates, biggest first.

        @param frame A BGR image.
        @return An (n, 15) array of YuNet rows; empty when nothing was found.
        """
        height, width = frame.shape[:2]
        scale = min(1.0, MAX_INPUT_EDGE / max(height, width))
        if scale < 1.0:
            resized = cv2.resize(
                frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            resized = frame

        self.detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self.detector.detect(resized)
        if faces is None or len(faces) == 0:
            return np.empty((0, 15), dtype=np.float32)

        faces = faces.copy()
        if scale < 1.0:
            # Coordinates only - column 14 is the score and must not be scaled.
            faces[:, :14] /= scale
        return faces[np.argsort(-faces[:, 3])]

    def embed(self, frame, row):
        """@brief Align one detected face and embed it.

        @param frame The full-resolution frame the detection came from.
        @param row   One YuNet detection row.
        @return A unit-length 128-D embedding.
        """
        aligned = self.recognizer.alignCrop(frame, row.reshape(1, -1))
        feature = self.recognizer.feature(aligned).flatten()
        return feature / (np.linalg.norm(feature) + 1e-9)


def load_gallery(path: Path):
    """@brief Read an rfd-enroll gallery.

    @param path The gallery JSON file.
    @return (identity names, template matrix, SFace digest).
    """
    if not path.is_file():
        sys.exit(f"error: no gallery at {path}\n       run: rfd-enroll external/data/gallery -o {path}")

    doc = json.loads(path.read_text())
    names = [entry["name"] for entry in doc["identities"]]
    templates = np.array([entry["embedding"] for entry in doc["identities"]], dtype=np.float32)
    return names, templates, doc.get("model", {}).get("sface_sha256", "")


def check_stills(gallery_dir: Path, pipeline: Pipeline, report: Report) -> dict[str, int]:
    """@brief Check every gallery still is decodable, colour and unambiguous.

    @param gallery_dir Directory holding one subdirectory per identity.
    @param pipeline    The Python detect/embed pipeline.
    @param report      Collects failures and warnings.
    @return Usable still count per identity.
    """
    print("gallery stills")
    counts: dict[str, int] = {}

    for identity_dir in sorted(p for p in gallery_dir.iterdir() if p.is_dir()):
        identity = identity_dir.name
        stills = sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        usable = 0

        for still in stills:
            image = cv2.imread(str(still), cv2.IMREAD_COLOR)
            label = f"{identity}/{still.name}"
            if image is None:
                report.fail(f"{label}: cannot decode")
                continue

            saturation = colour_saturation(image)
            if saturation < MIN_COLOUR_SATURATION_P99:
                report.fail(f"{label}: greyscale (p99 saturation {saturation:.0f})")
                continue

            faces = pipeline.detect(image)
            if len(faces) == 0:
                report.fail(f"{label}: no face detected")
                continue
            if len(faces) > 1:
                report.fail(f"{label}: {len(faces)} faces, subject is ambiguous")
                continue
            usable += 1

        counts[identity] = usable
        if usable < MIN_STILLS_PER_IDENTITY:
            report.fail(
                f"{identity}: only {usable} usable stills, want >= {MIN_STILLS_PER_IDENTITY}"
            )
        else:
            print(f"  ok   {identity}: {usable} stills")

    return counts


def check_clips(gallery_dir: Path, pipeline: Pipeline, names, templates, report: Report,
                sample_frames: int) -> set[str]:
    """@brief Check every probe clip, genuine and leave-one-out impostor.

    Also reports the operating point: the window every genuine score must sit
    above and every impostor score below, and where COSINE_THRESHOLD falls in it.

    @param gallery_dir   Directory holding the clips, one per identity folder.
    @param pipeline      The Python detect/embed pipeline.
    @param names         Enrolled identity names, in template-matrix order.
    @param templates     The gallery template matrix.
    @param report        Collects failures and warnings.
    @param sample_frames How many frames to sample per clip.
    @return The identities a clip was found for.
    """
    print("\nprobe clips")
    seen: set[str] = set()
    margins: list[tuple[str, float, float]] = []

    clips = sorted(p for p in gallery_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        report.fail("no probe clips found under the gallery directory")
        return seen

    for clip in clips:
        # The clip's identity is its parent directory, matching the gallery layout.
        identity = clip.parent.name
        seen.add(identity)

        megabytes = clip.stat().st_size / 1e6
        capture = cv2.VideoCapture(str(clip))
        if not capture.isOpened():
            report.fail(f"{clip.name}: cannot open")
            continue

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        saturations, with_face, rank1, wrong, unknown, sampled = [], 0, 0, 0, 0, 0
        held_out = names.index(identity) if identity in names else None
        false_accepts, best_impostor_score = 0, -1.0
        genuine_scores: list[float] = []
        step = max(1, int(round(fps / 3))) if fps else 1
        decoded = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = decoded
            decoded += 1
            if index % step:
                continue
            if sampled >= sample_frames:
                continue
            sampled += 1
            saturations.append(colour_saturation(frame))

            faces = pipeline.detect(frame)
            if len(faces) == 0:
                continue
            with_face += 1

            similarities = templates @ pipeline.embed(frame, faces[0])
            best = int(np.argmax(similarities))
            if similarities[best] < COSINE_THRESHOLD:
                unknown += 1
            elif names[best] == identity:
                rank1 += 1
                genuine_scores.append(float(similarities[best]))
            else:
                wrong += 1
                report.warn(
                    f"{clip.name} frame {index}: matched {names[best]} "
                    f"({similarities[best]:.3f}), expected {identity}"
                )

            # The impostor case: strip this subject and see what is left.
            impostor = np.delete(similarities, held_out) if held_out is not None else similarities
            if impostor.size:
                top = float(impostor.max())
                best_impostor_score = max(best_impostor_score, top)
                if top >= COSINE_THRESHOLD:
                    false_accepts += 1

        capture.release()
        duration = decoded / fps if fps else 0.0
        median_saturation = float(np.median(saturations)) if saturations else 0.0
        enrolled = identity in names

        worst_genuine = min(genuine_scores) if genuine_scores else float("nan")
        margins.append((identity, worst_genuine, best_impostor_score))

        print(
            f"  {identity:8} {width}x{height:<5} {decoded:4}f {duration:5.1f}s "
            f"{megabytes:5.1f}MB  faces {with_face}/{sampled}  "
            f"rank1 {rank1} wrong {wrong} unknown {unknown}  |  "
            f"genuine min {worst_genuine:.3f}  impostor max {best_impostor_score:.3f}  "
            f"({false_accepts} false accepts)"
        )

        if megabytes > MAX_CLIP_MEGABYTES:
            report.fail(f"{clip.name}: {megabytes:.1f}MB exceeds the {MAX_CLIP_MEGABYTES}MB budget")
        if median_saturation < MIN_COLOUR_SATURATION_P99:
            report.fail(f"{clip.name}: greyscale (median p99 saturation {median_saturation:.0f})")
        if sampled and with_face / sampled < 0.5:
            report.fail(f"{clip.name}: face found in only {with_face}/{sampled} sampled frames")

        if enrolled:
            if wrong:
                report.fail(f"{clip.name}: {wrong} frames matched the wrong identity")
            if with_face and rank1 / with_face < 0.8:
                report.fail(
                    f"{clip.name}: rank-1 {rank1}/{with_face} below 80% of frames with a face"
                )
        # Open-set rejection, measured for every clip via leave-one-out above.
        if false_accepts:
            report.fail(
                f"{clip.name}: with {identity} removed from the gallery, "
                f"{false_accepts}/{with_face} frames still matched someone "
                f"(up to {best_impostor_score:.3f} >= {COSINE_THRESHOLD}); "
                "the threshold is not rejecting impostors"
            )

    # --- operating point ----------------------------------------------------
    genuine_floor = min(m[1] for m in margins if m[1] == m[1])
    impostor_ceiling = max(m[2] for m in margins)

    print(f"\noperating point over {len(margins)} clips")
    print(f"  worst genuine   {genuine_floor:.3f}")
    print(f"  best impostor   {impostor_ceiling:.3f}")
    print(f"  usable window   {impostor_ceiling:.3f} .. {genuine_floor:.3f}"
          f"  (width {genuine_floor - impostor_ceiling:.3f})")
    print(f"  configured      {COSINE_THRESHOLD:.3f}")

    if genuine_floor <= impostor_ceiling:
        report.fail(
            f"genuine and impostor scores overlap ({genuine_floor:.3f} <= "
            f"{impostor_ceiling:.3f}); no single threshold separates them"
        )
    else:
        midpoint = (genuine_floor + impostor_ceiling) / 2
        print(f"  midpoint        {midpoint:.3f}  <- maximises margin both ways")

        width = genuine_floor - impostor_ceiling
        if width < 0.05:
            report.warn(
                f"separation window is only {width:.3f} wide; this data barely "
                "distinguishes these identities and the threshold is fragile"
            )

        margin = 0.25 * width
        if not (impostor_ceiling + margin <= COSINE_THRESHOLD <= genuine_floor - margin):
            report.warn(
                f"configured threshold {COSINE_THRESHOLD:.3f} is in the outer "
                f"quarter of the {impostor_ceiling:.3f}..{genuine_floor:.3f} "
                f"window; consider {midpoint:.3f}"
            )

    return seen


def main() -> int:
    """@brief Validate the stills, the clips and the committed data budget.

    @return 0 when everything passed, 1 on any failure (or any warning under
            --strict).
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery", type=Path, default=REPO / "external/data/gallery",
                        help="gallery directory (default external/data/gallery)")
    parser.add_argument("--models", type=Path, default=REPO / "external/models",
                        help="directory holding the ONNX weights")
    parser.add_argument("--json", type=Path, default=Path("gallery.json"),
                        help="gallery file produced by rfd-enroll (default gallery.json)")
    parser.add_argument("--frames", type=int, default=40,
                        help="frames to sample per clip (default 40)")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    args = parser.parse_args()

    if not args.gallery.is_dir():
        sys.exit(f"error: no gallery directory at {args.gallery}")

    names, templates, digest = load_gallery(args.json)
    pipeline = Pipeline(args.models)
    report = Report()

    print(f"gallery  {args.json}  ({len(names)} identities, sface {digest[:12]}…)\n")

    check_stills(args.gallery, pipeline, report)
    clip_identities = check_clips(args.gallery, pipeline, names, templates, report, args.frames)

    missing = set(names) - clip_identities
    if missing:
        report.warn(
            "no probe clip for enrolled identities: " + ", ".join(sorted(missing))
            + " - their rejection behaviour is untested"
        )

    total_megabytes = sum(p.stat().st_size for p in (REPO / "external/data").rglob("*")
                          if p.is_file()) / 1e6
    print(f"\ncommitted data {total_megabytes:.1f}MB")
    if total_megabytes > MAX_TOTAL_DATA_MEGABYTES:
        report.fail(f"external/data is {total_megabytes:.1f}MB, over the "
                    f"{MAX_TOTAL_DATA_MEGABYTES}MB budget")

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} problem(s), {len(report.warnings)} warning(s)")
        return 1
    if report.warnings and args.strict:
        print(f"FAILED (--strict): {len(report.warnings)} warning(s)")
        return 1
    print(f"PASSED ({len(report.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
