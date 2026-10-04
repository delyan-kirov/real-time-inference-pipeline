# external/data/ - vendored face data

Data for application testing.

## Layout

```text
gallery/<identity>/*.png|*.jpg   stills -> one template per identity
gallery/<identity>/*.mp4         probe clip for that identity
```

## Validation

```sh
./build/dev/rfd-enroll external/data/gallery -o gallery.json
python3 tools/validate_data.py
```

Checks, in order of how quietly they fail:

1. Every still decodes, is in colour, and has exactly **one** detectable face.
2. Every identity has at least three usable stills.
3. Every clip is colour, within budget, and its subject is found in most frames.
4. Every clip is matched against the full gallery (rank-1 must hold) **and**
   against a gallery with its own subject removed (must be rejected). The second
   pass is the open-set test; leave-one-out means all identities stay enrolled
   while rejection is still measured for each.
5. The score floor and impostor-score ceiling across all clips are
   reported as a threshold window, and the configured threshold is flagged if it
   hugs either edge.

The validator re-implements detect/align/embed in Python on purpose. The templates
it matches against were produced by the C++ path, so agreement cross-checks the two
implementations; disagreement means one of them is wrong.

## Not here yet

Multi-face footage for load testing - a wide shot with five to fifteen faces at
varying scale and motion blur. The clips here each have one well-framed subject,
which is right for measuring recognition and useless for proving the pipeline does
not stall or degrade under sustained load. That belongs with the pipeline work,
since it is a throughput concern rather than a recognition one.
