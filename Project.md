# Project: Real-Time Facial Recognition

- GStreamer - video ingestion and pipeline
- OpenCV - ML inference
- YuNet - face detection + 5 facial landmarks
- SFace - face embeddings / recognition
- C++ - pipeline logic, gallery matching, metadata
- CMake - build system
- Nix/Nixpkgs - provide OpenCV and GStreamer
- Vendor model weights in the repository
- JSON metadata - identities, similarity scores, bounding boxes, etc.

# Pipeline:

Video
|
GStreamer
|
YuNet
|
SFace
|
Compare embedding against gallery
|
JSON metadata

# Decision

1. Use SFace + YuNet models, both are in OpenCV.
2. Don't vendor the libraries, just the weights and the data.
3. No need to implement the ML algorithms.

