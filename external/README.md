# external/ - vendored third-party assets

Weights for YuNet and SFace. 

## models/

| File | Model | Role | Size |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet | Face detection + 5 landmarks | 227 KB |
| `face_recognition_sface_2021dec.onnx` | SFace | 128-D face embeddings | 37 MB |

### Provenance

Both files come from the OpenCV Model Zoo, pinned at commit
`47534e27c9851bb1128ccc0102f1145e27f23f98`:

- <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>

To re-fetch and verify against the checked-in digests:

```sh
base=https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models
curl -L -o external/models/face_detection_yunet_2023mar.onnx \
  "$base/face_detection_yunet/face_detection_yunet_2023mar.onnx"
curl -L -o external/models/face_recognition_sface_2021dec.onnx \
  "$base/face_recognition_sface/face_recognition_sface_2021dec.onnx"

(cd external/models && sha256sum -c SHA256SUMS)
```

### Licences

| Model | Licence | Copyright |
|---|---|---|
| YuNet | MIT | 2020 Shiqi Yu |
| SFace | Apache-2.0 | Zhong Yaoyao / insightface |

