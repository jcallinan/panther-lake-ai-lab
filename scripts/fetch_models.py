"""Download the models used by the NPU showcase examples into models/ (gitignored).

Run: python scripts/fetch_models.py [--only vision|depth|llm|all]
All models were verified to compile and run on the Core Ultra 5 322 NPU (see docs/DRIVER_SETUP.md for driver notes).
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"
ONNX = ["onnx/model.onnx", "*.json", "README.md"]

# group, Hugging Face repo, local folder, allow_patterns (None = whole repo), licence, approx size
CATALOG = [
    ("vision", "OpenVINO/yolox_tiny-int8-ov", "yolox_tiny-int8-ov", None, "Apache-2.0", "6 MB"),
    ("vision", "onnx-community/mediapipe_selfie_segmentation", "mediapipe_selfie_segmentation", ONNX,
     "Apache-2.0", "1 MB"),
    ("depth", "onnx-community/depth-anything-v2-small", "depth-anything-v2-small", ONNX, "Apache-2.0", "95 MB"),
    ("llm", "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov", "phi-3-mini-4k-instruct-int4-cw-ov", None, "MIT", "1.9 GB"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["vision", "depth", "llm", "all"], default="all")
    a = p.parse_args()
    for group, repo, folder, patterns, lic, size in CATALOG:
        if a.only not in ("all", group):
            continue
        print(f"{repo} ({lic}, ~{size}) -> models/{folder}")
        snapshot_download(repo, local_dir=MODELS / folder, allow_patterns=patterns)
    print("Done. The vision examples need 'vision'; offload needs 'vision'; depth_parallax needs 'depth'; "
          "llm_chat needs 'llm'.")


if __name__ == "__main__":
    main()
