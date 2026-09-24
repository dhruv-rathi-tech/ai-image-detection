"""
predict.py
==========
CLI entry point for running inference with a trained FDCS-Net V4 model.

Usage
-----
    python src/predict.py --model models/fdcsnet_v4_final.keras --image path/to/image.jpg
    python src/predict.py --model models/fdcsnet_v4_final.keras --dir path/to/folder/
"""

import argparse
import os
import sys

import tensorflow as tf

# Ensure src/ is on the path when called from project root
sys.path.insert(0, os.path.dirname(__file__))
# Ensure project root is on the path so `configs` resolves regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, predict_single_image
from configs.config import CLASSIFICATION_THRESHOLD


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SynthLens: AI Image Detection"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to saved .keras model file"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--dir",   help="Path to a directory of images")
    parser.add_argument(
        "--threshold", type=float, default=CLASSIFICATION_THRESHOLD,
        help=f"Classification threshold (default: {CLASSIFICATION_THRESHOLD}, "
             f"from configs/config.py)"
    )
    return parser.parse_args()


def run_on_image(model, image_path: str, threshold: float) -> None:
    result = predict_single_image(model, image_path, threshold=threshold)
    print(
        f"  {os.path.basename(image_path):<40s} "
        f"→  {result['label']:<16s} "
        f"(confidence: {result['confidence']:.2%}, raw: {result['prediction']:.4f})"
    )


def main():
    args = parse_args()

    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)
    print(f"Threshold: {args.threshold}\n")
    print(f"{'File':<40s}   {'Label':<16s}  Details")
    print("-" * 75)

    if args.image:
        run_on_image(model, args.image, args.threshold)
    else:
        images = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED
        ]
        if not images:
            print(f"No supported images found in {args.dir}")
            sys.exit(1)
        for img_path in images:
            run_on_image(model, img_path, args.threshold)

    print()


if __name__ == "__main__":
    main()