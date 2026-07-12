#!/usr/bin/env python3
"""Inspect a YOLO model used by the Aries vision pipeline."""

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "src/aries_vision_grasp/models/grasp.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model = YOLO(str(args.model))
    print(f"\nModel task: {model.task.upper()}")
    print("\n--- Architecture summary ---")
    model.info()


if __name__ == "__main__":
    main()
