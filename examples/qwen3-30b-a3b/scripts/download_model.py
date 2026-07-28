# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Download Qwen3-30B-A3B from HuggingFace.

Usage:
    python download_model.py --output_dir /mnt/raid0/models/Qwen3-30B-A3B
    python download_model.py --model_name Qwen/Qwen3-30B-A3B-Base --output_dir /data/model
"""

import argparse
import logging

from lumen.models.utils import download_hf_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-30B-A3B model")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/raid0/models/Qwen3-30B-A3B",
        help="Directory to save the downloaded model",
    )
    parser.add_argument("--verify", action="store_true", default=False)
    args = parser.parse_args()
    download_hf_model(args.model_name, args.output_dir, verify=args.verify)


if __name__ == "__main__":
    main()
