import argparse
import os
import sys

import torch

# Add parent directory to path to allow imports from fcos package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import FCOS
from datasets import VOC2007DetectionTiny
from engine import inference_with_detector
from utils import load_config, get_default_config, merge_config


def parse_args():
    parser = argparse.ArgumentParser(description="Test FCOS detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fcos_voc.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for mAP evaluation. If not specified, visualizes results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Auto-detected if not specified.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    default_config = get_default_config()
    custom_config = load_config(args.config)
    cfg = merge_config(default_config, custom_config)

    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create validation dataset
    val_dataset = VOC2007DetectionTiny(
        cfg["data"]["dataset_dir"],
        split="val",
        image_size=cfg["data"]["image_size"],
    )

    print(f"Validation dataset size: {len(val_dataset)}")

    # Use batch_size = 1 during inference - during inference we do not center crop
    # the image to detect all objects, hence they may be of different size.
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    # Create model
    detector = FCOS(
        num_classes=cfg["model"]["num_classes"],
        fpn_channels=cfg["model"]["fpn_channels"],
        stem_channels=cfg["model"]["stem_channels"],
    )

    # Load weights
    detector.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location="cpu")
    )
    print(f"Loaded weights from {args.weights}")

    # Run inference
    inference_with_detector(
        detector,
        val_loader,
        val_dataset.idx_to_class,
        score_thresh=cfg["inference"]["score_thresh"],
        nms_thresh=cfg["inference"]["nms_thresh"],
        output_dir=args.output_dir,
        device=device,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
