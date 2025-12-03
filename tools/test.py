import argparse
import os
import sys

import torch

# Add parent directory to path to allow imports from fcos package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import FCOS
from datasets import VOC2007Dataset
from engine import evaluate_detector
from utils import load_config


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
    cfg = load_config(args.config)

    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create test dataset
    test_dataset = VOC2007Dataset(
        root=cfg["data"]["dataset_dir"],
        split="test",
        image_size=cfg["data"]["image_size"],
        max_boxes=cfg["data"]["max_boxes"],
        exclude_difficult=cfg["data"]["exclude_difficult"],
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Use batch_size = 1 during inference - during inference we do not center crop
    # the image to detect all objects, hence they may be of different size.
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg["num_workers"],
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

    # Run evaluation
    evaluate_detector(
        detector,
        test_loader,
        test_dataset.idx_to_class,
        score_thresh=cfg["inference"]["score_thresh"],
        nms_thresh=cfg["inference"]["nms_thresh"],
        output_dir=args.output_dir,
        device=device,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
