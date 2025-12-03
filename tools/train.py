import argparse
import multiprocessing
import os
import sys

import torch

# Add parent directory to path to allow imports from fcos package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import FCOS
from datasets import VOC2007Dataset
from engine import train_detector
from utils import reset_seed, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCOS detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fcos_voc.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fcos_detector.pth",
        help="Path to save trained model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
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

    # Set random seed
    reset_seed(args.seed)

    # Create training dataset
    train_dataset = VOC2007Dataset(
        root=cfg["data"]["dataset_dir"],
        split="trainval",
        image_size=cfg["data"]["image_size"],
        max_boxes=cfg["data"]["max_boxes"],
        exclude_difficult=cfg["data"]["exclude_difficult"],
    )
    print(f"Training dataset size: {len(train_dataset)}")

    # Create validation dataset (using test split)
    val_dataset = VOC2007Dataset(
        root=cfg["data"]["dataset_dir"],
        split="test",
        image_size=cfg["data"]["image_size"],
        max_boxes=cfg["data"]["max_boxes"],
        exclude_difficult=cfg["data"]["exclude_difficult"],
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    num_workers = cfg["num_workers"]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Create model
    detector = FCOS(
        num_classes=cfg["model"]["num_classes"],
        fpn_channels=cfg["model"]["fpn_channels"],
        stem_channels=cfg["model"]["stem_channels"],
    )

    # Train
    train_detector(
        detector,
        train_loader,
        learning_rate=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        max_iters=cfg["train"]["max_iters"],
        log_period=cfg["train"]["log_period"],
        device=device,
        val_loader=val_loader,
        val_period=cfg["validation"]["val_period"],
        num_classes=cfg["model"]["num_classes"],
        score_thresh=cfg["inference"]["score_thresh"],
        nms_thresh=cfg["inference"]["nms_thresh"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    # Save final model
    torch.save(detector.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
