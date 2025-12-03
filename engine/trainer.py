"""
Trainer for FCOS detector with validation and checkpointing.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils import compute_map


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


@torch.no_grad()
def validate_detector(
    detector,
    val_loader,
    num_classes: int,
    score_thresh: float,
    nms_thresh: float,
    device: str = "cpu",
):
    """
    Run validation and compute mAP.

    Args:
        detector: FCOS detector model
        val_loader: DataLoader for validation dataset
        num_classes: Number of classes
        score_thresh: Score threshold for filtering predictions
        nms_thresh: NMS IoU threshold
        device: Device to run validation on

    Returns:
        mAP value
    """
    detector.eval()

    predictions = []
    ground_truths = []

    for image_paths, images, gt_boxes in val_loader:
        images = images.to(device)

        # Run inference
        pred_boxes, pred_classes, pred_scores = detector(
            images,
            test_score_thresh=score_thresh,
            test_nms_thresh=nms_thresh,
        )

        # Move to CPU for mAP computation
        pred_boxes = pred_boxes.cpu()
        pred_classes = pred_classes.cpu()
        pred_scores = pred_scores.cpu()

        # Store predictions and ground truths
        # Note: batch_size=1 for validation
        predictions.append((pred_boxes, pred_classes, pred_scores))
        ground_truths.append(gt_boxes[0].cpu())

    # Compute mAP
    mAP, ap_per_class = compute_map(
        predictions,
        ground_truths,
        num_classes=num_classes,
        iou_threshold=0.5,
    )

    detector.train()
    return mAP


def train_detector(
    detector,
    train_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    device: str = "cpu",
    val_loader: Optional[DataLoader] = None,
    val_period: int = 1000,
    num_classes: int = 20,
    score_thresh: float = 0.4,
    nms_thresh: float = 0.6,
    checkpoint_dir: Optional[str] = None,
):
    """
    Train the detector with optional validation and checkpointing.

    Args:
        detector: FCOS detector model
        train_loader: DataLoader for training dataset
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        max_iters: Maximum number of training iterations
        log_period: Log training loss every N iterations
        device: Device to train on
        val_loader: Optional DataLoader for validation dataset
        val_period: Validate every N iterations
        num_classes: Number of classes for mAP computation
        score_thresh: Score threshold for validation
        nms_thresh: NMS threshold for validation
        checkpoint_dir: Directory to save checkpoints
    """
    detector.to(device=device)

    # Optimizer: use SGD with momentum.
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        momentum=0.9,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # LR scheduler: use step decay at 60% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Keep track of training loss for plotting.
    loss_history = []
    best_map = 0.0

    train_loader = infinite_loader(train_loader)
    detector.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        _, images, gt_boxes = next(train_loader)

        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # Dictionary of loss scalars.
        losses = detector(images, gt_boxes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"
            print(loss_str)
            loss_history.append(total_loss.item())

        # Validation
        if val_loader is not None and (_iter + 1) % val_period == 0:
            mAP = validate_detector(
                detector,
                val_loader,
                num_classes=num_classes,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                device=device,
            )
            print(f"[Iter {_iter + 1}][Validation mAP: {mAP:.4f}]")

            # Save checkpoints
            if checkpoint_dir is not None:
                # Save latest
                latest_path = os.path.join(checkpoint_dir, "latest.pth")
                torch.save(detector.state_dict(), latest_path)

                # Save best
                if mAP > best_map:
                    best_map = mAP
                    best_path = os.path.join(checkpoint_dir, "best.pth")
                    torch.save(detector.state_dict(), best_path)
                    print(f"[Iter {_iter + 1}][New best mAP: {best_map:.4f}]")

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()
