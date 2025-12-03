"""
Evaluator for FCOS detector - runs inference and handles mAP computation/visualization.
"""

import time
from typing import Dict

import torch
from torchvision import transforms

from utils import detection_visualizer, compute_map


def evaluate_detector(
    detector,
    test_loader,
    idx_to_class: Dict[int, str],
    score_thresh: float,
    nms_thresh: float,
    visualize: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """
    Run evaluation on a detector with a test data loader.

    Args:
        detector: FCOS detector model
        test_loader: DataLoader for test/val dataset
        idx_to_class: Dictionary mapping class index to class name
        score_thresh: Score threshold for filtering predictions
        nms_thresh: NMS IoU threshold
        visualize: If True, visualize detections; otherwise compute mAP
        dtype: Data type for inference
        device: Device to run inference on
    """
    # Ship model to device
    detector.to(dtype=dtype, device=device)
    detector.eval()

    start_t = time.time()

    # Define an "inverse" transform for the image that un-normalizes by ImageNet
    # color. Without this, the images will NOT be visually understandable.
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    # Collect predictions and ground truths for mAP computation
    all_predictions = []
    all_ground_truths = []

    for iter_num, test_batch in enumerate(test_loader):
        image_paths, images, gt_boxes = test_batch
        images = images.to(dtype=dtype, device=device)

        with torch.no_grad():
            if score_thresh is not None and nms_thresh is not None:
                # shapes: (num_preds, 4) (num_preds, ) (num_preds, )
                pred_boxes, pred_classes, pred_scores = detector(
                    images,
                    test_score_thresh=score_thresh,
                    test_nms_thresh=nms_thresh,
                )

        # Get GT boxes and remove padding
        gt_boxes_single = gt_boxes[0]
        valid_gt = gt_boxes_single[:, 4] != -1
        gt_boxes_valid = gt_boxes_single[valid_gt].cpu()

        # Get valid predictions
        if pred_boxes.shape[0] > 0:
            valid_pred = pred_classes != -1
            pred_boxes_valid = pred_boxes[valid_pred].cpu()
            pred_classes_valid = pred_classes[valid_pred].cpu()
            pred_scores_valid = pred_scores[valid_pred].cpu()
        else:
            pred_boxes_valid = torch.zeros(0, 4)
            pred_classes_valid = torch.zeros(0, dtype=torch.long)
            pred_scores_valid = torch.zeros(0)

        if visualize:
            # Visualize detections
            if pred_boxes_valid.shape[0] == 0:
                continue

            # Un-normalize image tensor for visualization
            image = inverse_norm(images[0]).cpu()

            # Combine predicted classes and scores into boxes for visualization
            pred_boxes_combined = torch.cat(
                [pred_boxes_valid, pred_classes_valid.unsqueeze(1).float(), pred_scores_valid.unsqueeze(1)], dim=1
            )

            detection_visualizer(
                image, idx_to_class, gt_boxes_valid, pred_boxes_combined
            )
        else:
            # Collect for mAP computation
            all_predictions.append((pred_boxes_valid, pred_classes_valid, pred_scores_valid))
            all_ground_truths.append(gt_boxes_single.cpu())

    # Compute and print mAP
    if not visualize:
        num_classes = len(idx_to_class)
        mAP, ap_per_class = compute_map(
            all_predictions,
            all_ground_truths,
            num_classes=num_classes,
            iou_threshold=0.5,
        )

        # Print per-class AP
        for class_idx in range(num_classes):
            class_name = idx_to_class[class_idx]
            ap = ap_per_class[class_idx] * 100
            print(f"{ap:.2f}% = {class_name} AP")

        # Print mAP
        print(f"mAP = {mAP * 100:.2f}%")

    end_t = time.time()
    print(f"Total inference time: {end_t-start_t:.1f}s")
