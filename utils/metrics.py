"""
Evaluation metrics for object detection.
"""

import torch
from typing import Dict, List, Tuple


def compute_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between one box and multiple boxes.

    Args:
        box: Tensor of shape (4,) with (x1, y1, x2, y2)
        boxes: Tensor of shape (N, 4) with (x1, y1, x2, y2)

    Returns:
        Tensor of shape (N,) with IoU values
    """
    # Intersection
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / union


def compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """
    Compute Average Precision using 11-point interpolation (VOC style).

    Args:
        recalls: Tensor of recall values
        precisions: Tensor of precision values

    Returns:
        Average Precision value
    """
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max().item()
    return ap / 11


def compute_ap_per_class(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision for a single class.

    Args:
        pred_boxes: List of predicted boxes per image, each (N, 4)
        pred_scores: List of predicted scores per image, each (N,)
        gt_boxes: List of ground truth boxes per image, each (M, 4)
        iou_threshold: IoU threshold for matching

    Returns:
        Average Precision for this class
    """
    # Gather all predictions with image indices
    all_scores = []
    all_image_ids = []
    all_pred_boxes = []

    for img_id, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        if len(scores) > 0:
            all_scores.append(scores)
            all_image_ids.extend([img_id] * len(scores))
            all_pred_boxes.append(boxes)

    if len(all_scores) == 0:
        return 0.0

    all_scores = torch.cat(all_scores)
    all_pred_boxes = torch.cat(all_pred_boxes)
    all_image_ids = torch.tensor(all_image_ids)

    # Sort by score (descending)
    sorted_indices = torch.argsort(all_scores, descending=True)
    all_scores = all_scores[sorted_indices]
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_image_ids = all_image_ids[sorted_indices]

    # Track which GT boxes have been matched
    gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in gt_boxes]

    # Count total GT boxes
    total_gt = sum(len(gt) for gt in gt_boxes)
    if total_gt == 0:
        return 0.0

    # Compute TP/FP for each prediction
    tp = torch.zeros(len(all_scores))
    fp = torch.zeros(len(all_scores))

    for i, (pred_box, img_id) in enumerate(zip(all_pred_boxes, all_image_ids)):
        img_id = img_id.item()
        gt = gt_boxes[img_id]

        if len(gt) == 0:
            fp[i] = 1
            continue

        # Compute IoU with all GT boxes in this image
        ious = compute_iou(pred_box, gt)
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and not gt_matched[img_id][max_idx]:
            tp[i] = 1
            gt_matched[img_id][max_idx] = True
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return compute_ap(recalls, precisions)


def compute_map(
    predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ground_truths: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean Average Precision across all classes.

    Args:
        predictions: List of (pred_boxes, pred_classes, pred_scores) per image
            - pred_boxes: (N, 4) tensor
            - pred_classes: (N,) tensor
            - pred_scores: (N,) tensor
        ground_truths: List of GT boxes per image, each (M, 5) with (x1, y1, x2, y2, class)
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (mAP, dict of AP per class)
    """
    # Organize predictions and GT by class
    pred_boxes_per_class = {c: [] for c in range(num_classes)}
    pred_scores_per_class = {c: [] for c in range(num_classes)}
    gt_boxes_per_class = {c: [] for c in range(num_classes)}

    # Process each image
    for img_idx, (preds, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes, pred_classes, pred_scores = preds

        # Filter out invalid GT boxes (marked as -1)
        valid_gt = gt[:, 4] >= 0
        gt = gt[valid_gt]

        # Organize predictions by class for this image
        for c in range(num_classes):
            # Predictions for this class
            mask = pred_classes == c
            pred_boxes_per_class[c].append(pred_boxes[mask] if mask.any() else torch.zeros(0, 4))
            pred_scores_per_class[c].append(pred_scores[mask] if mask.any() else torch.zeros(0))

            # GT for this class
            gt_mask = gt[:, 4] == c
            gt_boxes_per_class[c].append(gt[gt_mask, :4] if gt_mask.any() else torch.zeros(0, 4))

    # Compute AP for each class
    ap_per_class = {}
    for c in range(num_classes):
        ap = compute_ap_per_class(
            pred_boxes_per_class[c],
            pred_scores_per_class[c],
            gt_boxes_per_class[c],
            iou_threshold=iou_threshold,
        )
        ap_per_class[c] = ap

    # Compute mAP (only over classes that have GT)
    classes_with_gt = [c for c in range(num_classes) if sum(len(gt) for gt in gt_boxes_per_class[c]) > 0]
    if len(classes_with_gt) == 0:
        return 0.0, ap_per_class

    mAP = sum(ap_per_class[c] for c in classes_with_gt) / len(classes_with_gt)

    return mAP, ap_per_class
