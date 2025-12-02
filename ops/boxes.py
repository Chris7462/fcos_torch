import torch


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    # Sort the boxes by scores in descending order
    sorted_indices = scores.argsort(descending=True)
    boxes = boxes[sorted_indices]

    keep = []
    while len(boxes) > 0:
        # Select the box with the highest score
        keep.append(sorted_indices[0].item())
        #if len(boxes) == 1:
        #    break

        # Compute IoU between the selected box and the rest
        base = boxes[0]
        rest = boxes[1:]

        # Compute Intersection
        x1 = torch.maximum(base[0], rest[:,0])
        y1 = torch.maximum(base[1], rest[:,1])
        x2 = torch.minimum(base[2], rest[:,2])
        y2 = torch.minimum(base[3], rest[:,3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Compute areas
        base_area = (base[2] - base[0]) * (base[3] - base[1])
        rest_area = (rest[:,2] - rest[:,0]) * (rest[:,3] - rest[:,1])

        # Compute union
        union = base_area + rest_area - intersection

        # Compute IoU
        iou = intersection / union

        # Suppress boxes with IoU greater than the threshold
        remain_indices = torch.where(iou <= iou_threshold)[0] + 1 # +1 to adjust index

        # Update boxes and scores
        sorted_indices = sorted_indices[remain_indices]
        boxes = boxes[remain_indices]

    keep = torch.tensor(keep, dtype=torch.long)

    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
