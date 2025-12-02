from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from ops import fcos_make_centerness_targets


TensorDict = Dict[str, torch.Tensor]


class FCOSLoss(nn.Module):
    """
    Loss computation for FCOS detector. Computes three losses:
        1. Classification loss: Sigmoid focal loss
        2. Box regression loss: L1 loss on LTRB deltas
        3. Centerness loss: Binary cross entropy loss

    Background locations (where GT is -1) are excluded from box and centerness losses.
    """

    def __init__(self, num_classes: int, normalizer_momentum: float = 0.9, initial_normalizer: float = 150.0):
        """
        Args:
            num_classes: Number of object classes (excluding background).
            normalizer_momentum: Momentum for EMA update of loss normalizer.
            initial_normalizer: Initial value for the loss normalizer.
        """
        super().__init__()

        self.num_classes = num_classes
        self.normalizer_momentum = normalizer_momentum

        # Averaging factor for training loss; EMA of foreground locations.
        self._normalizer = initial_normalizer

    @property
    def normalizer(self) -> float:
        """Current normalizer value (EMA of foreground locations per image)."""
        return self._normalizer

    def forward(
        self,
        pred_cls_logits: torch.Tensor,
        pred_boxreg_deltas: torch.Tensor,
        pred_ctr_logits: torch.Tensor,
        matched_gt_boxes: torch.Tensor,
        matched_gt_deltas: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute FCOS losses.

        Args:
            pred_cls_logits: Predicted class logits, shape (B, N, num_classes)
                where N is total number of locations across all FPN levels.
            pred_boxreg_deltas: Predicted box deltas, shape (B, N, 4).
            pred_ctr_logits: Predicted centerness logits, shape (B, N, 1).
            matched_gt_boxes: Matched GT boxes, shape (B, N, 5).
                Last dimension is (x1, y1, x2, y2, class). Background is -1.
            matched_gt_deltas: Matched GT deltas, shape (B, N, 4).
                Background locations have deltas of -1.

        Returns:
            Dictionary with keys "loss_cls", "loss_box", "loss_ctr".
        """
        batch_size = pred_cls_logits.shape[0]

        # Update normalizer with EMA of foreground locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / batch_size
        self._normalizer = (
            self.normalizer_momentum * self._normalizer
            + (1 - self.normalizer_momentum) * pos_loc_per_image
        )

        ######################################################################
        # Classification loss
        ######################################################################
        # Extract target class labels. -1: background. 0-19: foreground.
        target_classes = matched_gt_boxes[:, :, 4].to(torch.long)
        target_classes += 1  # make class values non-negative for one_hot method

        # Convert ground-truth class labels to one-hot format.
        target_one_hot = F.one_hot(target_classes, num_classes=self.num_classes + 1)

        # Remove the background class in the one-hot representation.
        target_one_hot = target_one_hot[:, :, 1:].to(torch.float32)

        # Calculate losses for classification
        loss_cls = sigmoid_focal_loss(inputs=pred_cls_logits, targets=target_one_hot)

        ######################################################################
        # Box regression loss
        ######################################################################
        # Calculate loss for box regression.
        # Multiply with 0.25 to average across four LTRB components.
        loss_box = 0.25 * F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")

        # No loss for background
        loss_box[matched_gt_deltas < 0] *= 0.0

        ######################################################################
        # Centerness loss
        ######################################################################
        # Calculate loss for centerness
        B, N, M = matched_gt_deltas.shape
        gt_centerness = fcos_make_centerness_targets(
            matched_gt_deltas.reshape(-1, M)
        ).reshape(B, N, -1)

        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits, gt_centerness, reduction="none"
        )

        # No loss for background
        loss_ctr[gt_centerness < 0] *= 0.0

        ######################################################################
        # Normalize and return losses
        ######################################################################
        normalizer = self._normalizer * batch_size

        return {
            "loss_cls": loss_cls.sum() / normalizer,
            "loss_box": loss_box.sum() / normalizer,
            "loss_ctr": loss_ctr.sum() / normalizer,
        }
