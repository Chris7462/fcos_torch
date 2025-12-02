from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from model.backbone import RegNetBackbone
from model.neck import FPN
from model.head import FCOSHead
from model.loss import FCOSLoss
from ops import (
    class_spec_nms,
    get_fpn_location_coords,
    fcos_match_locations_to_gt,
    fcos_get_deltas_from_locations,
    fcos_apply_deltas_to_locations,
)


TensorDict = Dict[str, torch.Tensor]


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything: a backbone with FPN, prediction head,
    and loss computation. It computes loss during training and predicts boxes
    during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # Initialize backbone, neck, head, and loss
        ######################################################################
        self.backbone = RegNetBackbone()
        self.neck = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
        )
        self.head = FCOSHead(
            num_classes=num_classes,
            in_channels=fpn_channels,
            stem_channels=stem_channels,
        )
        self.loss = FCOSLoss(num_classes=num_classes)

    @property
    def fpn_strides(self) -> Dict[str, int]:
        """Expose FPN strides from neck."""
        return self.neck.fpn_strides

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # Process the image through backbone, FPN, and prediction head
        ######################################################################
        backbone_feats = self.backbone(images)
        fpn_feats = self.neck(backbone_feats)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.head(fpn_feats)

        ######################################################################
        # Get absolute co-ordinates `(xc, yc)` for every location in FPN levels.
        ######################################################################
        shape_per_fpn_level = {
            level_name: feat.shape for level_name, feat in fpn_feats.items()
        }
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level,
            self.fpn_strides,
            dtype=images.dtype,
            device=images.device,
        )

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            return self.inference(
                images,
                locations_per_fpn_level,
                pred_cls_logits,
                pred_boxreg_deltas,
                pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )

        ######################################################################
        # Assign ground-truth boxes to feature locations.
        ######################################################################
        matched_gt_boxes = []
        matched_gt_deltas = []

        for i in range(images.shape[0]):
            # Match the GT boxes.
            matched_boxes = fcos_match_locations_to_gt(
                locations_per_fpn_level,
                self.fpn_strides,
                gt_boxes[i],
            )

            matched_gt_boxes.append(matched_boxes)

            # Calculate GT deltas for these matched boxes.
            gt_deltas = {
                level_name: fcos_get_deltas_from_locations(
                    locations_per_fpn_level[level_name],
                    matched_boxes[level_name],
                    self.fpn_strides[level_name],
                )
                for level_name in locations_per_fpn_level
            }

            matched_gt_deltas.append(gt_deltas)

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        ######################################################################
        # Compute losses
        ######################################################################
        return self.loss(
            pred_cls_logits,
            pred_boxreg_deltas,
            pred_ctr_logits,
            matched_gt_boxes,
            matched_gt_deltas,
        )

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )

            # Step 1: Get the most confident predicted class and its score for each box
            level_pred_scores, level_pred_classes = torch.max(level_pred_scores, dim=1)

            # Step 2: Only retain prediction that have a confidence score higher
            # than provided threshold in arguments.
            keep_idx = level_pred_scores > test_score_thresh
            level_pred_scores = level_pred_scores[keep_idx]
            level_pred_classes = level_pred_classes[keep_idx]
            level_locations = level_locations[keep_idx]
            level_deltas = level_deltas[keep_idx]

            # Step 3: Obtain predicted boxes using predicted deltas and locations
            stride = self.fpn_strides[level_name]
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas, level_locations, stride
            )

            # Step 4: Use `images` to get (height, width) for clipping.
            _, _, height, width = images.shape
            level_pred_boxes[:, 0::2] = level_pred_boxes[:, 0::2].clamp(min=0, max=width)
            level_pred_boxes[:, 1::2] = level_pred_boxes[:, 1::2].clamp(min=0, max=height)

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        ######################################################################
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]

        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
