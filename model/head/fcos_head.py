import math
from typing import Dict, List

import torch
from torch import nn


TensorDict = Dict[str, torch.Tensor]


class FCOSHead(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        # Initialize stem_cls and stem_box as sequences of alternating Conv2d and ReLU layers.
        stem_cls = []
        stem_box = []
        prev_channels_cls = in_channels
        prev_channels_box = in_channels

        for out_channels in stem_channels:
            # For classification stem.
            conv_cls = nn.Conv2d(prev_channels_cls, out_channels, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(conv_cls.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv_cls.bias, 0.0)
            stem_cls.extend([conv_cls, nn.ReLU()])
            prev_channels_cls = out_channels

            # For box regression stem
            conv_box = nn.Conv2d(prev_channels_box, out_channels, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(conv_box.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv_box.bias, 0.0)
            stem_box.extend([conv_box, nn.ReLU()])
            prev_channels_box = out_channels

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Class prediction conv
        self.pred_cls = nn.Conv2d(stem_channels[-1], num_classes, kernel_size=3, stride=1, padding=1)

        # Box regression conv
        self.pred_box = nn.Conv2d(stem_channels[-1], 4, kernel_size=3, stride=1, padding=1)

        # Centerness conv
        self.pred_ctr = nn.Conv2d(stem_channels[-1], 1, kernel_size=3, stride=1, padding=1)

        # Initialize weights for prediction layers
        nn.init.normal_(self.pred_cls.weight, mean=0.0, std=0.01)
        # Use a negative bias in `pred_cls` to improve training stability.
        # Without this, the training will most likely diverge.
        nn.init.constant_(self.pred_cls.bias, -math.log(99))

        nn.init.normal_(self.pred_box.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred_box.bias, 0.0)

        nn.init.normal_(self.pred_ctr.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred_ctr.bias, 0.0)

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level_name, feature_map in feats_per_fpn_level.items():
            # Pass feature map through stems.
            cls_feats = self.stem_cls(feature_map)
            box_feats = self.stem_box(feature_map)

            # Predict class logits, box deltas, and centerness
            cls_logits = self.pred_cls(cls_feats)
            box_deltas = self.pred_box(box_feats)
            ctr_logits = self.pred_ctr(box_feats)

            # Reshape outputs to flatten spatial dimensions (H, W) into one dimension.
            batch_size, _, _, _ = cls_logits.shape
            class_logits[level_name] = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.pred_cls.out_channels)
            boxreg_deltas[level_name] = box_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, self.pred_box.out_channels)
            centerness_logits[level_name] = ctr_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.pred_ctr.out_channels)

        return [class_logits, boxreg_deltas, centerness_logits]
