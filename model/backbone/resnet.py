"""
ResNet backbone for feature extraction.
"""

from typing import Dict

import torch
from torch import nn
from torchvision import models
from torchvision.models import feature_extraction


class ResNetBackbone(nn.Module):
    """
    ResNet backbone that extracts multi-scale features from input images.
    Uses a pretrained ResNet-101 model and returns intermediate features
    from different stages (c3, c4, c5) for use with FPN.

    Output feature strides:
        - c3: stride 8
        - c4: stride 16
        - c5: stride 32
    """

    def __init__(self):
        super().__init__()

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "layer2": "c3",  # stride 8, 512 channels
                "layer3": "c4",  # stride 16, 1024 channels
                "layer4": "c5",  # stride 32, 2048 channels
            },
        )

        # Infer output channels for each level using a dummy forward pass.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        self._out_channels = {
            level_name: feat.shape[1] for level_name, feat in dummy_out.items()
        }

    @property
    def out_channels(self) -> Dict[str, int]:
        """
        Returns a dictionary of output channels for each feature level.
        e.g., {"c3": 512, "c4": 1024, "c5": 2048}
        """
        return self._out_channels

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Batch of images, tensor of shape (B, 3, H, W).

        Returns:
            Dictionary with keys {"c3", "c4", "c5"} containing feature maps
            at different scales.
        """
        return self.backbone(images)
