from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) that takes multi-scale features from a
    backbone network and produces refined features at multiple scales.

    Takes features (c3, c4, c5) from backbone and produces (p3, p4, p5) with
    the same spatial dimensions but uniform channel dimension.

    Output feature strides (same as input):
        - p3: stride 8
        - p4: stride 16
        - p5: stride 32
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int):
        """
        Args:
            in_channels: Dictionary mapping feature level names to their
                channel dimensions, e.g., {"c3": 64, "c4": 160, "c5": 400}
            out_channels: Number of output channels for all FPN levels.
        """
        super().__init__()

        self.out_channels = out_channels

        # Lateral 1x1 convs to reduce channels to out_channels.
        self.lateral_c3 = nn.Conv2d(
            in_channels["c3"], out_channels, kernel_size=1, stride=1, padding=0
        )
        self.lateral_c4 = nn.Conv2d(
            in_channels["c4"], out_channels, kernel_size=1, stride=1, padding=0
        )
        self.lateral_c5 = nn.Conv2d(
            in_channels["c5"], out_channels, kernel_size=1, stride=1, padding=0
        )

        # Output 3x3 convs for FPN feature maps.
        self.output_p3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.output_p4 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.output_p5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def fpn_strides(self) -> Dict[str, int]:
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, backbone_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            backbone_feats: Dictionary with keys {"c3", "c4", "c5"} containing
                feature maps from the backbone at different scales.

        Returns:
            Dictionary with keys {"p3", "p4", "p5"} containing FPN feature maps.
        """
        # Extract backbone features
        c3, c4, c5 = backbone_feats["c3"], backbone_feats["c4"], backbone_feats["c5"]

        # FPN top-down pathway with lateral connections.
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        # FPN output layers.
        p5 = self.output_p5(p5)
        p4 = self.output_p4(p4)
        p3 = self.output_p3(p3)

        return {"p3": p3, "p4": p4, "p5": p5}
