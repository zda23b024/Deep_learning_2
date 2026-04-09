"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer for bounding box regression."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        freeze_encoder: bool = False,
    ):
        super(VGG11Localizer, self).__init__()

        # Encoder (feature extractor)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),  # [B, 512, 7, 7] → [B, 25088]

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(512, 4)  # [cx, cy, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Bounding boxes [B, 4] in pixel space (0–224)
            Format: [cx, cy, w, h]
        """
        # Extract features
        features = self.encoder(x)  # [B, 512, 7, 7]

        # Raw predictions
        bbox = self.regressor(features)  # [B, 4]

        # ✅ Constrain outputs to [0, 1]
        bbox = torch.sigmoid(bbox)

        # ✅ Scale to image size (224x224)
        bbox = bbox * 224.0

        return bbox
