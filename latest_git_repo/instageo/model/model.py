# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Model Module."""

import os
import time
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore
from absl import logging

from instageo.model.CoAtNet import CoAtNet 


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file."""
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class CoAtNetSeg(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        
        self.backbone = CoAtNet(
            image_size=(image_size, image_size),
            in_channels=3,
            num_blocks=[2, 2, 3, 5, 2],  # coatnet_0 configuration
            channels=[64, 96, 192, 384, 768],
            num_classes=num_classes,
            block_types=['C', 'C', 'T', 'T']
        )
        # Remove classification head
        self.backbone.pool = nn.Identity()
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3,
                    stride=2, padding=1, output_padding=1
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        embed_dims = [768 // (2**i) for i in range(6)]  # [768, 384, 192, 96, 48, 24]
        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(5)],
            nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        batch_size = img.shape[0]
        
        # Handle temporal dimension: average over time steps
        if img.ndim == 5:  # [batch_size, temporal_steps, channels, height, width]
            img = img.mean(dim=1)  # Average over temporal dimension
        elif img.ndim == 4:  # [batch_size, channels, height, width]
            pass  # Input already batched
        elif img.ndim == 3:  # [channels, height, width]
            img = img.unsqueeze(0)  # Add batch dimension
        elif img.ndim == 2:  # [height, width] -> assume grayscale; add channel and batch dims
            img = img.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected input shape: {img.shape}")

        # Pass through backbone
        features = self.backbone(img)

        # Now, ensure that the features have the right shape for the segmentation head.
        # If features is 2D, we assume it's a flattened spatial map with shape [num_tokens, channels].
        if features.ndim == 2:
            # At this point, features has shape [num_tokens, channels]
            # We expect channels to be 768 (to match segmentation_head's expected input).
            # For example, if num_tokens == 2048, we can reshape into a 2D grid of size 32 x 64.
            features = features.transpose(0, 1)  # Now shape [channels, num_tokens]
            features = features.view(batch_size, features.shape[0], 32, 64)  # [batch, channels, H, W]
        elif features.ndim == 3:
            # features with shape [channels, H, W] -> add batch dimension.
            features = features.unsqueeze(0)
        elif features.ndim != 4:
            raise ValueError(f"Unexpected features shape: {features.shape}")

        # Now features is a 4D tensor as required.
        return self.segmentation_head(features)

