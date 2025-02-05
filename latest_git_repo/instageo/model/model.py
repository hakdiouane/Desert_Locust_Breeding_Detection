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

from instageo.model.CoAtNet import CoAtNet  # Import CoAtNet instead of ViTEncoder


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
    """A normalization layer for 2D inputs."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class CoAtNetSeg(nn.Module):
    """CoAtNet Segmentation Model."""
    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        in_channels = 3  # Assuming 3 input channels per timestep
        self.temporal_step = temporal_step
        self.image_size = image_size
        self.num_classes = num_classes

        # Initialize CoAtNet backbone
        self.coatnet_backbone = CoAtNet(
            image_size=(image_size, image_size),
            in_channels=in_channels * temporal_step,
            num_blocks=[2, 2, 3, 5, 2],  # From coatnet_0 configuration
            channels=[64, 96, 192, 384, 768],
            num_classes=0,  # Disable classification head
            block_types=['C', 'C', 'T', 'T']
        )
        # Remove unused layers from CoAtNet
        del self.coatnet_backbone.pool
        del self.coatnet_backbone.fc

        if freeze_backbone:
            for param in self.coatnet_backbone.parameters():
                param.requires_grad = False

        # Segmentation head
        initial_channels = 768  # Output channels from CoAtNet's last stage
        embed_dims = [initial_channels // (2 ** i) for i in range(6)]  # [768, 384, 192, 96, 48, 24]

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

        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(5)],
            nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = img.shape
        # Combine temporal and channel dimensions
        x = img.view(B, C * T, H, W)
        
        # Forward through CoAtNet stages
        x = self.coatnet_backbone.s0(x)
        x = self.coatnet_backbone.s1(x)
        x = self.coatnet_backbone.s2(x)
        x = self.coatnet_backbone.s3(x)
        x = self.coatnet_backbone.s4(x)
        
        # Upsample to original resolution
        return self.segmentation_head(x)