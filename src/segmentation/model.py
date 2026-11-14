"""U-Net and U-Net++ architectures for Bone and Erosion Segmentation (PRD Section 8.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double 3x3 convolution with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Medical U-Net encoder-decoder for hand bone & erosion segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_ch: int = 24) -> None:
        super().__init__()
        self.architecture = "unet"
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class UNetPlusPlus(nn.Module):
    """Nested U-Net (U-Net++) with dense skip pathways for fine cortical erosion segmentation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_ch: int = 20) -> None:
        super().__init__()
        self.architecture = "unetplusplus"
        self.pool = nn.MaxPool2d(2, 2)

        self.x0_0 = ConvBlock(in_channels, base_ch)
        self.x1_0 = ConvBlock(base_ch, base_ch * 2)
        self.x2_0 = ConvBlock(base_ch * 2, base_ch * 4)

        self.x0_1 = ConvBlock(base_ch + base_ch * 2, base_ch)
        self.x1_1 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2)
        self.x0_2 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x0_1 = self.x0_1(
            torch.cat(
                [x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode="bilinear", align_corners=False)],
                dim=1,
            )
        )

        x2_0 = self.x2_0(self.pool(x1_0))
        x1_1 = self.x1_1(
            torch.cat(
                [x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode="bilinear", align_corners=False)],
                dim=1,
            )
        )
        x0_2 = self.x0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    F.interpolate(x1_1, size=x0_0.shape[2:], mode="bilinear", align_corners=False),
                ],
                dim=1,
            )
        )
        return self.out_conv(x0_2)


def build_segmentation_model(
    architecture: str = "unet",
    in_channels: int = 1,
    num_classes: int = 4,
) -> nn.Module:
    """Factory function for U-Net and U-Net++ segmentation models."""
    if architecture.lower() in {"unetplusplus", "unet++"}:
        return UNetPlusPlus(in_channels=in_channels, num_classes=num_classes)
    return UNet(in_channels=in_channels, num_classes=num_classes)
