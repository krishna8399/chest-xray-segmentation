"""U-Net for medical image segmentation — built from scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample_blocks = nn.ModuleList()

        ch = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(ch, feature))
            ch = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.upsample_blocks.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i, (upsample, decoder) in enumerate(
            zip(self.upsample_blocks, self.decoder_blocks)
        ):
            x = upsample(x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    assert out.shape[2:] == dummy.shape[2:]
    print("✅ U-Net working correctly")
