"""DeepLabV3+ with pretrained ResNet backbone for comparison."""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Segmenter(nn.Module):
    def __init__(self, pretrained=True, out_channels=1):
        super().__init__()
        self.model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)

        old_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.model.backbone.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True))

        self.model.classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=1)

    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)["out"]


if __name__ == "__main__":
    model = DeepLabV3Segmenter(pretrained=False, out_channels=1)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
