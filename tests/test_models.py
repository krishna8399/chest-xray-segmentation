"""Smoke tests for model forward passes and output shapes."""

import pytest
import torch

from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3Segmenter


@pytest.fixture
def dummy_batch():
    return torch.randn(2, 1, 256, 256)


class TestUNet:
    def test_output_shape(self, dummy_batch):
        model = UNet(in_channels=1, out_channels=1)
        out = model(dummy_batch)
        assert out.shape == (2, 1, 256, 256)

    def test_spatial_size_preserved(self, dummy_batch):
        model = UNet(in_channels=1, out_channels=1)
        out = model(dummy_batch)
        assert out.shape[2:] == dummy_batch.shape[2:]

    def test_custom_features(self, dummy_batch):
        model = UNet(in_channels=1, out_channels=1, features=[32, 64])
        out = model(dummy_batch)
        assert out.shape == (2, 1, 256, 256)

    def test_param_count(self):
        model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
        params = sum(p.numel() for p in model.parameters())
        assert 30_000_000 < params < 35_000_000


class TestDeepLabV3:
    def test_output_shape(self, dummy_batch):
        model = DeepLabV3Segmenter(pretrained=False, out_channels=1)
        out = model(dummy_batch)
        assert out.shape == (2, 1, 256, 256)

    def test_freeze_unfreeze(self, dummy_batch):
        model = DeepLabV3Segmenter(pretrained=False)
        model.freeze_backbone()
        for p in model.model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.model.backbone.parameters():
            assert p.requires_grad
