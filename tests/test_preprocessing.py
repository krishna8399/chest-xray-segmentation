"""Tests for preprocessing functions."""

import numpy as np
import pytest

from src.data.preprocessing import apply_clahe, normalize_image, process_mask, combine_lung_masks


class TestCLAHE:
    def test_output_shape(self):
        img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        out = apply_clahe(img)
        assert out.shape == img.shape

    def test_rgb_input(self):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        out = apply_clahe(img)
        assert out.ndim == 2


class TestNormalize:
    def test_uint8_to_float(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        out = normalize_image(img)
        assert out.max() == pytest.approx(1.0, abs=1e-5)
        assert out.dtype == np.float32

    def test_already_normalized(self):
        img = np.random.rand(64, 64).astype(np.float32)
        out = normalize_image(img)
        assert out.max() <= 1.0


class TestProcessMask:
    def test_binary_output(self):
        mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        out = process_mask(mask)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_color_mask(self):
        mask = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        out = process_mask(mask)
        assert out.ndim == 2


class TestCombineMasks:
    def test_combined_is_union(self):
        left = np.zeros((64, 64), dtype=np.uint8)
        left[:, :32] = 255
        right = np.zeros((64, 64), dtype=np.uint8)
        right[:, 32:] = 255
        combined = combine_lung_masks(left, right)
        assert combined.max() == pytest.approx(1.0, abs=1e-5)
        assert combined.min() == pytest.approx(1.0, abs=1e-5)

    def test_no_double_counting(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        combined = combine_lung_masks(mask, mask)
        assert combined.max() == pytest.approx(1.0, abs=1e-5)
