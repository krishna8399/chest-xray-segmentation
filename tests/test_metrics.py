"""Tests for segmentation metrics."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import dice_score, iou_score, sensitivity, specificity, compute_batch_metrics


def ones(shape):
    return np.ones(shape, dtype=np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


class TestDiceScore:
    def test_perfect_overlap(self):
        mask = ones((256, 256))
        assert dice_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        pred = ones((256, 256))
        target = zeros((256, 256))
        assert dice_score(pred, target) < 0.01

    def test_half_overlap(self):
        pred = np.zeros((100,), dtype=np.float32)
        pred[:50] = 1.0
        target = np.zeros((100,), dtype=np.float32)
        target[25:75] = 1.0
        score = dice_score(pred, target)
        assert 0.4 < score < 0.7


class TestIoU:
    def test_perfect(self):
        mask = ones((128, 128))
        assert iou_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_iou_less_than_dice(self):
        pred = ones((100,))
        pred[50:] = 0.0
        target = ones((100,))
        assert iou_score(pred, target) < dice_score(pred, target)


class TestSensitivitySpecificity:
    def test_all_correct(self):
        mask = ones((64, 64))
        assert sensitivity(mask, mask) == pytest.approx(1.0, abs=1e-5)
        bg = zeros((64, 64))
        assert specificity(bg, bg) == pytest.approx(1.0, abs=1e-5)

    def test_all_false_positives(self):
        pred = ones((64, 64))
        target = zeros((64, 64))
        assert specificity(pred, target) < 0.01


class TestBatchMetrics:
    def test_returns_all_keys(self):
        preds = torch.zeros(4, 1, 64, 64)
        targets = torch.zeros(4, 1, 64, 64)
        m = compute_batch_metrics(preds, targets)
        assert set(m.keys()) == {"dice", "iou", "sensitivity", "specificity", "pixel_accuracy"}

    def test_sigmoid_applied(self):
        large_logits = torch.ones(2, 1, 64, 64) * 10.0
        targets = torch.ones(2, 1, 64, 64)
        m = compute_batch_metrics(large_logits, targets)
        assert m["dice"] > 0.99
