"""Мэдэгдэж буй камераар проекцолсон самбараас фокусыг сэргээж шалгана."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

from src.calibration import camera_calibrate


def test_calibration_from_projected_boards(tmp_path, monkeypatch):
    board = np.kron((np.indices((7, 10)).sum(axis=0) % 2).astype("uint8") * 255,
                    np.ones((60, 60), np.uint8))
    corners = np.float32([[0, 0], [599, 0], [599, 419], [0, 419]])
    world = np.float32([[-125, -87.5, 0], [125, -87.5, 0], [125, 87.5, 0], [-125, 87.5, 0]])
    k = np.array([[600., 0, 319.5], [0, 600., 239.5], [0, 0, 1.]])
    rng = np.random.default_rng(12)
    for i in range(16):
        rotation = rng.uniform([-.48, -.45, -.17], [.48, .45, .17])
        translation = rng.uniform([-40, -30, 460], [40, 30, 680])
        projected = cv2.projectPoints(world, rotation, translation, k, None)[0].reshape(4, 2)
        transform = cv2.getPerspectiveTransform(corners, projected)
        frame = cv2.warpPerspective(board, transform, (640, 480), borderValue=165)
        assert cv2.imwrite(str(tmp_path / f"{i:02d}.png"), frame)
    output = tmp_path / "camera.yaml"
    monkeypatch.setattr(sys, "argv", ["camera_calibrate", "--images", str(tmp_path / "*.png"),
                                      "--square-mm", "25", "--output", str(output)])
    camera_calibrate.main()
    result = yaml.safe_load(output.read_text())
    assert result["calibrated"]
    assert result["rms_px"] < 0.5
    assert result["fx"] == pytest.approx(600., rel=0.02)
    assert result["fy"] == pytest.approx(600., rel=0.02)
    assert len(result["per_view_rms_px"]) == 16
    intrinsics, maps = camera_calibrate.rectification(result, 320, 240)
    assert intrinsics.fx == result["fx"] / 2
    assert maps[0].shape[:2] == (240, 320)
