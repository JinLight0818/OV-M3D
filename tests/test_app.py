"""Камергүйгээр үндсэн програмын горим солилт, дүрслэл ба агшин хадгалалтыг шалгана."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src import app
from src.models.detector import Detection
from src.sensor.esp32_tof import Packet


def test_live_toggle_and_snapshot(monkeypatch, tmp_path):
    cfg = yaml.safe_load(open("config/default_config.yaml"))
    cfg.update(snapshot_dir=str(tmp_path))
    cfg["fusion"].update(dx_m=0., dy_m=0.)
    frame = np.full((240, 320, 3), 45, np.uint8)
    cap = SimpleNamespace(isOpened=lambda: True, set=lambda *args: True,
                          read=lambda: (True, frame.copy()), release=lambda: None)
    monkeypatch.setattr(app.cv2, "VideoCapture", lambda source: cap)
    for name in ("namedWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(app.cv2, name, lambda *args: None)
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *args: 1)
    controls = iter("smsq")
    monkeypatch.setattr(app.cv2, "waitKey", lambda ms: ord(next(controls)))
    detect = lambda img: [Detection(i, "cup", (100 + i*15, 100, 112 + i*15, 120), 0.9)
                          for i in (1, 2)]
    monkeypatch.setattr(app, "Detector", lambda c: SimpleNamespace(
        device="CPU", prompts=("cup",), detect=detect))

    class Depth:
        device = "CPU"
        def estimate(self, image, index, fx):
            self.last_frame_index = index
            return np.ones(image.shape[:2], np.float32)

    monkeypatch.setattr(app, "DepthEstimator", lambda c: Depth())
    sensor = SimpleNamespace(start=lambda: None, close=lambda: None,
                             latest=lambda at: Packet("VL53L1X", 1, 2, (2000,), (0,), at))
    monkeypatch.setattr(app, "SerialReader", lambda **kwargs: sensor)
    app.run(cfg, 0)
    records = {rec["mode"]: rec for rec in [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]}
    assert set(records) == {"mono", "fusion"}
    for mode, z in [("mono", 1), ("fusion", 2)]:
        record = records[mode]
        assert record["objects"][0]["xyz_m"][2] == z
        assert record["pairs"][0]["distance_m"] == pytest.approx(15*z/300)
        assert record["pairs"][0]["ids"] == [1, 2]
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 2
    assert app.cv2.imread(str(files[0])).shape == (240, 560, 3)
