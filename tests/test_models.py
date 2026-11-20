"""Жин татахгүйгээр кадр алгасалт, ангилал солилт ба төхөөрөмжийн шилжилтийг шалгана."""

from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

from src.models import depth_estimator, detector
from src.models.runtime import Runtime


def test_depth_cache_focal_and_frame_size(monkeypatch):
    runtime = Mock(shape=(1, 3, 28, 42))
    runtime.return_value = np.full((1, 28, 42), 3., dtype=np.float32)
    monkeypatch.setattr(depth_estimator, "Runtime", lambda config: runtime)
    model = depth_estimator.DepthEstimator({
        "backend": "openvino", "input_height": 28, "input_width": 42,
        "train_focal_px": 40, "focal_reference_width": 80, "depth_every_n_frames": 2,
    })
    frame = np.zeros((30, 80, 3), np.uint8)
    first = model.estimate(frame, 0, fx=60)
    assert first.shape == (30, 80)
    np.testing.assert_allclose(first, 4.5)
    assert model.estimate(frame, 1, 60) is first
    second = model.estimate(frame, 2, 60)
    assert second is not first and model.last_frame_index == 2
    half_frame = frame[::2, ::2]
    third = model.estimate(half_frame, 3, 30)
    assert third.shape == (15, 40)
    np.testing.assert_allclose(third, 4.5)
    assert runtime.call_count == 3


@pytest.mark.parametrize("failure", ["compile", "infer"])
def test_gpu_fallback(monkeypatch, tmp_path, failure):
    attempted = []
    image = np.ones((1, 3, 28, 42), np.float32)

    def compile_model(model, device, options):
        attempted.append(device)
        if device == "GPU" and failure == "compile":
            raise RuntimeError("График төхөөрөмжид эрх хүрэхгүй байна.")
        compiled = Mock()
        request = compiled.create_infer_request.return_value
        if device == "GPU":
            request.infer.side_effect = RuntimeError("График төхөөрөмж салсан.")
        request.get_output_tensor.return_value.data = image[:, 0]
        compiled.input.return_value.shape = image.shape
        return compiled

    core = Mock(compile_model=compile_model)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    path = tmp_path / "depth.xml"
    path.touch()
    runtime = Runtime({"backend": "openvino", "device": "GPU", "ir_path": path})
    np.testing.assert_array_equal(runtime(image), image[:, 0])
    runtime(image)
    assert attempted == ["GPU", "CPU"]
    assert runtime.device == "CPU"


@pytest.fixture
def detector_setup(monkeypatch, tmp_path):
    tracker = Mock()
    dynamic = Mock()
    world = Mock(return_value=dynamic)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOWorld=world))
    monkeypatch.setitem(sys.modules, "ultralytics.trackers.byte_tracker", SimpleNamespace(BYTETracker=lambda args: tracker))
    monkeypatch.setitem(sys.modules, "ultralytics.engine.results", SimpleNamespace(Boxes=lambda data, shape: data))

    path = tmp_path / "detector.xml"
    path.with_suffix(".labels.json").write_text('["cup", "phone"]', encoding="utf-8")
    runtime = Mock(shape=(1, 3, 64, 64))
    monkeypatch.setattr(detector, "Runtime", lambda config: runtime)
    config = {
        "backend": "openvino", "device": "CPU", "ir_path": path, "weights": "test.pt",
        "image_size": 64, "prompts": ["cup", "phone"],
    }
    return config, runtime, tracker, dynamic, world


def test_vocabulary_cache(detector_setup):
    config, _, tracker, dynamic, world = detector_setup
    model = detector.Detector(config)
    model.set_prompts(["phone", "phone", " "])
    assert model.prompts == ("phone",)
    world.assert_not_called()
    model.set_prompts(["pear"])
    model.set_prompts([" pear "])
    model.set_prompts(["cup"])
    model.set_prompts(["pear"])
    dynamic.set_classes.assert_called_once_with(["pear"])
    world.assert_called_once()
    assert tracker.reset.call_count == 5
    assert not model.use_fixed


def test_detector_shape_empty_and_subset(detector_setup):
    config, runtime, tracker, _, _ = detector_setup
    model = detector.Detector(config)
    frame = np.zeros((32, 64, 3), np.uint8)
    tracker.update.return_value = []
    runtime.return_value = np.zeros((1, 6, 20), dtype=np.float32)
    assert model.detect(frame) == []
    assert tracker.update.call_args.args[0].shape == (0, 6)

    model.set_prompts(["phone"])
    runtime.return_value[0, :, 5] = [32, 32, 16, 12, 0.95, 0.85]
    tracker.update.side_effect = lambda boxes, image: [[*boxes[0, :4], 7, *boxes[0, 4:], 0]]
    tracks = model.detect(frame)
    assert len(tracks) == 1
    assert tracks[0].box == (24, 10, 40, 22)
    assert tracks[0].label == "phone"
    assert tracks[0].confidence == pytest.approx(0.85)
    assert tracks[0].track_id == 7

    runtime.shape = (1, 3, 128, 128)
    with pytest.raises(ValueError):
        detector.Detector(config)
