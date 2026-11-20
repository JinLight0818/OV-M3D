"""Бодит төхөөрөмжгүйгээр өгөгдлийн бүртгэл, нэгж, хурдны хугацааг шалгана."""

import csv
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import benchmark_runtime, collect_dataset, evaluate_results


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_collect_and_evaluate(tmp_path, monkeypatch):
    shot = tmp_path / "агшин.json"
    record = dict(mode="fusion", frame_index=17, objects=[
        dict(track_id=1, label="cup", xyz_m=[0, 0, 2], distance_m=2, width_cm=8, height_cm=10),
        dict(track_id=2, label="phone", xyz_m=[3, 0, 4], distance_m=5, width_cm=9, height_cm=11),
    ])
    shot.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "measurements.csv"
    args = Namespace(snapshot=shot, track_id=2, scene="баруун_1", distance_m=4.5,
                     width_cm=10, height_cm=10, output=output)
    collect_dataset.capture(args)
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="бүртгэсэн"):
        collect_dataset.capture(args)
    assert output.read_bytes() == previous
    sample, = read_rows(output)
    assert sample["label"] == "phone" and sample["mode"] == "fusion"
    assert float(sample["z_m"]) == 4 and float(sample["distance_m"]) == 5

    monkeypatch.setattr(evaluate_results, "ROOT", tmp_path)
    target = tmp_path / "үр_дүн"
    monkeypatch.setattr(sys, "argv", ["үнэлгээ", "--measurements", str(output), "--output", str(target)])
    evaluate_results.main()
    summaries = {r["quantity"]: r for r in read_rows(target / "metrics.csv")}
    for name, unit, error, percent in (("distance_m", "м", 0.5, 100 / 9),
                                       ("width_cm", "см", 1, 10), ("height_cm", "см", 1, 10)):
        row = summaries[name]
        assert (row["source"], row["count"], row["unit"]) == ("бодит", "1", unit)
        assert float(row["mae"]) == float(row["rmse"]) == error
        assert float(row["mape_pct"]) == pytest.approx(percent)
    assert len(list(target.glob("*.png"))) == 3
    assert not (target / "synthetic").exists()


def test_metrics_multiple_observations():
    actual = evaluate_results.metrics(np.array([1, 2]), np.array([3, 1]))
    assert actual == pytest.approx((1.5, np.sqrt(2.5), 125))


@pytest.mark.parametrize("explicit", [False, True])
def test_artificial_isolated(tmp_path, explicit, monkeypatch):
    monkeypatch.setattr(evaluate_results, "ROOT", tmp_path)
    target = tmp_path / "results"
    target.mkdir()
    real_metrics = target / "metrics.csv"
    real_metrics.write_text("Энэ бол өмнөх бодит хэмжилтийн үр дүн.\n", encoding="utf-8")
    old = real_metrics.read_bytes()
    args = ["үнэлгээ", "--output", str(target)]
    if explicit:
        args.append("--synthetic")
    monkeypatch.setattr(sys, "argv", args)
    evaluate_results.main()
    assert real_metrics.read_bytes() == old
    assert not (tmp_path / "data").exists()
    folder = target / "synthetic"
    assert {r["source"] for r in read_rows(folder / "metrics.csv")} == {"хиймэл"}
    assert {f.name for f in folder.glob("*.png")} == {
        "distance_accuracy.png", "size_accuracy.png", "speed_comparison.png"
    }


@pytest.mark.parametrize("switch", [False, True])
def test_benchmark_window(monkeypatch, switch):
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    state = {"read": 0, "closed": False}
    depth_calls = []

    def read():
        state["read"] += 1
        return True, frame

    def estimate(image, index, fx):
        depth_calls.append((index, fx))
        if switch and index == 2:
            depth.device = "CPU"
        return frame[:, :, 0]

    camera = SimpleNamespace(isOpened=lambda: True, read=read, release=lambda: state.update(closed=True))
    monkeypatch.setattr(benchmark_runtime.cv2, "VideoCapture", lambda path: camera)
    stamps = iter([0, 10, 10, 11, 11, 14])
    monkeypatch.setattr(benchmark_runtime, "perf_counter", lambda: next(stamps))
    depth = SimpleNamespace(device="GPU", estimate=estimate)
    det = SimpleNamespace(detect=lambda image: [])
    args = ("бичлэг", det, depth, {"fx": 100, "width": 80}, 1, 2)
    if switch:
        with pytest.raises(benchmark_runtime.DeviceChanged):
            benchmark_runtime.measure(*args)
    else:
        result = benchmark_runtime.measure(*args)
        assert (result["width"], result["height"]) == (40, 30)
        assert result["elapsed_s"] == 4 and result["fps"] == 0.5
        assert result["mean_ms"] == 2000 and result["p95_ms"] == pytest.approx(2900)
    assert state == {"read": 3, "closed": True}
    assert depth_calls == [(0, 50), (1, 50), (2, 50)]
