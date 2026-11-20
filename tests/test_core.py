"""Төхөөрөмж шаардахгүй геометр, шүүлтүүр, багц ба мэдрэгчийн нэгтгэлийн сорил."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from src.geometry.spatial_3d import (
    Intrinsics, Measurement, TrackSmoother, back_project, distance_3d, measure_object,
)
from src.sensor.esp32_tof import Packet, SerialReader, ToFFusion, parse_packet


CAMERA = Intrinsics(100.0, 120.0, 50.0, 50.0)


def test_pinhole_projection_and_distances():
    assert back_project(60, 38, 2, CAMERA) == pytest.approx((0.2, -0.2, 2))
    depth = np.full((100, 100), 2.0)
    obj = measure_object(depth, [50, 32, 70, 44], CAMERA)
    assert (obj.width_cm, obj.height_cm) == pytest.approx((40, 20))
    assert obj.distance_m == pytest.approx(math.sqrt(4.08))
    other = measure_object(depth, [60, 44, 80, 56], CAMERA)
    assert distance_3d(obj, other) == pytest.approx(math.hypot(.2, .2))


def test_foreground_and_missing_depth():
    depth = np.full((100, 100), 6.0)
    depth[34:66, 34:66] = 1.0
    depth[40:45, 43:46] = np.nan
    assert measure_object(depth, [10, 10, 90, 90], CAMERA).z == 1
    depth[33:67, 33:67] = -1
    assert measure_object(depth, [10, 10, 90, 90], CAMERA) is None
    assert measure_object(depth, [-20, 5, -10, 10], CAMERA) is None


@pytest.mark.parametrize("method", ["ema", "kalman"])
def test_track_smoother_keeps_tracks_separate(method):
    smoother = TrackSmoother(method=method, alpha=0.25, process_noise=0.0, measurement_noise=0.1)
    first = Measurement(0, 0, 1, 1, 10, 20)
    second = Measurement(0, 0, 3, 3, 30, 60)
    assert smoother.update(1, first) == first
    assert smoother.update(2, second) == second
    observed = smoother.update(1, second)
    expected = 1.5 if method == "ema" else 2.0
    assert observed.z == pytest.approx(expected)
    assert observed.distance_m == observed.z
    assert observed.width_cm == pytest.approx(expected * 10)
    assert observed.height_cm == pytest.approx(expected * 20)
    smoother.retain([2])
    assert smoother.update(1, first) == first
    smoother.clear()
    assert smoother.update(2, first) == first


def test_single_zone_parallax_and_last_good_scale():
    depth = np.full((100, 100), 3.0)
    # Мэдрэгч камерын баруун талд арван сантиметр, доор таван сантиметрт байна.
    depth[53, 55] = 1.0
    packet = Packet("VL53L1X", 1, 500, (2000,), (0,), 10)
    fusion = ToFFusion(dx_m=0.10, dy_m=0.05)
    assert fusion.update(depth, packet, CAMERA)
    assert fusion.scale == 2
    assert not fusion.update(depth, packet, CAMERA)
    bad = replace(packet, seq=2, mm=(500,), status=(4,))
    assert not fusion.update(depth, bad, CAMERA)
    assert not fusion.update(depth, None, CAMERA)
    assert (fusion.scale, fusion.shift) == (2, 0)
    assert fusion.apply(depth)[50, 50] == 6


def test_grid_recovers_affine_fit_and_rejects_bad_frame():
    fusion = ToFFusion()
    depth = np.zeros((100, 100))
    slopes = ((np.arange(8) + .5) / 4 - 1) * math.tan(math.pi / 8)
    distances = []
    for row, ry in enumerate(slopes):
        for col, rx in enumerate(slopes):
            v = round(50 + 120 * ry)
            u = round(50 + 100 * rx)
            z = 1 + (8 * row + col) / 100
            depth[v, u] = z
            distances.append(round(1000 * (1.2 * z + 0.15)))
    packet = Packet("VL53L5CX", 1, 1000, tuple(distances), (0,) * 64, 11)
    assert fusion.update(depth, packet, CAMERA)
    assert (fusion.scale, fusion.shift) == pytest.approx((1.2, 0.15))
    damaged = replace(packet, seq=2, status=(0,) * 63 + (255,))
    assert not fusion.update(depth, damaged, CAMERA)
    assert not fusion.update(np.ones_like(depth), replace(packet, seq=3), CAMERA)
    assert (fusion.scale, fusion.shift) == pytest.approx((1.2, 0.15))
    assert np.isnan(fusion.apply(depth)[0, 0])


def test_received_packet_shape_and_age():
    raw = dict(v=1, sensor="VL53L1X", seq=8, t_ms=500, mm=[2000], status=[0])
    p = parse_packet(json.dumps(raw).encode(), received_at=12.0)
    assert p.mm == (2000,)
    for update in ({"mm": [1, 2]}, {"sensor": "VL53L5CX"}, {"status": [False]}, {"v": 2}):
        with pytest.raises(ValueError):
            parse_packet(json.dumps(raw | update).encode(), 12)
    reader = SerialReader(max_age_s=0.15)
    reader._queue.extend([p, replace(p, seq=9, received_at=12.10)])
    assert reader.latest(12.02) is p
    assert reader.latest(12.11).seq == 9
    assert reader.latest(13) is None


def test_reader_without_hardware():
    reader = SerialReader(port="/эх-үүсвэргүй/мэдрэгч").start()
    reader.close()
    assert reader.latest() is None
