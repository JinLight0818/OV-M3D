"""Камерын гажилт зассан дүрс ба метрээр өгөгдсөн тэнхлэгийн гүнээс хэмжээс олно."""

from dataclasses import dataclass
from math import ceil, floor, hypot, isfinite
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole Camera Model (нүхэн камерын загвар)-ийн пикселээр өгсөн тогтмолууд."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if not all(map(isfinite, (self.fx, self.fy, self.cx, self.cy))) or min(self.fx, self.fy) <= 0:
            raise ValueError("Камерын фокусын урт эерэг, бүх тогтмол төгсгөлөг байна.")


@dataclass(frozen=True)
class Measurement:
    x: float
    y: float
    z: float
    distance_m: float
    width_cm: float
    height_cm: float


def back_project(u: float, v: float, z: float, k: Intrinsics) -> tuple[float, float, float]:
    """Камерын төвөөс баруун, доош, урагш чиглэсэн координатыг метрээр олно."""
    return (u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z


def distance_3d(first: Measurement, second: Measurement) -> float:
    return hypot(first.x - second.x, first.y - second.y, first.z - second.z)


def measure_object(depth: np.ndarray, box: Sequence[float], k: Intrinsics,
                   fraction: float = 0.38, percentile: float = 30) -> Measurement | None:
    """Хүрээний төв хэсгийн гүнийг ашиглан урд хавтгайд харагдах хэмжээг олно."""
    if depth.ndim != 2 or not 0 < fraction <= 1 or not 0 <= percentile <= 100:
        raise ValueError("Гүний зураг, түүврийн хувь эсвэл хувийн эрэмбэ буруу байна.")
    height, width = depth.shape
    x1, y1, x2, y2 = box
    if not all(map(isfinite, box)):
        return None
    x1, x2 = max(x1, 0), min(x2, width)
    y1, y2 = max(y1, 0), min(y2, height)
    if x2 <= x1 or y2 <= y1:
        return None
    du, dv = x2 - x1, y2 - y1
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    region = depth[floor(v - dv * fraction / 2):ceil(v + dv * fraction / 2),
                   floor(u - du * fraction / 2):ceil(u + du * fraction / 2)]
    samples = region[np.isfinite(region) & (region > 0)]
    if not samples.size:
        return None
    z = float(np.percentile(samples, percentile))
    x, y, _ = back_project(u, v, z, k)
    return Measurement(x, y, z, hypot(x, y, z), 100 * du * z / k.fx, 100 * dv * z / k.fy)


class TrackSmoother:
    """Объект тус бүрийн координат, хэмжээг метрээр жигдрүүлж, зайг дахин олно."""

    def __init__(self, method="ema", alpha=0.35, process_noise=0.001, measurement_noise=0.02):
        if method not in ("ema", "kalman") or not 0 < alpha <= 1:
            raise ValueError("Шүүлтүүрийн арга эсвэл жин буруу байна.")
        if not isfinite(process_noise) or process_noise < 0 or not isfinite(measurement_noise) or measurement_noise <= 0:
            raise ValueError("Дуу чимээний дисперсийг шалгана уу.")
        self.method = method
        self.alpha = alpha
        self.q = process_noise
        self.r = measurement_noise
        self._states = {}

    def update(self, track_id: int, measurement: Measurement) -> Measurement:
        current = np.array([measurement.x, measurement.y, measurement.z,
                            measurement.width_cm / 100, measurement.height_cm / 100])
        variance = self.r
        if track_id in self._states:
            previous, old_variance = self._states[track_id]
            gain = self.alpha
            if self.method == "kalman":
                # Тогтмол төлөвийн загварт таван үл хамаарах тэнхлэг ижил дисперстэй.
                predicted_variance = old_variance + self.q
                gain = predicted_variance / (predicted_variance + self.r)
                variance = (1 - gain) * predicted_variance
            current = previous + gain * (current - previous)
        self._states[track_id] = (current, variance)
        x, y, z, width, height = map(float, current)
        return Measurement(x, y, z, hypot(x, y, z), width * 100, height * 100)

    def retain(self, live_ids: Iterable[int]):
        for key in self._states.keys() - set(live_ids):
            del self._states[key]

    def clear(self):
        self._states.clear()
