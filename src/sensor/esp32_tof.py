"""ToF (гэрлийн нисэлтийн хугацаагаар зай хэмжих) өгөгдлийг уншиж гүнийг засна."""

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.geometry.spatial_3d import Intrinsics

LOG = logging.getLogger(__name__)
MAX_LINE = 2048


@dataclass(frozen=True)
class Packet:
    sensor: str
    seq: int
    t_ms: int
    mm: tuple[int, ...]
    status: tuple[int, ...]
    received_at: float


def parse_packet(line: bytes, received_at: float) -> Packet:
    """Хувилбар, урт, бүхэл тоон талбаруудыг төхөөрөмжийн зурвастай тулгана."""
    if len(line) > MAX_LINE:
        raise ValueError("Мэдрэгчийн мөр хэт урт байна.")
    obj = json.loads(line)
    if not isinstance(obj, dict) or type(obj.get("v")) is not int or obj["v"] != 1:
        raise ValueError("Мэдрэгчийн багцын хувилбар танигдсангүй.")
    sensor = obj.get("sensor")
    if sensor not in ("VL53L1X", "VL53L5CX"):
        raise ValueError("Мэдрэгчийн нэр буруу байна.")
    count = 1 if sensor == "VL53L1X" else 64
    for field in ("seq", "t_ms"):
        if type(obj.get(field)) is not int or not 0 <= obj[field] <= 0xffffffff:
            raise ValueError("Багцын дугаар эсвэл хугацаа буруу байна.")
    for field, lower, upper in (("mm", -32768, 65535), ("status", 0, 255)):
        values = obj.get(field)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError("Мэдрэгчийн бүсийн тоо буруу байна.")
        if any(type(n) is not int or not lower <= n <= upper for n in values):
            raise ValueError("Мэдрэгчийн утга зөвшөөрсөн мужаас гарлаа.")
    return Packet(sensor, obj["seq"], obj["t_ms"], tuple(obj["mm"]), tuple(obj["status"]), received_at)


class SerialReader:
    """USB Serial (цуваа холболт)-ийн сүүлийн багцуудыг тусдаа урсгалд хадгална."""

    def __init__(self, port=None, baud=921600, max_age_s=0.25):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Багцын зөвшөөрөх нас эерэг байна.")
        self.port, self.baud, self.max_age_s = port, baud, max_age_s
        self._queue = deque(maxlen=256)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._read, name="esp32-tof", daemon=True)
            self._thread.start()
        return self

    def _read(self):
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            LOG.info("Цуваа холболтын сан суулгаагүй тул дан камераар ажиллана.")
            return
        try:
            port = self.port
            if not port or port == "auto":
                candidates = sorted(p.device for p in list_ports.comports()
                                    if p.vid in (0x303a, 0x10c4, 0x1a86, 0x0403))
                if len(candidates) != 1:
                    LOG.info("Мэдрэгчийн оролтыг тохиргоонд зааж болно. Дан камераар ажиллана.")
                    return
                port = candidates[0]
            with serial.Serial(port, self.baud, timeout=0.15) as stream:
                while not self._stop.is_set():
                    line = stream.read_until(b"\n", MAX_LINE + 1)
                    if not line.endswith(b"\n"):
                        continue
                    try:
                        packet = parse_packet(line, time.monotonic())
                    except (ValueError, UnicodeError, RecursionError):
                        continue
                    with self._lock:
                        self._queue.append(packet)
        except (OSError, ValueError, serial.SerialException):
            LOG.info("Мэдрэгчийн холболт олдсонгүй эсвэл тасарлаа. Дан камераар ажиллана.")
        finally:
            with self._lock:
                self._queue.clear()

    def latest(self, at=None) -> Packet | None:
        """Камерын кадр ирэх агшинд хамгийн ойр, зөвшөөрсөн хугацаан дахь багцыг авна."""
        at = time.monotonic() if at is None else at
        with self._lock:
            if not self._queue:
                return None
            nearest = min(reversed(self._queue), key=lambda pkt: abs(pkt.received_at - at))
        return nearest if abs(nearest.received_at - at) <= self.max_age_s else None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


class ToFFusion:
    """Зэрэгцээ оптик тэнхлэгтэй бэхэлгээнд тохируулгын сүүлийн утгыг хадгална."""

    def __init__(self, dx_m=0.0, dy_m=0.0, hfov_deg=45.0, vfov_deg=45.0,
                 rotation_deg=0, min_zones=6, min_spread_m=0.05):
        if not 0 < hfov_deg < 180 or not 0 < vfov_deg < 180:
            raise ValueError("Мэдрэгчийн харах өнцөг буруу байна.")
        if not 2 <= min_zones <= 64 or not min_spread_m > 0:
            raise ValueError("Бүсийн тоо эсвэл гүний доод хэлбэлзэл буруу байна.")
        if not all(map(math.isfinite, (dx_m, dy_m, rotation_deg, min_spread_m))):
            raise ValueError("Бэхэлгээний тохиргоог шалгана уу.")
        self.dx_m, self.dy_m = dx_m, dy_m
        # Пикселийн координатад эерэг эргэлт цагийн зүүний дагуу байна.
        theta = math.radians(rotation_deg)
        grid = (np.arange(8) + 0.5) / 4 - 1
        x, y = np.meshgrid(grid * math.tan(math.radians(hfov_deg / 2)),
                           grid * math.tan(math.radians(vfov_deg / 2)))
        self.rays_x = (x * math.cos(theta) - y * math.sin(theta)).ravel()
        self.rays_y = (x * math.sin(theta) + y * math.cos(theta)).ravel()
        self.min_zones = min_zones
        self.min_spread_m = min_spread_m
        self.scale, self.shift = 1.0, 0.0
        self.calibrated = False
        self._last_packet = None

    def update(self, depth: np.ndarray, packet: Packet | None, k: Intrinsics) -> bool:
        """Хүчингүй төлөвтэй багцыг бүхлээр нь алгасаж өмнөх коэффициентийг хадгална."""
        if packet is None or any(packet.status):
            return False
        identity = (packet.sensor, packet.seq, packet.t_ms)
        if identity == self._last_packet:
            return False
        target = np.asarray(packet.mm, dtype=float) / 1000
        single = packet.sensor == "VL53L1X"
        if np.any(target <= 0) or np.any(target > 4):
            return False
        # Олон бүсийн мэдрэгчийн дотоод програм тэнхлэгт буулгасан гүнийг өгдөг.
        rx, ry = (0.0, 0.0) if single else (self.rays_x, self.rays_y)
        u = np.rint(k.cx + k.fx * (rx + self.dx_m / target)).astype(int)
        v = np.rint(k.cy + k.fy * (ry + self.dy_m / target)).astype(int)
        height, width = depth.shape
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ai, target = depth[v[inside], u[inside]], target[inside]
        valid = np.isfinite(ai) & (ai > 0)
        ai, target = ai[valid], target[valid]
        if single:
            if not ai.size:
                return False
            scale, shift = target[0] / ai[0], 0.0
        else:
            if len(ai) < self.min_zones or np.ptp(ai) < self.min_spread_m:
                return False
            design = np.column_stack((ai, np.ones(len(ai))))
            fit, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            if rank < 2 or np.ptp(target) < self.min_spread_m:
                return False
            scale, shift = fit
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(shift):
            return False
        self.scale, self.shift = float(scale), float(shift)
        self.calibrated, self._last_packet = True, identity
        return True

    def apply(self, depth: np.ndarray) -> np.ndarray:
        result = np.asarray(depth, dtype=np.float32) * self.scale + self.shift
        valid = np.isfinite(result) & (depth > 0) & (result > 0)
        return np.where(valid, result, np.nan)
