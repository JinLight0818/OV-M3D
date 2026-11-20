"""Хэмжилтийн хүснэгтийн бүтэц ба нэг мөр нэмэх үйлдэл."""

import csv
import fcntl
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASURE_FIELDS = (
    "snapshot", "track_id", "frame_index", "scene", "mode", "label",
    "x_m", "y_m", "z_m", "gt_distance_m", "distance_m",
    "gt_width_cm", "width_cm", "gt_height_cm", "height_cm",
)
BENCH_FIELDS = (
    "run_id", "video", "video_sha256", "config_sha256", "backend", "detector_device",
    "depth_device", "width", "height", "warmup", "frames", "depth_every_n_frames",
    "elapsed_s", "fps", "mean_ms", "p95_ms",
)


def positive(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} талбарт тоо оруулна уу.") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} талбарт тэгээс их төгсгөлөг тоо оруулна уу.")
    return number


def append_row(path, fields, row, unique=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        reader = csv.DictReader(stream)
        if reader.fieldnames is not None and tuple(reader.fieldnames) != fields:
            raise ValueError(f"{path} файлын баганууд хүснэгтийн бүтэцтэй зөрж байна.")
        if unique and any(all(old[k] == str(row[k]) for k in unique) for old in reader):
            raise ValueError("Энэ агшны сонгосон биетийг бүртгэсэн байна.")
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
