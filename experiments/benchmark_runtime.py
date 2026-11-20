"""Нэг видеоны ижил кадрыг танигч ба гүний загвараар зэрэгцээ боловсруулж хугацааг хэмжинэ."""

import hashlib
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import ArgumentParser
from experiments._records import BENCH_FIELDS, ROOT, append_row


class DeviceChanged(RuntimeError):
    """Хэмжих явцад гүний загварын төхөөрөмж өөрчлөгдсөнийг илэрхийлнэ."""


def measure(video, detector, depth, camera, warmup, count):
    reader = cv2.VideoCapture(str(video))
    try:
        if not reader.isOpened():
            raise OSError(f"{video} видеог нээж чадсангүй.")
        times = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            for index in range(warmup + count):
                ok, frame = reader.read()
                if not ok:
                    raise ValueError(f"Видео дор хаяж {warmup + count} уншигдах кадртай байх ёстой.")
                height, width = frame.shape[:2]
                fx = camera["fx"] * width / camera["width"]
                before = depth.device
                start = perf_counter()
                future = pool.submit(depth.estimate, frame, index, fx)
                detector.detect(frame)
                future.result()
                elapsed = perf_counter() - start
                if index >= warmup:
                    if before != depth.device:
                        raise DeviceChanged
                    times.append(elapsed)
        total = sum(times)
        return dict(width=width, height=height, elapsed_s=total, fps=count / total,
                    mean_ms=total * 1000 / count, p95_ms=float(np.percentile(times, 95)) * 1000)
    finally:
        reader.release()


def record_one(backend, args, identifiers):
    try:
        from src.models.depth_estimator import DepthEstimator
        from src.models.detector import Detector

        # Тохиргоон дахь загварын замууд төслийн үндэстэй харьцангуй.
        os.chdir(ROOT)
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        det_cfg = dict(cfg["detector"], backend=backend, device="CPU", dynamic_fallback=False)
        depth_cfg = dict(cfg["depth"], backend=backend, depth_every_n_frames=args.depth_every)
        depth_cfg["device"] = "GPU" if backend == "openvino" else "CPU"
        camera = yaml.safe_load(Path(cfg["intrinsics"]).read_text(encoding="utf-8"))
        detector = Detector(det_cfg)
        if backend != "pytorch" and tuple(detector.prompts) != tuple(detector.fixed):
            raise ValueError("Харьцуулахын тулд тохиргооны бүх ангиллыг ижил дарааллаар хөрвүүлнэ үү.")
        depth = DepthEstimator(depth_cfg)
        try:
            stats = measure(args.video, detector, depth, camera, args.warmup, args.frames)
        except DeviceChanged:
            print("Хэмжих явцад гүн тооцоолох төхөөрөмж солигдов. Төв процессороор эхнээс нь дахин хэмжинэ.")
            depth = DepthEstimator(dict(depth_cfg, device="CPU"))
            detector = Detector(det_cfg)
            stats = measure(args.video, detector, depth, camera, args.warmup, args.frames)
        row = dict(identifiers, **stats, backend=backend, detector_device=detector.device,
                   depth_device=depth.device, warmup=args.warmup, frames=args.frames,
                   depth_every_n_frames=args.depth_every)
        append_row(args.output, BENCH_FIELDS, row)
        print(f'{backend} ({detector.device} / {depth.device})  {stats["fps"]:.2f} кадр/с, '
              f'{stats["mean_ms"]:.2f} мс дундаж хоцролт.')
    except Exception as error:
        print(f"{backend} горимын хэмжилтийг хадгалсангүй. {error}", file=sys.stderr)
        raise SystemExit(1) from None


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Хөдөлгөөнтэй бодит дүрс бичлэг")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default_config.yaml", help="Төслийн тохиргоо")
    parser.add_argument("--warmup", type=int, default=10, help="Урьдчилан ажиллуулах кадрын тоо")
    parser.add_argument("--frames", type=int, default=60, help="Хугацааг нь хэмжих кадрын тоо")
    parser.add_argument("--depth-every", type=int, default=1, help="Гүн тооцоолох кадрын алхам")
    parser.add_argument("--backends", nargs="+", choices=["pytorch", "onnx", "openvino"],
                        default=["pytorch", "onnx", "openvino"], help="Хэмжих хөдөлгүүрүүд ба дараалал")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark.csv", help="Гаралтын хүснэгт")
    args = parser.parse_args()
    if min(args.warmup, args.frames, args.depth_every) < 1:
        parser.exit(2, "Кадрын тоо ба алхам эерэг бүхэл байна.\n")
    if len(set(args.backends)) != len(args.backends):
        parser.exit(2, "Хөдөлгүүр бүрийг нэг удаа сонгоно уу.\n")
    args.video, args.config, args.output = (p.resolve() for p in (args.video, args.config, args.output))
    try:
        identifiers = dict(run_id=uuid4().hex, video=args.video.name, video_sha256=file_digest(args.video),
                           config_sha256=file_digest(args.config))
    except OSError as error:
        parser.exit(1, f"Оролтын файлыг нээсэнгүй. {error}\n")
    print(f'Хурдны туршилтын дугаар {identifiers["run_id"]}', flush=True)
    failed = []
    for backend in args.backends:
        process = multiprocessing.get_context("spawn").Process(target=record_one, args=(backend, args, identifiers))
        process.start()
        process.join()
        if process.exitcode:
            failed.append(backend)
    if failed:
        parser.exit(1, f"Дуусаагүй хөдөлгүүрүүд ({', '.join(failed)}). Бусад хэмжилт хүснэгтэд хадгалагдсан.\n")


if __name__ == "__main__":
    main()
