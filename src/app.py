"""Объект илрүүлж, камерын координатаар байрлал, зай, хэмжээг харуулна."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import time
from uuid import uuid4

import cv2
import yaml

from src.cli import ArgumentParser
from src.calibration.camera_calibrate import rectification
from src.geometry.spatial_3d import TrackSmoother, distance_3d, measure_object
from src.models.depth_estimator import DepthEstimator
from src.models.detector import Detector
from src.sensor.esp32_tof import SerialReader, ToFFusion
from src.viz.hud_and_bev import render


def save_snapshot(directory, screen, record):
    """Зураг бүрэн бичигдсэний дараа түүнд харгалзах хэмжилтийг нийтэлнэ."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:12])
    png, pending, output = (base.with_suffix(ext) for ext in (".png", ".tmp", ".json"))
    try:
        if not cv2.imwrite(str(png), screen):
            raise OSError("Агшны зураг хадгалагдсангүй.")
        pending.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
        pending.replace(output)
    except Exception:
        png.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)
        raise
    print(f"Агшны зураг ба хэмжилт хадгалагдлаа. {output}")
    return output


def measure(detections, depth, k, smoothing, sampling):
    objects, measurements = [], {}
    for det in detections:
        value = measure_object(depth, det.box, k, **sampling)
        if value is None:
            continue
        track_id = int(det.track_id)
        m = smoothing.update(track_id, value)
        measurements[track_id] = m
        objects.append(dict(track_id=track_id, label=det.label, confidence=float(det.confidence),
                            box=list(map(float, det.box)), xyz_m=[m.x, m.y, m.z],
                            distance_m=m.distance_m, width_cm=m.width_cm, height_cm=m.height_cm))
    smoothing.retain([d.track_id for d in detections])
    pairs = [dict(ids=[a, b], distance_m=distance_3d(measurements[a], measurements[b]))
             for a, b in combinations(measurements, 2)]
    return objects, pairs


def run(config, source):
    camera = isinstance(source, int)
    intrinsics = yaml.safe_load(Path(config["intrinsics"]).read_text())
    if not intrinsics.get("calibrated", False):
        print("Камерын дотоод үзүүлэлтийг ойролцоолсон. Тоон үнэлгээний өмнө калибровк хийнэ үү.")
    with ExitStack() as stack:
        capture = cv2.VideoCapture(source)
        stack.callback(capture.release)
        if not capture.isOpened():
            raise ValueError(f"Дүрсний эх үүсвэр нээгдсэнгүй. {source}")
        if camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera_width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera_height"])
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        detector = Detector(config["detector"])
        estimator = DepthEstimator(config["depth"])
        sensor = SerialReader(**config["serial"])
        if camera:
            stack.callback(sensor.close)
            sensor.start()
        fusion = ToFFusion(**config["fusion"])
        smoothing = TrackSmoother(**config["smoothing"])
        worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        stack.callback(cv2.destroyAllWindows)
        requested = camera
        mode = "mono"
        index, fps, frame_time = 0, 0.0, 0.0
        shape = None
        last_packet = None
        status = "OV-M3D"
        cv2.namedWindow(status, cv2.WINDOW_NORMAL)
        print("M — горим, P — ангилал, S — агшин хадгалах, Q — гарах.")
        while True:
            start = time.monotonic()
            ok, frame = capture.read()
            captured = time.monotonic()
            stamp = datetime.now(timezone.utc).isoformat()
            if not ok:
                if index == 0:
                    raise ValueError("Эхний кадрыг уншиж чадсангүй.")
                break
            h, w = frame.shape[:2]
            if shape != (h, w):
                k, maps = rectification(intrinsics, w, h)
                shape = (h, w)
            if maps:
                frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
            packet = sensor.latest(at=captured) if requested else None
            job = worker.submit(estimator.estimate, frame, index, k.fx)
            detections = detector.detect(frame)
            ai = job.result()
            if (packet is not None and packet is not last_packet
                    and index == estimator.last_frame_index):
                fusion.update(ai, packet, k)
                last_packet = packet
            actual = "fusion" if packet is not None and fusion.calibrated else "mono"
            if actual != mode:
                smoothing.clear()
            mode = actual
            depth = fusion.apply(ai) if actual == "fusion" else ai
            objects, pairs = measure(detections, depth, k, smoothing, config["sampling"])
            screen = render(frame, depth, k, objects, pairs, mode, fps, estimator.device,
                            config["bev_range_m"], config["proximity_warning_m"])
            cv2.imshow(status, screen)
            key = cv2.waitKey(1) & 255
            elapsed = max(1e-6, time.monotonic() - start)
            frame_time = 0.9 * frame_time + 0.1 * elapsed if frame_time else elapsed
            fps = 1 / frame_time
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(status, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("m"), ord("M")):
                requested = camera and not requested
                print("Мэдрэгчийн горим сонгов." if requested else "Дан камерын горим сонгов.")
            if key in (ord("p"), ord("P")):
                try:
                    names = input("Ангиллын нэрсийг таслалаар зааглаж оруулна уу. ")
                    prompts = [name.strip() for name in names.split(",") if name.strip()]
                    if prompts:
                        detector.set_prompts(prompts)
                        smoothing.clear()
                except (ValueError, OSError, RuntimeError, EOFError) as exc:
                    print(f"Ангилал солих хүсэлт биелсэнгүй. {exc}")
            if key in (ord("s"), ord("S")):
                record = dict(timestamp_utc=stamp, source=source, mode=mode, frame_index=index,
                              objects=objects, pairs=pairs, calibrated=intrinsics.get("calibrated", False),
                              intrinsics=dict(fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy),
                              image_size=[w, h], prompts=list(detector.prompts),
                              depth_device=estimator.device, detector_device=detector.device,
                              depth_frame_index=estimator.last_frame_index,
                              sensor=packet.sensor if packet else None,
                              tof_seq=packet.seq if packet else None,
                              tof_age_ms=(captured - packet.received_at) * 1000 if packet else None,
                              tof_status=list(packet.status) if packet else None,
                              depth_every_n_frames=config["depth"]["depth_every_n_frames"],
                              scale=fusion.scale if mode == "fusion" else 1.0,
                              shift_m=fusion.shift if mode == "fusion" else 0.0)
                save_snapshot(config["snapshot_dir"], screen, record)
            index += 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default_config.yaml"),
                        help="тохиргооны файл")
    parser.add_argument("--source", help="камерын дугаар эсвэл бичлэгийн зам")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = args.source if args.source is not None else config["source"]
    if isinstance(source, str) and source.isdecimal():
        source = int(source)
    try:
        run(config, source)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Ажиллуулах үед алдаа гарлаа. {exc}\n")


if __name__ == "__main__":
    main()
