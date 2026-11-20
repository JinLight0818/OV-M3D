"""YOLO-World (үгээр заасан объект таних загвар) ба ByteTrack (олон объект мөрдөх арга)."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .runtime import Runtime


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    box: tuple[float, float, float, float]
    confidence: float


def vocabulary(words):
    if isinstance(words, str) or not all(isinstance(word, str) for word in words):
        raise ValueError("Ангиллыг үгийн жагсаалтаар өгнө.")
    names = tuple(dict.fromkeys(word.strip() for word in words if word.strip()))
    if not names:
        raise ValueError("Дор хаяж нэг ангилал оруулна уу.")
    return names


class Detector:
    def __init__(self, config):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.config = config
        self.backend = config["backend"]
        self.device = "CPU"
        self.size = int(config.get("image_size", 640))
        self.confidence = float(config.get("confidence", 0.1))
        self.fixed = ()
        self.runtime = None
        self.dynamic = None
        self.dynamic_names = ()
        self.prompts = ()
        if self.backend in ("openvino", "onnx"):
            if str(config.get("device", "CPU")).upper() != "CPU":
                raise ValueError("Объект танигчийг CPU төхөөрөмжөөр ажиллуулна.")
            self.runtime = Runtime(config)
            if self.runtime.shape != (1, 3, self.size, self.size):
                raise ValueError("Танигчийн оролтын хэмжээ тохиргооноос зөрлөө. Загварыг дахин хөрвүүлнэ үү.")
            key = "ir_path" if self.backend == "openvino" else "onnx_path"
            names_path = Path(config[key]).with_suffix(".labels.json")
            self.fixed = vocabulary(json.loads(names_path.read_text(encoding="utf-8")))
        elif self.backend != "pytorch":
            raise ValueError("Танигчийн горим openvino, onnx эсвэл pytorch байна.")
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
            track_buffer=30, match_thresh=0.8, fuse_score=True,
        ))
        self.set_prompts(config["prompts"])

    def set_prompts(self, words):
        from ultralytics import YOLOWorld

        names = vocabulary(words)
        if names == self.prompts:
            return
        use_fixed = self.runtime is not None and set(names).issubset(self.fixed)
        if not use_fixed and names != self.dynamic_names:
            if self.runtime is not None and not self.config.get("dynamic_fallback", True):
                raise ValueError("Энэ ангилал хөрвүүлсэн загварт байхгүй. dynamic_fallback тохиргоог идэвхжүүлнэ үү.")
            if self.dynamic is None:
                self.dynamic = YOLOWorld(str(self.config["weights"]))
                self.dynamic.model.eval().to("cpu")
            self.dynamic.set_classes(list(names))
            self.dynamic_names = names
        self.names = self.fixed if use_fixed else names
        self.use_fixed = use_fixed
        self.active_indices = np.array([self.names.index(word) for word in names], dtype=int)
        self.prompts = names
        self.tracker.reset()

    def detect(self, frame):
        from ultralytics.engine.results import Boxes

        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (self.size - rw) // 2, (self.size - rh) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[top:top + rh, left:left + rw] = cv2.resize(frame, (rw, rh))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        if self.use_fixed:
            output = self.runtime(tensor)
        else:
            import torch

            with torch.inference_mode():
                output = self.dynamic.model(torch.from_numpy(tensor))[0].numpy()

        # Сүлжээ төв, өргөн, өндөр болон ангилал бүрийн магадлалыг өгнө.
        rows = output[0].T
        if rows.shape[1] != 4 + len(self.names):
            raise ValueError("Танигчийн гаралт ангиллын жагсаалттай таарахгүй байна.")
        scores = rows[:, 4:][:, self.active_indices]
        labels = np.argmax(scores, axis=1)
        conf = scores[np.arange(len(labels)), labels]
        mask = conf >= self.confidence
        xywh = rows[mask, :4].copy()
        conf, labels = conf[mask], labels[mask]
        xywh[:, :2] -= xywh[:, 2:] / 2
        selected = []
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            keep = cv2.dnn.NMSBoxes(xywh[group].tolist(), conf[group].tolist(), 0.0, 0.45)
            selected.extend(group[np.asarray(keep, dtype=int).ravel()])
        selected = np.asarray(selected, dtype=int)
        boxes = xywh[selected].copy()
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) * width / rw, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) * height / rh, 0, height)
        det = np.column_stack((boxes, conf[selected], labels[selected])).astype(np.float32)
        det = det[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        tracks = self.tracker.update(Boxes(det, (height, width)), frame)
        return [
            Detection(int(row[4]), self.prompts[int(row[6])], tuple(map(float, row[:4])), float(row[5]))
            for row in tracks
        ]
