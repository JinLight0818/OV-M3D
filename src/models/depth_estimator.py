"""Depth Anything V2 (нэг дүрсээс метрээр гүн таамаглах загвар)."""

import cv2
import numpy as np

from .runtime import Runtime


def input_size(config):
    height = int(config.get("input_height", 392))
    width = int(config.get("input_width", 518))
    if min(height, width) <= 0 or height % 14 or width % 14:
        raise ValueError("Гүний загварын оролтын хоёр хэмжээ 14-т хуваагдах эерэг тоо байна.")
    return height, width


def load_metric_model(config):
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(config["hf_model"]).eval().to("cpu")
    if getattr(model.config, "depth_estimation_type", None) != "metric":
        raise ValueError("Сонгосон загвар метрийн нэгжээр гүн гаргадаг байх шаардлагатай.")
    return model


class DepthEstimator:
    def __init__(self, config):
        self.height, self.width = input_size(config)
        self.every = int(config.get("depth_every_n_frames", 2))
        if self.every < 1:
            raise ValueError("Гүн бодох кадрын алхам нэгээс бага байж болохгүй.")
        self.train_focal = config.get("train_focal_px")
        self.reference_width = float(config.get("focal_reference_width", 640))
        if self.train_focal is not None:
            if not np.isfinite(self.train_focal) or self.train_focal <= 0 or self.reference_width <= 0:
                raise ValueError("Фокусын урт ба жишиг дүрсний өргөн эерэг байна.")
        self.runtime = None
        if config["backend"] == "pytorch":
            self.model = load_metric_model(config)
        elif config["backend"] in ("onnx", "openvino"):
            self.runtime = Runtime(config)
            if self.runtime.shape != (1, 3, self.height, self.width):
                raise ValueError("Гүний загварын оролтын хэмжээ өөр байна. Загварыг дахин хөрвүүлнэ үү.")
        else:
            raise ValueError("Гүний горим openvino, onnx эсвэл pytorch байна.")
        self.previous = None
        self.last_frame_index = None
        self.last_fx = None
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    @property
    def device(self):
        return self.runtime.device if self.runtime else "CPU"

    def estimate(self, frame, frame_index, fx):
        """Завсрын кадрт өмнөх гүнийг өгнө. Буцаасан массивыг өөрчлөхгүй хэрэглэнэ."""
        height, width = frame.shape[:2]
        if self.previous is not None and self.previous.shape == (height, width) and self.last_fx == fx:
            if 0 <= frame_index - self.last_frame_index < self.every:
                return self.previous

        scale = min(self.width / width, self.height / height)
        rw, rh = max(1, round(scale * width)), max(1, round(scale * height))
        dx, dy = self.width - rw, self.height - rh
        left, top = dx // 2, dy // 2
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.copyMakeBorder(resized, top, dy - top, left, dx - left, cv2.BORDER_REPLICATE)[:, :, ::-1]
        rgb = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        if self.runtime:
            output = self.runtime(tensor)
        else:
            import torch

            with torch.inference_mode():
                output = self.model(pixel_values=torch.from_numpy(tensor)).predicted_depth.numpy()
        depth = np.squeeze(output)
        if depth.shape != (self.height, self.width):
            raise ValueError("Гүний зураглалын хэлбэр загварын оролттой таарахгүй байна.")
        depth = cv2.resize(depth[top:top + rh, left:left + rw], (width, height), interpolation=cv2.INTER_LINEAR)
        if self.train_focal is not None:
            # Камер ба жишиг фокусыг ижил өргөнтэй дүрсний пикселд шилжүүлнэ.
            depth *= fx * self.reference_width / (width * self.train_focal)
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        self.previous = depth
        self.last_frame_index = frame_index
        self.last_fx = fx
        return depth
