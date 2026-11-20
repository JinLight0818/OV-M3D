"""ONNX (загвар солилцох хэлбэр) ба OpenVINO IR (завсрын дүрслэл) үүсгэнэ."""

import json
from pathlib import Path
import shutil

import yaml

from src.cli import ArgumentParser
from .depth_estimator import input_size, load_metric_model
from .detector import vocabulary


def convert_ir(onnx_path, ir_path):
    import openvino as ov

    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(ir_path), compress_to_fp16=True)


def export_detector(config):
    from ultralytics import YOLOWorld

    weights, onnx_path, ir_path = (Path(config[k]) for k in ("weights", "onnx_path", "ir_path"))
    for path in (weights, onnx_path, ir_path):
        path.parent.mkdir(exist_ok=True, parents=True)
    names = vocabulary(config["prompts"])
    model = YOLOWorld(str(weights))
    model.set_classes(list(names))
    exported = Path(model.export(
        format="onnx", imgsz=int(config["image_size"]), device="cpu", batch=1,
        dynamic=False, half=False, nms=False, opset=17, simplify=False,
    ))
    if exported.resolve() != onnx_path.resolve():
        shutil.move(str(exported), onnx_path)
    convert_ir(onnx_path, ir_path)
    for path in {onnx_path.with_suffix(".labels.json"), ir_path.with_suffix(".labels.json")}:
        path.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_depth(config):
    import torch

    class MetricOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = load_metric_model(config)

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).predicted_depth

    onnx_path, ir_path = (Path(config[k]) for k in ("onnx_path", "ir_path"))
    for path in (onnx_path, ir_path):
        path.parent.mkdir(exist_ok=True, parents=True)
    example = torch.zeros(1, 3, *input_size(config))
    with torch.inference_mode():
        torch.onnx.export(
            MetricOutput().eval(), example, str(onnx_path), opset_version=17,
            dynamo=False, input_names=["pixel_values"], output_names=["depth_m"],
        )
    convert_ir(onnx_path, ir_path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default_config.yaml", help="Тохиргооны зам")
    parser.add_argument("--only", choices=["detector", "depth"], help="Зөвхөн сонгосон загварыг хөрвүүлнэ")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key, exporter in (("detector", export_detector), ("depth", export_depth)):
        if args.only is None or key == args.only:
            try:
                exporter(config[key])
            except ModuleNotFoundError as exc:
                parser.exit(1, f"{exc.name} сан олдсонгүй. scripts/setup_ubuntu.sh ажиллуулаад төслийн орчныг идэвхжүүлнэ үү.\n")
            print(f"Загварыг хөрвүүлж хадгаллаа. {config[key]['ir_path']}")


if __name__ == "__main__":
    main()
