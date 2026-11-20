"""Тогтмол оролттой ONNX болон OpenVINO загварын гүйцэтгэгч."""

from pathlib import Path
import sys


class Runtime:
    """Нэг дүрс хүлээн авч, нэг гаралт буцаах сүлжээг ажиллуулна."""

    def __init__(self, config):
        self.backend = config["backend"]
        key = "ir_path" if self.backend == "openvino" else "onnx_path"
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"{path} олдсонгүй. python -m src.models.export_openvino ажиллуулна уу.")
        if self.backend == "openvino":
            import openvino as ov

            self.core = ov.Core()
            self.model = self.core.read_model(str(path))
            self.device = str(config.get("device", "CPU")).upper()
            if self.device not in ("CPU", "GPU"):
                raise ValueError("Тооцоолох төхөөрөмжид CPU эсвэл GPU сонгоно.")
            try:
                self._compile()
            except RuntimeError:
                self._switch_to_cpu()
            self.shape = tuple(self.compiled.input(0).shape)
        elif self.backend == "onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.input = self.session.get_inputs()[0]
            self.shape = tuple(self.input.shape)
            self.device = "CPU"
        else:
            raise ValueError("Тооцооллын горим тодорхойгүй.")

    def _compile(self):
        self.compiled = self.core.compile_model(self.model, self.device, {"PERFORMANCE_HINT": "LATENCY"})
        self.request = self.compiled.create_infer_request()

    def _switch_to_cpu(self):
        if self.device == "CPU":
            raise
        self.device = "CPU"
        print("График төхөөрөмж ажиллахгүй тул төв процессорт шилжлээ.", file=sys.stderr)
        self._compile()

    def __call__(self, tensor):
        if self.backend == "onnx":
            return self.session.run(None, {self.input.name: tensor})[0]
        try:
            self.request.infer({0: tensor})
        except RuntimeError:
            self._switch_to_cpu()
            self.request.infer({0: tensor})
        return self.request.get_output_tensor(0).data.copy()
