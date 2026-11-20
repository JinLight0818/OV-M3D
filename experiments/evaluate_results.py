"""Зай, хэмжээний алдаа болон хоёр загварыг хамтад нь ажиллуулсан хурдыг дүрсэлнэ."""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import ArgumentParser
from experiments._records import ROOT, positive

MODES = {"mono": ("Дан камер", "#1f628c"), "fusion": ("Камер ба мэдрэгч", "#be5b35")}
RUNTIMES = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "openvino": "OpenVINO"}
QUANTITIES = ("distance_m", "width_cm", "height_cm")
METRIC_FIELDS = ("source", "mode", "quantity", "unit", "count", "mae", "rmse", "mape_pct")


def read_table(path, required):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} хүснэгтэд багана дутуу байна ({', '.join(sorted(missing))}).")
        return list(reader)


def artificial_example():
    rng = np.random.default_rng(57)
    measures = []
    for distance in np.arange(0.5, 3.51, 0.5):
        for mode, spread in (("mono", 0.08), ("fusion", 0.04)):
            for _ in range(4):
                row = {"mode": mode}
                for field, truth in zip(QUANTITIES, (distance, 16, 11)):
                    row["gt_" + field] = truth
                    row[field] = truth * (1 + rng.normal(0, spread))
                measures.append(row)
    timings = []
    for backend, device, ms in (("pytorch", "CPU", 210), ("onnx", "CPU", 160), ("openvino", "GPU", 110)):
        timings.append(dict(run_id="хиймэл", backend=backend, detector_device="CPU",
                            depth_device=device, fps=1000 / ms, p95_ms=ms * 1.2))
    return measures, timings


def metrics(truth, estimate):
    error = estimate - truth
    return np.abs(error).mean(), np.sqrt(np.mean(error**2)), 100 * np.mean(np.abs(error / truth))


def save_plot(fig, path, artificial):
    if artificial:
        fig.text(0.99, 0.01, "ХИЙМЭЛ ӨГӨГДӨЛ", ha="right", size=9, color="#9b382a")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def empty_chart(axis, text):
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def measurement_plots(rows, target, artificial):
    distance_fig, distance_ax = plt.subplots(figsize=(6.5, 4.5))
    size_fig, size_ax = plt.subplots(figsize=(6.5, 4.5))
    summaries = []
    limits = []
    if any(row["mode"] not in MODES for row in rows):
        raise ValueError("Хэмжилтийн хүснэгтэд үл таних горим байна.")
    for offset, (mode, (label, color)) in enumerate(MODES.items()):
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        size_errors = []
        for field in QUANTITIES:
            truth = np.array([positive(row["gt_" + field], field) for row in subset])
            estimate = np.array([positive(row[field], field) for row in subset])
            mae, rmse, mape = metrics(truth, estimate)
            unit = "м" if field == "distance_m" else "см"
            summaries.append(dict(source="хиймэл" if artificial else "бодит", mode=mode,
                                  quantity=field, unit=unit, count=len(subset), mae=mae,
                                  rmse=rmse, mape_pct=mape))
            if field == "distance_m":
                distance_ax.scatter(truth, estimate, alpha=0.68, s=24, color=color, label=label)
                limits.extend([truth.min(), truth.max(), estimate.min(), estimate.max()])
            else:
                size_errors.append(mae)
        size_ax.bar(np.arange(2) + (offset - 0.5) * 0.34, size_errors, width=0.32, color=color, label=label)
    if rows:
        low, high = min(limits), max(limits)
        distance_ax.plot([low, high], [low, high], color="#626262", ls="--", label="Төгс тохироо")
        distance_ax.set(xlabel="Шугамаар хэмжсэн шулуун зай (м)", ylabel="Тооцоолсон шулуун зай (м)")
        distance_ax.legend()
        size_ax.set_xticks([0, 1], ["Өргөн", "Өндөр"])
        size_ax.set_ylabel("Дундаж үнэмлэхүй алдаа (см)")
        size_ax.legend()
    else:
        for axis in (distance_ax, size_ax):
            empty_chart(axis, "Хэмжээсийн ажиглалт бүртгэгдээгүй.")
    distance_ax.set_title("Камераас биет хүртэлх зайн нарийвчлал")
    size_ax.set_title("Харагдах өргөн, өндрийн нарийвчлал")
    save_plot(distance_fig, target / "distance_accuracy.png", artificial)
    save_plot(size_fig, target / "size_accuracy.png", artificial)
    with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)


def speed_plot(rows, target, artificial, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    if rows:
        selected_id = run_id or rows[-1]["run_id"]
        batch = [row for row in rows if row["run_id"] == selected_id]
        if not batch:
            raise ValueError(f"{selected_id} дугаартай хурдны туршилт олдсонгүй.")
        labels = [f'{RUNTIMES[r["backend"]]}\n{r["detector_device"]} / {r["depth_device"]}' for r in batch]
        for axis, field, ylabel in zip(axes, ("fps", "p95_ms"), ("Кадр/с", "Миллисекунд")):
            values = [positive(row[field], field) for row in batch]
            axis.bar(range(len(batch)), values, color="#3d7473", width=0.65)
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(len(batch)), labels, fontsize=9)
        print(f"Дүрсэлсэн хурдны туршилтын дугаар {selected_id}")
    elif run_id:
        raise ValueError("Хүссэн дугаартай хурдны туршилт бүртгэгдээгүй.")
    else:
        for axis in axes:
            empty_chart(axis, "Хурдны хэмжилт бүртгэгдээгүй.")
    axes[0].set_title("Боловсруулалтын хурд")
    axes[1].set_title("Хоцролтын 95-р хувьт утга")
    save_plot(fig, target / "speed_comparison.png", artificial)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, help="Бодит хэмжилтийн хүснэгт")
    parser.add_argument("--benchmark", type=Path, help="Гүйцэтгэлийн хүснэгт")
    parser.add_argument("--run-id", help="Дүрслэх хурдны туршилтын дугаар, анхдагчаар хамгийн сүүлийнх")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="Үр дүнгийн хавтас")
    parser.add_argument("--synthetic", action="store_true", help="Хиймэл жишээг тусдаа хавтсанд дүрслэх")
    args = parser.parse_args()
    try:
        if args.synthetic:
            measures, timings = artificial_example()
            synthetic = True
        else:
            for path in (args.measurements, args.benchmark):
                if path is not None and not path.is_file():
                    raise ValueError(f"{path} файл олдсонгүй.")
            columns = ("mode", *QUANTITIES, *["gt_" + k for k in QUANTITIES])
            measures = read_table(args.measurements or ROOT / "data/measurements.csv", columns)
            timings = read_table(args.benchmark or ROOT / "data/benchmark.csv",
                                 ("backend", "run_id", "depth_device", "detector_device", "fps", "p95_ms"))
            synthetic = not (measures or timings)
            if synthetic:
                measures, timings = artificial_example()
        target = args.output / "synthetic" if synthetic else args.output
        target.mkdir(parents=True, exist_ok=True)
        speed_plot(timings, target, synthetic, args.run_id)
        measurement_plots(measures, target, synthetic)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Үнэлгээг дуусгасангүй. {error}\n")
    kind = "Хиймэл" if synthetic else "Бодит"
    print(f"{kind} өгөгдлийн гурван зураг ба алдааны хүснэгтийг {target} хавтсанд хадгалав.")


if __name__ == "__main__":
    main()
