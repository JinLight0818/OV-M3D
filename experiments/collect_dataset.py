"""Хадгалсан агшны хэмжилтийг шугамаар авсан утгатай хослуулна."""

import json
import math
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import ArgumentParser
from experiments._records import MEASURE_FIELDS, ROOT, append_row, positive


def capture(args):
    source = args.snapshot.resolve()
    packet = json.loads(source.read_text(encoding="utf-8"))
    if packet["mode"] not in ("mono", "fusion"):
        raise ValueError("Агшны хэмжилтийн горим танигдсангүй.")
    objects = {int(obj["track_id"]): obj for obj in packet["objects"]}
    if not objects:
        raise ValueError("Агшинд хэмжилттэй биет алга.")
    track = args.track_id
    if track is None:
        for key, obj in objects.items():
            print(f'{key:5d}  {obj["label"]}  {obj["distance_m"]:.3f} м')
        track = next(iter(objects)) if len(objects) == 1 else int(input("Биетийн дугаар оруулна уу → "))
    if track not in objects:
        raise ValueError("Сонгосон дугаартай биет энэ агшинд алга.")
    obj = objects[track]
    x, y, z = map(float, obj["xyz_m"])
    if not all(math.isfinite(v) for v in (x, y, z)) or z <= 0:
        raise ValueError("Агшны орон зайн координат хүчингүй байна.")
    scene = (args.scene or input("Байрлуулалт, давталтын тэмдэглэгээ → ")).strip()
    if not scene:
        raise ValueError("Байрлуулалтын тэмдэглэгээ хоосон байна.")
    row = {
        "snapshot": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        "track_id": track, "frame_index": int(packet["frame_index"]),
        "scene": scene, "mode": packet["mode"], "label": obj["label"],
        "x_m": x, "y_m": y, "z_m": z,
    }
    for field, label in (("distance_m", "Линзээс гадаргуугийн төв хүртэлх шулуун зай (м)"),
                         ("width_cm", "Урд талаас харагдах өргөн (см)"),
                         ("height_cm", "Урд талаас харагдах өндөр (см)")):
        value = getattr(args, field)
        if value is None:
            value = input(f"{label} → ")
        row[f"gt_{field}"] = positive(value, label)
        row[field] = positive(obj[field], field)
    append_row(args.output, MEASURE_FIELDS, row, unique=("snapshot", "track_id"))
    print(f"{args.output} файлд {track} дугаартай биетийн хэмжилтийг нэмэв.")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="S товчоор хадгалсан агшны JSON файл")
    parser.add_argument("--track-id", type=int, help="Хянаж буй биетийн дугаар")
    parser.add_argument("--scene", help="Хоёр горимд ижил байх байрлуулалт ба давталтын тэмдэглэгээ")
    parser.add_argument("--distance-m", type=float, help="Тулгуур шулуун зай, метрээр")
    parser.add_argument("--width-cm", type=float, help="Тулгуур өргөн, сантиметрээр")
    parser.add_argument("--height-cm", type=float, help="Тулгуур өндөр, сантиметрээр")
    parser.add_argument("--output", type=Path, default=ROOT / "data/measurements.csv", help="Гаралтын хүснэгт")
    try:
        capture(parser.parse_args())
    except (OSError, EOFError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Хэмжилтийг хадгалсангүй. {error}\n")


if __name__ == "__main__":
    main()
