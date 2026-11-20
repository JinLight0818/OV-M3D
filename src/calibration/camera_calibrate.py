"""Шатрын хөлгийн 9×6 дотоод булангаар камерын дотоод үзүүлэлтийг хэмжинэ."""

from glob import glob
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import yaml

from src.cli import ArgumentParser
from src.geometry.spatial_3d import Intrinsics


def rectification(settings, width, height):
    """Хэмжээ өөрчлөгдсөн кадрт таарах фокус, гол цэг, гажилт засах торыг гаргана."""
    sx, sy = width / settings["width"], height / settings["height"]
    k = Intrinsics(settings["fx"] * sx, settings["fy"] * sy,
                   (settings["cx"] + 0.5) * sx - 0.5, (settings["cy"] + 0.5) * sy - 0.5)
    distortion = np.asarray(settings["distortion"], dtype=float)
    if not np.any(distortion):
        return k, None
    matrix = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]], dtype=float)
    maps = cv2.initUndistortRectifyMap(matrix, distortion, None, matrix,
                                      (width, height), cv2.CV_16SC2)
    return k, maps


def corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, points = cv2.findChessboardCorners(gray, (9, 6))
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        return cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), criteria).reshape(-1, 1, 2)
    return None


def collect(args):
    images, size = [], None

    def add(frame, points):
        nonlocal size
        shape = (frame.shape[1], frame.shape[0])
        if size is not None and size != shape:
            raise ValueError("Бүх зураг ижил нягтралтай байх шаардлагатай.")
        size = shape
        if points is not None:
            if any(np.linalg.norm(points - old, axis=2).mean() < 4 for old in images):
                print("Өмнөх зурагтай давхцаж байна. Самбарын өнцөг, байрлалыг өөрчилнө үү.")
            else:
                images.append(points)
                print(f"Авсан байрлалын тоо {len(images)}")

    if args.images:
        for name in sorted(glob(args.images)):
            frame = cv2.imread(name)
            if frame is None:
                raise ValueError(f"Зураг унших боломжгүй байна. {name}")
            add(frame, corners(frame))
    else:
        capture = cv2.VideoCapture(args.camera)
        try:
            if not capture.isOpened():
                raise ValueError("Камер нээгдсэнгүй.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            print("Самбарыг өөр өөр өнцөгт барина. Зай авах товчоор хадгалж, Q товчоор дуусгана.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                points = corners(frame)
                if points is not None:
                    cv2.drawChessboardCorners(frame, (9, 6), points, True)
                cv2.imshow("OV-M3D", frame)
                key = cv2.waitKey(10) & 255
                if key == 32:
                    add(frame, points)
                if key in (27, ord("q"), ord("Q")):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
    if len(images) < 8:
        raise ValueError("Булан нь бүрэн харагдсан дор хаяж найман ялгаатай байрлал хэрэгтэй.")
    return images, size


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="камерын дугаар")
    parser.add_argument("--images", help="зургийн замын загвар")
    parser.add_argument("--width", type=int, default=640, help="кадрын өргөн")
    parser.add_argument("--height", type=int, default=480, help="кадрын өндөр")
    parser.add_argument("--square-mm", type=float, required=True, help="нэг нүдний талын урт, мм")
    parser.add_argument("--output", type=Path, default=Path("config/camera_intrinsics.yaml"),
                        help="үзүүлэлт хадгалах зам")
    args = parser.parse_args()
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("Нэг нүдний хэмжээ эерэг, төгсгөлтэй байх ёстой.")
    try:
        images, size = collect(args)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    grid = np.zeros((54, 3), np.float32)
    grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2) * args.square_mm
    rms, matrix, dist, rotations, offsets = cv2.calibrateCamera(
        [grid] * len(images), images, size, None, None)
    errors = []
    for pixels, rotation, offset in zip(images, rotations, offsets):
        predicted, _ = cv2.projectPoints(grid, rotation, offset, matrix, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((pixels.reshape(-1, 2) - predicted.reshape(-1, 2)) ** 2, axis=1)))))
    settings = dict(calibrated=True, width=size[0], height=size[1],
                    fx=float(matrix[0, 0]), fy=float(matrix[1, 1]), cx=float(matrix[0, 2]),
                    cy=float(matrix[1, 2]), distortion=dist.ravel().tolist(),
                    rms_px=float(rms), per_view_rms_px=errors, square_mm=args.square_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as tmp:
        yaml.safe_dump(settings, tmp, sort_keys=False)
    Path(tmp.name).replace(args.output)
    print(f"{len(images)} байрлал. Дахин проекцолсон алдаа {rms:.3f} пиксел. {args.output}")


if __name__ == "__main__":
    main()
