"""HUD (хэмжилтийн дэлгэц), гүний зураг, BEV (дээрээс харсан дүрс) зурна."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=4)
def font(size):
    return ImageFont.truetype("DejaVuSans.ttf", size)


def label(draw, position, value, color="white", size=14):
    draw.text(position, value, font=font(size), fill=color, stroke_width=2, stroke_fill="#121821")


def place_labels(draw, annotations, width, height):
    """Ойр объектын бичвэрүүдийг сул мөрөнд шилжүүлж, эх цэгтэй нь холбоно."""
    occupied = []
    for anchor, lines, color in annotations:
        face = font(13)
        sizes = [draw.textbbox((0, 0), text, font=face) for text in lines]
        w = min(width, max(box[2] for box in sizes) + 8)
        h = len(lines) * 17 + 4
        x = max(0, min(width - w, int(anchor[0])))
        preferred = max(0, min(height - h, int(anchor[1]) - h))
        candidates = sorted(set([preferred, *range(0, max(1, height - h), 5)]),
                            key=lambda y: abs(y - preferred))
        for y in candidates:
            if all(x >= right or x+w <= left or y >= bottom or y+h <= top
                   for left, top, right, bottom in occupied):
                break
        else:
            y = preferred
        if abs(y - preferred) > 5:
            draw.line([anchor, (x+3, y+h/2)], fill=color)
        occupied.append((x, y, x+w, y+h))
        draw.rectangle((x, y, x+w-1, y+h-1), fill="#172029")
        for row, text in enumerate(lines):
            draw.text((x+4, y+1+row*17), text, font=face, fill=color)


def depth_colors(depth, limit):
    normalized = np.clip(np.nan_to_num(depth, nan=0.0) / limit, 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * (1 - normalized)), cv2.COLORMAP_INFERNO)
    colored[(depth <= 0) | ~np.isfinite(depth)] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render(frame, depth, k, objects, pairs, mode, fps, device, limit, threshold):
    """Камерын координатын эхийг радарын ёроолд байрлуулж, гүнийг метрээр дүрсэлнэ."""
    height, width = frame.shape[:2]
    side = max(240, width // 2)
    mid = height // 2
    image = Image.new("RGB", (width + side, height), "#101b25")
    front = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(front)
    centers = {}
    annotations = []
    for obj in objects:
        box = obj["box"]
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        centers[obj["track_id"]] = center
        # Арын хүрээний зузаан нь зөвхөн дүрслэлийн зориулалттай.
        z = obj["xyz_m"][2]
        ratio = z / (z + 0.1)
        back = [(k.cx + (x - k.cx) * ratio, k.cy + (y - k.cy) * ratio)
                for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        draw.polygon(back, outline="#63c7bc", width=1)
        for near, far in zip([(x1,y1), (x2,y1), (x2,y2), (x1,y2)], back):
            draw.line([near, far], fill="#63c7bc")
        draw.rectangle(box, outline="#70ffe4", width=2)
        title = f"#{obj['track_id']} {obj['label']}  {obj['distance_m']:.2f} м"
        xyz = ", ".join(f"{v:.2f}" for v in obj["xyz_m"])
        annotations.append(((x1, y1), [title, f"({xyz}) м",
                            f"{obj['width_cm']:.1f} × {obj['height_cm']:.1f} см"], "#baffee"))
    for pair in pairs:
        a, b = [centers[i] for i in pair["ids"]]
        close = pair["distance_m"] < threshold
        color = "#ff3d48" if close else "#ffd266"
        draw.line([a, b], fill=color, width=4 if close else 1)
        prefix = "ОЙР " if close else ""
        annotations.append((((a[0]+b[0])/2, (a[1]+b[1])/2),
                            [f"{prefix}{pair['distance_m']:.2f} м"], color))
    place_labels(draw, annotations, width, height)
    image.paste(front, (0, 0))
    inset = Image.fromarray(depth_colors(depth, limit))
    inset.thumbnail((side, mid))
    image.paste(inset, (width + (side - inset.width) // 2, (mid - inset.height) // 2))
    draw = ImageDraw.Draw(image)
    label(draw, (width + 8, 5), f"Гүн   0–{limit:g} м")
    origin = (width + side // 2, height - 10)
    slopes = [-k.cx / k.fx, (width - k.cx) / k.fx]
    scale = min(max(1, height - mid - 63) / limit,
                (side / 2 - 8) / (limit * max(map(abs, slopes))))
    far = [(origin[0] + slope * limit * scale, origin[1] - limit * scale) for slope in slopes]
    draw.polygon([origin, *far], fill="#173740", outline="#487c82")
    for z in range(1, int(limit) + 1):
        y = origin[1] - z * scale
        draw.line([(origin[0] + slope * z * scale, y) for slope in slopes], fill="#3c5a65")
        label(draw, (width + 5, y - 9), f"{z} м", size=11)
    for obj in objects:
        x, _, z = obj["xyz_m"]
        px, py = origin[0] + x * scale, origin[1] - z * scale
        if z <= limit and width <= px < width + side:
            draw.ellipse((px-3, py-3, px+3, py+3), fill="#70ffe4")
            label(draw, (px+4, py-12), str(obj["track_id"]), size=12)
    mode_name = "Камер + ToF" if mode == "fusion" else "Дан камер"
    label(draw, (width + 8, mid + 3), f"{mode_name}   {device}")
    label(draw, (width + 8, mid + 23), f"Дээрээс   {fps:.1f} кадр/с", size=13)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
