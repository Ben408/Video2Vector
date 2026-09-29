"""Build demo videos from public still images."""
from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "video"
OUT.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg",
    "https://ultralytics.com/images/bus.jpg",
]


def fetch(url: str, dest: Path) -> Path:
    print("download", url)
    req = urllib.request.Request(url, headers={"User-Agent": "video2vector/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print("saved", dest, dest.stat().st_size)
    return dest


def image_to_video(image_path: Path, video_path: Path, seconds: float = 6.0, fps: int = 10) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read {image_path}")
    h, w = img.shape[:2]
    scale = min(1.0, 640 / max(w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    n = int(seconds * fps)
    for i in range(n):
        shift = int(4 * np.sin(i / 4))
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
    writer.release()
    print("wrote", video_path, video_path.stat().st_size)
    return video_path


def main():
    img_path = OUT / "person_stock.jpg"
    if not (img_path.exists() and img_path.stat().st_size > 1000):
        last_err = None
        for url in URLS:
            try:
                fetch(url, img_path)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print("retry after", exc)
        if last_err and not img_path.exists():
            raise SystemExit(f"Could not download demo image: {last_err}")
    image_to_video(img_path, OUT / "people_demo.mp4", seconds=8.0, fps=8)
    image_to_video(img_path, OUT / "people_demo_short.mp4", seconds=3.0, fps=8)


if __name__ == "__main__":
    main()
