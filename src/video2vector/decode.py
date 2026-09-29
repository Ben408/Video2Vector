from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    has_audio: bool


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 1e-3:
        fps = 25.0
    duration = frame_count / fps if frame_count else 0.0
    # Audio presence is best-effort; pipeline tolerates missing audio.
    has_audio = True
    return VideoInfo(path, fps, frame_count, width, height, duration, has_audio)


def iter_sampled_frames(path: str | Path, sample_fps: float, max_seconds: float | None = None):
    """Yield (timestamp_seconds, frame_bgr) at approximately sample_fps."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if native_fps <= 1e-3:
        native_fps = 25.0
    step = max(native_fps / max(sample_fps, 0.1), 1.0)
    next_idx = 0.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / native_fps
            if max_seconds is not None and t > max_seconds:
                break
            if idx + 0.5 >= next_idx:
                yield t, frame
                next_idx += step
            idx += 1
    finally:
        cap.release()


def frame_quality_flags(frame_bgr: np.ndarray) -> dict[str, bool]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "too_dark": mean < 20.0,
        "too_bright": mean > 235.0,
        "too_blurry": blur < 20.0,
    }
