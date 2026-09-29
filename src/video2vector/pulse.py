from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PulseResult:
    bpm: float | None
    measurable: bool
    reason: str | None = None


class PulseEstimator:
    """POS remote-PPG with a motion/face-presence gate."""

    def __init__(self, window_seconds: float = 4.0, sample_fps: float = 5.0, max_motion: float = 25.0, min_face_fraction: float = 0.6):
        self.window_seconds = window_seconds
        self.sample_fps = sample_fps
        self.max_motion = max_motion
        self.min_face_fraction = min_face_fraction
        self._maxlen = max(int(window_seconds * sample_fps), 8)
        self._rgb = deque(maxlen=self._maxlen)
        self._face_ok = deque(maxlen=self._maxlen)
        self._motion = deque(maxlen=self._maxlen)

    def update(self, face_crop: np.ndarray | None, face_ok: bool, motion: float) -> PulseResult:
        self._face_ok.append(face_ok)
        self._motion.append(motion)
        if face_crop is None or not face_ok:
            self._rgb.append((0.0, 0.0, 0.0))
            return PulseResult(None, False, "no_face")
        if motion > self.max_motion:
            self._rgb.append((0.0, 0.0, 0.0))
            return PulseResult(None, False, "motion")
        mean = face_crop.reshape(-1, 3).mean(axis=0)  # BGR
        self._rgb.append((float(mean[2]), float(mean[1]), float(mean[0])))  # RGB
        face_frac = sum(self._face_ok) / max(len(self._face_ok), 1)
        if len(self._rgb) < self._maxlen or face_frac < self.min_face_fraction:
            return PulseResult(None, False, "warming_up")
        bpm = _pos_bpm(np.asarray(self._rgb, dtype=np.float64), self.sample_fps)
        if bpm is None:
            return PulseResult(None, False, "unstable")
        return PulseResult(bpm, True, None)


def _pos_bpm(rgb: np.ndarray, fps: float) -> float | None:
    """Wang et al. POS algorithm (simplified)."""
    if rgb.shape[0] < 8:
        return None
    # Temporal normalize
    c = rgb - rgb.mean(axis=0, keepdims=True)
    # Projection
    s1 = c[:, 1] - c[:, 2]  # G - B
    s2 = -2 * c[:, 0] + c[:, 1] + c[:, 2]  # -2R + G + B
    alpha = (s1.std() + 1e-8) / (s2.std() + 1e-8)
    h = s1 + alpha * s2
    h = h - h.mean()
    # Band-limit via FFT peak in 0.7–3.0 Hz (~42–180 bpm)
    n = len(h)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec = np.abs(np.fft.rfft(h * np.hanning(n)))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return None
    peak = freqs[mask][int(np.argmax(spec[mask]))]
    bpm = float(peak * 60.0)
    if bpm < 40 or bpm > 180:
        return None
    return bpm
