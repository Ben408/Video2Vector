from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseResult:
    keypoints: np.ndarray | None  # (17, 3) x,y,conf
    posture: str
    motion: float
    ok: bool


class PoseEstimator:
    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        self.model_name = model_name
        self._model = None
        self._prev_xy: np.ndarray | None = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def estimate(self, crop_bgr: np.ndarray | None) -> PoseResult:
        if crop_bgr is None or crop_bgr.size == 0:
            return PoseResult(None, "unknown", 0.0, False)
        self._ensure()
        results = self._model.predict(source=crop_bgr, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return PoseResult(None, "unknown", 0.0, False)
        kps = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        posture = classify_posture(kps)
        motion = 0.0
        xy = kps[:, :2]
        if self._prev_xy is not None and self._prev_xy.shape == xy.shape:
            motion = float(np.linalg.norm(xy - self._prev_xy, axis=1).mean())
        self._prev_xy = xy.copy()
        return PoseResult(kps, posture, motion, True)


def classify_posture(kps: np.ndarray) -> str:
    """Rough posture from COCO-17 keypoints."""
    # indices: 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles
    def avg(idxs):
        pts = [kps[i] for i in idxs if kps[i, 2] > 0.3]
        if not pts:
            return None
        return np.mean(pts, axis=0)

    shoulders = avg([5, 6])
    hips = avg([11, 12])
    ankles = avg([15, 16])
    if shoulders is None or hips is None:
        return "unknown"
    torso = hips[:2] - shoulders[:2]
    torso_len = float(np.linalg.norm(torso)) + 1e-6
    vertical = abs(torso[1]) / torso_len
    if vertical < 0.45:
        return "lying"
    if ankles is not None:
        hip_to_ankle = abs(ankles[1] - hips[1])
        if hip_to_ankle < 0.35 * torso_len:
            return "sitting"
    return "upright"
