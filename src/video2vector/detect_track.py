from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    track_id: int | None
    cls_id: int
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class DetectorTracker:
    """YOLO11 detection + ByteTrack tracking (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.35,
        person_class: int = 0,
        animal_classes: list[int] | None = None,
        tracker: str = "bytetrack.yaml",
    ):
        self.model_name = model_name
        self.conf = conf
        self.person_class = person_class
        self.animal_classes = set(animal_classes or [14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        self.tracker = tracker
        self._model = None

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def process(self, frame_bgr: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """Return (people, animals) with track ids for people when available."""
        self._ensure()
        try:
            results = self._model.track(
                source=frame_bgr,
                persist=True,
                conf=self.conf,
                tracker=self.tracker,
                verbose=False,
            )
        except Exception:
            # Tracking deps missing or frame issue — fall back to detect-only.
            results = self._model.predict(source=frame_bgr, conf=self.conf, verbose=False)
        people: list[Detection] = []
        animals: list[Detection] = []
        if not results:
            return people, animals
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return people, animals
        names = r0.names or {}
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id
        track_ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(cls)
        for i, c in enumerate(cls):
            det = Detection(
                track_id=int(track_ids[i]) if track_ids[i] is not None else None,
                cls_id=int(c),
                label=str(names.get(int(c), str(c))),
                conf=float(conf[i]),
                xyxy=tuple(float(v) for v in xyxy[i]),
            )
            if c == self.person_class:
                people.append(det)
            elif c in self.animal_classes:
                animals.append(det)
        return people, animals


def crop_xyxy(frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float], pad: float = 0.05):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def crop_head_xyxy(frame_bgr: np.ndarray, xyxy: tuple[float, float, float, float]):
    """Crop the upper portion of a person box for face analysis."""
    x1, y1, x2, y2 = xyxy
    head = (x1, y1, x2, y1 + 0.45 * (y2 - y1))
    return crop_xyxy(frame_bgr, head, pad=0.08)
