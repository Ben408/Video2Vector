from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# MediaPipe Face Landmarker blendshape names we map toward FACS-like scores.
BLENDSHAPE_MAP = {
    "brow_lowerer": ["browDownLeft", "browDownRight"],
    "cheek_raiser": ["cheekSquintLeft", "cheekSquintRight"],
    "lid_tightener": ["eyeSquintLeft", "eyeSquintRight"],
    "nose_wrinkler": ["noseSneerLeft", "noseSneerRight"],
    "upper_lip_raiser": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "eye_closure": ["eyeBlinkLeft", "eyeBlinkRight"],
}


@dataclass
class FaceResult:
    ok: bool
    landmarks_count: int = 0
    landmarks_xy: np.ndarray | None = None  # (N, 2) pixel coords in face crop
    scores: dict[str, float] = field(default_factory=dict)
    left_scores: dict[str, float] = field(default_factory=dict)
    right_scores: dict[str, float] = field(default_factory=dict)
    pain_score: float = 0.0
    asymmetry: float = 0.0
    eye_aspect: float = 0.0
    age_band: str | None = None
    sex_estimate: str | None = None


class FaceAnalyzer:
    """MediaPipe Face Landmarker (blendshapes + landmarks). Apache-2.0 friendly path."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._landmarker = None
        self._mp = None
        self._age_votes: list[str] = []
        self._sex_votes: list[str] = []

    def _ensure(self):
        if self._landmarker is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = _ensure_face_landmarker_model()
        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def analyze(self, crop_bgr: np.ndarray | None) -> FaceResult:
        if crop_bgr is None or crop_bgr.size == 0:
            return FaceResult(ok=False)
        self._ensure()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return FaceResult(ok=False)
        blend = {
            b.category_name: float(b.score)
            for b in (result.face_blendshapes[0] if result.face_blendshapes else [])
        }
        scores = {
            name: float(np.mean([blend.get(k, 0.0) for k in keys]))
            for name, keys in BLENDSHAPE_MAP.items()
        }
        left_scores, right_scores = _split_side_scores(blend)
        ear = _eye_aspect_ratio(result.face_landmarks[0], crop_bgr.shape)
        scores["eye_closure"] = max(scores.get("eye_closure", 0.0), 1.0 - min(ear / 0.25, 1.0))
        pain = _pain_score(scores)
        asym = _asymmetry(left_scores, right_scores)
        age_band, sex_est = self._estimate_demographics(crop_bgr)
        h, w = crop_bgr.shape[:2]
        lm = result.face_landmarks[0]
        landmarks_xy = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return FaceResult(
            ok=True,
            landmarks_count=len(lm),
            landmarks_xy=landmarks_xy,
            scores=scores,
            left_scores=left_scores,
            right_scores=right_scores,
            pain_score=pain,
            asymmetry=asym,
            eye_aspect=ear,
            age_band=age_band,
            sex_estimate=sex_est,
        )

    def _estimate_demographics(self, crop_bgr: np.ndarray) -> tuple[str | None, str | None]:
        """Lightweight heuristic placeholder — not a clinical age/sex model.

        A production build can swap in MiVOLO. Here we only record that a face
        was seen; age_band/sex stay None unless a model is plugged in later.
        """
        return None, None


def _pain_score(scores: dict[str, float]) -> float:
    # Prkachin–Solomon style cluster on mapped blendshapes.
    brow = scores.get("brow_lowerer", 0.0)
    orbit = max(scores.get("cheek_raiser", 0.0), scores.get("lid_tightener", 0.0))
    nose_lip = max(scores.get("nose_wrinkler", 0.0), scores.get("upper_lip_raiser", 0.0))
    close = scores.get("eye_closure", 0.0)
    return float(brow + orbit + nose_lip + close)


def _split_side_scores(blend: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    left = {
        "brow_lowerer": blend.get("browDownLeft", 0.0),
        "cheek_raiser": blend.get("cheekSquintLeft", 0.0),
        "lid_tightener": blend.get("eyeSquintLeft", 0.0),
        "nose_wrinkler": blend.get("noseSneerLeft", 0.0),
        "upper_lip_raiser": blend.get("mouthUpperUpLeft", 0.0),
        "eye_closure": blend.get("eyeBlinkLeft", 0.0),
    }
    right = {
        "brow_lowerer": blend.get("browDownRight", 0.0),
        "cheek_raiser": blend.get("cheekSquintRight", 0.0),
        "lid_tightener": blend.get("eyeSquintRight", 0.0),
        "nose_wrinkler": blend.get("noseSneerRight", 0.0),
        "upper_lip_raiser": blend.get("mouthUpperUpRight", 0.0),
        "eye_closure": blend.get("eyeBlinkRight", 0.0),
    }
    return left, right


def _asymmetry(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return float(np.mean([abs(left.get(k, 0.0) - right.get(k, 0.0)) for k in keys]))


def _eye_aspect_ratio(landmarks, shape) -> float:
    # MediaPipe face mesh indices for eyes (approximate).
    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]
    h, w = shape[:2]

    def ear(idxs):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hdist = np.linalg.norm(pts[0] - pts[3]) + 1e-6
        return float((v1 + v2) / (2.0 * hdist))

    try:
        return (ear(left_idx) + ear(right_idx)) / 2.0
    except Exception:
        return 0.25


def _ensure_face_landmarker_model():
    from pathlib import Path
    import urllib.request

    root = Path(__file__).resolve().parents[2] / "data" / "models"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "face_landmarker.task"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    urllib.request.urlretrieve(url, dest)
    return dest
