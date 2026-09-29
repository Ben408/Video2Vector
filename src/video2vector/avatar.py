"""Flat toon avatar: silhouette + face/body fill + wireframe (privacy-friendly)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO-17 skeleton edges (Ultralytics YOLO pose).
BODY_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]

# MediaPipe Face Mesh contour subsets (landmark indices).
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


@dataclass
class AvatarStyle:
    bg_blur: int = 25
    bg_darken: float = 0.55
    wire_bgr: tuple[int, int, int] = (30, 30, 45)
    cloth_bgr: tuple[int, int, int] = (92, 118, 150)
    limb_scale: float = 0.16
    face_wire: bool = True


def estimate_skin_tone(bgr: np.ndarray | None, fallback: tuple[int, int, int] = (180, 150, 130)) -> tuple[int, int, int]:
    """Median tone from central crop region; lightly desaturated for toon look."""
    if bgr is None or bgr.size == 0:
        return fallback
    h, w = bgr.shape[:2]
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        patch = bgr
    med = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)
    # Pull toward mid gray for flat avatar paint.
    gray = float(med.mean())
    toon = 0.65 * med + 0.35 * gray
    return tuple(int(np.clip(c, 40, 230)) for c in toon)  # type: ignore[return-value]


def crop_with_box(
    frame_bgr: np.ndarray,
    xyxy: tuple[float, float, float, float],
    pad: float = 0.05,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1i = max(0, int(x1 - pad * bw))
    y1i = max(0, int(y1 - pad * bh))
    x2i = min(w, int(x2 + pad * bw))
    y2i = min(h, int(y2 + pad * bh))
    if x2i <= x1i or y2i <= y1i:
        return None, None
    return frame_bgr[y1i:y2i, x1i:x2i].copy(), (x1i, y1i, x2i, y2i)


def head_box_from_person(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1, y1, x2, y1 + 0.45 * (y2 - y1))


def _valid_kp(kps: np.ndarray, idx: int, thr: float = 0.25) -> bool:
    return idx < len(kps) and float(kps[idx, 2]) > thr


def _pt(kps: np.ndarray, idx: int, ox: int = 0, oy: int = 0) -> tuple[int, int]:
    return (int(kps[idx, 0] + ox), int(kps[idx, 1] + oy))


def _draw_capsule(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.line(img, p1, p2, color, max(2, thickness), cv2.LINE_AA)
    r = max(2, thickness // 2)
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)


def paint_body_avatar(
    canvas: np.ndarray,
    kps: np.ndarray | None,
    origin: tuple[int, int],
    skin: tuple[int, int, int],
    style: AvatarStyle,
) -> None:
    if kps is None or len(kps) < 17:
        return
    ox, oy = origin
    # Torso fill.
    torso_idx = [5, 6, 12, 11]
    if all(_valid_kp(kps, i) for i in torso_idx):
        poly = np.array([_pt(kps, i, ox, oy) for i in torso_idx], dtype=np.int32)
        cv2.fillConvexPoly(canvas, poly, style.cloth_bgr, cv2.LINE_AA)
        cv2.polylines(canvas, [poly], True, style.wire_bgr, 2, cv2.LINE_AA)

    # Limb thickness from shoulder span.
    if _valid_kp(kps, 5) and _valid_kp(kps, 6):
        span = float(np.linalg.norm(kps[5, :2] - kps[6, :2]))
    else:
        span = 40.0
    thick = max(6, int(span * style.limb_scale))

    for a, b in BODY_EDGES:
        if not (_valid_kp(kps, a) and _valid_kp(kps, b)):
            continue
        # Head + limbs = skin; shoulder/hip torso links = cloth.
        if {a, b} <= {5, 6, 11, 12}:
            color = style.cloth_bgr
        else:
            color = skin
        _draw_capsule(canvas, _pt(kps, a, ox, oy), _pt(kps, b, ox, oy), color, thick)

    # Wireframe on top.
    for a, b in BODY_EDGES:
        if _valid_kp(kps, a) and _valid_kp(kps, b):
            cv2.line(canvas, _pt(kps, a, ox, oy), _pt(kps, b, ox, oy), style.wire_bgr, 2, cv2.LINE_AA)
    for i in range(17):
        if _valid_kp(kps, i):
            cv2.circle(canvas, _pt(kps, i, ox, oy), 3, style.wire_bgr, -1, cv2.LINE_AA)

    # Head disc if nose present.
    if _valid_kp(kps, 0):
        nose = _pt(kps, 0, ox, oy)
        head_r = max(10, int(span * 0.35))
        overlay = canvas.copy()
        cv2.circle(overlay, nose, head_r, skin, -1, cv2.LINE_AA)
        cv2.circle(overlay, nose, head_r, style.wire_bgr, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)


def _face_pts(landmarks_xy: np.ndarray, idxs: list[int], ox: int, oy: int) -> np.ndarray:
    pts = []
    n = len(landmarks_xy)
    for i in idxs:
        if i < n:
            pts.append([int(landmarks_xy[i, 0] + ox), int(landmarks_xy[i, 1] + oy)])
    return np.array(pts, dtype=np.int32) if pts else np.zeros((0, 2), dtype=np.int32)


def paint_face_avatar(
    canvas: np.ndarray,
    landmarks_xy: np.ndarray | None,
    origin: tuple[int, int],
    skin: tuple[int, int, int],
    style: AvatarStyle,
) -> None:
    if landmarks_xy is None or len(landmarks_xy) < 100:
        return
    ox, oy = origin
    oval = _face_pts(landmarks_xy, FACE_OVAL, ox, oy)
    if len(oval) >= 3:
        cv2.fillConvexPoly(canvas, cv2.convexHull(oval), skin, cv2.LINE_AA)
        cv2.polylines(canvas, [oval], True, style.wire_bgr, 2, cv2.LINE_AA)

    # Soft feature fills.
    for idxs, fill in (
        (LEFT_EYE, (245, 245, 250)),
        (RIGHT_EYE, (245, 245, 250)),
        (LIPS_OUTER, (90, 70, 160)),
    ):
        pts = _face_pts(landmarks_xy, idxs, ox, oy)
        if len(pts) >= 3:
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), fill, cv2.LINE_AA)

    if style.face_wire:
        for idxs in (LEFT_EYE, RIGHT_EYE, LIPS_OUTER, LEFT_BROW, RIGHT_BROW, FACE_OVAL):
            pts = _face_pts(landmarks_xy, idxs, ox, oy)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], True, style.wire_bgr, 1, cv2.LINE_AA)


def make_background(frame_bgr: np.ndarray, style: AvatarStyle) -> np.ndarray:
    k = style.bg_blur if style.bg_blur % 2 == 1 else style.bg_blur + 1
    blur = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    return (blur.astype(np.float32) * style.bg_darken).astype(np.uint8)


def render_avatar_frame(
    frame_bgr: np.ndarray,
    body_kps: np.ndarray | None,
    body_origin: tuple[int, int] | None,
    face_landmarks_xy: np.ndarray | None,
    face_origin: tuple[int, int] | None,
    skin: tuple[int, int, int],
    style: AvatarStyle | None = None,
    person_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Compose flat avatar over blurred background."""
    style = style or AvatarStyle()
    out = make_background(frame_bgr, style)

    # Soft silhouette under painting (from person box).
    if person_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in person_xyxy]
        overlay = out.copy()
        cv2.ellipse(
            overlay,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
            0,
            0,
            360,
            style.cloth_bgr,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    if body_kps is not None and body_origin is not None:
        paint_body_avatar(out, body_kps, body_origin, skin, style)
    if face_landmarks_xy is not None and face_origin is not None:
        paint_face_avatar(out, face_landmarks_xy, face_origin, skin, style)
    return out
