"""Render a flat-avatar privacy video alongside the timeline pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from .avatar import (
    AvatarStyle,
    crop_with_box,
    estimate_skin_tone,
    head_box_from_person,
    render_avatar_frame,
)
from .config import load_config
from .decode import iter_sampled_frames, probe_video
from .detect_track import DetectorTracker
from .face import FaceAnalyzer
from .patient_select import PatientSelector
from .pose import PoseEstimator
from .sslutil import allow_insecure_downloads_if_needed


def render_avatar_video(
    video_path: str | Path,
    out_path: str | Path,
    config: dict[str, Any] | None = None,
    show_progress: bool = True,
    patient_track_id: int | None = None,
) -> Path:
    """Write a flat toon/wireframe avatar video for the selected patient."""
    allow_insecure_downloads_if_needed()
    cfg = config or load_config()
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    sample_fps = float(cfg.get("sample_fps") or 5)
    max_seconds = cfg.get("max_seconds")
    style = AvatarStyle()

    detector = DetectorTracker(
        model_name=cfg["detection"]["model"],
        conf=cfg["detection"]["conf"],
        person_class=cfg["detection"]["person_class"],
        animal_classes=cfg["detection"]["animal_classes"],
        tracker=cfg["tracking"]["tracker"],
    )
    selector = PatientSelector(
        mode=cfg["patient"]["mode"],
        min_track_seconds=cfg["patient"]["min_track_seconds"],
    )
    pose_est = PoseEstimator()
    face_est = FaceAnalyzer(min_detection_confidence=cfg["face"]["min_detection_confidence"])

    frames = list(iter_sampled_frames(video_path, sample_fps, max_seconds))
    # Always observe+lock in this pass. ByteTrack IDs rematch after a detector
    # reset, so a patient_track_id from a prior pipeline run is only a hint.
    for t, frame in frames:
        people, _ = detector.process(frame)
        selector.observe(t, people)
    locked = selector.lock()
    if patient_track_id is None:
        patient_track_id = locked
    detector = DetectorTracker(
        model_name=cfg["detection"]["model"],
        conf=cfg["detection"]["conf"],
        person_class=cfg["detection"]["person_class"],
        animal_classes=cfg["detection"]["animal_classes"],
        tracker=cfg["tracking"]["tracker"],
    )
    # Prefer prior ID when it reappears; otherwise use freshly locked ID.
    preferred_id = patient_track_id
    selector.patient_id = preferred_id if preferred_id is not None else locked
    selector.locked = True

    # Use full video fps for smoother avatar playback when source is short.
    writer_fps = max(sample_fps, min(float(info.fps or sample_fps), 15.0))
    # Probe size from first frame
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    h, w = frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (w, h))

    skin_ema: tuple[int, int, int] | None = None
    iterator = frames
    if show_progress:
        iterator = tqdm(frames, desc="avatar", unit="frame")

    for _t, frame in iterator:
        people, _ = detector.process(frame)
        patient, _others = selector.choose(people)
        # If preferred track missing this frame, fall back to largest person.
        if patient is None and people:
            patient = max(people, key=lambda p: p.area)
            if preferred_id is not None:
                # Once preferred track shows up, stick to it.
                for p in people:
                    if p.track_id == preferred_id:
                        patient = p
                        selector.patient_id = preferred_id
                        break

        body_kps = None
        body_origin = None
        face_lm = None
        face_origin = None
        person_xyxy = patient.xyxy if patient else None
        skin = skin_ema or (180, 150, 130)

        if patient is not None:
            body_crop, body_box = crop_with_box(frame, patient.xyxy, pad=0.05)
            pose = pose_est.estimate(body_crop)
            if pose.ok and pose.keypoints is not None and body_box is not None:
                body_kps = pose.keypoints
                body_origin = (body_box[0], body_box[1])

            head_xyxy = head_box_from_person(patient.xyxy)
            face_crop, face_box = crop_with_box(frame, head_xyxy, pad=0.08)
            face = face_est.analyze(face_crop)
            tone_src = face_crop if face.ok else body_crop
            sample = estimate_skin_tone(tone_src)
            if skin_ema is None:
                skin_ema = sample
            else:
                skin_ema = tuple(
                    int(0.85 * a + 0.15 * b) for a, b in zip(skin_ema, sample)
                )  # type: ignore[assignment]
            skin = skin_ema
            if face.ok and face.landmarks_xy is not None and face_box is not None:
                face_lm = face.landmarks_xy
                face_origin = (face_box[0], face_box[1])

        avatar = render_avatar_frame(
            frame,
            body_kps=body_kps,
            body_origin=body_origin,
            face_landmarks_xy=face_lm,
            face_origin=face_origin,
            skin=skin,
            style=style,
            person_xyxy=person_xyxy,
        )
        writer.write(avatar)

    writer.release()
    return out_path
