from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .audio_features import extract_audio_features
from .config import load_config
from .decode import frame_quality_flags, iter_sampled_frames, probe_video
from .detect_track import DetectorTracker, crop_head_xyxy, crop_xyxy
from .face import FaceAnalyzer
from .patient_select import PatientSelector
from .pose import PoseEstimator
from .pulse import PulseEstimator
from .sslutil import allow_insecure_downloads_if_needed
from .timeline import rows_to_dataframe, summarize, write_timeline


def run_video_pipeline(
    video_path: str | Path,
    out_path: str | Path,
    config: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    allow_insecure_downloads_if_needed()
    cfg = config or load_config()
    video_path = Path(video_path)
    info = probe_video(video_path)

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
    pulse_est = PulseEstimator(
        window_seconds=cfg["pulse"]["window_seconds"],
        sample_fps=cfg["sample_fps"],
        max_motion=cfg["pulse"]["max_motion"],
        min_face_fraction=cfg["pulse"]["min_face_fraction"],
    )

    # Pass 1: observe tracks to lock patient identity.
    sample_fps = float(cfg["sample_fps"])
    max_seconds = cfg.get("max_seconds")
    frames = list(iter_sampled_frames(video_path, sample_fps, max_seconds))
    for t, frame in frames:
        people, _animals = detector.process(frame)
        selector.observe(t, people)
    patient_id = selector.lock()

    # Reset tracker state for a clean second pass (new model instance).
    detector = DetectorTracker(
        model_name=cfg["detection"]["model"],
        conf=cfg["detection"]["conf"],
        person_class=cfg["detection"]["person_class"],
        animal_classes=cfg["detection"]["animal_classes"],
        tracker=cfg["tracking"]["tracker"],
    )

    audio_rows = {
        round(a.t): a
        for a in extract_audio_features(
            video_path,
            hop_seconds=cfg["audio"]["hop_seconds"],
            max_seconds=max_seconds,
        )
    }

    # Aggregate sampled frames into 1-second buckets.
    buckets: dict[int, list[dict]] = {}
    iterator = frames
    if show_progress:
        iterator = tqdm(frames, desc="video2vector", unit="frame")

    for t, frame in iterator:
        q = frame_quality_flags(frame)
        people, animals = detector.process(frame)
        # Re-observe is unnecessary; patient already locked.
        patient, others = selector.choose(people)
        body_crop = crop_xyxy(frame, patient.xyxy) if patient is not None else None
        face_crop = crop_head_xyxy(frame, patient.xyxy) if patient is not None else None
        pose = pose_est.estimate(body_crop)
        face = face_est.analyze(face_crop)
        pulse = pulse_est.update(face_crop if face.ok else None, face.ok, pose.motion)
        sec = int(t)
        buckets.setdefault(sec, []).append(
            {
                "t_sample": t,
                "patient_track_id": patient.track_id if patient else patient_id,
                "n_people": len(people),
                "n_others": len(others),
                "n_animals": len(animals),
                "posture": pose.posture,
                "pose_motion": pose.motion,
                "pose_ok": pose.ok,
                "face_ok": face.ok,
                "pain_score": face.pain_score,
                "asymmetry": face.asymmetry,
                "eye_aspect": face.eye_aspect,
                "au_brow_lowerer": face.scores.get("brow_lowerer", 0.0),
                "au_cheek_raiser": face.scores.get("cheek_raiser", 0.0),
                "au_lid_tightener": face.scores.get("lid_tightener", 0.0),
                "au_nose_wrinkler": face.scores.get("nose_wrinkler", 0.0),
                "au_upper_lip_raiser": face.scores.get("upper_lip_raiser", 0.0),
                "au_eye_closure": face.scores.get("eye_closure", 0.0),
                "au_left_brow": face.left_scores.get("brow_lowerer", 0.0),
                "au_right_brow": face.right_scores.get("brow_lowerer", 0.0),
                "pulse_bpm": pulse.bpm,
                "pulse_measurable": pulse.measurable,
                "pulse_reason": pulse.reason,
                "too_dark": q["too_dark"],
                "too_bright": q["too_bright"],
                "too_blurry": q["too_blurry"],
                "tracker_unsure": patient is None,
            }
        )

    rows = []
    for sec in sorted(buckets):
        samples = buckets[sec]
        # Prefer last sample in the second (most recent measurements).
        s = samples[-1]
        audio = audio_rows.get(sec)
        rows.append(
            {
                "t": float(sec),
                **{k: v for k, v in s.items() if k != "t_sample"},
                "breathing_rate": audio.breathing_rate if audio else None,
                "vocal_strain": audio.vocal_strain if audio else None,
                "has_voice": audio.has_voice if audio else False,
            }
        )

    df = rows_to_dataframe(rows)
    out_path = write_timeline(df, out_path)
    summary = {
        "video": str(video_path),
        "output": str(out_path),
        "probe": {
            "fps": info.fps,
            "duration_seconds": info.duration_seconds,
            "width": info.width,
            "height": info.height,
        },
        "patient_track_id": patient_id,
        "timeline": summarize(df),
    }
    summary_path = Path(out_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
