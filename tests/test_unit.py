from pathlib import Path

import numpy as np
import pandas as pd

from video2vector.patient_select import PatientSelector
from video2vector.detect_track import Detection
from video2vector.timeline import rows_to_dataframe, summarize, write_timeline
from video2vector.pulse import _pos_bpm


def test_patient_selector_longest_track():
    sel = PatientSelector(mode="longest_track", min_track_seconds=0.5)
    for t in [0.0, 0.5, 1.0, 1.5]:
        people = [
            Detection(1, 0, "person", 0.9, (0, 0, 50, 100)),
            Detection(2, 0, "person", 0.8, (100, 0, 140, 80)),
        ]
        # track 1 present every frame; track 2 only early
        if t > 0.6:
            people = [people[0]]
        sel.observe(t, people)
    pid = sel.lock()
    assert pid == 1
    patient, others = sel.choose([Detection(1, 0, "person", 0.9, (0, 0, 50, 100)), Detection(2, 0, "person", 0.8, (100, 0, 140, 80))])
    assert patient is not None and patient.track_id == 1
    assert len(others) == 1


def test_timeline_roundtrip(tmp_path: Path):
    rows = [
        {
            "t": 0.0,
            "patient_track_id": 1,
            "n_people": 1,
            "n_others": 0,
            "n_animals": 0,
            "posture": "sitting",
            "pose_motion": 1.2,
            "pose_ok": True,
            "face_ok": True,
            "pain_score": 0.4,
            "asymmetry": 0.1,
            "eye_aspect": 0.2,
            "au_brow_lowerer": 0.1,
            "au_cheek_raiser": 0.0,
            "au_lid_tightener": 0.0,
            "au_nose_wrinkler": 0.0,
            "au_upper_lip_raiser": 0.0,
            "au_eye_closure": 0.0,
            "au_left_brow": 0.1,
            "au_right_brow": 0.05,
            "pulse_bpm": 72.0,
            "pulse_measurable": True,
            "pulse_reason": None,
            "breathing_rate": 12.0,
            "vocal_strain": 0.0,
            "has_voice": False,
            "too_dark": False,
            "too_bright": False,
            "too_blurry": False,
            "tracker_unsure": False,
        }
    ]
    df = rows_to_dataframe(rows)
    out = write_timeline(df, tmp_path / "t.parquet")
    assert out.exists()
    s = summarize(df)
    assert s["seconds"] == 1
    assert s["face_ok_frac"] == 1.0


def test_pos_bpm_synthetic_sine():
    fps = 30.0
    t = np.arange(0, 8, 1 / fps)
    # 1.2 Hz ~ 72 bpm reflected into RGB channels
    sig = np.sin(2 * np.pi * 1.2 * t)
    rgb = np.stack([0.1 * sig + 0.5, 0.2 * sig + 0.5, 0.05 * sig + 0.5], axis=1)
    bpm = _pos_bpm(rgb, fps)
    assert bpm is not None
    assert 55 <= bpm <= 95


def test_avatar_flat_paint():
    from video2vector.avatar import estimate_skin_tone, render_avatar_frame

    frame = np.full((240, 160, 3), 80, dtype=np.uint8)
    # Fake COCO-17 upright pose in crop coords.
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 2] = 0.9
    # nose
    kps[0, :2] = (80, 40)
    kps[5, :2] = (60, 70)  # L shoulder
    kps[6, :2] = (100, 70)
    kps[11, :2] = (65, 130)
    kps[12, :2] = (95, 130)
    kps[7, :2] = (45, 100)
    kps[8, :2] = (115, 100)
    kps[9, :2] = (40, 130)
    kps[10, :2] = (120, 130)
    kps[13, :2] = (65, 180)
    kps[14, :2] = (95, 180)
    kps[15, :2] = (65, 220)
    kps[16, :2] = (95, 220)
    skin = estimate_skin_tone(frame)
    out = render_avatar_frame(
        frame,
        body_kps=kps,
        body_origin=(0, 0),
        face_landmarks_xy=None,
        face_origin=None,
        skin=skin,
        person_xyxy=(40, 20, 120, 230),
    )
    assert out.shape == frame.shape
    # Avatar paint should differ from pure darkened blur.
    assert float(np.mean(np.abs(out.astype(float) - frame.astype(float)))) > 1.0
