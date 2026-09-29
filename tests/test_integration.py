from pathlib import Path

import pytest

from video2vector.eeg import eeg_window_features, load_eeg
from video2vector.pipeline import run_video_pipeline
from video2vector.timeline import write_timeline

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures():
    import importlib.util

    fixtures_script = Path(__file__).resolve().parents[1] / "scripts" / "make_fixtures.py"
    spec = importlib.util.spec_from_file_location("make_fixtures", fixtures_script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    video = FIXTURES / "synthetic_person.mp4"
    edf = FIXTURES / "synthetic.edf"
    if not video.exists():
        mod.write_synthetic_person_video(video, seconds=2.5, fps=8)
    if not edf.exists():
        mod.write_synthetic_eeg_edf(edf, seconds=3.0, sfreq=128.0)
    return video, edf


def test_video_pipeline_smoke(ensure_fixtures, tmp_path):
    video, _ = ensure_fixtures
    out = tmp_path / "timeline.parquet"
    summary = run_video_pipeline(
        video,
        out,
        config={
            "sample_fps": 4,
            "write_every_seconds": 1.0,
            "max_seconds": 2.0,
            "detection": {
                "model": "yolo11n.pt",
                "conf": 0.1,
                "person_class": 0,
                "animal_classes": [14, 15, 16],
            },
            "tracking": {"tracker": "bytetrack.yaml"},
            "patient": {"mode": "longest_track", "min_track_seconds": 0.2},
            "face": {"max_faces": 1, "min_detection_confidence": 0.3},
            "pulse": {"window_seconds": 2.0, "min_face_fraction": 0.2, "max_motion": 100.0},
            "audio": {"hop_seconds": 1.0},
            "eeg": {"target_sfreq": 128, "window_seconds": 1.0, "bandpass_hz": [0.5, 40.0]},
        },
        show_progress=False,
    )
    assert out.exists()
    assert summary["timeline"]["seconds"] >= 1


def test_eeg_pipeline_smoke(ensure_fixtures, tmp_path):
    _, edf = ensure_fixtures
    info, data = load_eeg(edf, target_sfreq=128.0)
    df = eeg_window_features(data, info.sfreq, window_seconds=1.0)
    out = write_timeline(df, tmp_path / "eeg.parquet")
    assert out.exists()
    assert len(df) >= 2
    assert "bp_alpha" in df.columns
