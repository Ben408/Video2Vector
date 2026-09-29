"""Headless smoke test for Gradio handlers (no browser required)."""
from pathlib import Path

from video2vector.app import run_eeg, run_video
from video2vector.config import project_root

ROOT = project_root()


def test_ui_video_handler():
    video = ROOT / "data" / "video" / "web_people_demo_short.mp4"
    if not video.exists():
        video = ROOT / "data" / "video" / "people_demo_short.mp4"
    assert video.exists()
    headline, summary, table, chart, out, summary_path, avatar = run_video(str(video), 3, 4, False)
    assert "Patient track" in headline or "patient" in headline.lower()
    assert Path(out).exists()
    assert len(table) >= 1
    assert avatar is None


def test_ui_avatar_render(tmp_path):
    video = ROOT / "data" / "video" / "web_people_demo_short.mp4"
    if not video.exists():
        video = ROOT / "data" / "video" / "people_demo_short.mp4"
    from video2vector.avatar_video import render_avatar_video
    from video2vector.config import load_config

    cfg = load_config()
    cfg["max_seconds"] = 2
    cfg["sample_fps"] = 4
    out = tmp_path / "avatar.mp4"
    path = render_avatar_video(video, out, config=cfg, show_progress=False)
    assert path.exists()
    assert path.stat().st_size > 1000


def test_ui_eeg_handler():
    edf = ROOT / "tests" / "fixtures" / "synthetic.edf"
    headline, summary, table, chart, out = run_eeg(str(edf), 5)
    assert "channels" in headline.lower()
    assert Path(out).exists()
    assert len(table) >= 1
