"""Record full demo including flat avatar videos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)


def _panel(text_lines: list[str], w: int = 1280, h: int = 720, title: str = "Video2Vector UI") -> np.ndarray:
    img = np.full((h, w, 3), 28, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w, 72), (40, 40, 48), -1)
    cv2.putText(img, title, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 240), 2, cv2.LINE_AA)
    y = 120
    for line in text_lines:
        cv2.putText(img, line[:90], (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 220), 2, cv2.LINE_AA)
        y += 36
    return img


def _write_clip(frames: list[np.ndarray], path: Path, fps: float = 8.0) -> Path:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f.shape[0] != h or f.shape[1] != w:
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    return path


def _hold(frame: np.ndarray, seconds: float, fps: float = 8.0) -> list[np.ndarray]:
    return [frame] * max(1, int(seconds * fps))


def _resize_vid(path: Path, w: int = 1280, h: int = 720, max_seconds: float = 8.0) -> list[np.ndarray]:
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 8)
    out = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx / fps > max_seconds:
            break
        out.append(cv2.resize(frame, (w, h)))
        idx += 1
    cap.release()
    step = max(1, int(fps / 8))
    return out[::step]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from video2vector.avatar_video import render_avatar_video
    from video2vector.config import load_config
    from video2vector.pipeline import run_video_pipeline

    face = ROOT / "data" / "video" / "web_face_wikitongues.mp4"
    if not face.exists():
        face = ROOT / "data" / "video" / "face_wikitongues.mp4"
    crowd = ROOT / "data" / "video" / "web_crowd_street.mp4"
    if not crowd.exists():
        crowd = ROOT / "data" / "video" / "crowd_street.mp4"

    cfg = load_config()
    cfg["max_seconds"] = 12
    cfg["sample_fps"] = 5

    results = {}
    print("timeline + avatar: face...")
    tl = OUT / "face_timeline.parquet"
    summary = run_video_pipeline(face, tl, config=cfg, show_progress=True)
    av_face = OUT / "demo_face_avatar.mp4"
    render_avatar_video(face, av_face, config=cfg, show_progress=True, patient_track_id=summary.get("patient_track_id"))
    results["face"] = {"summary": summary, "avatar": str(av_face)}

    print("timeline + avatar: crowd...")
    cfg2 = dict(cfg)
    cfg2["max_seconds"] = 15
    tl2 = OUT / "crowd_timeline.parquet"
    summary2 = run_video_pipeline(crowd, tl2, config=cfg2, show_progress=True)
    av_crowd = OUT / "demo_crowd_avatar.mp4"
    render_avatar_video(
        crowd, av_crowd, config=cfg2, show_progress=True, patient_track_id=summary2.get("patient_track_id")
    )
    results["crowd"] = {"summary": summary2, "avatar": str(av_crowd)}

    # Optional Gradio API if server is up
    try:
        c = Client("http://127.0.0.1:7860", verbose=False)
        print("UI API face+avatar...")
        r = c.predict(handle_file(str(face)), 10.0, 5.0, True, api_name="/run_video")
        results["ui_face"] = {"headline": r[0], "avatar": r[6]}
    except Exception as e:
        results["ui_face"] = {"error": str(e)}

    frames: list[np.ndarray] = []
    frames += _hold(
        _panel(
            [
                "Flat avatar mode (tier B)",
                "Blurred background + toon silhouette + face mesh wireframe",
                "Skin tone sampled from the patient crop",
                "Timeline metrics unchanged — avatar is a privacy view",
            ],
            title="Video2Vector — avatar demo",
        ),
        3,
    )
    frames += _hold(_panel(["Face clip — flat avatar"], title="Run 1"), 2)
    frames += _resize_vid(av_face, max_seconds=8)
    frames += _hold(_panel(["Crowd clip — patient avatar, others blurred"], title="Run 2"), 2)
    frames += _resize_vid(av_crowd, max_seconds=8)
    frames += _hold(
        _panel(
            [
                "Artifacts: outputs/demo/demo_*_avatar.mp4",
                "UI: checkbox 'Render flat avatar video'",
                "Open http://127.0.0.1:7860",
            ],
            title="Video2Vector — done",
        ),
        3,
    )

    demo_path = OUT / "full_ui_demo.mp4"
    _write_clip(frames, demo_path, fps=8.0)
    manifest = {
        "full_ui_demo": str(demo_path),
        "face_avatar": str(av_face),
        "crowd_avatar": str(av_crowd),
        "results": {
            "face_patient": summary.get("patient_track_id"),
            "face_timeline": summary.get("timeline"),
            "crowd_patient": summary2.get("patient_track_id"),
            "crowd_timeline": summary2.get("timeline"),
            "ui": results.get("ui_face"),
        },
    }
    (OUT / "full_ui_demo_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"wrote": str(demo_path), "frames": len(frames), "avatars": [str(av_face), str(av_crowd)]}, indent=2))


if __name__ == "__main__":
    main()
