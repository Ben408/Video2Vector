"""Record a self-contained demo video of Video2Vector results (no screen capture needed)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from video2vector.config import load_config
from video2vector.pipeline import run_video_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)


def annotate_video(video_path: Path, timeline: Path, out_path: Path, max_seconds: float = 20.0) -> Path:
    df = pd.read_parquet(timeline).set_index("t")
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # side panel for HUD
    panel_w = 420
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w + panel_w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t > max_seconds:
            break
        sec = int(t)
        row = df.loc[sec] if sec in df.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = (24, 24, 28)
        lines = [
            "Video2Vector demo",
            f"t = {t:5.1f}s",
            "",
        ]
        if row is not None:
            lines += [
                f"people  {int(row.get('n_people', 0))}",
                f"others  {int(row.get('n_others', 0))}",
                f"animals {int(row.get('n_animals', 0))}",
                f"posture {row.get('posture')}",
                f"face_ok {bool(row.get('face_ok'))}",
                f"pose_ok {bool(row.get('pose_ok'))}",
                f"pain    {float(row.get('pain_score') or 0):.2f}",
                f"asym    {float(row.get('asymmetry') or 0):.2f}",
                f"pulse   {row.get('pulse_bpm') if row.get('pulse_measurable') else 'n/a'}",
            ]
        y0 = 36
        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (w + 20, y0 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 230),
                2,
                cv2.LINE_AA,
            )
        writer.write(canvas)
        idx += 1
    cap.release()
    writer.release()
    return out_path


def main():
    face = ROOT / "data" / "video" / "face_wikitongues.mp4"
    crowd = ROOT / "data" / "video" / "crowd_street.mp4"
    cfg = load_config()
    cfg["max_seconds"] = 15
    cfg["sample_fps"] = 5

    demos = []
    if face.exists():
        tl = OUT / "face_timeline.parquet"
        summary = run_video_pipeline(face, tl, config=cfg, show_progress=True)
        demo = annotate_video(face, tl, OUT / "demo_face_hud.mp4", max_seconds=15)
        demos.append({"clip": "face", "video": str(demo), "summary": summary})
    if crowd.exists():
        tl = OUT / "crowd_timeline.parquet"
        cfg2 = dict(cfg)
        cfg2["max_seconds"] = 20
        summary = run_video_pipeline(crowd, tl, config=cfg2, show_progress=True)
        demo = annotate_video(crowd, tl, OUT / "demo_crowd_hud.mp4", max_seconds=20)
        demos.append({"clip": "crowd", "video": str(demo), "summary": summary})

    manifest = OUT / "demo_manifest.json"
    manifest.write_text(json.dumps(demos, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "demos": demos}, indent=2, default=str))


if __name__ == "__main__":
    main()
