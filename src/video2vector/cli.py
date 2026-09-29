from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, project_root
from .eeg import eeg_window_features, load_bids_events, load_eeg
from .pipeline import run_video_pipeline
from .timeline import write_timeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="video2vector", description="Generic video/EEG → timeline features")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_video = sub.add_parser("video", help="Run Video2Vector on an MP4/AVI")
    p_video.add_argument("input", type=Path, help="Path to video file")
    p_video.add_argument("-o", "--output", type=Path, required=True, help="Output .parquet/.csv/.sqlite")
    p_video.add_argument("-c", "--config", type=Path, default=None)
    p_video.add_argument("--max-seconds", type=float, default=None)

    p_eeg = sub.add_parser("eeg", help="Extract per-second EEG features from EDF")
    p_eeg.add_argument("input", type=Path, help="Path to .edf")
    p_eeg.add_argument("-o", "--output", type=Path, required=True)
    p_eeg.add_argument("-c", "--config", type=Path, default=None)
    p_eeg.add_argument("--events", type=Path, default=None, help="Optional BIDS events.tsv")

    p_info = sub.add_parser("info", help="Show project paths")

    args = parser.parse_args(argv)
    if args.cmd == "info":
        print(json.dumps({"root": str(project_root())}, indent=2))
        return 0

    cfg = load_config(args.config)
    if args.cmd == "video":
        if args.max_seconds is not None:
            cfg["max_seconds"] = args.max_seconds
        summary = run_video_pipeline(args.input, args.output, config=cfg)
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "eeg":
        info, data = load_eeg(
            args.input,
            target_sfreq=cfg["eeg"]["target_sfreq"],
            bandpass_hz=tuple(cfg["eeg"]["bandpass_hz"]),
        )
        df = eeg_window_features(data, info.sfreq, window_seconds=cfg["eeg"]["window_seconds"])
        if args.events:
            events = load_bids_events(args.events)
            df.attrs["events_rows"] = len(events)
        out = write_timeline(df, args.output)
        print(
            json.dumps(
                {
                    "eeg": str(args.input),
                    "output": str(out),
                    "sfreq": info.sfreq,
                    "n_channels": info.n_channels,
                    "duration_seconds": info.duration_seconds,
                    "feature_rows": int(len(df)),
                    "channels": info.channel_names[:12],
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
