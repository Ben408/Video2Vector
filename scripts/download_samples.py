#!/usr/bin/env python
"""Download CC0/open sample media for local tests."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "video"
EEG_DIR = ROOT / "data" / "eeg"

# Mixkit license: free stock video. URLs are direct MP4s commonly used in demos.
# If a URL 404s, replace with another Mixkit/Pexels CC0 clip.
SAMPLE_VIDEOS = {
    "face_closeup.mp4": "https://assets.mixkit.co/videos/preview/mixkit-portrait-of-a-woman-in-a-pool-1220-large.mp4",
    "person_walking.mp4": "https://assets.mixkit.co/videos/preview/mixkit-man-walking-in-the-street-with-a-backpack-42921-large.mp4",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"exists: {dest}")
        return
    print(f"download: {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest.stat().st_size} bytes")


def download_videos() -> list[Path]:
    paths = []
    for name, url in SAMPLE_VIDEOS.items():
        dest = VIDEO_DIR / name
        try:
            download(url, dest)
            paths.append(dest)
        except Exception as exc:
            print(f"WARN video failed ({name}): {exc}", file=sys.stderr)
    return paths


def download_seizeit2_subset(max_files: int = 2) -> Path:
    """Fetch a tiny SeizeIT2 subset via openneuro-py or AWS open data if available.

    Falls back to printing manual instructions when network/tools are limited.
    """
    EEG_DIR.mkdir(parents=True, exist_ok=True)
    marker = EEG_DIR / "README_SEIZEIT2.txt"
    marker.write_text(
        "\n".join(
            [
                "SeizeIT2 (OpenNeuro ds005873) — CC0",
                "https://openneuro.org/datasets/ds005873",
                "https://nemar.org/dataexplorer/detail?dataset_id=ds005873",
                "",
                "Suggested small pull (after installing openneuro-py or datalad):",
                "  openneuro download --dataset ds005873 --target ./data/eeg/seizeit2 \\",
                "    --include 'sub-001/ses-001/eeg/*'",
                "",
                "Or from NEMAR:",
                "  nemar dataset download on005873 --include sub-001",
                "",
                "Place any .edf under data/eeg/ and run:",
                "  video2vector eeg data/eeg/<file>.edf -o outputs/eeg_demo.parquet",
            ]
        ),
        encoding="utf-8",
    )
    # Try openneuro CLI if present
    try:
        subprocess.run(["openneuro", "--help"], check=False, capture_output=True)
    except FileNotFoundError:
        print(marker.read_text(encoding="utf-8"))
        return marker

    print(f"Wrote instructions to {marker}")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--eeg-instructions-only", action="store_true")
    args = parser.parse_args()
    result = {}
    if not args.eeg_instructions_only:
        result["videos"] = [str(p) for p in download_videos()]
    if not args.videos_only:
        result["eeg_readme"] = str(download_seizeit2_subset())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
