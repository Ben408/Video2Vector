"""Create synthetic short videos for CI (no real faces required)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_synthetic_person_video(path: Path, seconds: float = 3.0, fps: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    n = int(seconds * fps)
    for i in range(n):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # moving "person" rectangle + face circle
        x = 80 + int(20 * np.sin(i / 5))
        y = 60
        cv2.rectangle(frame, (x, y), (x + 80, y + 140), (180, 160, 140), -1)
        cv2.circle(frame, (x + 40, y + 35), 22, (200, 180, 160), -1)
        # eyes / mouth so face mesh has a chance on real models; still synthetic
        cv2.circle(frame, (x + 32, y + 32), 3, (20, 20, 20), -1)
        cv2.circle(frame, (x + 48, y + 32), 3, (20, 20, 20), -1)
        cv2.ellipse(frame, (x + 40, y + 45), (8, 4), 0, 0, 180, (20, 20, 20), 1)
        # second person briefly
        if 10 <= i <= 18:
            cv2.rectangle(frame, (220, 50), (300, 200), (100, 100, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def write_synthetic_eeg_edf(path: Path, seconds: float = 5.0, sfreq: float = 128.0) -> Path:
    """Write a minimal EDF via MNE for EEG path tests."""
    import mne
    from mne.export import export_raw

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    data = np.vstack(
        [
            1e-5 * np.sin(2 * np.pi * 10 * t),
            1e-5 * np.sin(2 * np.pi * 6 * t),
            1e-5 * np.random.randn(n) * 0.1,
        ]
    )
    info = mne.create_info(["Fz", "Cz", "Pz"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    # MNE export to EDF
    if path.exists():
        path.unlink()
    export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    print(write_synthetic_person_video(root / "synthetic_person.mp4"))
    print(write_synthetic_eeg_edf(root / "synthetic.edf"))
