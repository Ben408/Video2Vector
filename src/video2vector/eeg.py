from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EegInfo:
    path: Path
    sfreq: float
    n_channels: int
    duration_seconds: float
    channel_names: list[str]


def load_eeg(path: str | Path, target_sfreq: float = 128.0, bandpass_hz=(0.5, 40.0)):
    """Load EDF/BDF (SeizeIT2 / PhysioNet style) with MNE."""
    import mne

    path = Path(path)
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    raw.pick(picks="eeg", exclude="bads", verbose="ERROR")
    if bandpass_hz:
        raw.filter(bandpass_hz[0], bandpass_hz[1], verbose="ERROR")
    if target_sfreq and abs(raw.info["sfreq"] - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, verbose="ERROR")
    data = raw.get_data()  # (n_channels, n_times)
    info = EegInfo(
        path=path,
        sfreq=float(raw.info["sfreq"]),
        n_channels=int(data.shape[0]),
        duration_seconds=float(data.shape[1] / raw.info["sfreq"]),
        channel_names=list(raw.ch_names),
    )
    return info, data


def eeg_window_features(data: np.ndarray, sfreq: float, window_seconds: float = 1.0) -> pd.DataFrame:
    """Per-second bandpower and line-length features across channels."""
    win = max(int(window_seconds * sfreq), 1)
    n = data.shape[1]
    rows = []
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    for i, start in enumerate(range(0, n - win + 1, win)):
        seg = data[:, start : start + win]
        row = {"t": float(i * window_seconds), "n_channels": int(seg.shape[0])}
        # line length (seizure-ish activity proxy)
        row["line_length"] = float(np.abs(np.diff(seg, axis=1)).mean())
        row["rms"] = float(np.sqrt((seg**2).mean()))
        freqs = np.fft.rfftfreq(seg.shape[1], d=1.0 / sfreq)
        spec = np.abs(np.fft.rfft(seg, axis=1)).mean(axis=0)
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            row[f"bp_{name}"] = float(spec[mask].mean()) if np.any(mask) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def load_bids_events(events_tsv: str | Path) -> pd.DataFrame:
    path = Path(events_tsv)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    return df
