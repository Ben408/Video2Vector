from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioSecond:
    t: float
    breathing_rate: float | None
    vocal_strain: float | None
    has_voice: bool
    ok: bool
    reason: str | None = None


def extract_audio_features(video_path: str | Path, hop_seconds: float = 1.0, max_seconds: float | None = None) -> list[AudioSecond]:
    """Derive per-second breathing/voice proxies from the audio track.

    Uses energy envelope heuristics (no speaker embedding). If demucs/YAMNet
    are unavailable, this still produces usable activity and strain scores.
    """
    video_path = Path(video_path)
    try:
        import librosa
        import soundfile as sf  # noqa: F401
    except ImportError:
        return []

    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=max_seconds)
    except Exception:
        return [AudioSecond(0.0, None, None, False, False, "no_audio")]

    if y.size == 0:
        return [AudioSecond(0.0, None, None, False, False, "empty_audio")]

    hop = int(hop_seconds * sr)
    out: list[AudioSecond] = []
    for i, start in enumerate(range(0, len(y), hop)):
        chunk = y[start : start + hop]
        if chunk.size < hop // 4:
            break
        t = float(i * hop_seconds)
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        # Simple voice activity: mid-band energy
        spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
        voice_band = (freqs >= 300) & (freqs <= 3400)
        breath_band = (freqs >= 100) & (freqs <= 800)
        voice_e = float(spec[voice_band].mean()) if np.any(voice_band) else 0.0
        breath_e = float(spec[breath_band].mean()) if np.any(breath_band) else 0.0
        has_voice = voice_e > 0.02 and rms > 0.01
        # Breathing rate proxy: zero-crossings of lowpassed envelope in breath band
        env = np.abs(chunk)
        env = np.convolve(env, np.ones(int(0.05 * sr)) / max(int(0.05 * sr), 1), mode="same")
        env = env - env.mean()
        zc = np.where(np.diff(np.sign(env)) != 0)[0]
        br = None
        if not has_voice and breath_e > 0.005:
            # half the zero-crossings ~ cycles; clamp to plausible breaths/min
            cycles = len(zc) / 2.0
            br = float(np.clip(cycles / hop_seconds * 60.0 / 8.0, 0.0, 40.0))
        strain = float(np.clip(rms * 10.0, 0.0, 1.0)) if has_voice else 0.0
        out.append(
            AudioSecond(
                t=t,
                breathing_rate=br,
                vocal_strain=strain if has_voice else 0.0,
                has_voice=has_voice,
                ok=True,
            )
        )
    return out
