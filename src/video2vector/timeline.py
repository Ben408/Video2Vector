from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "t",
    "patient_track_id",
    "n_people",
    "n_others",
    "n_animals",
    "posture",
    "pose_motion",
    "pose_ok",
    "face_ok",
    "pain_score",
    "asymmetry",
    "eye_aspect",
    "au_brow_lowerer",
    "au_cheek_raiser",
    "au_lid_tightener",
    "au_nose_wrinkler",
    "au_upper_lip_raiser",
    "au_eye_closure",
    "au_left_brow",
    "au_right_brow",
    "pulse_bpm",
    "pulse_measurable",
    "pulse_reason",
    "breathing_rate",
    "vocal_strain",
    "has_voice",
    "too_dark",
    "too_bright",
    "too_blurry",
    "tracker_unsure",
]


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FEATURE_COLUMNS].sort_values("t").reset_index(drop=True)


def write_timeline(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix in {".csv", ".tsv"}:
        df.to_csv(out_path, index=False, sep="\t" if suffix == ".tsv" else ",")
    elif suffix in {".sqlite", ".db"}:
        import sqlite3

        with sqlite3.connect(out_path) as conn:
            df.to_sql("timeline", conn, if_exists="replace", index=False)
    else:
        raise ValueError(f"Unsupported timeline suffix: {suffix}")
    return out_path


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"seconds": 0}
    return {
        "seconds": int(len(df)),
        "face_ok_frac": float(df["face_ok"].fillna(False).mean()),
        "pose_ok_frac": float(df["pose_ok"].fillna(False).mean()),
        "pulse_measurable_frac": float(df["pulse_measurable"].fillna(False).mean()),
        "mean_pain_score": float(df["pain_score"].fillna(0).mean()),
        "mean_asymmetry": float(df["asymmetry"].fillna(0).mean()),
        "posture_counts": df["posture"].fillna("unknown").value_counts().to_dict(),
        "max_others": int(df["n_others"].fillna(0).max()),
        "max_animals": int(df["n_animals"].fillna(0).max()),
    }
