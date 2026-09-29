"""Lightweight Gradio UI for Video2Vector."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from video2vector.avatar_video import render_avatar_video
from video2vector.config import load_config, project_root
from video2vector.eeg import eeg_window_features, load_eeg
from video2vector.pipeline import run_video_pipeline
from video2vector.timeline import write_timeline

ROOT = project_root()
OUT = ROOT / "outputs" / "ui"
OUT.mkdir(parents=True, exist_ok=True)


def _preview_table(df: pd.DataFrame, cols: list[str], n: int = 30) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    return df[use].head(n)


def _long_chart(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    if "t" not in df.columns or not use:
        return pd.DataFrame({"t": [], "value": [], "metric": []})
    long = df[["t", *use]].melt(id_vars="t", value_vars=use, var_name="metric", value_name="value")
    return long.dropna()


def run_video(
    video_path: str | None,
    max_seconds: float,
    sample_fps: float,
    render_avatar: bool = True,
):
    if not video_path:
        raise gr.Error("Upload or choose a video first.")
    # Gradio may pass a dict in newer versions
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name")
    cfg = load_config()
    cfg["max_seconds"] = float(max_seconds) if max_seconds and max_seconds > 0 else None
    cfg["sample_fps"] = float(sample_fps)
    stem = Path(str(video_path)).stem
    out = OUT / f"ui_video_{stem}.parquet"
    summary = run_video_pipeline(str(video_path), out, config=cfg, show_progress=True)
    avatar_path = ""
    if render_avatar:
        avatar_out = OUT / f"ui_avatar_{stem}.mp4"
        render_avatar_video(
            str(video_path),
            avatar_out,
            config=cfg,
            show_progress=True,
            patient_track_id=summary.get("patient_track_id"),
        )
        avatar_path = str(avatar_out)
        summary["avatar_video"] = avatar_path
    df = pd.read_parquet(out)
    preview_cols = [
        "t",
        "n_people",
        "n_others",
        "n_animals",
        "posture",
        "pose_ok",
        "face_ok",
        "pain_score",
        "asymmetry",
        "au_brow_lowerer",
        "au_left_brow",
        "au_right_brow",
        "pulse_bpm",
        "pulse_measurable",
        "has_voice",
    ]
    stats = summary.get("timeline", {})
    headline = (
        f"### Patient track `{summary.get('patient_track_id')}` · "
        f"**{stats.get('seconds', 0)}s** · "
        f"face **{stats.get('face_ok_frac', 0):.0%}** · "
        f"pose **{stats.get('pose_ok_frac', 0):.0%}** · "
        f"others≤{stats.get('max_others', 0)} · animals≤{stats.get('max_animals', 0)}"
    )
    if avatar_path:
        headline += " · **avatar on**"
    chart_df = _long_chart(df, ["pain_score", "asymmetry", "pose_motion"])
    return (
        headline,
        json.dumps(summary, indent=2),
        _preview_table(df, preview_cols),
        chart_df,
        str(out),
        str(Path(out).with_suffix(".summary.json")),
        avatar_path if avatar_path else None,
    )


def run_eeg(eeg_file, max_seconds: float):
    if eeg_file is None:
        raise gr.Error("Upload an EDF file first.")
    eeg_path = eeg_file if isinstance(eeg_file, str) else getattr(eeg_file, "name", None) or str(eeg_file)
    cfg = load_config()
    info, data = load_eeg(
        eeg_path,
        target_sfreq=cfg["eeg"]["target_sfreq"],
        bandpass_hz=tuple(cfg["eeg"]["bandpass_hz"]),
    )
    if max_seconds and max_seconds > 0:
        n = int(max_seconds * info.sfreq)
        data = data[:, : min(n, data.shape[1])]
        info.duration_seconds = float(data.shape[1] / info.sfreq)
    df = eeg_window_features(data, info.sfreq, window_seconds=cfg["eeg"]["window_seconds"])
    out = OUT / f"ui_eeg_{Path(eeg_path).stem}.parquet"
    write_timeline(df, out)
    summary = {
        "eeg": str(eeg_path),
        "output": str(out),
        "sfreq": info.sfreq,
        "n_channels": info.n_channels,
        "duration_seconds": info.duration_seconds,
        "feature_rows": int(len(df)),
        "channels": info.channel_names[:16],
    }
    headline = (
        f"### {info.n_channels} channels · **{info.duration_seconds:.1f}s** · "
        f"**{len(df)}** feature rows · {info.sfreq:.0f} Hz"
    )
    preview_cols = ["t", "rms", "line_length", "bp_delta", "bp_theta", "bp_alpha", "bp_beta"]
    chart_df = _long_chart(df, ["rms", "line_length", "bp_alpha"])
    return headline, json.dumps(summary, indent=2), _preview_table(df, preview_cols), chart_df, str(out)


def _example_paths():
    video_dir = ROOT / "data" / "video"
    names = [
        "web_face_wikitongues.mp4",
        "web_crowd_street.mp4",
        "web_people_demo_short.mp4",
        "face_wikitongues.mp4",
        "crowd_street.mp4",
        "people_demo_short.mp4",
    ]
    return [str(video_dir / n) for n in names if (video_dir / n).exists()]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Video2Vector") as demo:
        gr.Markdown(
            """
            # Video2Vector
            Turn a study video into a **per-second patient timeline**, or an EDF into EEG features.
            Optional **flat avatar** replaces the body/face with a toon silhouette + wireframe.
            Demo uses open / CC0-style clips and local fixtures only.
            """
        )
        with gr.Tabs():
            with gr.Tab("Video → timeline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(label="Video", sources=["upload"])
                        max_sec = gr.Slider(0, 120, value=15, step=1, label="Max seconds (0 = full)")
                        sample_fps = gr.Slider(1, 15, value=5, step=1, label="Sample FPS")
                        avatar_toggle = gr.Checkbox(value=True, label="Render flat avatar video")
                        run_btn = gr.Button("Run Video2Vector", variant="primary")
                        examples = _example_paths()
                    with gr.Column(scale=1):
                        headline = gr.Markdown("Results appear here.")
                        summary = gr.Code(label="Summary JSON", language="json")
                        table = gr.Dataframe(label="Timeline preview")
                        chart = gr.LinePlot(
                            x="t",
                            y="value",
                            color="metric",
                            title="Face / motion scores over time",
                            x_title="Time (s)",
                            y_title="Score",
                        )
                        avatar_out = gr.Video(label="Flat avatar preview", interactive=False)
                        out_path = gr.Textbox(label="Parquet path")
                        summary_path = gr.Textbox(label="Summary path")
                run_btn.click(
                    run_video,
                    inputs=[video_in, max_sec, sample_fps, avatar_toggle],
                    outputs=[headline, summary, table, chart, out_path, summary_path, avatar_out],
                    api_name="run_video",
                )
                if examples:
                    gr.Examples(
                        examples=[[p, 12, 5, True] for p in examples[:3]],
                        inputs=[video_in, max_sec, sample_fps, avatar_toggle],
                        outputs=[headline, summary, table, chart, out_path, summary_path, avatar_out],
                        fn=run_video,
                        cache_examples=False,
                        run_on_click=True,
                        label="Local demo clips (click to run)",
                    )

            with gr.Tab("EEG → features"):
                with gr.Row():
                    with gr.Column(scale=1):
                        eeg_in = gr.File(label="EDF file", file_types=[".edf", ".bdf"])
                        eeg_max = gr.Slider(0, 300, value=30, step=1, label="Max seconds (0 = full)")
                        eeg_btn = gr.Button("Run EEG features", variant="primary")
                        synth = ROOT / "tests" / "fixtures" / "synthetic.edf"
                        if synth.exists():
                            gr.Examples(examples=[str(synth)], inputs=eeg_in, label="Synthetic EDF")
                    with gr.Column(scale=1):
                        eeg_headline = gr.Markdown("Results appear here.")
                        eeg_summary = gr.Code(label="Summary JSON", language="json")
                        eeg_table = gr.Dataframe(label="EEG feature preview")
                        eeg_chart = gr.LinePlot(
                            x="t",
                            y="value",
                            color="metric",
                            title="EEG feature time series",
                            x_title="Time (s)",
                            y_title="Value",
                        )
                        eeg_out = gr.Textbox(label="Parquet path")
                eeg_btn.click(
                    run_eeg,
                    inputs=[eeg_in, eeg_max],
                    outputs=[eeg_headline, eeg_summary, eeg_table, eeg_chart, eeg_out],
                    api_name="run_eeg",
                )

        gr.Markdown(
            "_Prototype · patient-only measurements · originals stay on your machine · "
            "see `LIVE_TEST_PLAN.md`._"
        )
    return demo


def main():
    demo = build_app()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="slate"),
    )


if __name__ == "__main__":
    main()
