# Video2Vector

Open prototype that turns **study video** into a **per-second patient timeline**, and **EEG (EDF)** into windowed bandpower features — with an optional **flat avatar** view that replaces the subject with a toon silhouette + wireframe.

No proprietary hospital formats. Inputs are ordinary **MP4/AVI** and **EDF/BDF**.

---

## Build plan (generic)

| # | Milestone | Status |
|---|-----------|--------|
| 1 | **Ingest** — decode video; load EDF; write one-row-per-second tables | **Done** |
| 2 | **Detect & track** — person/animal detection, multi-object tracking | **Done** (YOLO11 + ByteTrack) |
| 3 | **Patient selection** — pick subject track; count others separately | **Done** |
| 4 | **Pose** — body keypoints, posture, motion | **Done** (YOLO pose) |
| 5 | **Face** — landmarks, AU-like blendshape proxies, asymmetry, pain proxy | **Done** (MediaPipe Face Landmarker) |
| 6 | **Physiology proxies** — gated rPPG pulse; audio voice / breathing proxies | **Done** (baseline) |
| 7 | **EEG features** — resample, bandpass, per-window bandpowers | **Done** (synthetic + SeizeIT2-ready) |
| 8 | **CLI + tests** — `video2vector` commands; automated regression | **Done** |
| 9 | **Lightweight UI** — Gradio app for video/EEG runs | **Done** |
| 10 | **Privacy view** — flat avatar (blurred bg + toon body/face + wireframe) | **Done** (tier B) |
| 11 | **Open sample media** — CC0 / open clips + fixture generator | **Done** (download scripts; fixtures in-repo) |
| 12 | **Richer avatar** — 3D / Ready-Player-Me style driven by pose + blendshapes | Remaining |
| 13 | **Stronger demographics** — optional age/sex model (e.g. MiVOLO) behind a flag | Remaining |
| 14 | **Clinical validation set** — labeled bedside / EMU-style clips + metrics report | Remaining |
| 15 | **Packaging** — pinned releases, optional GPU wheel notes, CI | Remaining |
| 16 | **Downstream model** — timeline → risk / event head (train/eval loop) | Remaining |

---

## What’s in this demo

- **Video → timeline:** detection, tracking, patient lock, pose, face AUs, gated pulse, audio proxies, quality flags → Parquet + summary JSON  
- **EEG → features:** EDF load, bandpass, windowed RMS / line-length / δ θ α β  
- **UI:** Gradio at `http://127.0.0.1:7860` — example clips, charts, optional avatar render  
- **Avatar:** privacy-oriented flat toon over a blurred background  

Automated tests: `pytest -q` (currently **9 passed**).

---

## Setup

```bash
cd video2vector
python -m pip install -e ".[dev]"
python scripts/make_fixtures.py
python scripts/build_demo_video.py
```

Optional open media (not committed — large files):

```bash
python scripts/download_samples.py --eeg-instructions-only
python scripts/download_live_media.py
```

---

## Quick start

```bash
# CLI — video timeline
video2vector video tests/fixtures/synthetic_person.mp4 -o outputs/synthetic.parquet --max-seconds 2

# After building local demos
video2vector video data/video/people_demo.mp4 -o outputs/people.parquet --max-seconds 8

# EEG
video2vector eeg tests/fixtures/synthetic.edf -o outputs/eeg.parquet

# Gradio UI
video2vector-ui
```

Live checklist: see [`LIVE_TEST_PLAN.md`](LIVE_TEST_PLAN.md).

---

## Outputs

| Artifact | Description |
|----------|-------------|
| `*.parquet` | One row per second (video) or per EEG window |
| `*.summary.json` | Probe + coverage stats |
| `outputs/ui/ui_avatar_*.mp4` | Flat avatar render when enabled in the UI |

Timeline columns include posture, pose motion, face AU proxies (left/right), pain score, asymmetry, gated pulse, breathing/voice proxies, and quality flags.

---

## Design notes

- **Patient-only measurements** — other people/animals are counted, not used as the subject.  
- **Open data only in the demo path** — synthetic fixtures ship in-repo; stock/CC0 clips stay local under `data/` (gitignored).  
- **EEG** — works with synthetic EDF; SeizeIT2 (CC0) instructions under `data/eeg/` when you add files locally.  
- **Avatar** is a visualization/privacy layer; the numeric timeline is unchanged.

---

## License / attribution

Code is provided as an open prototype for research and demos. Third-party weights (Ultralytics YOLO, MediaPipe) and any stock media you download remain under their respective licenses. Do not commit clinical recordings or identifiable patient media.
