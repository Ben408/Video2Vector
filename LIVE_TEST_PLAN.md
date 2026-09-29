# Live test plan — Video2Vector

Automated tests should pass (`pytest -q`, currently **9 passed**). Use this checklist for interactive checks on your machine.

## 0. One-time setup

```bash
cd video2vector
python -m pip install -e ".[dev]"
python scripts/make_fixtures.py
python scripts/build_demo_video.py
```

Confirm:

```bash
python -m pytest -q
video2vector info
```

---

## 1. Smoke — synthetic video

```bash
video2vector video tests/fixtures/synthetic_person.mp4 -o outputs/live_synthetic.parquet --max-seconds 2
```

**Pass if:** exit 0, parquet exists, summary `"seconds"` ≥ 1 (`face_ok_frac` may be 0).

---

## 2. People + pose — bundled demo

```bash
video2vector video data/video/people_demo.mp4 -o outputs/live_people.parquet --max-seconds 8
```

**Pass if:** `patient_track_id` set, `pose_ok_frac` > 0.5, `max_others` ≥ 1 when multiple people are visible. Distant faces may keep `face_ok_frac` low.

---

## 3. Close-up face (open stock)

Place a short open/CC0 close-face clip at `data/video/my_face_clip.mp4` (or use `scripts/download_live_media.py`), then:

```bash
video2vector video data/video/my_face_clip.mp4 -o outputs/live_face.parquet --max-seconds 20
```

**Pass if:** `face_ok_frac` high; AU / pain / asymmetry columns populate; pulse may gate on/off.

---

## 4. Multi-person clip

```bash
video2vector video data/video/crowd_street.mp4 -o outputs/live_crowd.parquet --max-seconds 20
```

**Pass if:** patient selected; `n_others` ≥ 1 on some seconds.

---

## 5. EEG

```bash
video2vector eeg tests/fixtures/synthetic.edf -o outputs/live_eeg.parquet
# or a local SeizeIT2 EDF under data/eeg/
```

**Pass if:** feature rows written; bandpower columns present.

---

## 6. Regression

```bash
python -m pytest -q
```

Must stay green before trusting a new clip result.

---

## 7. UI + avatar (optional)

```bash
video2vector-ui
```

Open `http://127.0.0.1:7860`, run a demo clip with **Render flat avatar video** enabled.

---

## What “good” looks like by stage

| Stage | Live signal |
|---|---|
| Decode | Probe fps/duration in summary JSON |
| Detect/track | `n_people` ≥ 1, stable `patient_track_id` |
| Drop others | `n_others` / `n_animals` counted but not used as patient |
| Pose | `pose_ok_frac` high; posture labels when body is visible |
| Face | `face_ok_frac` high on close-ups; AU columns populate |
| Pulse | `pulse_measurable` true only on steady face seconds |
| Audio | `has_voice` / `breathing_rate` when the file has audio |
| EEG | Parquet bandpowers for synthetic or local EDFs |
| Avatar | Flat toon + wireframe video under `outputs/ui/` |

---

## Known prototype limits (not bugs)

- Inputs are MP4/AVI and EDF/BDF only
- Face models need a reasonably large face; distant cameras report low `face_ok_frac`
- Age/sex estimates are placeholders (`null`) until an optional model is plugged in
- Open EEG corpora (e.g. SeizeIT2) typically have no paired room camera — video and EEG stay separate files here
- Do not commit `data/` media or `outputs/` — they are gitignored
- On some Windows networks, set `VIDEO2VECTOR_INSECURE_SSL=1` if model downloads fail TLS inspection

---

## When to stop and report back

1. CLI traceback (last ~30 lines)
2. People demo has `patient_track_id: null`
3. Close-up face clip has `face_ok_frac` = 0 for the whole run
4. Local EDF fails to open in `video2vector eeg`
