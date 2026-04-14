# 3D_sway — Postural Sway Assessment System

Real-time biomechanical postural sway assessment combining an RGB camera with a TI IWR6843 mmWave radar.

---

## Overview

The system tracks whole-body centre of mass (CoM) movement and displays a real-time stabilogram — the same type of output produced by a laboratory force plate, but using only a camera and radar.

- **Mediolateral (ML) axis** — from hip midpoint detected by the RGB camera
- **Anteroposterior (AP) axis** — from TI IWR6843 mmWave radar range data (camera trunk-length proxy used until radar is connected)

---

## Hardware

| Component | Details |
|-----------|---------|
| RGB camera | Any USB webcam (tested: laptop webcam, PlayStation Eye) |
| Radar | TI IWR6843ISK mmWave radar via XDS110 USB |

---

## Software Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `opencv-python`, `mediapipe`, `numpy`, `scipy`

---

## Files

| File | Purpose |
|------|---------|
| `live_preview.py` | Main live display — run this |
| `camera_reader.py` | Camera capture + MediaPipe pose estimation |
| `radar_reader.py` | TI IWR6843 radar interface |
| `sensor_fusion.py` | CoM estimation, CoP correction |
| `acquire_session.py` | Batch session recording |
| `requirements.txt` | Python dependencies |

---

## Usage

```bash
python live_preview.py
```

| Key | Action |
|-----|--------|
| `R` | Start / stop recording |
| `C` | Clear stabilogram history |
| `Q` | Quit |

Buttons on the stabilogram panel toggle:
- **Reset** — clear history
- **Honest / Fudged** — AP axis scaling correction
- **CoM / CoP** — display mode
- **Hip / Dempster** — ML axis source

---

## Output

On recording stop, two files are saved to the working directory:

- `recording_<timestamp>.pkl` — full frame data (Python pickle)
- `recording_<timestamp>.csv` — cm-calibrated columns:

```
time_s, hip_ml_cm, ap_cm, com_ml_cm, com_y_cm
```

---

## Stabilogram

- Fixed **±6 cm** axes (isotropic, like a force plate display)
- **95% confidence ellipse** computed over a rolling 5-second window
- Metrics (CE, Path, MVELO) averaged over 5 seconds and displayed below the stabilogram

---

## Git Workflow

```bash
git add -A
git commit -m "brief description of what changed"
git push
```
