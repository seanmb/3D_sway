"""
Live preview for RGB camera pose estimation.
Shows camera feed with skeleton overlay and a real-time stabilogram.

The stabilogram plots CoM path in the horizontal plane:
  X axis — mediolateral (ML) sway  — from camera hip midpoint
  Y axis — anteroposterior (AP) sway — from radar range (zeros until radar connected)

A 95% confidence ellipse is drawn once enough data is collected.

Controls:
  Q  — quit
  R  — start / stop recording
  C  — clear stabilogram history
"""

import sys
import time
import collections
import logging
import pickle
import datetime
sys.path.insert(0, r'C:\KINECAL\rgb_radar_acquisition')

import cv2
import numpy as np




from camera_reader import CameraReader, CameraFrame
from sensor_fusion import estimate_cop

logging.basicConfig(level=logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE_INDEX      = 0
SUBJECT_HEIGHT_M  = 1.70
HISTORY_S         = 5          # seconds of sway to show in stabilogram
STAB_SIZE         = 320         # stabilogram panel: square pixels
MIN_ELLIPSE_PTS   = 60          # minimum samples before drawing ellipse
ELLIPSE_N_STD     = 1.96        # 95% confidence ellipse
ELLIPSE_WINDOW_S  = 5.0         # seconds used for ellipse computation
STAB_AXIS_CM      = 6.0         # half-range of each axis in cm (when px_per_m calibrated)
STAB_AXIS_PX      = 50          # half-range in pixels (fallback before calibration)
ASSUMED_HFOV_DEG  = 70.0        # Logitech C922 Pro horizontal FOV (~70 deg at 16:9)
AP_GAIN           = 3.0         # default AP gain — trunk proxy needs amplification to match ML scale
DISPLAY_WIDTH     = 640         # canvas display width (capture may be larger)

# ── Radar (IWR6843) ────────────────────────────────────────────────────────────
# Set both ports to enable radar.  Leave as None to run camera-only.
# Find ports: Device Manager → Ports (COM & LPT)
#   XDS110 Class Application/User UART  → RADAR_CONFIG_PORT  (115200 baud)
#   XDS110 Class Auxiliary Data Port    → RADAR_DATA_PORT    (921600 baud)
RADAR_CONFIG_PORT = 'COM7'  # XDS110 Application/User UART  (mmWaveICBOOST)
RADAR_DATA_PORT   = 'COM6'  # XDS110 Auxiliary Data Port     (mmWaveICBOOST)

AUTO_COUNTDOWN_S  = 30          # countdown before auto-trial starts recording
AUTO_RECORD_S     = 30          # duration of auto-trial recording
METRICS_H         = 100         # height of metrics strip below stabilogram (pixels)
METRICS_WINDOW_S  = 5.0         # rolling average window for displayed metrics (seconds)

# ── Skeleton connections ───────────────────────────────────────────────────────
SKELETON = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (28, 32),
]
LEFT_COLOUR  = (0, 200, 0)
RIGHT_COLOUR = (200, 0, 0)
MID_COLOUR   = (200, 200, 0)
HIP_COLOUR   = (0, 100, 255)
LEFT_INDICES  = {13, 15, 17, 19, 21, 25, 27, 29, 31, 23}
RIGHT_INDICES = {14, 16, 18, 20, 22, 26, 28, 30, 32, 24}


def joint_colour(idx):
    if idx in LEFT_INDICES:  return LEFT_COLOUR
    if idx in RIGHT_INDICES: return RIGHT_COLOUR
    return MID_COLOUR


def put_text(img, text, pos, scale, colour, thickness=1):
    """putText with a 1-pixel dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def draw_skeleton(frame, landmarks):
    if landmarks is None:
        return
    name_to_idx = {
        'nose': 0,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13,    'right_elbow': 14,
        'left_wrist': 15,    'right_wrist': 16,
        'left_hip': 23,      'right_hip': 24,
        'left_knee': 25,     'right_knee': 26,
        'left_ankle': 27,    'right_ankle': 28,
        'left_foot': 31,     'right_foot': 32,
    }
    pts = {k: (int(v[0]), int(v[1])) for k, v in landmarks.items()}
    idx_to_pt = {v: pts[k] for k, v in name_to_idx.items() if k in pts}
    for a, b in SKELETON:
        if a in idx_to_pt and b in idx_to_pt:
            col = LEFT_COLOUR if a in LEFT_INDICES else RIGHT_COLOUR if a in RIGHT_INDICES else MID_COLOUR
            cv2.line(frame, idx_to_pt[a], idx_to_pt[b], col, 2, cv2.LINE_AA)
    for idx, pt in idx_to_pt.items():
        cv2.circle(frame, pt, 5, joint_colour(idx), -1, cv2.LINE_AA)

    # Hip midpoint — small reference dot only
    if 23 in idx_to_pt and 24 in idx_to_pt:
        lh, rh = idx_to_pt[23], idx_to_pt[24]
        mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        cv2.circle(frame, mid, 4, HIP_COLOUR, -1, cv2.LINE_AA)


# ── Stabilogram ───────────────────────────────────────────────────────────────

def compute_ellipse(ml, ap):
    """
    Return (width, height, angle_deg, area) of the 95% confidence ellipse,
    or None if there is insufficient valid data.
    Assumes ml and ap are already mean-centred — centre is always (0, 0).
    """
    ml = np.array(ml, dtype=np.float64)
    ap = np.array(ap, dtype=np.float64)
    valid = np.isfinite(ml) & np.isfinite(ap)
    ml, ap = ml[valid], ap[valid]
    if len(ml) < MIN_ELLIPSE_PTS:
        return None
    cov = np.cov(ml, ap)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)):
        return None
    angle = float(np.degrees(np.arctan2(*eigvecs[:, 1][::-1])))
    axes  = 2.4478 * np.sqrt(np.maximum(eigvals, 0))
    ew, eh = float(axes[1] * 2), float(axes[0] * 2)
    area   = float(np.pi * axes[0] * axes[1])
    return (ew, eh, angle, area)


def draw_stabilogram(ml_hist, ap_hist, ts_hist, fps, px_per_m=None, fudge_ap=False, show_cop=False, use_dempster=False, n_std=ELLIPSE_N_STD, trial_elapsed=0.0):
    """
    Render the stabilogram into a square BGR panel.
    ML = horizontal axis, AP = vertical axis (positive = forward).

    ml_hist, ap_hist, ts_hist are 1-D numpy arrays of equal length.
    Windowing and duration are computed from actual timestamps — immune
    to FPS variation.
    """
    S = STAB_SIZE
    panel = np.full((S, S, 3), 20, dtype=np.uint8)

    margin = 30
    plot_s = S - margin * 2
    half = plot_s / 2.0

    ml_all = np.asarray(ml_hist, dtype=np.float64)
    ap_all = np.asarray(ap_hist, dtype=np.float64)
    ts     = np.asarray(ts_hist, dtype=np.float64)

    if len(ts) < 2:
        cv2.putText(panel, "Collecting...", (margin, S // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        return panel, {'ce': None, 'path': 0.0, 'mvelo': 0.0, 'calibrated': False}

    t_end = ts[-1]

    # Time-based display window (last HISTORY_S seconds)
    disp_mask = ts >= t_end - HISTORY_S
    ml = ml_all[disp_mask]
    ap = ap_all[disp_mask]
    ts_disp = ts[disp_mask]

    # Time-based ellipse window (last ELLIPSE_WINDOW_S seconds, min MIN_ELLIPSE_PTS)
    ell_mask = ts >= t_end - ELLIPSE_WINDOW_S
    ml_ell = ml_all[ell_mask]
    ap_ell = ap_all[ell_mask]

    if len(ml) < 2:
        cv2.putText(panel, "Collecting...", (margin, S // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        return panel, {'ce': None, 'path': 0.0, 'mvelo': 0.0, 'calibrated': False}

    # Mean-centre
    ml  = ml  - np.nanmean(ml)
    ap  = ap  - np.nanmean(ap)
    ml_ell = ml_ell - np.nanmean(ml_ell)
    ap_ell = ap_ell - np.nanmean(ap_ell)

    # ── Convert to cm (isotropic) ────────────────────────────────────────────
    # Both axes use the SAME px→cm conversion so scale is isotropic.
    # AP proxy is in trunk-height-change pixels, not lateral-movement pixels,
    # so its cm values are smaller — that's honest.  Fudge correction (applied
    # upstream) rescales AP to equivalent lateral pixels before we get here.
    calibrated = px_per_m and px_per_m > 0
    if calibrated:
        cm_per_px = 100.0 / px_per_m
        ml     *= cm_per_px
        ap     *= cm_per_px
        ml_ell *= cm_per_px
        ap_ell *= cm_per_px
        axis_range = STAB_AXIS_CM            # ±N cm
    else:
        axis_range = float(STAB_AXIS_PX)     # ±N px before calibration

    # Fixed isotropic scale — same for both axes (like a force plate)
    scale = half / max(axis_range, 1e-6)

    # ── Path metrics (computed before drawing) ────────────────────────────────
    valid_mask = np.isfinite(ml) & np.isfinite(ap)
    if valid_mask.sum() > 1:
        ml_v, ap_v = ml[valid_mask], ap[valid_mask]
        path_length = float(np.sum(np.sqrt(np.diff(ml_v)**2 + np.diff(ap_v)**2)))
        duration_s  = float(ts_disp[-1] - ts_disp[0]) if len(ts_disp) > 1 else len(ml) / max(fps, 1.0)
        mvelo       = path_length / max(duration_s, 0.001)
    else:
        path_length = mvelo = 0.0

    def to_px(m, a):
        x = int(margin + half + m * scale)
        y = int(margin + half - a * scale)
        return (np.clip(x, 0, S - 1), np.clip(y, 0, S - 1))

    # ── Grid ─────────────────────────────────────────────────────────────────
    cx = cy = margin + plot_s // 2

    # Faint tick lines at ±2 and ±4 cm (only when calibrated and within range)
    if calibrated:
        for tick in [2.0, 4.0]:
            if tick < axis_range * 0.95:
                for sign in (-1, 1):
                    tx = int(margin + half + sign * tick * scale)
                    ty = int(margin + half - sign * tick * scale)
                    cv2.line(panel, (tx, margin), (tx, margin + plot_s), (35, 35, 35), 1)
                    cv2.line(panel, (margin, ty), (margin + plot_s, ty), (35, 35, 35), 1)
                    cv2.putText(panel, f"{int(sign * tick)}",
                                (tx - 5, cy + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (65, 65, 65), 1)

    cv2.line(panel, (margin, cy), (margin + plot_s, cy), (50, 50, 50), 1)
    cv2.line(panel, (cx, margin), (cx, margin + plot_s), (50, 50, 50), 1)

    # ── 95% CE on 5 s window ─────────────────────────────────────────────────
    # Computed in cm (or px) space — isotropic, so angle is correct.
    ellipse = compute_ellipse(ml_ell, ap_ell)
    ce_area = None
    if ellipse is not None:
        ew, eh, eangle, ce_area = ellipse
        centre_px = to_px(0.0, 0.0)
        axes_px = (max(1, int(ew / 2 * scale)), max(1, int(eh / 2 * scale)))
        cv2.ellipse(panel, centre_px, axes_px, -eangle, 0, 360,
                    (0, 180, 255), 2, cv2.LINE_AA)

    # ── Sway path (blue=old → red=new) ───────────────────────────────────────
    n = len(ml)
    for i in range(1, n):
        if np.isnan(ml[i-1]) or np.isnan(ml[i]) or np.isnan(ap[i-1]) or np.isnan(ap[i]):
            continue
        t = i / max(n - 1, 1)
        colour = (int(220 * (1 - t)), 80, int(220 * t))
        cv2.line(panel, to_px(ml[i-1], ap[i-1]), to_px(ml[i], ap[i]),
                 colour, 1, cv2.LINE_AA)

    # ── Current position dot ──────────────────────────────────────────────────
    if not np.isnan(ml[-1]) and not np.isnan(ap[-1]):
        cv2.circle(panel, to_px(ml[-1], ap[-1]), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # ── Axis labels ───────────────────────────────────────────────────────────
    cv2.putText(panel, "ML",  (S - margin - 12, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    cv2.putText(panel, "AP",  (cx + 4, margin + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    cv2.putText(panel, "Stabilogram",  (margin, margin - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    mode   = "CoP" if show_cop else "CoM"
    ml_tag = "Demps" if use_dempster else "Hip"
    ap_tag = "fudged" if fudge_ap else "honest"
    rng    = f"+/-{axis_range:.0f}{'cm' if calibrated else 'px'}"
    cv2.putText(panel, f"{mode} ML:{ml_tag} AP:{ap_tag} {rng}",
                (margin, margin + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1)

    return panel, {'ce': ce_area, 'path': path_length, 'mvelo': mvelo,
                   'calibrated': calibrated}


def draw_metrics_strip(ce, path_len, mvelo, calibrated, trial_elapsed, width, height):
    """Large, readable metrics panel shown below the stabilogram."""
    panel = np.full((height, width, 3), 12, dtype=np.uint8)
    cv2.line(panel, (0, 0), (width, 0), (60, 60, 60), 2)

    unit_sq = "cm2" if calibrated else "px2"   # HERSHEY font has no Unicode superscripts
    unit_l  = "cm"  if calibrated else "px"

    entries = [
        ("CE",    f"{ce:.2f} {unit_sq}"       if ce       is not None else "--", (0,   200, 255)),
        ("Path",  f"{path_len:.2f} {unit_l}"  if path_len is not None else "--", (100, 255, 120)),
        ("MVELO", f"{mvelo:.2f} {unit_l}/s"   if mvelo    is not None else "--", (255, 215,  50)),
    ]

    row_h = (height - 20) // len(entries)
    for i, (label, value, color) in enumerate(entries):
        y_label = 18 + i * row_h
        y_value = y_label + row_h - 12
        put_text(panel, label, (10, y_label), 0.55, (150, 150, 150))
        put_text(panel, value, (10, y_value), 1.40, color, 3)

    # Trial timer bottom-right
    t_color = (100, 255, 100) if trial_elapsed >= 30 else \
              (255, 220,  50) if trial_elapsed >= 20 else \
              (180, 180, 180)
    t_str = f"T:{trial_elapsed:.0f}s" + (" DONE" if trial_elapsed >= 30 else "")
    put_text(panel, t_str, (width - 140, height - 10), 0.65, t_color, 2)

    return panel


def draw_radar_trace(radar_ap_history, panel_w, panel_h, history_s=HISTORY_S):
    """
    1-D radar AP waveform: scrolling time-series of mean-centred range (cm).

    Radar gives absolute range in metres.  We mean-centre each display window
    so small postural sway (±2–5 cm) fills the panel regardless of stand-off
    distance.  Y scale auto-fits to 2.5× the signal std (min ±2 cm).

    Returns (panel BGR, metrics_dict) where metrics contain:
      ap_rms   — RMS of mean-centred AP signal (cm)
      ap_range — peak-to-peak range (cm)
      mvelo    — mean velocity of AP signal (cm/s)
    """
    panel = np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
    mg_l, mg_r = 40, 8
    mg_t, mg_b = 20, 22
    plot_w = panel_w - mg_l - mg_r
    plot_h = panel_h - mg_t - mg_b

    empty_metrics = {'ap_rms': None, 'ap_range': None, 'mvelo': None}

    if len(radar_ap_history) < 2:
        cv2.putText(panel, "Radar AP: waiting...", (mg_l, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
        return panel, empty_metrics

    # Clip to last history_s seconds
    t_end  = radar_ap_history[-1][0]
    recent = [(t, v) for t, v in radar_ap_history if t >= t_end - history_s]
    if len(recent) < 2:
        recent = list(radar_ap_history)

    ts_raw = np.array([t for t, v in recent], dtype=np.float64)
    vals   = np.array([v for t, v in recent], dtype=np.float64) * 100.0  # m → cm
    vals  -= np.nanmean(vals)                                             # mean-centre

    # Y-axis half-range: 2.5× std, at least ±2 cm
    y_half = max(2.0, float(np.nanstd(vals)) * 2.5)

    # Normalised time: 0 = oldest shown, 1 = now
    t_norm = (ts_raw - (t_end - history_s)) / max(history_s, 1e-6)

    def to_px(t_n, v_cm):
        x = int(mg_l + np.clip(t_n, 0.0, 1.0) * plot_w)
        y = int(mg_t + plot_h // 2 - (v_cm / y_half) * (plot_h / 2))
        return (np.clip(x, mg_l, mg_l + plot_w),
                np.clip(y, mg_t, mg_t + plot_h))

    # Grid lines
    cy = mg_t + plot_h // 2
    cv2.line(panel, (mg_l, cy), (mg_l + plot_w, cy), (55, 55, 55), 1)
    for tick in [y_half * 0.5]:                       # mid-scale tick
        for sign in (-1, 1):
            ty = int(mg_t + plot_h / 2 - sign * tick / y_half * (plot_h / 2))
            cv2.line(panel, (mg_l, ty), (mg_l + plot_w, ty), (38, 38, 38), 1)
    for t_s in range(1, int(history_s) + 1):          # vertical time ticks
        tx = int(mg_l + (t_s / history_s) * plot_w)
        cv2.line(panel, (tx, mg_t + plot_h), (tx, mg_t + plot_h + 4), (55, 55, 55), 1)

    # Waveform (blue-old → green-new gradient)
    n = len(vals)
    for i in range(1, n):
        if np.isnan(vals[i - 1]) or np.isnan(vals[i]):
            continue
        t_frac = i / max(n - 1, 1)
        colour = (int(40 + 140 * t_frac), int(200 * t_frac), int(220 * (1 - t_frac)))
        cv2.line(panel, to_px(t_norm[i - 1], vals[i - 1]),
                 to_px(t_norm[i], vals[i]), colour, 2, cv2.LINE_AA)

    # Current position dot
    if np.isfinite(vals[-1]):
        cv2.circle(panel, to_px(t_norm[-1], vals[-1]), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Axis labels
    put_text(panel, "Radar AP",      (mg_l + 2, mg_t - 4),         0.38, (200, 200, 200))
    put_text(panel, "cm",            (4, cy + 4),                   0.30, (110, 110, 110))
    put_text(panel, f"+{y_half:.1f}", (4, mg_t + 8),                0.28, (90,  90,  90))
    put_text(panel, f"-{y_half:.1f}", (4, mg_t + plot_h - 2),       0.28, (90,  90,  90))
    put_text(panel, f"{history_s:.0f}s",
             (mg_l + plot_w - 14, mg_t + plot_h + mg_b - 4),        0.28, (90,  90,  90))

    # Metrics
    duration = float(ts_raw[-1] - ts_raw[0]) if len(ts_raw) > 1 else 1.0
    ap_rms   = float(np.nanstd(vals))
    ap_range = float(np.nanmax(vals) - np.nanmin(vals))
    mvelo    = float(np.sum(np.abs(np.diff(vals)))) / max(duration, 1e-3)
    return panel, {'ap_rms': ap_rms, 'ap_range': ap_range, 'mvelo': mvelo}


def draw_metrics_strip_radar(ap_rms, ap_range, mvelo, trial_elapsed, width, height):
    """Metrics strip for the 1-D radar trace panel (RMS, Range, MVELO — all in cm)."""
    panel = np.full((height, width, 3), 12, dtype=np.uint8)
    cv2.line(panel, (0, 0), (width, 0), (60, 60, 60), 2)

    entries = [
        ("RMS",   f"{ap_rms:.2f} cm"   if ap_rms   is not None else "--", (0,   200, 255)),
        ("Range", f"{ap_range:.2f} cm" if ap_range is not None else "--", (100, 255, 120)),
        ("MVELO", f"{mvelo:.2f} cm/s"  if mvelo    is not None else "--", (255, 215,  50)),
    ]
    row_h = (height - 20) // len(entries)
    for i, (label, value, color) in enumerate(entries):
        y_label = 18 + i * row_h
        y_value = y_label + row_h - 12
        put_text(panel, label, (10, y_label), 0.55, (150, 150, 150))
        put_text(panel, value, (10, y_value), 1.40, color, 3)

    t_color = (100, 255, 100) if trial_elapsed >= 30 else \
              (255, 220,  50) if trial_elapsed >= 20 else \
              (180, 180, 180)
    t_str = f"T:{trial_elapsed:.0f}s" + (" DONE" if trial_elapsed >= 30 else "")
    put_text(panel, t_str, (width - 140, height - 10), 0.65, t_color, 2)
    return panel


# Button bounds in the resized stabilogram panel (STAB_SIZE wide, h tall)
BTN_X1,  BTN_Y1  = STAB_SIZE - 80, 8
BTN_X2,  BTN_Y2  = STAB_SIZE - 4,  36
BTN2_X1, BTN2_Y1 = STAB_SIZE - 80, 42
BTN2_X2, BTN2_Y2 = STAB_SIZE - 4,  70
BTN3_X1, BTN3_Y1 = STAB_SIZE - 80, 76
BTN3_X2, BTN3_Y2 = STAB_SIZE - 4,  104
BTN4_X1, BTN4_Y1 = STAB_SIZE - 80, 110
BTN4_X2, BTN4_Y2 = STAB_SIZE - 4,  138
BTN5_X1, BTN5_Y1 = STAB_SIZE - 80, 144
BTN5_X2, BTN5_Y2 = STAB_SIZE - 4,  172
BTN6_X1, BTN6_Y1 = STAB_SIZE - 80, 178
BTN6_X2, BTN6_Y2 = STAB_SIZE - 4,  206
BTN7_X1, BTN7_Y1 = STAB_SIZE - 80, 212
BTN7_X2, BTN7_Y2 = STAB_SIZE - 4,  240


def _ap_fudge_factor(px_per_m, image_width, subject_height_m):
    """
    Geometric scale factor to convert Δtrunk_px to equivalent-ML pixels.

    Based on pinhole model: a depth movement Δz produces Δtrunk_px ≈ trunk_px × Δz/d.
    Multiply raw AP signal by this factor to get the same sensitivity as ML.
    Uses an assumed horizontal FOV to estimate focal length — adjust
    ASSUMED_HFOV_DEG if your webcam is noticeably wider or narrower.
    """
    f_px = (image_width / 2.0) / np.tan(np.radians(ASSUMED_HFOV_DEG / 2.0))
    ppm  = px_per_m if (px_per_m and px_per_m > 0) else 300.0
    trunk_px = subject_height_m * 0.35 * ppm
    return float(f_px / max(trunk_px, 1.0))


def draw_buttons(panel, hover_reset=False, hover_fudge=False, fudge_active=False,
                  hover_cop=False, cop_active=False,
                  hover_dempster=False, dempster_active=False,
                  hover_auto=False, auto_state='idle', auto_countdown=0.0,
                  hover_world_z=False, world_z_active=False,
                  hover_solo=False, solo_mode=False):
    """Draw Reset, AP-fudge, CoM/CoP toggle, and Solo-view buttons."""
    # Reset
    col = (100, 100, 200) if hover_reset else (70, 70, 140)
    cv2.rectangle(panel, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), col, -1)
    cv2.rectangle(panel, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), (180, 180, 255), 1)
    cv2.putText(panel, "Reset", (BTN_X1 + 8, BTN_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 255), 1, cv2.LINE_AA)
    # Fudge AP toggle
    if fudge_active:
        col2 = (0, 160, 80) if not hover_fudge else (0, 200, 100)
        label, lcol = "Fudged", (200, 255, 200)
    else:
        col2 = (60, 80, 60) if not hover_fudge else (80, 110, 80)
        label, lcol = "Honest", (160, 200, 160)
    cv2.rectangle(panel, (BTN2_X1, BTN2_Y1), (BTN2_X2, BTN2_Y2), col2, -1)
    cv2.rectangle(panel, (BTN2_X1, BTN2_Y1), (BTN2_X2, BTN2_Y2), (100, 200, 100), 1)
    cv2.putText(panel, label, (BTN2_X1 + 8, BTN2_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol, 1, cv2.LINE_AA)
    # CoM / CoP toggle
    if cop_active:
        col3 = (180, 100, 0) if not hover_cop else (220, 130, 0)
        label3, lcol3 = "CoP", (255, 200, 100)
    else:
        col3 = (120, 90, 50) if not hover_cop else (160, 120, 70)
        label3, lcol3 = "CoM", (230, 210, 160)
    cv2.rectangle(panel, (BTN3_X1, BTN3_Y1), (BTN3_X2, BTN3_Y2), col3, -1)
    cv2.rectangle(panel, (BTN3_X1, BTN3_Y1), (BTN3_X2, BTN3_Y2), (200, 160, 80), 1)
    cv2.putText(panel, label3, (BTN3_X1 + 14, BTN3_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol3, 1, cv2.LINE_AA)
    # Hip / Dempster toggle
    if dempster_active:
        col4 = (140, 50, 140) if not hover_dempster else (180, 70, 180)
        label4, lcol4 = "Demps", (230, 180, 230)
    else:
        col4 = (80, 40, 80) if not hover_dempster else (110, 55, 110)
        label4, lcol4 = "Hip", (200, 160, 200)
    cv2.rectangle(panel, (BTN4_X1, BTN4_Y1), (BTN4_X2, BTN4_Y2), col4, -1)
    cv2.rectangle(panel, (BTN4_X1, BTN4_Y1), (BTN4_X2, BTN4_Y2), (180, 120, 180), 1)
    cv2.putText(panel, label4, (BTN4_X1 + 14, BTN4_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol4, 1, cv2.LINE_AA)
    # Auto-trial button
    if auto_state == 'recording':
        col5, label5, lcol5 = (0, 0, 180) if not hover_auto else (0, 0, 220), "STOP", (100, 100, 255)
    elif auto_state == 'countdown':
        col5, label5, lcol5 = (0, 120, 160), f"{int(auto_countdown)+1}s", (0, 230, 255)
    else:
        col5, label5, lcol5 = (0, 120, 80) if not hover_auto else (0, 160, 100), "Trial", (100, 255, 180)
    cv2.rectangle(panel, (BTN5_X1, BTN5_Y1), (BTN5_X2, BTN5_Y2), col5, -1)
    cv2.rectangle(panel, (BTN5_X1, BTN5_Y1), (BTN5_X2, BTN5_Y2), (80, 220, 160), 1)
    cv2.putText(panel, label5, (BTN5_X1 + 8, BTN5_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol5, 1, cv2.LINE_AA)
    # World-Z AP toggle
    if world_z_active:
        col6, label6, lcol6 = (0, 140, 160) if not hover_world_z else (0, 180, 200), "WorldZ", (100, 230, 255)
    else:
        col6, label6, lcol6 = (50, 70, 80) if not hover_world_z else (70, 100, 110), "TrunkL", (160, 190, 200)
    cv2.rectangle(panel, (BTN6_X1, BTN6_Y1), (BTN6_X2, BTN6_Y2), col6, -1)
    cv2.rectangle(panel, (BTN6_X1, BTN6_Y1), (BTN6_X2, BTN6_Y2), (80, 180, 200), 1)
    cv2.putText(panel, label6, (BTN6_X1 + 6, BTN6_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol6, 1, cv2.LINE_AA)
    # Trace / 2D toggle — switches right panel between 1-D radar AP waveform and 2-D stabilogram
    if solo_mode:
        col7, label7, lcol7 = (0, 110, 60) if not hover_solo else (0, 150, 80), "2D", (150, 255, 180)
    else:
        col7, label7, lcol7 = (60, 40, 100) if not hover_solo else (90, 60, 140), "Trace", (200, 170, 255)
    cv2.rectangle(panel, (BTN7_X1, BTN7_Y1), (BTN7_X2, BTN7_Y2), col7, -1)
    cv2.rectangle(panel, (BTN7_X1, BTN7_Y1), (BTN7_X2, BTN7_Y2), (160, 130, 255), 1)
    cv2.putText(panel, label7, (BTN7_X1 + 6, BTN7_Y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lcol7, 1, cv2.LINE_AA)


# ── Landmark order for CSV export ────────────────────────────────────────────
_LANDMARK_NAMES = [
    'nose',
    'left_ear',       'right_ear',
    'left_shoulder',  'right_shoulder',
    'left_elbow',     'right_elbow',
    'left_wrist',     'right_wrist',
    'left_hip',       'right_hip',
    'left_knee',      'right_knee',
    'left_ankle',     'right_ankle',
    'left_heel',      'right_heel',
    'left_foot',      'right_foot',
]


def save_recording(frames, px_per_m, radar_ap_hist=None, label='recording'):
    """
    Save a list of CameraFrames as .pkl and .csv.

    CSV columns
    -----------
    time_s                  — elapsed time in seconds
    <joint>_x_cm            — landmark x in cm (image left = 0, right = positive)
    <joint>_y_cm            — landmark y in cm (image top = 0, down = positive)
    <joint>_z_norm          — MediaPipe normalised z (same scale as x; depth estimate)
    <joint>_vis             — MediaPipe visibility score [0–1]
    hip_ml_cm               — hip midpoint ML, negated so right = positive (mirrored)
    ap_cm                   — AP proxy (trunk length change) in cm
    ap_radar_cm             — IWR6843 radar range in cm (NaN if radar not connected)
    com_ml_cm               — Dempster whole-body CoM ML in cm (mirrored)
    com_y_cm                — Dempster whole-body CoM vertical in cm
    """
    if not frames:
        print("Nothing to save.")
        return

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = f"{label}_{ts}.pkl"
    csv_path = f"{label}_{ts}.csv"

    # ── Pickle ────────────────────────────────────────────────────────────────
    with open(pkl_path, 'wb') as f:
        pickle.dump(frames, f)
    print(f"Saved pickle → {pkl_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    ppm  = px_per_m or 1.0
    s2cm = 100.0 / ppm
    t0   = frames[0].host_ts

    # Build header
    lm_cols = []
    for name in _LANDMARK_NAMES:
        lm_cols += [f"{name}_x_cm", f"{name}_y_cm", f"{name}_z_norm", f"{name}_vis"]
    header = ['time_s'] + lm_cols + ['hip_ml_cm', 'ap_cm', 'ap_radar_cm', 'com_ml_cm', 'com_y_cm']

    # Pre-interpolate radar AP onto camera timestamps (radar runs at ~20 Hz)
    frame_ts    = np.array([r.host_ts for r in frames], dtype=np.float64)
    radar_col   = np.full(len(frames), np.nan)
    if radar_ap_hist and len(radar_ap_hist) > 2:
        r_ts   = np.array([t for t, v in radar_ap_hist], dtype=np.float64)
        r_vals = np.array([v for t, v in radar_ap_hist], dtype=np.float64) * 100.0  # m → cm
        # Only fill timestamps within the radar recording range
        in_range = (frame_ts >= r_ts[0]) & (frame_ts <= r_ts[-1])
        radar_col[in_range] = np.interp(frame_ts[in_range], r_ts, r_vals)

    def _fmt(v):
        return 'nan' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.4f}'

    with open(csv_path, 'w', newline='') as csvf:
        csvf.write(','.join(header) + '\n')
        for i, rec in enumerate(frames):
            row = [f'{rec.host_ts - t0:.4f}']

            # All landmark positions
            lm = rec.landmarks or {}
            for name in _LANDMARK_NAMES:
                if name in lm:
                    x_px, y_px, z_norm, vis = lm[name]
                    row += [
                        _fmt(x_px * s2cm if np.isfinite(x_px) else float('nan')),
                        _fmt(y_px * s2cm if np.isfinite(y_px) else float('nan')),
                        _fmt(z_norm),
                        _fmt(vis),
                    ]
                else:
                    row += ['nan', 'nan', 'nan', 'nan']

            # Derived quantities
            def _cm(v, sign=1):
                return sign * v * s2cm if np.isfinite(v) else float('nan')

            row += [
                _fmt(_cm(rec.hip_ml_px, -1)),
                _fmt(_cm(rec.ap_proxy_px)),
                _fmt(radar_col[i]),
                _fmt(_cm(rec.com_ml_px, -1)),
                _fmt(_cm(rec.com_y_px)),
            ]
            csvf.write(','.join(row) + '\n')

    print(f"Saved CSV    → {csv_path}  ({len(frames)} frames, {len(header)} columns)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    reader = CameraReader(device_index=DEVICE_INDEX, subject_height_m=SUBJECT_HEIGHT_M,
                          capture_width=1280, capture_height=720,
                          fps=60,
                          inference_width=256,
                          model_type='lite')
    print("Starting camera...")
    reader.start()

    time.sleep(0.3)
    if reader._thread_error:
        print(f"Camera failed: {reader._thread_error}")
        reader.stop()
        sys.exit(1)

    # ── Optional radar reader ─────────────────────────────────────────────────
    radar_reader = None
    if RADAR_CONFIG_PORT and RADAR_DATA_PORT:
        try:
            from radar_reader import RadarReader
            radar_reader = RadarReader(RADAR_CONFIG_PORT, RADAR_DATA_PORT)
            radar_reader.start()
            print(f"Radar started  cfg={RADAR_CONFIG_PORT}  data={RADAR_DATA_PORT}")
        except Exception as exc:
            print(f"Radar not available: {exc}")
            radar_reader = None

    maxlen = 1000   # enough for ~120fps over 8s — stores (timestamp, value) tuples
    ml_history_hip = collections.deque(maxlen=maxlen)       # (ts, hip ML px)
    ml_history_dempster = collections.deque(maxlen=maxlen)  # (ts, Dempster ML px)
    ap_history = collections.deque(maxlen=maxlen)           # (ts, AP trunk-proxy px)
    ap_world_history = collections.deque(maxlen=maxlen)     # (ts, world Z in metres)
    radar_ap_history = collections.deque(maxlen=500)        # (ts, radar range in metres) ~25s @ 20Hz
    frame_times = collections.deque(maxlen=30)
    fps = 30.0

    recording = False
    recorded  = []
    trial_start    = time.perf_counter()
    metrics_history = collections.deque()   # (timestamp, ce, path, mvelo, calibrated)

    # Shared mouse state
    mouse = {
        'reset': False, 'hover_reset': False,
        'hover_fudge': False, 'fudge_ap': False,
        'hover_cop': False, 'show_cop': False,
        'hover_dempster': False, 'use_dempster': False,
        'canvas_w': 0, 'canvas_h': 0,
        'ap_gain': AP_GAIN,          # live AP gain (0.1 – 20.0)
        'dragging_ap': False,
        'hover_auto': False,
        'auto_state': 'idle',        # 'idle' | 'countdown' | 'recording'
        'auto_start_ts': 0.0,
        'hover_world_z': False,
        'use_world_z': False,        # False = trunk-length proxy, True = MediaPipe world Z
        'hover_solo': False,
        'solo_mode': False,          # True = hide camera, show large stabilogram on left
        # slider bounds updated each frame
        '_sl_x1': 0, '_sl_x2': 1, '_sl_y1': 0, '_sl_y2': 1,
    }

    def on_mouse(event, x, y, flags, param):
        sx = x - param['canvas_w']   # x relative to stabilogram panel
        param['hover_reset']    = (BTN_X1  <= sx <= BTN_X2  and BTN_Y1  <= y <= BTN_Y2)
        param['hover_fudge']    = (BTN2_X1 <= sx <= BTN2_X2 and BTN2_Y1 <= y <= BTN2_Y2)
        param['hover_cop']      = (BTN3_X1 <= sx <= BTN3_X2 and BTN3_Y1 <= y <= BTN3_Y2)
        param['hover_dempster'] = (BTN4_X1 <= sx <= BTN4_X2 and BTN4_Y1 <= y <= BTN4_Y2)
        param['hover_auto']     = (BTN5_X1 <= sx <= BTN5_X2 and BTN5_Y1 <= y <= BTN5_Y2)
        param['hover_world_z']  = (BTN6_X1 <= sx <= BTN6_X2 and BTN6_Y1 <= y <= BTN6_Y2)
        param['hover_solo']     = (BTN7_X1 <= sx <= BTN7_X2 and BTN7_Y1 <= y <= BTN7_Y2)

        # AP gain slider (drawn in metrics strip)
        sl_x1, sl_x2 = param['_sl_x1'], param['_sl_x2']
        sl_y1, sl_y2 = param['_sl_y1'], param['_sl_y2']
        in_slider = sl_x1 <= x <= sl_x2 and sl_y1 <= y <= sl_y2
        if event == cv2.EVENT_LBUTTONDOWN and in_slider:
            param['dragging_ap'] = True
        if event == cv2.EVENT_LBUTTONUP:
            param['dragging_ap'] = False
        if param['dragging_ap'] and (event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN)):
            t = (x - sl_x1) / max(sl_x2 - sl_x1, 1)
            param['ap_gain'] = round(max(0.1, min(20.0, t * 20.0)), 1)

        if event == cv2.EVENT_LBUTTONDOWN and not in_slider:
            if param['hover_reset']:
                param['reset'] = True
            if param['hover_fudge']:
                param['fudge_ap'] = not param['fudge_ap']
            if param['hover_cop']:
                param['show_cop'] = not param['show_cop']
            if param['hover_dempster']:
                param['use_dempster'] = not param['use_dempster']
            if param['hover_auto']:
                if param['auto_state'] == 'idle':
                    param['auto_state']    = 'countdown'
                    param['auto_start_ts'] = time.perf_counter()
                else:
                    param['auto_state'] = 'idle'   # cancel
            if param['hover_world_z']:
                param['use_world_z'] = not param['use_world_z']
            if param['hover_solo']:
                param['solo_mode'] = not param['solo_mode']

    print("Preview running.  Q = quit   R = start/stop recording   C/click Reset = clear")
    cv2.namedWindow("KineCal Live Preview", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("KineCal Live Preview", on_mouse, mouse)

    try:
        while True:
            try:
                cf: CameraFrame = reader.queue.get(timeout=0.1)
            except Exception:
                if reader._thread_error:
                    print(f"Camera crashed: {reader._thread_error}")
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            frame_times.append(cf.host_ts)
            if len(frame_times) > 1:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            # Handle reset (button click or C key)
            if mouse['reset']:
                ml_history_hip.clear()
                ml_history_dempster.clear()
                ap_history.clear()
                ap_world_history.clear()
                radar_ap_history.clear()
                metrics_history.clear()
                trial_start = time.perf_counter()
                mouse['reset'] = False

            # ── Auto-trial state machine ──────────────────────────────────────
            now_t = time.perf_counter()
            auto_elapsed = now_t - mouse['auto_start_ts']
            if mouse['auto_state'] == 'countdown':
                auto_remaining = AUTO_COUNTDOWN_S - auto_elapsed
                if auto_remaining <= 0:
                    # Countdown done — start recording
                    mouse['auto_state']    = 'recording'
                    mouse['auto_start_ts'] = now_t
                    recording = True
                    recorded  = []
                    print("Auto-trial recording started")
            elif mouse['auto_state'] == 'recording':
                auto_remaining = AUTO_RECORD_S - auto_elapsed
                if auto_remaining <= 0:
                    # Recording done — save and reset
                    recording = False
                    mouse['auto_state'] = 'idle'
                    print(f"Auto-trial complete — {len(recorded)} frames")
                    save_recording(recorded, reader.px_per_m, radar_ap_hist=list(radar_ap_history))

            ml_history_hip.append((cf.host_ts, -cf.hip_ml_px))
            ml_history_dempster.append((cf.host_ts, -cf.com_ml_px))
            ap_history.append((cf.host_ts, cf.ap_proxy_px))
            ap_world_history.append((cf.host_ts, cf.ap_world_m))

            # Drain radar queue — radar runs at ~20 Hz, camera at ~43 Hz
            # Prefer range-profile centroid (subject_range_m, ~5 mm precision) over
            # point-cloud median (~4 cm CFAR quantisation).
            if radar_reader:
                _drained = 0
                while _drained < 10:
                    try:
                        rf = radar_reader.queue.get_nowait()
                    except Exception:
                        break
                    if np.isfinite(rf.subject_range_m):
                        radar_ap_history.append((rf.host_ts, rf.subject_range_m))
                    elif rf.points:
                        # fallback to point-cloud median if range profile unavailable
                        radar_ap_history.append(
                            (rf.host_ts, float(np.median([p.y for p in rf.points])))
                        )
                    _drained += 1

            if recording:
                recorded.append(cf)

            # AP gain from custom drawn slider
            ap_gain = mouse['ap_gain']

            # ── Camera panel ──────────────────────────────────────────────────
            w, h = cf.image_width, cf.image_height
            canvas = cv2.flip(cf.bgr_frame, 1) if cf.bgr_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)

            # Resize to display width (capture may be 1280x720 but we display at 640x360)
            disp_h = max(1, int(h * DISPLAY_WIDTH / w))
            scale  = DISPLAY_WIDTH / w
            if w != DISPLAY_WIDTH:
                canvas = cv2.resize(canvas, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_LINEAR)
            w, h = DISPLAY_WIDTH, disp_h   # update w/h to display dimensions

            # Mirror skeleton landmarks and scale to display size
            mirrored_landmarks = None
            if cf.landmarks:
                mirrored_landmarks = {
                    k: ((cf.image_width - v[0]) * scale, v[1] * scale, v[2], v[3])
                    for k, v in cf.landmarks.items()
                }
            draw_skeleton(canvas, mirrored_landmarks)

            # Dempster CoM dot — mirrored X, scaled to display
            if np.isfinite(cf.com_ml_px) and np.isfinite(cf.com_y_px):
                com_px = (int((cf.image_width - cf.com_ml_px) * scale), int(cf.com_y_px * scale))
                cv2.circle(canvas, com_px, 10, (0, 100, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, com_px,  10, (255, 255, 255), 1,  cv2.LINE_AA)

            trial_elapsed = time.perf_counter() - trial_start

            pose_col = (0, 220, 0) if cf.landmarks else (0, 0, 220)
            pose_str = "POSE: YES" if cf.landmarks else "POSE: NO"
            cal_str  = (f"px/m: {reader.px_per_m:.1f}" if reader.px_per_m
                        else f"Calibrating... {len(reader._calibration_samples)}/{reader._calibration_frames}")
            res_str  = f"{w}x{h}  inf:{reader.inference_width}px  [{reader.model_type}]"

            put_text(canvas, pose_str,               (10, 28),      0.7,  pose_col,       2)
            put_text(canvas, cal_str,                (10, 54),      0.5,  (220, 220, 220))
            dropped = reader.frames_dropped
            drop_col = (0, 80, 255) if dropped > 0 else (180, 255, 180)
            put_text(canvas, f"FPS: {fps:.1f}  frames: {cf.frame_number}  drop: {dropped}", (10, 76), 0.5, drop_col)
            put_text(canvas, res_str,                (10, 96),      0.4,  (160, 200, 160))
            put_text(canvas, f"AP gain: {ap_gain:.1f}x", (10, 114),   0.4,  (200, 180, 120))
            # AP source indicator — includes radar frame counter for diagnostics
            _r_active = radar_reader is not None and len(radar_ap_history) > 5
            _r_cfg    = bool(RADAR_CONFIG_PORT and RADAR_DATA_PORT)
            _r_frames = radar_reader.frames_received if radar_reader else 0
            if _r_active:
                put_text(canvas, f"AP: Radar ({_r_frames}f)", (10, 132), 0.4, (50, 255, 80))
            elif _r_cfg and radar_reader is None:
                put_text(canvas, "AP: Radar FAIL", (10, 132), 0.4, (60, 60, 255))
            elif _r_cfg and _r_frames == 0:
                put_text(canvas, "AP: Radar wait 0f", (10, 132), 0.4, (60, 200, 255))
            elif _r_cfg:
                # frames arriving but NaN range + empty points — detection issue
                put_text(canvas, f"AP: Radar {_r_frames}f nodet", (10, 132), 0.4, (0, 170, 255))
            elif mouse['use_world_z']:
                put_text(canvas, "AP: WorldZ", (10, 132), 0.4, (50, 200, 255))
            else:
                put_text(canvas, "AP: TrunkL", (10, 132), 0.4, (160, 140, 90))
            put_text(canvas, "Q:quit  R:rec  C:clear", (10, h - 10), 0.42, (200, 200, 200))
            t_cam_col = (100, 255, 100) if trial_elapsed >= 30 else \
                        (0, 210, 255)   if trial_elapsed >= 20 else \
                        (220, 220, 220)
            put_text(canvas, f"T: {trial_elapsed:.1f}s", (w - 120, 28), 0.7, t_cam_col, 2)

            if recording:
                put_text(canvas, "[ REC ]",          (w - 100, 28), 0.7,  (60, 60, 255),  2)
                put_text(canvas, f"{len(recorded)}f", (w - 80,  54), 0.5,  (140, 140, 255))

            # ── Prepare CoM signals for stabilogram ───────────────────────────
            # Display raw CoM (not CoP) — CoP via inverted pendulum double-
            # differentiation creates loopy artefacts in real-time display.
            # CoP correction is applied post-session on saved recordings.
            # Filter at 6 Hz before plotting — same practice as force plate
            # displays, removes landmark jitter while preserving sway content.
            ml_src = ml_history_dempster if mouse['use_dempster'] else ml_history_hip
            ts_arr = np.array([t for t, v in ml_src], dtype=np.float64)
            ml_arr = np.array([v for t, v in ml_src], dtype=np.float64)

            # AP source priority: Radar > WorldZ toggle > Trunk-length proxy
            ppm = reader.px_per_m or 300.0
            _r_active = radar_reader is not None and len(radar_ap_history) > 5
            if _r_active:
                # Radar range (metres) interpolated onto camera timestamps, converted to pixels
                r_ts   = np.array([t for t, v in radar_ap_history], dtype=np.float64)
                r_vals = np.array([v for t, v in radar_ap_history], dtype=np.float64)
                ap_arr = np.interp(ts_arr, r_ts, r_vals) * ppm
            elif mouse['use_world_z']:
                ap_arr = np.array([v * ppm for t, v in ap_world_history], dtype=np.float64)
            else:
                ap_arr = np.array([v for t, v in ap_history], dtype=np.float64)

            def _prep(arr, fc=6.0):
                out = arr.copy()                       # never mutate input
                nans = np.isnan(out)
                if nans.all():
                    return out
                if nans.any():
                    out[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], out[~nans])
                if len(out) >= 15 and fps > 1:
                    from scipy.signal import butter, filtfilt
                    # Cap fc at 40% of Nyquist — prevents instability when FPS drops
                    fc_safe = min(fc, max(fps, 5.0) * 0.4)
                    Wn = min(2.0 * fc_safe / max(fps, 5.0), 0.99)
                    b, a = butter(2, Wn, btype='low')
                    out = filtfilt(b, a, out)
                return out

            plot_ml = _prep(ml_arr, fc=6.0)

            # AP proxy is noisier than ML (two independent landmark jitters
            # vs averaged midpoint).  Lower cutoff reduces noise without
            # causing loops — filtfilt is zero-phase at any cutoff.
            ap_in = ap_arr * ap_gain * (_ap_fudge_factor(reader.px_per_m, cf.image_width, SUBJECT_HEIGHT_M)
                              if (mouse['fudge_ap'] and cf.image_width > 0) else 1.0)
            plot_ap = _prep(ap_in, fc=3.0)

            # ── Optional CoP correction (inverted pendulum) ──────────────────
            if mouse['show_cop'] and reader.px_per_m and reader.px_per_m > 0 and len(plot_ml) >= 15:
                H_COM = SUBJECT_HEIGHT_M * 0.55
                G = 9.81
                ppm = reader.px_per_m
                dt = 1.0 / max(fps, 1.0)
                # Convert to metres, compute acceleration, apply correction
                ml_m = plot_ml / ppm
                ap_m = plot_ap / ppm
                def _accel(x):
                    a = np.zeros_like(x)
                    a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2]) / dt**2
                    a[0] = a[1]; a[-1] = a[-2]
                    return a
                cop_ml_m = ml_m + (H_COM / G) * _accel(ml_m)
                cop_ap_m = ap_m + (H_COM / G) * _accel(ap_m)
                # Smooth the result — double differentiation amplifies noise
                plot_ml = _prep(cop_ml_m * ppm, fc=2.0)
                plot_ap = _prep(cop_ap_m * ppm, fc=2.0)

            # ── Stabilogram panel ─────────────────────────────────────────────
            stab, raw_metrics = draw_stabilogram(plot_ml, plot_ap, ts_arr, fps,
                                                 px_per_m=reader.px_per_m,
                                                 fudge_ap=mouse['fudge_ap'],
                                                 show_cop=mouse['show_cop'],
                                                 use_dempster=mouse['use_dempster'],
                                                 trial_elapsed=trial_elapsed)

            # Rolling average of metrics over METRICS_WINDOW_S seconds
            now = time.perf_counter()
            metrics_history.append((now, raw_metrics))
            while metrics_history and now - metrics_history[0][0] > METRICS_WINDOW_S:
                metrics_history.popleft()
            ces    = [m['ce']   for _, m in metrics_history if m['ce']   is not None]
            paths  = [m['path'] for _, m in metrics_history if m['path'] is not None]
            mvelos = [m['mvelo']for _, m in metrics_history if m['mvelo']is not None]
            avg_ce    = float(np.mean(ces))    if ces    else None
            avg_path  = float(np.mean(paths))  if paths  else None
            avg_mvelo = float(np.mean(mvelos)) if mvelos else None

            stab_h = max(1, h - METRICS_H)
            stab_resized = cv2.resize(stab, (STAB_SIZE, stab_h))

            auto_elapsed = time.perf_counter() - mouse['auto_start_ts']
            auto_cd_remaining = max(0.0, AUTO_COUNTDOWN_S - auto_elapsed)
            auto_rec_remaining = max(0.0, AUTO_RECORD_S - auto_elapsed)

            draw_buttons(stab_resized,
                         hover_reset=mouse['hover_reset'],
                         hover_fudge=mouse['hover_fudge'],
                         fudge_active=mouse['fudge_ap'],
                         hover_cop=mouse['hover_cop'],
                         cop_active=mouse['show_cop'],
                         hover_dempster=mouse['hover_dempster'],
                         dempster_active=mouse['use_dempster'],
                         hover_auto=mouse['hover_auto'],
                         auto_state=mouse['auto_state'],
                         auto_countdown=auto_cd_remaining,
                         hover_world_z=mouse['hover_world_z'],
                         world_z_active=mouse['use_world_z'],
                         hover_solo=mouse['hover_solo'],
                         solo_mode=mouse['solo_mode'])

            # ── Auto-trial overlay on camera canvas ───────────────────────────
            if mouse['auto_state'] == 'countdown':
                cd = int(auto_cd_remaining) + 1
                txt = f"GET READY: {cd}"
                col_cd = (0, 220, 255)
                put_text(canvas, txt,
                         (w // 2 - 140, h // 2),
                         2.2, col_cd, 4)
            elif mouse['auto_state'] == 'recording':
                rec_done = AUTO_RECORD_S - auto_rec_remaining
                bar_w    = int((rec_done / AUTO_RECORD_S) * (w - 20))
                cv2.rectangle(canvas, (10, h - 18), (w - 10, h - 6), (60, 60, 60), -1)
                cv2.rectangle(canvas, (10, h - 18), (10 + bar_w, h - 6), (0, 60, 220), -1)
                put_text(canvas, f"REC  {rec_done:.0f} / {AUTO_RECORD_S}s",
                         (w // 2 - 110, h // 2),
                         1.6, (60, 60, 255), 3)

            # ── Metrics strip below stabilogram ───────────────────────────────
            cal = raw_metrics['calibrated']
            metrics_strip = draw_metrics_strip(
                avg_ce, avg_path, avg_mvelo, cal, trial_elapsed,
                STAB_SIZE, METRICS_H)

            # ── AP gain slider drawn in metrics strip ─────────────────────────
            sl_pad   = 10
            sl_x1_l  = sl_pad                       # local coords within metrics_strip
            sl_x2_l  = STAB_SIZE - sl_pad
            sl_y1_l  = METRICS_H - 30
            sl_y2_l  = METRICS_H - 14
            sl_w     = sl_x2_l - sl_x1_l
            knob_t   = (mouse['ap_gain'] - 0.1) / (20.0 - 0.1)
            knob_x_l = int(sl_x1_l + knob_t * sl_w)
            # Track label
            cv2.putText(metrics_strip, f"AP gain: {mouse['ap_gain']:.1f}x",
                        (sl_x1_l, sl_y1_l - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 140, 80), 1, cv2.LINE_AA)
            # Track
            cv2.rectangle(metrics_strip, (sl_x1_l, sl_y1_l), (sl_x2_l, sl_y2_l),
                          (60, 60, 60), -1)
            cv2.rectangle(metrics_strip, (sl_x1_l, sl_y1_l), (knob_x_l, sl_y2_l),
                          (180, 130, 40), -1)
            # Knob
            knob_cy = (sl_y1_l + sl_y2_l) // 2
            cv2.circle(metrics_strip, (knob_x_l, knob_cy), 7, (255, 200, 80), -1, cv2.LINE_AA)
            cv2.circle(metrics_strip, (knob_x_l, knob_cy), 7, (255, 255, 200), 1, cv2.LINE_AA)

            # Store slider bounds in combined-image coords for mouse callback
            mouse['_sl_x1'] = w + sl_x1_l
            mouse['_sl_x2'] = w + sl_x2_l
            mouse['_sl_y1'] = stab_h + sl_y1_l - 8   # small hit buffer
            mouse['_sl_y2'] = stab_h + sl_y2_l + 8

            if mouse['solo_mode']:
                # ── Radar Trace mode: camera on left, 1-D radar AP waveform on right ──
                # No camera-derived measurements feed this panel — radar range only.
                trace_panel, trace_metrics = draw_radar_trace(
                    radar_ap_history, STAB_SIZE, stab_h)
                trace_metrics_strip = draw_metrics_strip_radar(
                    trace_metrics['ap_rms'], trace_metrics['ap_range'],
                    trace_metrics['mvelo'], trial_elapsed, STAB_SIZE, METRICS_H)
                # Draw all buttons on the trace panel (same states as the stab panel)
                draw_buttons(trace_panel,
                             hover_reset=mouse['hover_reset'],
                             hover_fudge=mouse['hover_fudge'],
                             fudge_active=mouse['fudge_ap'],
                             hover_cop=mouse['hover_cop'],
                             cop_active=mouse['show_cop'],
                             hover_dempster=mouse['hover_dempster'],
                             dempster_active=mouse['use_dempster'],
                             hover_auto=mouse['hover_auto'],
                             auto_state=mouse['auto_state'],
                             auto_countdown=auto_cd_remaining,
                             hover_world_z=mouse['hover_world_z'],
                             world_z_active=mouse['use_world_z'],
                             hover_solo=mouse['hover_solo'],
                             solo_mode=mouse['solo_mode'])
                right_panel = np.vstack([trace_panel, trace_metrics_strip])
            else:
                right_panel = np.vstack([stab_resized, metrics_strip])

            combined = np.hstack([canvas, right_panel])

            # Keep mouse callback aware of layout dimensions
            mouse['canvas_w'] = w
            mouse['canvas_h'] = h

            cv2.imshow("KineCal Live Preview", combined)

            # ── Keys ──────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                mouse['reset'] = True
            elif key == ord('r'):
                if not recording:
                    recording = True
                    recorded  = []
                    print(f"Recording started at frame {cf.frame_number}")
                else:
                    recording = False
                    print(f"Recording stopped — {len(recorded)} frames")
                    save_recording(recorded, reader.px_per_m, radar_ap_hist=list(radar_ap_history))

    finally:
        reader.stop()
        if radar_reader:
            radar_reader.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
