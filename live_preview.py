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
ASSUMED_HFOV_DEG  = 65.0        # assumed webcam horizontal FOV for AP fudge correction
METRICS_H         = 200         # height of metrics strip below stabilogram (pixels)
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


def draw_stabilogram(ml_hist, ap_hist, fps, px_per_m=None, fudge_ap=False, show_cop=False, use_dempster=False, n_std=ELLIPSE_N_STD, trial_elapsed=0.0):
    """
    Render the stabilogram into a square BGR panel.
    ML = horizontal axis, AP = vertical axis (positive = forward).

    Force-plate-style display:
      • Both axes converted to cm (isotropic scale) when calibrated.
      • Fixed axis range (±STAB_AXIS_CM) so the plot doesn't rescale.
      • Ellipse computed in cm space so its orientation is correct.
    """
    S = STAB_SIZE
    panel = np.full((S, S, 3), 20, dtype=np.uint8)

    margin = 30
    plot_s = S - margin * 2
    half = plot_s / 2.0

    # Display window
    n_display = max(2, int(HISTORY_S * fps))
    ml = np.array(list(ml_hist)[-n_display:], dtype=np.float64)
    ap = np.array(list(ap_hist)[-n_display:], dtype=np.float64)

    # Ellipse window — fixed 5 s
    n_ellipse = max(MIN_ELLIPSE_PTS, int(ELLIPSE_WINDOW_S * fps))
    ml_ell = np.array(list(ml_hist)[-n_ellipse:], dtype=np.float64)
    ap_ell = np.array(list(ap_hist)[-n_ellipse:], dtype=np.float64)

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
        duration_s  = len(ml) / max(fps, 1.0)
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

    unit_sq = "cm\u00b2" if calibrated else "px\u00b2"
    unit_l  = "cm"     if calibrated else "px"

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


# Button bounds in the resized stabilogram panel (STAB_SIZE wide, h tall)
BTN_X1,  BTN_Y1  = STAB_SIZE - 80, 8
BTN_X2,  BTN_Y2  = STAB_SIZE - 4,  36
BTN2_X1, BTN2_Y1 = STAB_SIZE - 80, 42
BTN2_X2, BTN2_Y2 = STAB_SIZE - 4,  70
BTN3_X1, BTN3_Y1 = STAB_SIZE - 80, 76
BTN3_X2, BTN3_Y2 = STAB_SIZE - 4,  104
BTN4_X1, BTN4_Y1 = STAB_SIZE - 80, 110
BTN4_X2, BTN4_Y2 = STAB_SIZE - 4,  138


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
                  hover_dempster=False, dempster_active=False):
    """Draw Reset, AP-fudge, and CoM/CoP toggle buttons."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    reader = CameraReader(device_index=DEVICE_INDEX, subject_height_m=SUBJECT_HEIGHT_M,
                          capture_width=640, capture_height=480,
                          fps=60,
                          inference_width=320,
                          model_type='lite')
    print("Starting camera... (downloading pose model on first run)")
    reader.start()

    time.sleep(1.5)
    if reader._thread_error:
        print(f"Camera failed: {reader._thread_error}")
        reader.stop()
        sys.exit(1)

    maxlen = int(HISTORY_S * 35)
    ml_history_hip = collections.deque(maxlen=maxlen)      # hip midpoint ML
    ml_history_dempster = collections.deque(maxlen=maxlen)  # Dempster whole-body ML
    ap_history = collections.deque(maxlen=maxlen)
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
        'canvas_w': 0,
    }

    def on_mouse(event, x, y, flags, param):
        sx = x - param['canvas_w']   # x relative to stabilogram panel
        param['hover_reset']    = (BTN_X1  <= sx <= BTN_X2  and BTN_Y1  <= y <= BTN_Y2)
        param['hover_fudge']    = (BTN2_X1 <= sx <= BTN2_X2 and BTN2_Y1 <= y <= BTN2_Y2)
        param['hover_cop']      = (BTN3_X1 <= sx <= BTN3_X2 and BTN3_Y1 <= y <= BTN3_Y2)
        param['hover_dempster'] = (BTN4_X1 <= sx <= BTN4_X2 and BTN4_Y1 <= y <= BTN4_Y2)
        if event == cv2.EVENT_LBUTTONDOWN:
            if param['hover_reset']:
                param['reset'] = True
            if param['hover_fudge']:
                param['fudge_ap'] = not param['fudge_ap']
            if param['hover_cop']:
                param['show_cop'] = not param['show_cop']
            if param['hover_dempster']:
                param['use_dempster'] = not param['use_dempster']

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
                metrics_history.clear()
                trial_start = time.perf_counter()
                mouse['reset'] = False

            ml_history_hip.append(-cf.hip_ml_px)        # hip midpoint; negate for mirror
            ml_history_dempster.append(-cf.com_ml_px)    # Dempster CoM; negate for mirror
            ap_history.append(cf.ap_proxy_px)            # forward = larger trunk = positive AP

            if recording:
                recorded.append(cf)

            # ── Camera panel ──────────────────────────────────────────────────
            w, h = cf.image_width, cf.image_height
            canvas = cv2.flip(cf.bgr_frame, 1) if cf.bgr_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
            # Mirror skeleton landmarks to match flipped frame
            mirrored_landmarks = None
            if cf.landmarks:
                mirrored_landmarks = {k: (w - v[0], v[1], v[2], v[3]) for k, v in cf.landmarks.items()}
            draw_skeleton(canvas, mirrored_landmarks)

            # Dempster CoM dot — mirrored X to match flipped canvas
            if np.isfinite(cf.com_ml_px) and np.isfinite(cf.com_y_px):
                com_px = (int(cf.image_width - cf.com_ml_px), int(cf.com_y_px))
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
            put_text(canvas, f"FPS: {fps:.1f}",      (10, 76),      0.5,  (180, 255, 180))
            put_text(canvas, res_str,                (10, 96),      0.4,  (160, 200, 160))
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
            ml_arr = np.array(ml_src, dtype=np.float64)
            ap_arr = np.array(ap_history, dtype=np.float64)

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
                    Wn = min(np.pi * fc_safe / (2 * max(fps, 5.0)), 0.99)
                    b, a = butter(2, Wn, btype='low')
                    out = filtfilt(b, a, out)
                return out

            plot_ml = _prep(ml_arr, fc=6.0)

            # AP proxy is noisier than ML (two independent landmark jitters
            # vs averaged midpoint).  Lower cutoff reduces noise without
            # causing loops — filtfilt is zero-phase at any cutoff.
            ap_in = ap_arr * (_ap_fudge_factor(reader.px_per_m, cf.image_width, SUBJECT_HEIGHT_M)
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
            stab, raw_metrics = draw_stabilogram(plot_ml, plot_ap, fps,
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

            draw_buttons(stab_resized,
                         hover_reset=mouse['hover_reset'],
                         hover_fudge=mouse['hover_fudge'],
                         fudge_active=mouse['fudge_ap'],
                         hover_cop=mouse['hover_cop'],
                         cop_active=mouse['show_cop'],
                         hover_dempster=mouse['hover_dempster'],
                         dempster_active=mouse['use_dempster'])

            # ── Metrics strip below stabilogram ───────────────────────────────
            cal = raw_metrics['calibrated']
            metrics_strip = draw_metrics_strip(
                avg_ce, avg_path, avg_mvelo, cal, trial_elapsed,
                STAB_SIZE, METRICS_H)

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
                    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = f"recording_{ts}.pkl"
                    with open(out, 'wb') as f:
                        pickle.dump(recorded, f)
                    print(f"Saved to {out}")
                    # CSV export — cm units, mean-centering left to post-processing
                    csv_out = f"recording_{ts}.csv"
                    ppm    = reader.px_per_m or 1.0
                    s2cm   = 100.0 / ppm
                    t0     = recorded[0].host_ts
                    with open(csv_out, 'w', newline='') as csvf:
                        csvf.write("time_s,hip_ml_cm,ap_cm,com_ml_cm,com_y_cm\n")
                        for rec in recorded:
                            def _c(v, sign=1):
                                return sign * v * s2cm if np.isfinite(v) else float('nan')
                            row = [rec.host_ts - t0,
                                   _c(rec.hip_ml_px, -1),
                                   _c(rec.ap_proxy_px),
                                   _c(rec.com_ml_px, -1),
                                   _c(rec.com_y_px)]
                            csvf.write(','.join('nan' if np.isnan(v) else f'{v:.4f}' for v in row) + '\n')
                    print(f"CSV  saved to {csv_out}")

    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
