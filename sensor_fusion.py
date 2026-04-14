#!/usr/bin/env python3
"""
Sensor fusion and time alignment for RGB camera + TI IWR6843 radar.

Time alignment strategy
-----------------------
Both sensors write host-side timestamps using time.perf_counter().
This gives a common monotonic clock with ~1 µs resolution, but each sensor
introduces variable UART/USB buffering latency.

We apply two-stage alignment:

  Stage 1 — Nearest-neighbour frame matching
    For each radar frame (lower rate, ~20 Hz) find the camera frame whose
    host_ts is closest in time. This handles the rate difference and random
    jitter with no assumptions about systematic offset.

  Stage 2 — Cross-correlation offset calibration
    After collecting ≥ MIN_CAL_SECONDS of data, compute the systematic
    (fixed) time offset between the two streams using cross-correlation
    of a shared motion signal:
      - Radar signal:  mean absolute Doppler per frame (AP motion proxy)
      - Camera signal: hip midpoint velocity magnitude per frame (ML motion proxy)
    Both track whole-body movement and should be correlated during quiet
    standing sway. The lag at maximum cross-correlation is the systematic
    offset to subtract from all subsequent nearest-neighbour matches.

    Accuracy: this calibrates the fixed latency difference (typically 5-30 ms).
    Residual jitter (~1 frame period) is irreducible in software.

Coordinate system (fused)
--------------------------
  ML  (mediolateral):  from camera, hip midpoint X, calibrated to metres
  AP  (anteroposterior): from radar, mean range of detected points, metres
  Both are mean-centred over the recording (sway = deviation from mean)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, correlate

from camera_reader import CameraFrame, camera_ml_sway, camera_hip_velocity
from radar_reader  import RadarFrame,  radar_ap_sway,  radar_doppler_signal

logger = logging.getLogger(__name__)

MIN_CAL_SECONDS = 10.0   # minimum seconds of data before calibrating offset
BUTTERWORTH_N   = 2
BUTTERWORTH_FC  = 10.0   # Hz — matches existing sway_utils/metrics.py


# ── Fused frame ───────────────────────────────────────────────────────────────

@dataclass
class FusedFrame:
    """Single time-aligned sample from both sensors."""
    ts: float           # representative timestamp (radar host_ts after offset correction)
    radar_frame_num: int
    camera_frame_num: int
    dt_alignment: float # |t_radar_corrected - t_camera| in seconds (alignment quality)

    # Raw fused values (unfiltered, uncentred)
    hip_ml_px: float    # camera: ML hip midpoint in pixels
    ap_range_m: float   # radar:  mean range to subject in metres

    # Populated after the full session (see fuse_session())
    ml_m: float = 0.0   # calibrated, filtered, mean-centred ML sway (metres)
    ap_m: float = 0.0   # filtered, mean-centred AP sway (metres)


# ── Real-time fusion buffer ───────────────────────────────────────────────────

class FusionBuffer:
    """
    Continuously aligns incoming radar and camera frames in real time.

    Call push_radar() and push_camera() from their respective reader threads.
    Call pop_fused() from the processing thread to get aligned FusedFrames.

    The buffer uses Stage 1 (nearest-neighbour) alignment continuously.
    Stage 2 (cross-correlation offset) is applied automatically once
    MIN_CAL_SECONDS of data has been collected.
    """

    def __init__(self, camera_buffer_secs: float = 2.0):
        self._radar_buf:  deque[RadarFrame]  = deque()
        self._camera_buf: deque[CameraFrame] = deque()
        self._fused_buf:  deque[FusedFrame]  = deque()

        self._camera_buffer_secs = camera_buffer_secs
        self._ts_offset: float = 0.0   # systematic offset: t_radar_true ≈ t_radar_host - offset
        self._calibrated = False

        # For cross-correlation calibration
        self._cal_radar_signal:  list[float] = []
        self._cal_camera_signal: list[float] = []
        self._cal_radar_ts:  list[float] = []
        self._cal_camera_ts: list[float] = []
        self._session_start: float | None = None

    # ── Push / Pop ────────────────────────────────────────────────────────────

    def push_radar(self, frame: RadarFrame) -> None:
        if self._session_start is None:
            self._session_start = frame.host_ts
        self._radar_buf.append(frame)
        self._try_align()

    def push_camera(self, frame: CameraFrame) -> None:
        self._camera_buf.append(frame)
        self._trim_camera_buf(frame.host_ts)

    def pop_fused(self) -> FusedFrame | None:
        if self._fused_buf:
            return self._fused_buf.popleft()
        return None

    def pending(self) -> int:
        return len(self._fused_buf)

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _try_align(self) -> None:
        """
        For each radar frame in the buffer, try to find the best camera frame.
        A camera frame is 'best' if it is the nearest in time after applying the
        current offset estimate.
        """
        while self._radar_buf:
            r = self._radar_buf[0]
            corrected_r_ts = r.host_ts - self._ts_offset

            # Need at least one camera frame on each side of the radar timestamp
            # (or the camera buffer must be clearly past it) before committing.
            cam_list = list(self._camera_buf)
            if len(cam_list) < 2:
                return

            latest_cam_ts = cam_list[-1].host_ts
            if latest_cam_ts < corrected_r_ts + 0.05:
                # Wait for camera to catch up (50 ms headroom)
                return

            # Find nearest camera frame
            best_cam = min(cam_list, key=lambda c: abs(c.host_ts - corrected_r_ts))
            dt = abs(best_cam.host_ts - corrected_r_ts)

            # Extract AP range from radar
            if r.points:
                ap_range = float(np.mean([p.y for p in r.points]))
            else:
                ap_range = float('nan')

            fused = FusedFrame(
                ts=r.host_ts,
                radar_frame_num=r.frame_number,
                camera_frame_num=best_cam.frame_number,
                dt_alignment=dt,
                hip_ml_px=best_cam.hip_ml_px,
                ap_range_m=ap_range,
            )
            self._fused_buf.append(fused)
            self._radar_buf.popleft()

            # Accumulate signals for cross-correlation calibration
            if not self._calibrated:
                if r.points:
                    self._cal_radar_signal.append(float(np.mean([abs(p.doppler) for p in r.points])))
                else:
                    self._cal_radar_signal.append(0.0)
                self._cal_radar_ts.append(r.host_ts)

                if not np.isnan(best_cam.hip_ml_px):
                    self._cal_camera_signal.append(abs(best_cam.hip_ml_px))
                else:
                    self._cal_camera_signal.append(0.0)
                self._cal_camera_ts.append(best_cam.host_ts)

                elapsed = r.host_ts - self._session_start
                if elapsed >= MIN_CAL_SECONDS:
                    self._calibrate_offset()

    def _trim_camera_buf(self, current_ts: float) -> None:
        """Remove old camera frames that are too old to match any future radar frame."""
        cutoff = current_ts - self._camera_buffer_secs
        while self._camera_buf and self._camera_buf[0].host_ts < cutoff:
            self._camera_buf.popleft()

    def _calibrate_offset(self) -> None:
        """
        Stage 2: compute systematic time offset via cross-correlation.

        Both signals are resampled to a common 100 Hz grid before correlation
        to handle the rate mismatch (radar ~20 Hz, camera ~30 Hz).
        """
        try:
            radar_ts = np.array(self._cal_radar_ts)
            cam_ts   = np.array(self._cal_camera_ts)
            radar_sig = np.array(self._cal_radar_signal)
            cam_sig   = np.array(self._cal_camera_signal)

            if len(radar_ts) < 20 or len(cam_ts) < 20:
                return

            # Resample both to 100 Hz common grid
            t_start = max(radar_ts[0], cam_ts[0])
            t_end   = min(radar_ts[-1], cam_ts[-1])
            if t_end <= t_start:
                return

            t_grid = np.arange(t_start, t_end, 0.01)  # 100 Hz
            r_interp = np.interp(t_grid, radar_ts, radar_sig)
            c_interp = np.interp(t_grid, cam_ts,   cam_sig)

            # Normalise before cross-correlation
            def _norm(x):
                x = x - np.mean(x)
                std = np.std(x)
                return x / std if std > 1e-9 else x

            xcorr = correlate(_norm(r_interp), _norm(c_interp), mode='full')
            lags  = np.arange(-len(t_grid) + 1, len(t_grid))
            best_lag = lags[np.argmax(xcorr)]

            # Offset in seconds: positive means radar lags behind camera
            offset_s = best_lag * 0.01
            max_plausible_offset = 0.5   # >500 ms is likely a false correlation peak
            if abs(offset_s) < max_plausible_offset:
                self._ts_offset = offset_s
                self._calibrated = True
                logger.info(
                    "Temporal calibration complete: systematic offset = %.1f ms "
                    "(radar host_ts - %.1f ms → aligned to camera)",
                    offset_s * 1000, offset_s * 1000,
                )
            else:
                logger.warning(
                    "Cross-correlation offset %.1f ms exceeds plausible range; "
                    "keeping offset=0 ms. Check sensor connectivity.",
                    offset_s * 1000,
                )
                self._calibrated = True  # stop retrying

        except Exception as exc:
            logger.warning("Temporal calibration failed: %s", exc)
            self._calibrated = True


# ── Post-session fusion ───────────────────────────────────────────────────────

def fuse_session(
    radar_frames: list[RadarFrame],
    camera_frames: list[CameraFrame],
    px_per_m: float,
    fps_radar: float = 20.0,
    fps_camera: float = 30.0,
    ts_offset_s: float = 0.0,
) -> dict:
    """
    Offline fusion of a complete recorded session.

    Parameters
    ----------
    radar_frames   : list of RadarFrame from the session
    camera_frames  : list of CameraFrame from the session
    px_per_m       : calibrated pixel-to-metre scale factor from CameraReader
    fps_radar      : nominal radar frame rate
    fps_camera     : nominal camera frame rate
    ts_offset_s    : systematic time offset (from cross-correlation calibration
                     if available; otherwise 0)

    Returns
    -------
    dict with keys:
      'ml_m'         : np.ndarray – filtered, mean-centred ML sway (metres), at radar rate
      'ap_m'         : np.ndarray – filtered, mean-centred AP sway (metres), at radar rate
      'timestamps'   : np.ndarray – radar host timestamps (seconds)
      'dt_alignment' : np.ndarray – per-frame alignment quality (seconds)
      'fused_frames' : list[FusedFrame]
      'fps'          : effective frame rate of fused output
      'ts_offset_s'  : the offset applied
    """

    if not radar_frames or not camera_frames:
        raise ValueError("Both radar_frames and camera_frames must be non-empty")

    # ── Stage 1: nearest-neighbour matching ──────────────────────────────────
    cam_ts  = np.array([f.host_ts for f in camera_frames])
    cam_idx = np.arange(len(camera_frames))

    fused = []
    for r in radar_frames:
        corrected_ts = r.host_ts - ts_offset_s
        # Find nearest camera frame
        best_i = int(np.argmin(np.abs(cam_ts - corrected_ts)))
        c = camera_frames[best_i]
        dt = abs(c.host_ts - corrected_ts)

        if r.points:
            ap_range = float(np.mean([p.y for p in r.points]))
        else:
            ap_range = float('nan')

        fused.append(FusedFrame(
            ts=r.host_ts,
            radar_frame_num=r.frame_number,
            camera_frame_num=c.frame_number,
            dt_alignment=dt,
            hip_ml_px=c.hip_ml_px,
            ap_range_m=ap_range,
        ))

    # ── Stage 2: extract and filter ML and AP signals ─────────────────────────
    ml_px  = np.array([f.hip_ml_px  for f in fused], dtype=np.float64)
    ap_raw = np.array([f.ap_range_m for f in fused], dtype=np.float64)
    timestamps = np.array([f.ts for f in fused], dtype=np.float64)

    # Convert ML from pixels to metres
    ml_m = ml_px / px_per_m

    # Interpolate any NaN gaps before filtering
    for arr in (ml_m, ap_raw):
        nans = np.isnan(arr)
        if nans.any() and not nans.all():
            x = np.arange(len(arr))
            arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    # Mean-centre (sway = deviation from mean standing position)
    ml_m   -= np.nanmean(ml_m)
    ap_raw -= np.nanmean(ap_raw)

    # Butterworth low-pass filter (matches sway_utils/metrics.py parameters)
    ml_m   = _butterworth(ml_m,   fc=BUTTERWORTH_FC, fs=fps_radar, N=BUTTERWORTH_N)
    ap_m   = _butterworth(ap_raw, fc=BUTTERWORTH_FC, fs=fps_radar, N=BUTTERWORTH_N)

    # Write back to fused frames
    for i, f in enumerate(fused):
        f.ml_m = float(ml_m[i])
        f.ap_m = float(ap_m[i])

    # Compute effective fps from actual timestamps
    if len(timestamps) > 1:
        effective_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    else:
        effective_fps = fps_radar

    dt_alignment = np.array([f.dt_alignment for f in fused])
    logger.info(
        "Session fused: %d frames, %.1f Hz effective, "
        "mean alignment dt=%.1f ms, max=%.1f ms",
        len(fused), effective_fps,
        dt_alignment.mean() * 1000, dt_alignment.max() * 1000,
    )

    return {
        'ml_m':         ml_m,
        'ap_m':         ap_m,
        'timestamps':   timestamps,
        'dt_alignment': dt_alignment,
        'fused_frames': fused,
        'fps':          effective_fps,
        'ts_offset_s':  ts_offset_s,
    }


def to_sway_dataframe(fused_result: dict):
    """
    Convert fused session result to a pandas DataFrame compatible with
    sway_utils/metrics.py calculate_sway_from_recording().

    The existing code expects columns 'CoGx' (ML) and 'CoGz' (AP)
    in centimetres (Kinect units were metres, but AREA_CE is reported in cm
    because the data was scaled ×100 in recordings.py).

    We output metres here — note that if you pass this directly to
    calculate_sway_from_recording() you should set fc=10 in filter_signal()
    and be aware the metrics will be in metres, not the original cm scale.
    """
    import pandas as pd
    return pd.DataFrame({
        'CoGx':         fused_result['ml_m'],   # ML — maps to Kinect X axis
        'CoGz':         fused_result['ap_m'],   # AP — maps to Kinect Z axis
        'timestamp':    fused_result['timestamps'],
        'dt_alignment': fused_result['dt_alignment'],
    })


def estimate_offset_xcorr(
    radar_frames: list[RadarFrame],
    camera_frames: list[CameraFrame],
    resample_hz: float = 100.0,
) -> float:
    """
    Estimate systematic time offset between radar and camera streams using
    cross-correlation of motion-proxy signals.

    Returns offset_s such that: t_radar_true ≈ t_radar_host - offset_s

    Motion proxies used:
      Radar:  mean absolute Doppler velocity per frame
      Camera: hip midpoint X velocity (absolute) per frame
    """
    radar_ts  = np.array([f.host_ts for f in radar_frames])
    cam_ts    = np.array([f.host_ts for f in camera_frames])

    radar_sig = radar_doppler_signal(radar_frames)
    cam_sig   = camera_hip_velocity(camera_frames)

    # Resample both to common grid
    t_start = max(radar_ts[0], cam_ts[0])
    t_end   = min(radar_ts[-1], cam_ts[-1])
    dt = 1.0 / resample_hz
    t_grid = np.arange(t_start, t_end, dt)

    if len(t_grid) < 20:
        logger.warning("Insufficient overlap for cross-correlation calibration")
        return 0.0

    r_interp = np.interp(t_grid, radar_ts, radar_sig)
    c_interp = np.interp(t_grid, cam_ts,   cam_sig)

    def _norm(x):
        x = x - np.mean(x)
        s = np.std(x)
        return x / s if s > 1e-9 else x

    xcorr    = correlate(_norm(r_interp), _norm(c_interp), mode='full')
    lags     = np.arange(-len(t_grid) + 1, len(t_grid))
    best_lag = lags[np.argmax(xcorr)]
    offset_s = best_lag * dt

    logger.info("Cross-correlation offset: %.1f ms (lag=%d at %.0f Hz)",
                offset_s * 1000, best_lag, resample_hz)
    return float(offset_s)


# ── CoP estimation ───────────────────────────────────────────────────────────

def estimate_cop(
    com_ml: np.ndarray,
    com_ap: np.ndarray,
    fps: float,
    body_height_m: float,
    filter_fc: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate Centre of Pressure (CoP) from Centre of Mass (CoM) trajectory
    using the inverted pendulum model:

        CoP = CoM + (h_CoM / g) × CoM_acceleration

    Where h_CoM ≈ 55% of body height (Dempster's whole-body CoM height).

    Parameters
    ----------
    com_ml      : ML CoM trajectory (metres, mean-centred)
    com_ap      : AP CoM trajectory (metres, mean-centred)
    fps         : sample rate (Hz)
    body_height_m : subject height in metres
    filter_fc   : low-pass cutoff before differentiation (Hz)

    Returns
    -------
    cop_ml, cop_ap : CoP trajectories in metres

    Notes
    -----
    - The acceleration term amplifies noise, so CoM must be well-filtered
      before differentiation. A second Butterworth pass is applied here.
    - At typical quiet standing frequencies (<1 Hz), the acceleration
      correction is small (~5-15% of CoM amplitude) but meaningful for
      validation against force plate CoP.
    - This is the same GRF estimation approach used in compute_biomechanics.py.
    """
    G = 9.81
    h_com = body_height_m * 0.55   # CoM height ≈ 55% of body height

    dt = 1.0 / fps

    # Quiet standing sway is essentially all below 3 Hz.
    # Filter aggressively before differentiation — double differentiation
    # squares the noise amplification, so even small jitter at 10 Hz becomes
    # enormous in the acceleration signal. 3 Hz cutoff is standard practice
    # for kinematic-derived CoP estimation.
    fc_safe = min(filter_fc, fps * 0.4)  # never exceed 40% of Nyquist
    com_ml_f = _butterworth(com_ml, fc=fc_safe, fs=fps, N=2)
    com_ap_f = _butterworth(com_ap, fc=fc_safe, fs=fps, N=2)

    # Central difference acceleration
    def accel(x):
        a = np.zeros_like(x)
        a[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt**2
        a[0]    = a[1]
        a[-1]   = a[-2]
        return a

    acc_ml = accel(com_ml_f)
    acc_ap = accel(com_ap_f)

    # Apply the inverted pendulum correction then smooth the result once more
    cop_ml_raw = com_ml_f + (h_com / G) * acc_ml
    cop_ap_raw = com_ap_f + (h_com / G) * acc_ap

    cop_ml = _butterworth(cop_ml_raw, fc=fc_safe, fs=fps, N=2)
    cop_ap = _butterworth(cop_ap_raw, fc=fc_safe, fs=fps, N=2)

    return cop_ml, cop_ap


# ── Internal helpers ──────────────────────────────────────────────────────────

def _butterworth(signal: np.ndarray, fc: float, fs: float, N: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (matches sway_utils/metrics.py)."""
    Wn = np.pi * fc / (2 * fs)   # same normalisation as metrics.py
    Wn = max(1e-3, min(Wn, 0.99))  # clamp: protects against low fps at startup
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, signal)
