#!/usr/bin/env python3
"""
RGB camera reader using MediaPipe Pose for frontal-view quiet standing.

Each frame is timestamped with time.perf_counter() at capture and placed
into a thread-safe queue as a CameraFrame namedtuple.

Coordinate conventions (frontal view, subject facing camera):
  ML axis: landmark.x in image coordinates → left/right = mediolateral
  AP axis: NOT available from RGB alone (depth) — supplied by radar
  Vertical axis: landmark.y (inverted: 0=top, 1=bottom in MediaPipe)

The ML COM position is estimated from the hip midpoint X coordinate,
calibrated to metres using the subject's known height.
"""

import threading
import time
import logging
import traceback
import urllib.request
import os
from collections import namedtuple
from queue import Queue, Full
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

logger = logging.getLogger(__name__)

# ── Model download ────────────────────────────────────────────────────────────
# Approximate CPU speed on a typical laptop:
#   lite  : ~30 fps  (less accurate but sufficient for sway landmarks)
#   full  : ~20 fps  (good balance — default)
#   heavy : ~10 fps  (best accuracy, needed only if z-depth is used)
_MODEL_VARIANTS = {
    'lite':  'pose_landmarker_lite',
    'full':  'pose_landmarker_full',
    'heavy': 'pose_landmarker_heavy',
}
_MODEL_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{name}/float16/latest/{name}.task"
)

def _ensure_model(model_type: str = 'full') -> str:
    name = _MODEL_VARIANTS.get(model_type, _MODEL_VARIANTS['full'])
    path = os.path.join(os.path.dirname(__file__), f"{name}.task")
    if not os.path.exists(path):
        url = _MODEL_BASE_URL.format(name=name)
        logger.info("Downloading MediaPipe %s model...", name)
        print(f"Downloading {name}.task (first run)...")
        urllib.request.urlretrieve(url, path)
        logger.info("Model saved to %s", path)
    return path

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
LEFT_HEEL      = 29
RIGHT_HEEL     = 30
LEFT_FOOT      = 31
RIGHT_FOOT     = 32

# ── Dempster (1955) segment mass fractions (Winter 2009, Table 4.1) ───────────
# Each bilateral entry covers both sides combined.
# Segment CoM is estimated as the midpoint of proximal/distal joints.
DEMPSTER_WEIGHTS = {
    'head':      0.081,   # ears midpoint (or nose fallback)
    'trunk':     0.497,   # shoulder–hip midpoint
    'upper_arm': 0.028,   # shoulder–elbow midpoint  (per side; applied ×2)
    'forearm':   0.016,   # elbow–wrist midpoint      (per side; applied ×2)
    'hand':      0.006,   # wrist point               (per side; applied ×2)
    'thigh':     0.100,   # hip–knee midpoint         (per side; applied ×2)
    'shank':     0.0465,  # knee–ankle midpoint       (per side; applied ×2)
    'foot':      0.0145,  # ankle–foot midpoint       (per side; applied ×2)
}
# Sum = 0.081 + 0.497 + 2*(0.028+0.016+0.006+0.100+0.0465+0.0145) = 1.000

_VIS_THRESHOLD = 0.3   # minimum MediaPipe visibility to trust a landmark

# ── Dempster CoM estimation ───────────────────────────────────────────────────

def _dempster_com(lm: dict) -> tuple:
    """
    Compute Dempster-weighted whole-body CoM in pixel coordinates.

    Returns (com_x, com_y) — both in image pixels.
    com_x = ML position; com_y = vertical position in image (Y increases downward).

    Segment CoM = midpoint of proximal/distal joints.
    Landmarks below _VIS_THRESHOLD are excluded; remaining weights are
    re-normalised so the result is always a proper weighted mean.

    Returns (nan, nan) if fewer than 3 segments are visible.
    """
    def _coord(key: str, axis: int):
        entry = lm.get(key)
        if entry is None:
            return float('nan')
        return entry[axis] if entry[3] >= _VIS_THRESHOLD else float('nan')

    def _mid(a: float, b: float) -> float:
        if np.isfinite(a) and np.isfinite(b):
            return (a + b) / 2.0
        return a if np.isfinite(a) else b if np.isfinite(b) else float('nan')

    def _seg(axis: int):
        """Return (head, trunk, upper_arm, forearm, hand, thigh, shank, foot) along axis."""
        v = lambda k: _coord(k, axis)

        head = _mid(v('left_ear'), v('right_ear'))
        if not np.isfinite(head):
            head = v('nose')

        sho   = _mid(v('left_shoulder'), v('right_shoulder'))
        hip   = _mid(v('left_hip'),      v('right_hip'))
        trunk = _mid(sho, hip)

        def _bi(pl, dl, pr, dr):
            return _mid(_mid(v(pl), v(dl)), _mid(v(pr), v(dr)))

        upper_arm = _bi('left_shoulder', 'left_elbow',  'right_shoulder', 'right_elbow')
        forearm   = _bi('left_elbow',    'left_wrist',  'right_elbow',    'right_wrist')
        hand      = _mid(v('left_wrist'), v('right_wrist'))
        thigh     = _bi('left_hip',  'left_knee',  'right_hip',  'right_knee')
        shank     = _bi('left_knee', 'left_ankle', 'right_knee', 'right_ankle')
        foot      = _mid(_mid(v('left_ankle'),  v('left_foot')),
                         _mid(v('right_ankle'), v('right_foot')))
        return [head, trunk, upper_arm, forearm, hand, thigh, shank, foot]

    W = DEMPSTER_WEIGHTS
    weights = [W['head'], W['trunk'], W['upper_arm'], W['forearm'],
               W['hand'], W['thigh'], W['shank'],     W['foot']]

    results = []
    for axis in (0, 1):   # 0 = X (ML), 1 = Y (vertical)
        vals = _seg(axis)
        visible = [(w, v) for w, v in zip(weights, vals) if np.isfinite(v)]
        if len(visible) < 3:
            results.append(float('nan'))
        else:
            total_w = sum(w for w, _ in visible)
            results.append(sum(w * v for w, v in visible) / total_w)

    return tuple(results)   # (com_x, com_y)


# ── Data structures ───────────────────────────────────────────────────────────
CameraFrame = namedtuple('CameraFrame', [
    'host_ts',        # float: time.perf_counter() at frame capture
    'frame_number',   # int
    'landmarks',      # dict[str, (x_px, y_px, z_mp, visibility)] or None if no pose
    'hip_ml_px',      # float: hip midpoint X in pixels (ML direction), or NaN
    'hip_height_px',  # float: vertical hip midpoint Y in pixels, or NaN
    'shoulder_ml_px', # float: shoulder midpoint X in pixels
    'ap_proxy_px',    # float: AP sway proxy from apparent body height (px) — larger = closer, or NaN
    'com_ml_px',      # float: Dempster-weighted whole-body CoM X in pixels, or NaN
    'com_y_px',       # float: Dempster-weighted whole-body CoM Y in pixels (image coords), or NaN
    'image_width',    # int
    'image_height',   # int
    'bgr_frame',      # np.ndarray (H, W, 3) uint8 — raw camera frame for display
])


class CameraReader:
    """
    Captures frames from an RGB camera, runs MediaPipe Pose, and emits
    CameraFrame objects into a queue.

    Usage
    -----
    reader = CameraReader(device_index=0, subject_height_m=1.75)
    reader.start()
    frame = reader.queue.get(timeout=1.0)
    reader.stop()

    Calibration
    -----------
    subject_height_m is used to establish a px/m scale factor.
    It is estimated at startup from the first few frames by measuring the
    pixel distance from ankle to shoulder, then refined over the session.
    If subject_height_m is None, pixel units are used (metrics will be in pixels).
    """

    def __init__(
        self,
        device_index: int = 0,
        subject_height_m: float | None = None,
        fps: int = 60,
        queue_size: int = 128,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        capture_width: int = 640,     # capture resolution (MJPEG allows 30fps at 640x480 over USB)
        capture_height: int = 480,
        inference_width: int = 320,   # downscale to this before pose inference; 0 = no downscale
        model_type: str = 'full',     # 'lite' ~30fps / 'full' ~20fps / 'heavy' ~10fps
    ):
        self.device_index = device_index
        self.subject_height_m = subject_height_m
        self.fps = fps
        self.queue: Queue[CameraFrame] = Queue(maxsize=queue_size)
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.inference_width = inference_width
        self.model_type = model_type

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread_error: Optional[Exception] = None  # captures crash from background thread

        # Calibration: pixels per metre
        self._px_per_m: Optional[float] = None
        self._calibration_samples = []
        self._calibration_frames = 30   # number of frames to average for calibration
        self._calib_lock = threading.Lock()  # guards calibration state (callback runs on MP thread)

        # Statistics
        self.frames_captured = 0
        self.frames_no_pose = 0
        self.frames_dropped = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        # Try Media Foundation first (same API as Windows Camera app — better compatibility
        # with some drivers e.g. PS3 Universal), then fall back to DirectShow.
        for backend, name in [(cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_DSHOW, 'DSHOW'), (0, 'default')]:
            self._cap = cv2.VideoCapture(self.device_index, backend)
            if self._cap.isOpened():
                print(f"Camera opened with {name} backend")
                break
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.device_index}. "
                f"Run the camera index check snippet to find available devices."
            )
        # Set resolution and FPS — do not force MJPEG codec as some drivers
        # (e.g. PS3 Universal) use their own internal format and ignore it.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # discard stale frames, minimise latency
        # Report what the driver actually negotiated (may differ from what was requested)
        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        actual_codec = ''.join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        print(f"Camera: requested {self.capture_width}x{self.capture_height} @ {self.fps}fps MJPG")
        print(f"Camera: got       {actual_w}x{actual_h} @ {actual_fps:.0f}fps {actual_codec!r}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name='CameraReader')
        self._thread.start()
        logger.info("CameraReader started on device %d", self.device_index)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        if self._thread_error:
            logger.error("CameraReader thread crashed: %s", self._thread_error)
        logger.info(
            "CameraReader stopped. frames_captured=%d no_pose=%d dropped=%d px_per_m=%s",
            self.frames_captured, self.frames_no_pose, self.frames_dropped,
            f"{self._px_per_m:.1f}" if self._px_per_m else "not calibrated",
        )

    @property
    def px_per_m(self) -> Optional[float]:
        """Calibrated pixel-to-metre scale factor (set after ~30 frames)."""
        return self._px_per_m

    # ── Capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        try:
            model_path = _ensure_model(self.model_type)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
            )
            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                frame_num = 0
                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ret, bgr = self._cap.read()
                    host_ts = time.perf_counter()
                    if not ret:
                        consecutive_failures += 1
                        logger.warning("Camera read failed (attempt %d)", consecutive_failures)
                        if consecutive_failures > 10:
                            raise RuntimeError("Camera read failed 10 times consecutively — device lost")
                        time.sleep(0.05)
                        continue
                    consecutive_failures = 0

                    self.frames_captured += 1
                    frame_num += 1
                    h, w = bgr.shape[:2]

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if self.inference_width > 0 and w > self.inference_width:
                        inf_h = max(1, int(h * self.inference_width / w))
                        inf_rgb = cv2.resize(rgb, (self.inference_width, inf_h),
                                             interpolation=cv2.INTER_LINEAR)
                    else:
                        inf_rgb = rgb

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_rgb)
                    timestamp_ms = int(host_ts * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if not result.pose_landmarks:
                        self.frames_no_pose += 1
                        cf = CameraFrame(
                            host_ts=host_ts, frame_number=frame_num,
                            landmarks=None, hip_ml_px=float('nan'),
                            hip_height_px=float('nan'), shoulder_ml_px=float('nan'),
                            ap_proxy_px=float('nan'), com_ml_px=float('nan'),
                            com_y_px=float('nan'),
                            image_width=w, image_height=h, bgr_frame=bgr,
                        )
                    else:
                        cf = self._build_frame(
                            result.pose_landmarks[0], host_ts, frame_num, w, h, bgr
                        )
                    self._update_calibration(cf)
                    self._enqueue(cf)

        except Exception as exc:
            self._thread_error = exc
            logger.error("CameraReader thread crashed:\n%s", traceback.format_exc())

    def _build_frame(self, lm, host_ts, frame_num, w, h, bgr) -> CameraFrame:
        def px(idx):
            l = lm[idx]
            return l.x * w, l.y * h, l.z, getattr(l, 'visibility', 1.0)

        landmarks = {
            'nose':            px(NOSE),
            'left_ear':        px(LEFT_EAR),
            'right_ear':       px(RIGHT_EAR),
            'left_shoulder':   px(LEFT_SHOULDER),
            'right_shoulder':  px(RIGHT_SHOULDER),
            'left_elbow':      px(LEFT_ELBOW),
            'right_elbow':     px(RIGHT_ELBOW),
            'left_wrist':      px(LEFT_WRIST),
            'right_wrist':     px(RIGHT_WRIST),
            'left_hip':        px(LEFT_HIP),
            'right_hip':       px(RIGHT_HIP),
            'left_knee':       px(LEFT_KNEE),
            'right_knee':      px(RIGHT_KNEE),
            'left_ankle':      px(LEFT_ANKLE),
            'right_ankle':     px(RIGHT_ANKLE),
            'left_heel':       px(LEFT_HEEL),
            'right_heel':      px(RIGHT_HEEL),
            'left_foot':       px(LEFT_FOOT),
            'right_foot':      px(RIGHT_FOOT),
        }

        lh_x, lh_y = landmarks['left_hip'][:2]
        rh_x, rh_y = landmarks['right_hip'][:2]
        hip_ml_px   = (lh_x + rh_x) / 2.0
        hip_vert_px = (lh_y + rh_y) / 2.0

        ls_x = landmarks['left_shoulder'][0]
        rs_x = landmarks['right_shoulder'][0]
        shoulder_ml_px = (ls_x + rs_x) / 2.0

        # AP proxy: hip-to-upper-body vertical pixel distance.
        # Forward lean → trunk tilts → apparent vertical distance changes.
        # Use shoulders + ears (where visible) for a more robust upper-body
        # reference point — reduces noise when a single shoulder landmark jitters.
        hip_y_mid  = (lh_y + rh_y) / 2.0
        ls_y = landmarks['left_shoulder'][1]
        rs_y = landmarks['right_shoulder'][1]
        upper_y_pts = [ls_y, rs_y]
        if landmarks['left_ear'][3]  >= _VIS_THRESHOLD:
            upper_y_pts.append(landmarks['left_ear'][1])
        if landmarks['right_ear'][3] >= _VIS_THRESHOLD:
            upper_y_pts.append(landmarks['right_ear'][1])
        upper_y_mid = float(np.mean(upper_y_pts))
        ap_proxy_px = abs(hip_y_mid - upper_y_mid)

        com_ml_px, com_y_px = _dempster_com(landmarks)

        return CameraFrame(
            host_ts=host_ts, frame_number=frame_num,
            landmarks=landmarks,
            hip_ml_px=hip_ml_px,
            hip_height_px=hip_vert_px,
            shoulder_ml_px=shoulder_ml_px,
            ap_proxy_px=ap_proxy_px,
            com_ml_px=com_ml_px,
            com_y_px=com_y_px,
            image_width=w, image_height=h, bgr_frame=bgr,
        )

    # ── Calibration ───────────────────────────────────────────────────────────

    def _update_calibration(self, frame: CameraFrame) -> None:
        """
        Estimate px/m from ankle-to-shoulder pixel distance and subject height.
        Uses the first _calibration_frames good frames then locks the value.
        Called from the MediaPipe callback thread — protected by _calib_lock.
        """
        with self._calib_lock:
            if self._px_per_m is not None:
                return  # already calibrated
            if self.subject_height_m is None:
                return  # no reference height provided
            if frame.landmarks is None:
                return

            lm = frame.landmarks
            la_x, la_y = lm['left_ankle'][:2]
            ra_x, ra_y = lm['right_ankle'][:2]
            ls_x, ls_y = lm['left_shoulder'][:2]
            rs_x, rs_y = lm['right_shoulder'][:2]

            ankle_y    = (la_y + ra_y) / 2.0
            shoulder_y = (ls_y + rs_y) / 2.0

            # Shoulder-to-ankle ≈ 80% of total height (empirical approximation)
            segment_height_m = self.subject_height_m * 0.80
            px_span = abs(ankle_y - shoulder_y)

            if px_span > 10:   # ignore degenerate frames
                self._calibration_samples.append(px_span / segment_height_m)

            if len(self._calibration_samples) >= self._calibration_frames:
                self._px_per_m = float(np.median(self._calibration_samples))
            logger.info("Camera calibrated: %.1f px/m (from %d frames)",
                        self._px_per_m, len(self._calibration_samples))

    def _enqueue(self, frame: CameraFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except Full:
            self.frames_dropped += 1
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except Exception:
                pass


# ── Utilities ─────────────────────────────────────────────────────────────────

def camera_ml_sway(frames: list[CameraFrame], px_per_m: float) -> np.ndarray:
    """
    Extract the ML sway signal (metres) from a list of CameraFrames.
    NaN frames are linearly interpolated.
    """
    values = np.array([f.hip_ml_px for f in frames], dtype=np.float64)
    values /= px_per_m   # convert pixels → metres
    # Mean-centre (remove standing position offset)
    nans = np.isnan(values)
    if not nans.all():
        values -= np.nanmean(values)
    # Interpolate NaN gaps
    if nans.any() and not nans.all():
        x = np.arange(len(values))
        values[nans] = np.interp(x[nans], x[~nans], values[~nans])
    return values


def camera_hip_velocity(frames: list[CameraFrame]) -> np.ndarray:
    """
    Frame-to-frame hip displacement magnitude in pixels (unnormalised).
    Used as a 1-D motion signal for cross-correlation temporal calibration.
    Returns array of length len(frames); first element is 0.
    """
    ml = np.array([f.hip_ml_px for f in frames], dtype=np.float64)
    nans = np.isnan(ml)
    if not nans.all():
        ml[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], ml[~nans])
    vel = np.abs(np.diff(ml, prepend=ml[0]))
    return vel


def extract_frontal_joint_angles(frames: list[CameraFrame], px_per_m: float) -> dict[str, np.ndarray]:
    """
    Extract frontal-plane joint angles (degrees) over the recording.

    Returns a dict with keys:
      'trunk_lean_ml' - lateral trunk lean angle relative to vertical (degrees)
      'hip_width_m'   - inter-hip distance in metres (postural width indicator)
      'knee_sym'      - knee symmetry index: |left_knee_x - right_knee_x| / hip_width_px
    """
    trunk_lean = []
    hip_width  = []
    knee_sym   = []

    for f in frames:
        if f.landmarks is None:
            trunk_lean.append(np.nan)
            hip_width.append(np.nan)
            knee_sym.append(np.nan)
            continue

        lm = f.landmarks
        lh_x, lh_y = lm['left_hip'][:2]
        rh_x, rh_y = lm['right_hip'][:2]
        ls_x, ls_y = lm['left_shoulder'][:2]
        rs_x, rs_y = lm['right_shoulder'][:2]
        lk_x = lm['left_knee'][0]
        rk_x = lm['right_knee'][0]

        hip_mid_x = (lh_x + rh_x) / 2.0
        hip_mid_y = (lh_y + rh_y) / 2.0
        sho_mid_x = (ls_x + rs_x) / 2.0
        sho_mid_y = (ls_y + rs_y) / 2.0

        # Lateral lean: angle of trunk midline from vertical
        dx = sho_mid_x - hip_mid_x
        dy = hip_mid_y - sho_mid_y   # invert Y (pixels increase downward)
        lean_deg = np.degrees(np.arctan2(dx, dy))
        trunk_lean.append(lean_deg)

        hw_px = abs(lh_x - rh_x)
        hip_width.append(hw_px / px_per_m)

        if hw_px > 0:
            ks = abs(lk_x - rk_x) / hw_px
        else:
            ks = np.nan
        knee_sym.append(ks)

    def _interp(arr):
        a = np.array(arr, dtype=np.float64)
        nans = np.isnan(a)
        if nans.any() and not nans.all():
            x = np.arange(len(a))
            a[nans] = np.interp(x[nans], x[~nans], a[~nans])
        return a

    return {
        'trunk_lean_ml': _interp(trunk_lean),
        'hip_width_m':   _interp(hip_width),
        'knee_sym':      _interp(knee_sym),
    }
