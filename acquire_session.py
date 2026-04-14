#!/usr/bin/env python3
"""
Main acquisition script: RGB camera + TI IWR6843 radar quiet standing session.

Usage
-----
python acquire_session.py \
    --config-port COM3 \
    --data-port   COM4 \
    --camera      0 \
    --height      1.75 \
    --duration    30 \
    --output      ./sessions/participant_01

The output directory will contain:
  radar_frames.pkl   - list of RadarFrame namedtuples
  camera_frames.pkl  - list of CameraFrame namedtuples
  fused.pkl          - fused session result dict
  sway_input.csv     - CoGx (ML) and CoGz (AP) in metres, ready for sway_utils
  metadata.json      - session metadata (ts_offset, fps, alignment stats, etc.)
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np

# Allow running from this directory with sway_utils on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from radar_reader  import RadarReader
from camera_reader import CameraReader
from sensor_fusion import FusionBuffer, fuse_session, estimate_offset_xcorr, to_sway_dataframe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='RGB + Radar quiet standing acquisition')
    p.add_argument('--config-port', required=True, help='IWR6843 config UART (e.g. COM3)')
    p.add_argument('--data-port',   required=True, help='IWR6843 data  UART (e.g. COM4)')
    p.add_argument('--camera',      type=int, default=0, help='OpenCV camera device index')
    p.add_argument('--height',      type=float, required=True, help='Subject height in metres')
    p.add_argument('--duration',    type=float, default=30.0,  help='Recording duration (seconds)')
    p.add_argument('--output',      required=True, help='Output directory path')
    p.add_argument('--radar-fps',   type=float, default=20.0, help='Nominal radar frame rate')
    p.add_argument('--camera-fps',  type=float, default=30.0, help='Camera capture frame rate')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Initialise readers ────────────────────────────────────────────────────
    radar  = RadarReader(config_port=args.config_port, data_port=args.data_port)
    camera = CameraReader(
        device_index=args.camera,
        subject_height_m=args.height,
        fps=int(args.camera_fps),
    )

    radar_frames:  list = []
    camera_frames: list = []
    fusion_buf = FusionBuffer()

    logger.info("Starting acquisition. Duration: %.0f s", args.duration)
    logger.info("Stand still facing the camera/radar. Recording starts in 3 seconds...")
    time.sleep(3.0)

    # ── Start readers ─────────────────────────────────────────────────────────
    camera.start()
    time.sleep(0.5)   # let camera warm up before radar fires
    radar.start()

    t_start = time.perf_counter()
    t_end   = t_start + args.duration

    # ── Acquisition loop ──────────────────────────────────────────────────────
    logger.info("Recording...")
    try:
        while time.perf_counter() < t_end:
            # Drain radar queue
            while not radar.queue.empty():
                r = radar.queue.get_nowait()
                radar_frames.append(r)
                fusion_buf.push_radar(r)

            # Drain camera queue
            while not camera.queue.empty():
                c = camera.queue.get_nowait()
                camera_frames.append(c)
                fusion_buf.push_camera(c)

            elapsed = time.perf_counter() - t_start
            remaining = args.duration - elapsed
            if int(elapsed) % 5 == 0 and elapsed > 0:
                n_aligned = fusion_buf.pending()
                cal_status = "calibrated" if fusion_buf._calibrated else "calibrating..."
                logger.info("  %.0f s remaining | radar=%d cam=%d aligned=%d | %s",
                            remaining, len(radar_frames), len(camera_frames),
                            n_aligned, cal_status)
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    finally:
        radar.stop()
        camera.stop()

    logger.info("Acquisition complete: %d radar frames, %d camera frames",
                len(radar_frames), len(camera_frames))

    if not radar_frames or not camera_frames:
        logger.error("No data collected — check sensor connections")
        sys.exit(1)

    # ── Post-session temporal calibration ─────────────────────────────────────
    logger.info("Running cross-correlation temporal calibration...")
    ts_offset = estimate_offset_xcorr(radar_frames, camera_frames)
    logger.info("Applying offset: %.1f ms", ts_offset * 1000)

    # ── Fuse session ──────────────────────────────────────────────────────────
    px_per_m = camera.px_per_m
    if px_per_m is None:
        # Fallback: estimate from subject height and typical image geometry
        # Assume subject fills ~80% of frame height at ~2m distance
        # This is a rough fallback — real value comes from calibration
        w = camera_frames[0].image_width
        h = camera_frames[0].image_height
        px_per_m = h * 0.80 / (args.height * 0.80)
        logger.warning("Camera calibration not complete — using fallback %.1f px/m", px_per_m)

    logger.info("Fusing session with px_per_m=%.1f, ts_offset=%.1f ms",
                px_per_m, ts_offset * 1000)
    fused = fuse_session(
        radar_frames=radar_frames,
        camera_frames=camera_frames,
        px_per_m=px_per_m,
        fps_radar=args.radar_fps,
        fps_camera=args.camera_fps,
        ts_offset_s=ts_offset,
    )

    dt_align = fused['dt_alignment']
    logger.info(
        "Alignment quality — mean: %.1f ms  max: %.1f ms  >50ms: %d/%d frames",
        dt_align.mean() * 1000, dt_align.max() * 1000,
        int((dt_align > 0.05).sum()), len(dt_align),
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    logger.info("Saving to %s ...", args.output)

    # Raw frames
    with open(os.path.join(args.output, 'radar_frames.pkl'), 'wb') as f:
        pickle.dump(radar_frames, f)
    with open(os.path.join(args.output, 'camera_frames.pkl'), 'wb') as f:
        pickle.dump(camera_frames, f)

    # Fused result
    with open(os.path.join(args.output, 'fused.pkl'), 'wb') as f:
        pickle.dump(fused, f)

    # CSV compatible with sway_utils/metrics.py
    df = to_sway_dataframe(fused)
    df.to_csv(os.path.join(args.output, 'sway_input.csv'), index=False)

    # Metadata
    metadata = {
        'subject_height_m': args.height,
        'duration_s':       args.duration,
        'px_per_m':         px_per_m,
        'ts_offset_ms':     ts_offset * 1000,
        'calibrated':       fusion_buf._calibrated,
        'radar_frames':     len(radar_frames),
        'camera_frames':    len(camera_frames),
        'fused_frames':     len(fused['fused_frames']),
        'effective_fps':    fused['fps'],
        'alignment_mean_ms': float(dt_align.mean() * 1000),
        'alignment_max_ms':  float(dt_align.max() * 1000),
        'alignment_p95_ms':  float(np.percentile(dt_align, 95) * 1000),
        'radar_config_port': args.config_port,
        'radar_data_port':   args.data_port,
        'camera_device':     args.camera,
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Done. Outputs written to %s", args.output)
    logger.info("Next step: pass sway_input.csv to sway_utils/metrics.py "
                "calculate_sway_from_recording() to compute biomechanical metrics")


if __name__ == '__main__':
    main()
