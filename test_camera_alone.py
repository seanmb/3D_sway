import sys
import time
import logging
sys.path.insert(0, r'C:\KINECAL\rgb_radar_acquisition')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from camera_reader import CameraReader

reader = CameraReader(device_index=0, subject_height_m=1.75)
reader.start()

# Give the thread a moment to start and catch any immediate crash
time.sleep(1.0)

if reader._thread_error:
    print(f"Thread crashed on startup: {reader._thread_error}")
    reader.stop()
    sys.exit(1)

try:
    for _ in range(100):
        frame = reader.queue.get(timeout=2.0)
        print(f"frame={frame.frame_number:4d}  "
              f"hip_ml={frame.hip_ml_px:7.1f}px  "
              f"ts={frame.host_ts:.3f}  "
              f"pose={'YES' if frame.landmarks else 'NO '}")
        if reader._thread_error:
            print(f"Thread crashed mid-run: {reader._thread_error}")
            break
finally:
    reader.stop()
