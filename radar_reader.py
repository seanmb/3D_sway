#!/usr/bin/env python3
"""
TI IWR6843 radar reader.

Connects to the IWR6843 over two serial ports:
  - Config port  (115200 baud): sends chirp configuration commands
  - Data port    (921600 baud): receives TLV frames

Each frame is timestamped with time.perf_counter() on receipt and placed
into a thread-safe queue as a RadarFrame namedtuple.

TLV types parsed:
  - Type 1: Detected point cloud (x, y, z, doppler)
  - Type 2: Range profile magnitude
  - Type 5: Range-Doppler heatmap (for micro-Doppler accumulation)
"""

import struct
import threading
import time
import logging
from collections import namedtuple
from queue import Queue, Full

import numpy as np
import serial

logger = logging.getLogger(__name__)

# ── TLV constants ────────────────────────────────────────────────────────────
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_FMT = '<Q4I3H2I'   # magic(8) ver(4) len(4) platform(4) frame_num(4)
                           # time_cpu_cycles(4) num_obj(2) num_tlvs(2) subframe(2)
                           # (note: some SDK versions differ slightly — adjust if needed)
HEADER_SIZE = 40           # bytes
TLV_HEADER_FMT = '<II'     # type(4) length(4)
TLV_HEADER_SIZE = 8

TLV_POINT_CLOUD  = 1
TLV_RANGE_PROF   = 2
TLV_RANGE_DOPPLER = 5

DSP_CLOCK_HZ = 600e6  # IWR6843 C674x DSP clock

# ── Data structures ──────────────────────────────────────────────────────────
Point = namedtuple('Point', ['x', 'y', 'z', 'doppler'])

RadarFrame = namedtuple('RadarFrame', [
    'host_ts',        # float: time.perf_counter() at first byte received
    'frame_number',   # int
    'cpu_cycles',     # int: DSP timestamp (monotonic within session)
    'points',         # list[Point] – detected point cloud
    'range_profile',  # np.ndarray shape (N,) float32, or None
    'range_doppler',  # np.ndarray shape (N_range, N_doppler) float32, or None
])


# ── Default chirp configuration for 1-3 m indoor quiet standing ─────────────
# Produces ~20 Hz frame rate, ~4 cm range resolution
DEFAULT_CONFIG = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",           # 4 RX, 3 TX enabled
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 60 329 7 57.14 0 0 70 1 256 5209 0 0 30",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "chirpCfg 2 2 0 0 0 0 0 4",
    "frameCfg 0 2 64 0 50 1 0",    # 64 chirps/frame, 50 ms period → 20 Hz
    "lowPower 0 0",
    "guiMonitor -1 1 1 0 0 0 1",   # enable: point cloud, range profile, R-D map
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 4 2 3 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "lvdsStreamCfg -1 0 0 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "nearFieldCfg -1 0 0 0",
    "sensorStart",
]


class RadarReader:
    """
    Reads TLV frames from the IWR6843 data UART in a background thread.

    Usage
    -----
    reader = RadarReader(config_port='COM3', data_port='COM4')
    reader.start()
    frame = reader.queue.get(timeout=1.0)   # RadarFrame
    reader.stop()
    """

    def __init__(
        self,
        config_port: str,
        data_port: str,
        config_baud: int = 115200,
        data_baud: int = 921600,
        config_commands: list[str] | None = None,
        queue_size: int = 128,
    ):
        self.config_port = config_port
        self.data_port = data_port
        self.config_baud = config_baud
        self.data_baud = data_baud
        self.config_commands = config_commands or DEFAULT_CONFIG
        self.queue: Queue[RadarFrame] = Queue(maxsize=queue_size)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ser_data: serial.Serial | None = None
        self._ser_cfg: serial.Serial | None = None

        # Statistics
        self.frames_received = 0
        self.frames_dropped = 0
        self.parse_errors = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open serial ports, send config, start background reader thread."""
        self._configure()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name='RadarReader')
        self._thread.start()
        logger.info("RadarReader started on data port %s", self.data_port)

    def stop(self) -> None:
        """Signal the reader thread to stop and close serial ports."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._close_ports()
        logger.info(
            "RadarReader stopped. frames_received=%d dropped=%d parse_errors=%d",
            self.frames_received, self.frames_dropped, self.parse_errors,
        )

    # ── Configuration ─────────────────────────────────────────────────────────

    def _configure(self) -> None:
        self._ser_cfg = serial.Serial(self.config_port, self.config_baud, timeout=1)
        time.sleep(0.5)
        for cmd in self.config_commands:
            self._ser_cfg.write((cmd + '\n').encode())
            time.sleep(0.05)
            response = self._ser_cfg.read_all().decode(errors='ignore')
            if 'Error' in response:
                logger.warning("Config command '%s' returned: %s", cmd, response.strip())
        self._ser_data = serial.Serial(self.data_port, self.data_baud, timeout=1)
        logger.info("IWR6843 configured and sensor started")

    def _close_ports(self) -> None:
        for port in (self._ser_data, self._ser_cfg):
            if port and port.is_open:
                try:
                    port.close()
                except Exception:
                    pass

    # ── Read loop ─────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        buf = bytearray()
        while not self._stop_event.is_set():
            try:
                chunk = self._ser_data.read(self._ser_data.in_waiting or 1)
                if not chunk:
                    continue
                buf.extend(chunk)
                buf = self._drain_frames(buf)
            except serial.SerialException as exc:
                logger.error("Serial read error: %s", exc)
                break

    def _drain_frames(self, buf: bytearray) -> bytearray:
        """Extract and process all complete frames from the buffer."""
        while True:
            magic_pos = buf.find(MAGIC_WORD)
            if magic_pos == -1:
                # No magic word — keep last 7 bytes in case it straddles a read
                return bytearray(buf[-7:]) if len(buf) >= 7 else buf
            if magic_pos > 0:
                buf = buf[magic_pos:]  # discard bytes before magic word

            if len(buf) < HEADER_SIZE:
                return buf  # wait for more data

            # Parse header to get total packet length
            try:
                header = self._parse_header(bytes(buf[:HEADER_SIZE]))
            except struct.error:
                self.parse_errors += 1
                buf = buf[len(MAGIC_WORD):]  # skip this magic word
                continue

            total_len = header['packet_len']
            if len(buf) < total_len:
                return buf  # wait for rest of packet

            # Timestamp at first byte of this frame
            host_ts = time.perf_counter()

            packet = bytes(buf[:total_len])
            buf = buf[total_len:]

            try:
                frame = self._parse_frame(packet, header, host_ts)
                self._enqueue(frame)
            except Exception as exc:
                self.parse_errors += 1
                logger.debug("Frame parse error: %s", exc)

        return buf

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_header(data: bytes) -> dict:
        """
        Parse the 40-byte frame header.

        Field layout (little-endian):
          magic          8 bytes
          version        4 bytes
          packet_len     4 bytes
          platform       4 bytes
          frame_number   4 bytes
          cpu_cycles     4 bytes  (DSP timestamp)
          num_objects    4 bytes
          num_tlvs       4 bytes
          subframe_num   4 bytes
          (total = 40 bytes; the remaining 4 bytes are padding in some SDK versions)
        """
        fields = struct.unpack_from('<8sIIIIIIII', data)
        return {
            'magic':        fields[0],
            'version':      fields[1],
            'packet_len':   fields[2],
            'platform':     fields[3],
            'frame_number': fields[4],
            'cpu_cycles':   fields[5],
            'num_objects':  fields[6],
            'num_tlvs':     fields[7],
            'subframe_num': fields[8],
        }

    def _parse_frame(self, packet: bytes, header: dict, host_ts: float) -> RadarFrame:
        offset = HEADER_SIZE
        num_tlvs = header['num_tlvs']

        points = []
        range_profile = None
        range_doppler = None

        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > len(packet):
                break
            tlv_type, tlv_len = struct.unpack_from(TLV_HEADER_FMT, packet, offset)
            offset += TLV_HEADER_SIZE
            tlv_data = packet[offset: offset + tlv_len]
            offset += tlv_len

            if tlv_type == TLV_POINT_CLOUD:
                points = self._parse_points(tlv_data)
            elif tlv_type == TLV_RANGE_PROF:
                range_profile = self._parse_range_profile(tlv_data)
            elif tlv_type == TLV_RANGE_DOPPLER:
                range_doppler = self._parse_range_doppler(tlv_data)

        self.frames_received += 1
        return RadarFrame(
            host_ts=host_ts,
            frame_number=header['frame_number'],
            cpu_cycles=header['cpu_cycles'],
            points=points,
            range_profile=range_profile,
            range_doppler=range_doppler,
        )

    @staticmethod
    def _parse_points(data: bytes) -> list[Point]:
        """Each point: x(f32), y(f32), z(f32), doppler(f32) = 16 bytes."""
        n = len(data) // 16
        pts = []
        for i in range(n):
            x, y, z, d = struct.unpack_from('<4f', data, i * 16)
            pts.append(Point(x, y, z, d))
        return pts

    @staticmethod
    def _parse_range_profile(data: bytes) -> np.ndarray:
        """Range profile: N × uint16 (log-magnitude). Return as float32."""
        n = len(data) // 2
        arr = np.frombuffer(data, dtype=np.uint16, count=n).astype(np.float32)
        return arr

    @staticmethod
    def _parse_range_doppler(data: bytes, n_range: int = 256, n_doppler: int = 64) -> np.ndarray:
        """
        Range-Doppler heatmap: n_range × n_doppler uint16 values.
        Shape returned: (n_range, n_doppler) float32.
        n_range and n_doppler must match your chirp config ADC samples and chirps/frame.
        """
        expected = n_range * n_doppler * 2
        if len(data) < expected:
            # Infer dimensions from actual data length
            total = len(data) // 2
            n_range = int(np.sqrt(total * 4))   # heuristic; caller should override
            n_doppler = total // n_range
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
        arr = arr[:n_range * n_doppler].reshape(n_range, n_doppler)
        return arr

    def _enqueue(self, frame: RadarFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except Full:
            self.frames_dropped += 1
            try:
                self.queue.get_nowait()   # drop oldest
                self.queue.put_nowait(frame)
            except Exception:
                pass


# ── Utilities ─────────────────────────────────────────────────────────────────

def radar_ap_sway(frames: list[RadarFrame]) -> np.ndarray:
    """
    Extract the AP (range) sway signal from a list of RadarFrames.

    Returns the mean range (y-coordinate in IWR6843 coordinate system, which is
    range in front of the sensor) of all detected points per frame.
    Frames with no detected points are interpolated.
    """
    values = []
    for f in frames:
        if f.points:
            # y is the forward/range axis in IWR6843 coordinate system
            values.append(np.mean([p.y for p in f.points]))
        else:
            values.append(np.nan)
    arr = np.array(values, dtype=np.float64)
    # Linear interpolation over NaN gaps
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def radar_doppler_signal(frames: list[RadarFrame]) -> np.ndarray:
    """
    Aggregate Doppler signal per frame: mean absolute Doppler across all points.
    Useful as a 1-D motion signal for cross-correlation temporal calibration.
    """
    values = []
    for f in frames:
        if f.points:
            values.append(np.mean([abs(p.doppler) for p in f.points]))
        else:
            values.append(0.0)
    return np.array(values, dtype=np.float64)
