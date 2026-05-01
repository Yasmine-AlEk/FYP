import csv
import json
import math
import serial
import os
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pymavlink import mavutil
from transformers import SegformerForSemanticSegmentation

try:
    import board
    import busio
    import adafruit_vl53l1x
    VL53L1X_AVAILABLE = True
except Exception:
    VL53L1X_AVAILABLE = False


# =========================================================
# MAIN CONNECTION CONFIG
# =========================================================

# Jetson receives MAVLink from WSL MAVProxy on this port.
MAVLINK_URL = "udpin:0.0.0.0:14550"


# =========================================================
# DEFAULT LOCAL TEST SCAN CONFIG
# Used when no GUI JSON file is given.
# =========================================================

LOCAL_SCAN_SPEED_MPS = 0.5
LOCAL_START_ALT_M = 2.0
LOCAL_MAX_ALT_M = 5.0
LOCAL_LANE_SPACING_M = 1.0
LOCAL_STANDOFF_M = 2.0
LOCAL_OVERSHOOT_M = 0.5

# Wall in local NE coordinates.
LOCAL_WALL_A_NE = (0.0, 0.0)
LOCAL_WALL_B_NE = (0.0, 6.0)

YAW_HOLD_DEG = 0.0


# =========================================================
# PID CONFIG
# =========================================================

# Row-start correction using two ultrasonic sensors on Arduino.
# The drone uses position commands for the scan rows.
# At the beginning of each row, it briefly switches to BODY_NED velocity commands
# to correct wall standoff distance and yaw alignment.

ULTRASONIC_PID_ENABLED = True

# Arduino ultrasonic serial port.
SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 9600
SERIAL_TIMEOUT_S = 0.02
SENSOR_LOST_TIMEOUT_S = 0.50

# PID correction timing.
ROW_START_CORRECTION_HZ = 20.0
ROW_START_CORRECTION_TIME_S = 5.0

# Stop the row-start PID early if both errors are stable.
ROW_START_STOP_WHEN_STABLE = True
ROW_START_STABLE_HOLD_TIME_S = 1.0

# Deadbands.
# Distance: ignore errors within ±10 cm.
# Yaw: correct if the two ultrasonic readings differ by more than 3 cm.
# Height: ignore altitude errors within ±10 cm.
DIST_DEADBAND_M = 0.10
YAW_DIFF_DEADBAND_M = 0.03
HEIGHT_DEADBAND_M = 0.10

# Sensor averaging.
SENSOR_AVG_WINDOW = 3
SENSOR_MIN_VALID_SAMPLES = 2

# Simple PID logging.
PID_PRINT_EVERY_N = 1
PID_LOG_CSV = "row_start_pid_log.csv"

# Distance PID gains.
KP_WALL_DIST, KI_WALL_DIST, KD_WALL_DIST = 0.80, 0.00, 0.05

# Yaw PID gains.
KP_YAW, KI_YAW, KD_YAW = 1.20, 0.00, 0.08

# Height PID gains.
# This corrects altitude at the beginning of each row using AirSim/MAVLink altitude.
KP_HEIGHT, KI_HEIGHT, KD_HEIGHT = 0.60, 0.00, 0.05

MAX_VX_CORR = 0.35
MAX_YAW_RATE_CORR = 0.50
MAX_VZ_CORR = 0.30

# Flip these if correction direction is wrong.
DIST_SIGN = 1.0
YAW_SIGN = -1.0
HEIGHT_SIGN = 1.0

# Opened in main().
ULTRASONIC_SER = None


# =========================================================
# VISION CONFIG
# =========================================================

CHECKPOINT_PATH = "segformer_corrosion.pth"
VISION_OUTPUTS_DIR = Path("realtime_outputs")

IMAGE_SIZE = 512
TARGET_FPS = 30

CORROSION_THRESHOLD_PERCENT = 2.0
COOLDOWN_SECONDS = 4.0

# Set to 0.0 to run vision at every scan-setpoint command.
# If the scan becomes too slow, try 0.3 or 0.5.
VISION_MIN_INTERVAL_S = 0.0

GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=1280, height=720, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink drop=1"
)

CONFIDENCE_THRESHOLD = 0.7
USE_FP16 = True
MIN_REGION_AREA = 500
SCENE_CHANGE_THRESHOLD = 15.0
TEMPORAL_BUFFER_SIZE = 5

BLUR_THRESHOLD = 50.0
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 230

IMX219_HFOV_DEG = 62.2
IMX219_VFOV_DEG = 51.1

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================================================
# LOGGING CONFIG
# =========================================================

DEFAULT_DETECTIONS_JSON = "corrosion_detections.json"
DETECTION_DEDUP_THRESHOLD_M = 0.5


# =========================================================
# TYPE ALIASES
# =========================================================

Vec2 = Tuple[float, float]             # north, east
Vec3 = Tuple[float, float, float]      # north, east, down


# =========================================================
# GRACEFUL SHUTDOWN
# =========================================================

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Finishing current step...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =========================================================
# MAVLINK HELPERS
# =========================================================

_T0_MS = None


def _now_ms_u32():
    global _T0_MS
    t = time.monotonic()
    if _T0_MS is None:
        _T0_MS = t
    return int((t - _T0_MS) * 1000) & 0xFFFFFFFF


def connect_sitl(url: str = MAVLINK_URL):
    print(f"[MAVLink] Connecting using: {url}")
    master = mavutil.mavlink_connection(url)

    print("[MAVLink] Waiting for heartbeat...")
    master.wait_heartbeat()

    print(
        f"[MAVLink] Connected: system={master.target_system}, "
        f"component={master.target_component}"
    )
    return master


def set_mode(master, mode: str):
    mode_map = master.mode_mapping()

    if mode not in mode_map:
        raise RuntimeError(
            f"Mode {mode} not available. Available modes: {list(mode_map.keys())}"
        )

    mode_id = mode_map[mode]

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )

    print(f"[MAVLink] Mode command sent: {mode}")
    time.sleep(1.0)


def arm(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0,
    )

    print("[MAVLink] Arming...")
    time.sleep(2.0)


def takeoff(master, target_alt_m: float):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        target_alt_m,
    )

    print(f"[MAVLink] Taking off to {target_alt_m:.2f} m...")
    time.sleep(1.0)


def rtl(master):
    try:
        set_mode(master, "RTL")
        print("[MAVLink] RTL command sent.")
    except Exception as e:
        print(f"[MAVLink] Could not switch to RTL: {e}")


def goto_position_target_local_ned(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)

    # Enable position + yaw.
    # Ignore velocity, acceleration, and yaw rate.
    m = mavutil.mavlink
    type_mask = (
        m.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        m.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        m.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        m.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        _now_ms_u32(),
        master.target_system,
        master.target_component,
        m.MAV_FRAME_LOCAL_NED,
        type_mask,
        float(x), float(y), float(z),
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        float(yaw_rad),
        0.0,
    )


def send_body_velocity_yawrate(master, vx: float, vy: float, vz: float, yaw_rate: float):
    """
    BODY_NED:
    +x forward
    +y right
    +z down
    yaw_rate in rad/s

    Used only during short row-start PID correction.
    """
    m = mavutil.mavlink

    type_mask = (
        m.POSITION_TARGET_TYPEMASK_X_IGNORE |
        m.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        m.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        m.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        m.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        _now_ms_u32(),
        master.target_system,
        master.target_component,
        m.MAV_FRAME_BODY_NED,
        type_mask,
        0.0, 0.0, 0.0,
        float(vx), float(vy), float(vz),
        0.0, 0.0, 0.0,
        0.0,
        float(yaw_rate),
    )


def get_global_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(
        type="GLOBAL_POSITION_INT",
        blocking=True,
        timeout=timeout_s,
    )

    if msg is None:
        raise TimeoutError("No GLOBAL_POSITION_INT received.")

    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    rel_alt = msg.relative_alt / 1000.0

    return lat, lon, rel_alt


def get_local_position(master, timeout_s: float = 2.0) -> Vec3:
    msg = master.recv_match(
        type="LOCAL_POSITION_NED",
        blocking=True,
        timeout=timeout_s,
    )

    if msg is None:
        raise TimeoutError("No LOCAL_POSITION_NED received.")

    return float(msg.x), float(msg.y), float(msg.z)


def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# VECTOR HELPERS
# =========================================================

def v2_add(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def v2_sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def v2_scale(a: Vec2, s: float) -> Vec2:
    return a[0] * s, a[1] * s


def v2_norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def v2_unit(a: Vec2) -> Vec2:
    n = v2_norm(a)
    if n < 1e-6:
        raise ValueError("Zero-length vector.")
    return a[0] / n, a[1] / n


def v2_midpoint(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def v3_dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


# =========================================================
# GPS / LOCAL HELPERS
# =========================================================

def latlon_to_ne_m(lat0: float, lon0: float, lat: float, lon: float) -> Vec2:
    R = 6378137.0

    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)

    north = dlat * R
    east = dlon * R * math.cos(math.radians(lat0))

    return north, east


def current_gps_to_target_local_ned(
    current_lat: float,
    current_lon: float,
    current_local_ned: Vec3,
    target_lat: float,
    target_lon: float,
    target_alt_m: float,
) -> Vec3:
    dn, de = latlon_to_ne_m(
        current_lat,
        current_lon,
        target_lat,
        target_lon,
    )

    cx, cy, _cz = current_local_ned

    return cx + dn, cy + de, -target_alt_m


# =========================================================
# PID HELPER
# =========================================================

class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        out_limit: Optional[float] = None,
        i_limit: Optional[float] = None,
        deadband: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.deadband = deadband
        self.ei = 0.0
        self.prev = None
        self.tprev = None

    def reset(self):
        self.ei = 0.0
        self.prev = None
        self.tprev = None

    def __call__(self, e: float) -> float:
        now = time.time()

        if self.deadband and abs(e) < self.deadband:
            e = 0.0

        if self.tprev is None:
            dt = 0.0
        else:
            dt = max(1e-3, now - self.tprev)

        de = 0.0 if self.prev is None else e - self.prev

        self.ei += e * dt

        if self.i_limit is not None:
            self.ei = max(-self.i_limit, min(self.i_limit, self.ei))

        u = self.kp * e + self.ki * self.ei

        if dt > 0.0:
            u += self.kd * (de / dt)

        if self.out_limit is not None:
            u = max(-self.out_limit, min(self.out_limit, u))

        self.prev = e
        self.tprev = now

        return u


# =========================================================
# ULTRASONIC SERIAL HELPERS FOR ROW-START PID
# =========================================================

def parse_ultrasonic_value(value: str) -> Optional[float]:
    value = value.strip()

    if value in ["OUT_OF_RANGE", "N/A", ""]:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def open_ultrasonic_serial():
    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=SERIAL_TIMEOUT_S,
        write_timeout=SERIAL_TIMEOUT_S,
    )

    time.sleep(2.0)
    ser.reset_input_buffer()

    print(f"[PID] Connected to Arduino ultrasonic sensors on {SERIAL_PORT} at {BAUD_RATE} baud.")
    print(f"[PID] Serial timeout: {SERIAL_TIMEOUT_S}s")

    return ser


def read_ultrasonic_sensors_cm(
    ser,
    max_attempts: int = 5,
) -> Optional[Tuple[float, float, float]]:
    """
    Expected Arduino line:
        sensor1_cm,sensor2_cm,error_cm

    Example:
        195.0,203.0,-8.0
    """

    latest_valid = None

    for _ in range(max_attempts):
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        if line.startswith("sensor_1_cm"):
            continue

        parts = line.split(",")

        if len(parts) != 3:
            continue

        sensor1 = parse_ultrasonic_value(parts[0])
        sensor2 = parse_ultrasonic_value(parts[1])
        error = parse_ultrasonic_value(parts[2])

        if sensor1 is None or sensor2 is None:
            continue

        if error is None:
            error = sensor1 - sensor2

        latest_valid = (sensor1, sensor2, error)

    return latest_valid


class UltrasonicRollingAverage:
    def __init__(
        self,
        window_size: int = SENSOR_AVG_WINDOW,
        min_samples: int = SENSOR_MIN_VALID_SAMPLES,
    ):
        self.window_size = window_size
        self.min_samples = min_samples
        self.s1_buffer = deque(maxlen=window_size)
        self.s2_buffer = deque(maxlen=window_size)

    def reset(self):
        self.s1_buffer.clear()
        self.s2_buffer.clear()

    def update(self, sensor1_cm: float, sensor2_cm: float):
        self.s1_buffer.append(sensor1_cm)
        self.s2_buffer.append(sensor2_cm)

        if len(self.s1_buffer) < self.min_samples:
            return None

        avg_s1_cm = sum(self.s1_buffer) / len(self.s1_buffer)
        avg_s2_cm = sum(self.s2_buffer) / len(self.s2_buffer)

        # Positive means sensor 1 is farther than sensor 2.
        avg_error_cm = avg_s1_cm - avg_s2_cm

        return avg_s1_cm, avg_s2_cm, avg_error_cm


def apply_deadband(error: float, deadband: float) -> float:
    if abs(error) <= deadband:
        return 0.0

    return error


def stop_motion(master):
    send_body_velocity_yawrate(
        master,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        yaw_rate=0.0,
    )


def build_row_start_pid_controllers() -> Dict[str, PID]:
    return {
        "wall_distance": PID(
            KP_WALL_DIST,
            KI_WALL_DIST,
            KD_WALL_DIST,
            out_limit=MAX_VX_CORR,
        ),
        "yaw_rate": PID(
            KP_YAW,
            KI_YAW,
            KD_YAW,
            out_limit=MAX_YAW_RATE_CORR,
        ),
        "height": PID(
            KP_HEIGHT,
            KI_HEIGHT,
            KD_HEIGHT,
            out_limit=MAX_VZ_CORR,
        ),
    }


def wall_distance_pid(
    master,
    pid: PID,
    desired_standoff_m: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_standoff_m, row_start_local_ned, plan)
    return 0.0


def yawrate_pid(
    master,
    pid: PID,
    desired_yaw_deg: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_yaw_deg, row_start_local_ned, plan)
    return 0.0


def height_pid(
    master,
    pid: PID,
    desired_alt_m: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_alt_m, row_start_local_ned, plan)
    return 0.0


def run_row_start_correction(
    master,
    plan: Dict,
    row_start: Vec3,
    yaw_hold_deg: float,
    duration_s: float = ROW_START_CORRECTION_TIME_S,
    hz: float = ROW_START_CORRECTION_HZ,
):
    """
    Row-start correction using:
    1. ultrasonic average distance for standoff PID
    2. ultrasonic difference for yaw PID
    3. AirSim/MAVLink local altitude for height PID

    This runs before each scan row. The row itself still uses LOCAL_NED position commands.
    """

    _ = yaw_hold_deg

    if not ULTRASONIC_PID_ENABLED:
        print("[PID] Ultrasonic row-start PID disabled. Skipping correction.")
        return

    if ULTRASONIC_SER is None:
        print("[PID] No ultrasonic serial connection. Skipping row-start correction.")
        return

    print("[PID] Running ultrasonic + height row-start correction...")

    controllers = build_row_start_pid_controllers()

    sensor_filter = UltrasonicRollingAverage(
        window_size=SENSOR_AVG_WINDOW,
        min_samples=SENSOR_MIN_VALID_SAMPLES,
    )

    desired_standoff_m = float(plan["params"]["standoff_m"])

    # LOCAL_NED z is down, so altitude is -z.
    desired_alt_m = -float(row_start[2])

    dt = 1.0 / max(hz, 1e-6)
    steps = max(1, int(duration_s * hz))

    stable_start_time = None
    last_sensor_time = time.time()

    csv_exists = Path(PID_LOG_CSV).exists()

    with open(PID_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not csv_exists:
            writer.writerow([
                "timestamp_s",
                "row_start_x",
                "row_start_y",
                "row_start_z",
                "desired_alt_m",
                "current_alt_m",
                "height_error_m",
                "vz_cmd_mps",
                "avg_distance_m",
                "distance_error_m",
                "vx_cmd_mps",
                "yaw_error_m",
                "yaw_rate_cmd_rad_s",
                "stable",
            ])

        for k in range(steps):
            if _shutdown_requested:
                break

            # -------------------------------------------------
            # 1. Read ultrasonic sensors for distance + yaw
            # -------------------------------------------------
            reading = read_ultrasonic_sensors_cm(ULTRASONIC_SER)

            if reading is None:
                if time.time() - last_sensor_time > SENSOR_LOST_TIMEOUT_S:
                    print("[PID] Sensor timeout. Sending zero correction.")
                    stop_motion(master)

                time.sleep(0.005)
                continue

            last_sensor_time = time.time()

            raw_sensor1_cm, raw_sensor2_cm, _raw_error_cm = reading

            filtered = sensor_filter.update(
                raw_sensor1_cm,
                raw_sensor2_cm,
            )

            if filtered is None:
                print(
                    f"[PID] Averaging sensors... "
                    f"s1={raw_sensor1_cm:.1f}cm | "
                    f"s2={raw_sensor2_cm:.1f}cm"
                )
                stop_motion(master)
                time.sleep(dt)
                continue

            sensor1_cm, sensor2_cm, yaw_error_cm = filtered

            sensor1_m = sensor1_cm / 100.0
            sensor2_m = sensor2_cm / 100.0

            avg_distance_m = (sensor1_m + sensor2_m) / 2.0

            # Positive means drone is too far from wall.
            # Negative means drone is too close to wall.
            distance_error_raw_m = avg_distance_m - desired_standoff_m

            # Positive/negative depends on physical ultrasonic placement.
            yaw_error_raw_m = yaw_error_cm / 100.0

            distance_error_control_m = apply_deadband(
                distance_error_raw_m,
                DIST_DEADBAND_M,
            )

            yaw_error_control_m = apply_deadband(
                yaw_error_raw_m,
                YAW_DIFF_DEADBAND_M,
            )

            # -------------------------------------------------
            # 2. Read AirSim/MAVLink altitude for height PID
            # -------------------------------------------------
            try:
                _x, _y, current_z_down = get_local_position(master, timeout_s=0.3)
                current_alt_m = -float(current_z_down)
            except Exception:
                current_alt_m = None

            if current_alt_m is None:
                height_error_raw_m = 0.0
                height_error_control_m = 0.0
                vz = 0.0
                height_stable = False
                print("[PID] No altitude reading. Height correction set to zero.")
            else:
                # Positive means drone is too high.
                # Negative means drone is too low.
                #
                # BODY_NED vz:
                # +vz = down
                # -vz = up
                #
                # So this sign naturally works:
                # too high  -> positive error -> positive vz -> move down
                # too low   -> negative error -> negative vz -> move up
                height_error_raw_m = current_alt_m - desired_alt_m

                height_error_control_m = apply_deadband(
                    height_error_raw_m,
                    HEIGHT_DEADBAND_M,
                )

                vz = HEIGHT_SIGN * controllers["height"](height_error_control_m)

                height_stable = abs(height_error_raw_m) <= HEIGHT_DEADBAND_M

            # -------------------------------------------------
            # 3. Compute distance + yaw commands
            # -------------------------------------------------
            vx = DIST_SIGN * controllers["wall_distance"](distance_error_control_m)
            yaw_rate = YAW_SIGN * controllers["yaw_rate"](yaw_error_control_m)

            distance_stable = abs(distance_error_raw_m) <= DIST_DEADBAND_M
            yaw_stable = abs(yaw_error_raw_m) <= YAW_DIFF_DEADBAND_M

            stable = distance_stable and yaw_stable and height_stable

            if stable:
                if stable_start_time is None:
                    stable_start_time = time.time()
            else:
                stable_start_time = None

            # -------------------------------------------------
            # 4. Send combined BODY_NED command
            # -------------------------------------------------
            send_body_velocity_yawrate(
                master,
                vx=vx,
                vy=0.0,
                vz=vz,
                yaw_rate=yaw_rate,
            )

            writer.writerow([
                round(time.time(), 3),
                round(row_start[0], 4),
                round(row_start[1], 4),
                round(row_start[2], 4),
                round(desired_alt_m, 4),
                round(current_alt_m, 4) if current_alt_m is not None else None,
                round(height_error_raw_m, 4),
                round(vz, 4),
                round(avg_distance_m, 4),
                round(distance_error_raw_m, 4),
                round(vx, 4),
                round(yaw_error_raw_m, 4),
                round(yaw_rate, 4),
                int(stable),
            ])

            f.flush()

            if k % PID_PRINT_EVERY_N == 0:
                alt_text = (
                    f"{current_alt_m:.2f}m"
                    if current_alt_m is not None
                    else "N/A"
                )

                print(
                    f"[PID] "
                    f"dist={avg_distance_m:.2f}m | "
                    f"dist_err={distance_error_raw_m:+.2f}m | "
                    f"vx={vx:+.2f}m/s || "
                    f"yaw_err={yaw_error_raw_m:+.2f}m | "
                    f"yaw_rate={yaw_rate:+.2f}rad/s || "
                    f"alt={alt_text} | "
                    f"alt_ref={desired_alt_m:.2f}m | "
                    f"alt_err={height_error_raw_m:+.2f}m | "
                    f"vz={vz:+.2f}m/s | "
                    f"stable={stable}"
                )

            if ROW_START_STOP_WHEN_STABLE and stable_start_time is not None:
                stable_duration = time.time() - stable_start_time

                if stable_duration >= ROW_START_STABLE_HOLD_TIME_S:
                    print(
                        f"[PID] Stable for {ROW_START_STABLE_HOLD_TIME_S:.1f}s. "
                        f"Ending row-start correction."
                    )
                    break

            time.sleep(dt)

    stop_motion(master)
    print("[PID] Row-start correction finished. Body velocity command set to zero.")


# =========================================================
# VISION HELPERS
# =========================================================

def init_vl53l1x():
    if not VL53L1X_AVAILABLE:
        print("[VL53L1X] Library not installed. Running without distance sensor.")
        return None

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_vl53l1x.VL53L1X(i2c)
        sensor.distance_mode = 1
        sensor.timing_budget = 50
        sensor.start_ranging()
        print("[VL53L1X] Sensor initialised successfully.")
        return sensor

    except Exception as e:
        print(f"[VL53L1X] Sensor init failed. Running without distance sensor. ({e})")
        return None


def read_distance_mm(sensor):
    if sensor is None:
        return None

    try:
        if sensor.data_ready:
            d = sensor.distance
            sensor.clear_interrupt()

            if d is None or d <= 0:
                return None

            return d * 1000.0

    except Exception:
        pass

    return None


def corrosion_area_from_mask(mask01, distance_mm, hfov_deg, vfov_deg):
    h, w = mask01.shape[:2]

    Z = distance_mm / 1000.0

    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    scene_w_m = 2.0 * Z * math.tan(hfov / 2.0)
    scene_h_m = 2.0 * Z * math.tan(vfov / 2.0)

    m_per_px_x = scene_w_m / w
    m_per_px_y = scene_h_m / h

    pixel_count = int(mask01.sum())

    return pixel_count * m_per_px_x * m_per_px_y


def check_frame_quality(
    frame,
    blur_threshold=None,
    brightness_min=None,
    brightness_max=None,
):
    if blur_threshold is None:
        blur_threshold = BLUR_THRESHOLD
    if brightness_min is None:
        brightness_min = BRIGHTNESS_MIN
    if brightness_max is None:
        brightness_max = BRIGHTNESS_MAX

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(gray.mean())

    issues = []

    if laplacian_var < blur_threshold:
        issues.append("BLURRY")

    if mean_brightness < brightness_min:
        issues.append("TOO_DARK")

    if mean_brightness > brightness_max:
        issues.append("OVEREXPOSED")

    return len(issues) == 0, issues


def compute_frame_similarity(frame_a, frame_b, thumbnail_size=(64, 64)):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    small_a = cv2.resize(gray_a, thumbnail_size).astype(np.float32)
    small_b = cv2.resize(gray_b, thumbnail_size).astype(np.float32)

    return float(np.mean(np.abs(small_a - small_b)))


def load_model(device):
    use_fp16 = USE_FP16 and device.type == "cuda"

    print("[Vision] Loading SegFormer model...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )

    model.to(device)
    model.eval()

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print(f"[Vision] Loading weights from: {CHECKPOINT_PATH}")

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    if use_fp16:
        model.half()
        print("[Vision] FP16 inference enabled.")

    print("[Vision] Model loaded successfully.")
    return model


def preprocess_frame(frame, device):
    use_fp16 = USE_FP16 and device.type == "cuda"

    h, w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    norm = resized.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))

    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)

    if use_fp16:
        tensor = tensor.half()

    return tensor, (h, w)


def run_inference(model, tensor, original_size):
    h, w = original_size

    inference_start = time.time()

    with torch.no_grad():
        forward_start = time.time()
        logits = model(pixel_values=tensor).logits
        forward_time = time.time() - forward_start

        interp_start = time.time()
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        interp_time = time.time() - interp_start

        classify_start = time.time()

        if CONFIDENCE_THRESHOLD > 0:
            probs = F.softmax(logits, dim=1)
            confidence_vals, raw_preds = probs.max(dim=1)

            preds = raw_preds[0].cpu().numpy().astype(np.uint8)
            conf_map = confidence_vals[0].cpu().numpy()

            preds[conf_map < CONFIDENCE_THRESHOLD] = 0

        else:
            preds = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        classify_time = time.time() - classify_start

    total_inference_time = time.time() - inference_start

    timing_info = {
        "forward_pass_ms": forward_time * 1000.0,
        "interpolation_ms": interp_time * 1000.0,
        "classify_ms": classify_time * 1000.0,
        "total_inference_ms": total_inference_time * 1000.0,
    }

    return preds, timing_info


def postprocess_predictions(preds):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = np.zeros_like(preds)

    for cls in [1, 2]:
        cls_mask = (preds == cls).astype(np.uint8)

        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, kernel)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)

        if MIN_REGION_AREA > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < MIN_REGION_AREA:
                    cls_mask[labels == i] = 0

        cleaned[cls_mask == 1] = cls

    return cleaned


def calculate_metrics(preds, frame_shape):
    h, w = frame_shape[:2]
    total_pixels = h * w

    class0_pixels = int(np.sum(preds == 0))
    class1_pixels = int(np.sum(preds == 1))
    class2_pixels = int(np.sum(preds == 2))

    class1_percent = class1_pixels / total_pixels * 100.0
    class2_percent = class2_pixels / total_pixels * 100.0
    corroded_total_percent = class1_percent + class2_percent

    return {
        "total_pixels": int(total_pixels),
        "pixel_counts": {
            "class0": class0_pixels,
            "class1": class1_pixels,
            "class2": class2_pixels,
        },
        "area_percent": {
            "class1": round(class1_percent, 2),
            "class2": round(class2_percent, 2),
            "corroded_total": round(corroded_total_percent, 2),
        },
    }


def analyze_regions(preds, distance_mm=None):
    binary = ((preds == 1) | (preds == 2)).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    h, w = preds.shape[:2]

    m_per_px_x = None
    m_per_px_y = None

    if distance_mm is not None and distance_mm > 0:
        Z = distance_mm / 1000.0

        scene_w = 2.0 * Z * math.tan(math.radians(IMX219_HFOV_DEG) / 2.0)
        scene_h = 2.0 * Z * math.tan(math.radians(IMX219_VFOV_DEG) / 2.0)

        m_per_px_x = scene_w / w
        m_per_px_y = scene_h / h

    regions = []

    for i in range(1, num_labels):
        x, y, rw, rh, area_px = stats[i]

        region_pixels = preds[labels == i]

        severe_count = int(np.sum(region_pixels == 2))
        fair_count = int(np.sum(region_pixels == 1))

        region = {
            "id": int(i),
            "bbox": {
                "x": int(x),
                "y": int(y),
                "w": int(rw),
                "h": int(rh),
            },
            "area_pixels": int(area_px),
            "centroid": {
                "x": round(float(centroids[i][0]), 1),
                "y": round(float(centroids[i][1]), 1),
            },
            "severity_breakdown": {
                "fair_pixels": fair_count,
                "severe_pixels": severe_count,
            },
            "dominant_severity": "severe" if severe_count > fair_count else "fair",
        }

        if m_per_px_x is not None:
            area_m2 = area_px * m_per_px_x * m_per_px_y
            region["area_cm2"] = round(area_m2 * 10000.0, 4)

        regions.append(region)

    return sorted(regions, key=lambda r: r["area_pixels"], reverse=True)


def grade_severity(metrics):
    total = metrics["area_percent"]["corroded_total"]
    severe = metrics["area_percent"]["class2"]

    if severe > 5.0 or total > 15.0:
        return "CRITICAL", "Immediate maintenance required"

    if severe > 2.0 or total > 8.0:
        return "HIGH", "Schedule maintenance soon"

    if total > 3.0:
        return "MODERATE", "Monitor closely"

    if total > 0.3:
        return "LOW", "Minor corrosion detected"

    return "CLEAN", "No significant corrosion"


class TemporalSmoother:
    def __init__(self, buffer_size=TEMPORAL_BUFFER_SIZE, num_classes=3):
        self._buffer = deque(maxlen=buffer_size)
        self._num_classes = num_classes

    def smooth(self, preds):
        if self._buffer and self._buffer[-1].shape != preds.shape:
            self._buffer.clear()

        self._buffer.append(preds.copy())

        if len(self._buffer) < 3:
            return preds

        stacked = np.stack(list(self._buffer), axis=0)

        h, w = preds.shape[:2]
        counts = np.zeros((self._num_classes, h, w), dtype=np.int32)

        for c in range(self._num_classes):
            counts[c] = np.sum(stacked == c, axis=0)

        return np.argmax(counts, axis=0).astype(np.uint8)

    def reset(self):
        self._buffer.clear()


def create_overlay(frame, preds, alpha=0.3, regions=None):
    h, w = frame.shape[:2]

    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[preds == 1] = (0, 255, 0)
    color_mask[preds == 2] = (0, 0, 255)

    overlay = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

    if regions:
        for region in regions:
            bbox = region["bbox"]

            color = (
                (0, 0, 255)
                if region["dominant_severity"] == "severe"
                else (0, 255, 0)
            )

            x = bbox["x"]
            y = bbox["y"]
            rw = bbox["w"]
            rh = bbox["h"]

            cv2.rectangle(
                overlay,
                (x, y),
                (x + rw, y + rh),
                color,
                2,
            )

            label = f"R{region['id']}: {region['dominant_severity']}"

            if "area_cm2" in region:
                label += f" ({region['area_cm2']:.1f}cm2)"

            cv2.putText(
                overlay,
                label,
                (x, max(y - 5, 15)),
                FONT,
                0.4,
                color,
                1,
            )

    return overlay


def get_unique_filename(base_path, extension):
    path = Path(f"{base_path}{extension}")

    if not path.exists():
        return str(path)

    counter = 1

    while True:
        path = Path(f"{base_path}_{counter}{extension}")

        if not path.exists():
            return str(path)

        counter += 1


def save_vision_snapshot(frame, overlay, metrics, regions, grade_info):
    VISION_OUTPUTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = VISION_OUTPUTS_DIR / timestamp

    overlay_path = get_unique_filename(f"{base_name}", ".jpg")
    json_path = get_unique_filename(base_name, ".json")

    cv2.imwrite(overlay_path, overlay)

    h, w = frame.shape[:2]

    json_data = {
        "timestamp": timestamp,
        "frame_size": {
            "width": w,
            "height": h,
        },
        "pixel_counts": metrics["pixel_counts"],
        "area_percent": metrics["area_percent"],
        "severity": {
            "grade": grade_info[0],
            "description": grade_info[1],
        },
        "regions": {
            "count": len(regions),
            "details": regions,
        },
    }

    if metrics.get("distance_mm") is not None:
        json_data["distance_mm"] = metrics["distance_mm"]

    if metrics.get("area_cm2") is not None:
        json_data["total_corrosion_area_cm2"] = metrics["area_cm2"]

    if "timing" in metrics:
        json_data["performance"] = {
            "timing": metrics["timing"],
            "gpu": metrics.get("gpu", {}),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    return {
        "image": Path(overlay_path).name,
        "json": Path(json_path).name,
    }


class VisionCorrosionDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = USE_FP16 and self.device.type == "cuda"

        self.model = None
        self.cap = None
        self.tof_sensor = None
        self.smoother = TemporalSmoother()

        self.last_detection_time = 0.0
        self.last_saved_frame = None
        self.last_process_time = 0.0

    def open(self):
        print(f"[Vision] Using device: {self.device}")
        print(f"[Vision] FP16: {'enabled' if self.use_fp16 else 'disabled'}")
        print(f"[Vision] Corrosion threshold: {CORROSION_THRESHOLD_PERCENT}%")
        print(f"[Vision] Output directory: {VISION_OUTPUTS_DIR.absolute()}")

        self.model = load_model(self.device)

        if self.device.type == "cuda":
            print("[Vision] Running warm-up inference...")

            dummy = torch.zeros(
                1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                device=self.device,
            )

            if self.use_fp16:
                dummy = dummy.half()

            with torch.no_grad():
                self.model(pixel_values=dummy)

            print("[Vision] Warm-up complete.")

        self.tof_sensor = init_vl53l1x()

        print("[Vision] Opening CSI camera...")
        print(f"[Vision] Pipeline: {GSTREAMER_PIPELINE}")

        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open CSI camera via GStreamer.\n"
                "Make sure no other script is using the camera."
            )

        print("[Vision] Camera opened successfully.")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Vision] Camera released.")

    def process_scan_step(self, master, commanded_target_local_ned: Vec3) -> Dict:
        now = time.time()

        if VISION_MIN_INTERVAL_S > 0.0:
            if now - self.last_process_time < VISION_MIN_INTERVAL_S:
                return {
                    "detected": False,
                    "skipped": True,
                    "reason": "rate_limit",
                }

        self.last_process_time = now

        if self.cap is None:
            return {
                "detected": False,
                "error": "camera_not_open",
            }

        ret, frame = self.cap.read()

        if not ret:
            print("[Vision] Failed to read camera frame.")
            return {
                "detected": False,
                "error": "frame_read_failed",
            }

        quality_ok, quality_issues = check_frame_quality(frame)

        if not quality_ok:
            print(f"[Vision] Skipped frame. Quality issues: {quality_issues}")
            return {
                "detected": False,
                "quality_ok": False,
                "quality_issues": quality_issues,
            }

        preprocess_start = time.time()
        tensor, original_size = preprocess_frame(frame, self.device)
        preprocess_ms = (time.time() - preprocess_start) * 1000.0

        raw_preds, inference_timing = run_inference(
            self.model,
            tensor,
            original_size,
        )

        postprocess_start = time.time()
        cleaned_preds = postprocess_predictions(raw_preds)
        preds = self.smoother.smooth(cleaned_preds)

        metrics = calculate_metrics(preds, frame.shape)
        postprocess_ms = (time.time() - postprocess_start) * 1000.0

        distance_mm = read_distance_mm(self.tof_sensor)

        area_cm2 = None

        if distance_mm is not None:
            binary_mask = ((preds == 1) | (preds == 2)).astype(np.uint8)

            area_m2 = corrosion_area_from_mask(
                binary_mask,
                distance_mm,
                IMX219_HFOV_DEG,
                IMX219_VFOV_DEG,
            )

            area_cm2 = area_m2 * 10000.0

        metrics["distance_mm"] = (
            round(distance_mm, 1)
            if distance_mm is not None
            else None
        )

        metrics["area_cm2"] = (
            round(area_cm2, 4)
            if area_cm2 is not None
            else None
        )

        regions = analyze_regions(preds, distance_mm=distance_mm)

        grade, grade_desc = grade_severity(metrics)

        total_frame_ms = (
            preprocess_ms +
            inference_timing["total_inference_ms"] +
            postprocess_ms
        )

        if self.device.type == "cuda":
            gpu_mem_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            gpu_mem_mb = 0.0

        metrics["timing"] = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_timing["total_inference_ms"], 2),
            "forward_pass_ms": round(inference_timing["forward_pass_ms"], 2),
            "interpolation_ms": round(inference_timing["interpolation_ms"], 2),
            "classify_ms": round(inference_timing["classify_ms"], 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_frame_ms": round(total_frame_ms, 2),
            "fps_capacity": (
                round(1000.0 / total_frame_ms, 2)
                if total_frame_ms > 0
                else 0.0
            ),
        }

        metrics["gpu"] = {
            "memory_allocated_mb": round(gpu_mem_mb, 2),
            "device": str(self.device),
        }

        corroded_pct = metrics["area_percent"]["corroded_total"]

        print(
            f"[Vision] {grade} | "
            f"corrosion={corroded_pct:.2f}% | "
            f"regions={len(regions)} | "
            f"frame={total_frame_ms:.1f}ms"
        )

        if corroded_pct < CORROSION_THRESHOLD_PERCENT:
            return {
                "detected": False,
                "corroded_area": None,
                "extra": {
                    "corrosion_percent": corroded_pct,
                    "severity_grade": grade,
                    "commanded_target_ned": commanded_target_local_ned,
                },
            }

        cooldown_elapsed = (now - self.last_detection_time) >= COOLDOWN_SECONDS

        if not cooldown_elapsed:
            return {
                "detected": False,
                "skipped": True,
                "reason": "cooldown",
                "extra": {
                    "corrosion_percent": corroded_pct,
                    "severity_grade": grade,
                },
            }

        scene_changed = True

        if self.last_saved_frame is not None:
            diff = compute_frame_similarity(frame, self.last_saved_frame)
            scene_changed = diff >= SCENE_CHANGE_THRESHOLD

        if not scene_changed:
            return {
                "detected": False,
                "skipped": True,
                "reason": "scene_not_changed",
                "extra": {
                    "corrosion_percent": corroded_pct,
                    "severity_grade": grade,
                },
            }

        overlay = create_overlay(
            frame,
            preds,
            regions=regions,
        )

        saved_files = save_vision_snapshot(
            frame,
            overlay,
            metrics,
            regions,
            grade_info=(grade, grade_desc),
        )

        self.last_detection_time = now
        self.last_saved_frame = frame.copy()

        return {
            "detected": True,
            "corroded_area": metrics.get("area_cm2"),
            "extra": {
                "corrosion_percent": corroded_pct,
                "severity_grade": grade,
                "severity_description": grade_desc,
                "num_regions": len(regions),
                "regions": regions,
                "distance_mm": metrics.get("distance_mm"),
                "area_cm2": metrics.get("area_cm2"),
                "vision_files": saved_files,
                "commanded_target_ned": commanded_target_local_ned,
                "vision_metrics": metrics,
            },
        }


# =========================================================
# WAIT / HOLD HELPERS
# =========================================================

def wait_until_altitude(
    master,
    target_alt_m: float,
    tol_m: float = 0.25,
    timeout_s: float = 45.0,
):
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        _x, _y, z = get_local_position(master, timeout_s=1.0)
        current_alt_m = -z

        if abs(current_alt_m - target_alt_m) <= tol_m:
            return

        time.sleep(0.1)

    raise TimeoutError(f"Did not reach altitude {target_alt_m:.2f} m in time.")


def hold_local_target_until_reached(
    master,
    target_xyz: Vec3,
    yaw_deg: float = 0.0,
    xy_tol_m: float = 0.40,
    z_tol_m: float = 0.25,
    timeout_s: float = 60.0,
    hz: float = 10.0,
    on_position_command: Optional[Callable[[object, Vec3], None]] = None,
):
    dt = 1.0 / hz
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        goto_position_target_local_ned(
            master,
            target_xyz[0],
            target_xyz[1],
            target_xyz[2],
            yaw_deg=yaw_deg,
        )

        if on_position_command is not None:
            on_position_command(master, target_xyz)

        cx, cy, cz = get_local_position(master, timeout_s=1.0)

        err_xy = math.hypot(
            cx - target_xyz[0],
            cy - target_xyz[1],
        )
        err_z = abs(cz - target_xyz[2])

        if err_xy <= xy_tol_m and err_z <= z_tol_m:
            return

        time.sleep(dt)

    raise TimeoutError(f"Did not reach local target {target_xyz} in time.")


# =========================================================
# MOVEMENT FUNCTIONS
# =========================================================

def fly_line_local_streamed(
    master,
    start_xyz: Vec3,
    end_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = 10.0,
    min_time_s: float = 0.5,
    on_position_command: Optional[Callable[[object, Vec3], None]] = None,
):
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz

    dx = ex - sx
    dy = ey - sy
    dz = ez - sz

    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    speed = max(float(speed_mps), 0.05)
    total_time = max(min_time_s, dist / speed)

    steps = max(2, int(total_time * hz))
    dt = 1.0 / hz

    for k in range(steps + 1):
        if _shutdown_requested:
            return

        a = k / steps

        x = sx + a * dx
        y = sy + a * dy
        z = sz + a * dz

        target = (x, y, z)

        goto_position_target_local_ned(
            master,
            x,
            y,
            z,
            yaw_deg=yaw_deg,
        )

        if on_position_command is not None:
            on_position_command(master, target)

        time.sleep(dt)


def fly_to_local_point(
    master,
    target_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = 10.0,
    on_position_command: Optional[Callable[[object, Vec3], None]] = None,
):
    start_xyz = get_local_position(master, timeout_s=2.0)

    fly_line_local_streamed(
        master=master,
        start_xyz=start_xyz,
        end_xyz=target_xyz,
        yaw_deg=yaw_deg,
        speed_mps=speed_mps,
        hz=hz,
        on_position_command=on_position_command,
    )

    hold_local_target_until_reached(
        master=master,
        target_xyz=target_xyz,
        yaw_deg=yaw_deg,
        hz=hz,
        on_position_command=on_position_command,
    )


# =========================================================
# SCAN GEOMETRY
# =========================================================

def load_scan_parameters(spec: dict) -> Dict[str, float]:
    start_alt_m = float(spec["scan_settings"]["start_alt_m"])
    speed_mps = float(spec["scan_settings"]["speed_mps"])
    standoff_m = float(spec["scan_settings"]["standoff_m"])
    lane_spacing_m = float(spec["scan_settings"]["lane_spacing_m"])

    if "max_alt_m" in spec["scan_settings"]:
        max_alt_m = float(spec["scan_settings"]["max_alt_m"])
    else:
        max_alt_m = float(spec["structure"]["dimensions_m"]["height"])

    overshoot_m = float(spec["scan_settings"].get("overshoot_m", 2.0))

    if max_alt_m < start_alt_m:
        raise ValueError("max_alt_m must be greater than or equal to start_alt_m")

    if lane_spacing_m <= 0.0:
        raise ValueError("lane_spacing_m must be > 0")

    if speed_mps <= 0.0:
        raise ValueError("speed_mps must be > 0")

    return {
        "start_alt_m": start_alt_m,
        "max_alt_m": max_alt_m,
        "speed_mps": speed_mps,
        "standoff_m": standoff_m,
        "lane_spacing_m": lane_spacing_m,
        "overshoot_m": overshoot_m,
    }


def load_local_scan_parameters() -> Dict[str, float]:
    return {
        "start_alt_m": LOCAL_START_ALT_M,
        "max_alt_m": LOCAL_MAX_ALT_M,
        "speed_mps": LOCAL_SCAN_SPEED_MPS,
        "standoff_m": LOCAL_STANDOFF_M,
        "lane_spacing_m": LOCAL_LANE_SPACING_M,
        "overshoot_m": LOCAL_OVERSHOOT_M,
    }


def compute_wall_unit_vector(a_ne: Vec2, b_ne: Vec2) -> Vec2:
    wall_vec = v2_sub(b_ne, a_ne)

    if v2_norm(wall_vec) < 1e-6:
        raise ValueError("A and B are too close to define a wall direction.")

    return v2_unit(wall_vec)


def compute_candidate_wall_normals(u_wall: Vec2) -> Tuple[Vec2, Vec2]:
    n1 = (-u_wall[1], u_wall[0])
    n2 = (u_wall[1], -u_wall[0])
    return n1, n2


def choose_scan_normal_same_side_as_drone(
    current_ne: Vec2,
    a_ne: Vec2,
    b_ne: Vec2,
    standoff_m: float,
) -> Tuple[Vec2, Vec2]:
    u_wall = compute_wall_unit_vector(a_ne, b_ne)
    n1, n2 = compute_candidate_wall_normals(u_wall)

    wall_mid = v2_midpoint(a_ne, b_ne)

    candidate_mid_1 = v2_add(
        wall_mid,
        v2_scale(n1, standoff_m),
    )

    candidate_mid_2 = v2_add(
        wall_mid,
        v2_scale(n2, standoff_m),
    )

    d1 = v2_norm(v2_sub(current_ne, candidate_mid_1))
    d2 = v2_norm(v2_sub(current_ne, candidate_mid_2))

    n_side = n1 if d1 <= d2 else n2

    return u_wall, n_side


def build_scan_endpoints_ne(
    a_ne: Vec2,
    b_ne: Vec2,
    u_wall: Vec2,
    n_side: Vec2,
    overshoot_m: float,
    standoff_m: float,
) -> Tuple[Vec2, Vec2]:
    left_scan_ne = v2_add(
        v2_sub(a_ne, v2_scale(u_wall, overshoot_m)),
        v2_scale(n_side, standoff_m),
    )

    right_scan_ne = v2_add(
        v2_add(b_ne, v2_scale(u_wall, overshoot_m)),
        v2_scale(n_side, standoff_m),
    )

    return left_scan_ne, right_scan_ne


def build_row_altitudes(
    start_alt_m: float,
    max_alt_m: float,
    lane_spacing_m: float,
) -> List[float]:
    alts = []
    z = start_alt_m

    while z <= max_alt_m + 1e-6:
        alts.append(round(z, 6))
        z += lane_spacing_m

    if not alts:
        raise RuntimeError("No row altitudes generated.")

    return alts


def ne_alt_to_local_ned(ne: Vec2, alt_m: float) -> Vec3:
    return ne[0], ne[1], -alt_m


def build_serpentine_rows(
    left_scan_ne: Vec2,
    right_scan_ne: Vec2,
    row_altitudes_m: List[float],
) -> List[Tuple[Vec3, Vec3]]:
    rows = []

    for i, alt_m in enumerate(row_altitudes_m):
        left_pt = ne_alt_to_local_ned(left_scan_ne, alt_m)
        right_pt = ne_alt_to_local_ned(right_scan_ne, alt_m)

        if i % 2 == 0:
            rows.append((left_pt, right_pt))
        else:
            rows.append((right_pt, left_pt))

    return rows


def convert_wall_corners_to_local_ne(master, spec: dict) -> Tuple[Vec2, Vec2, Vec3]:
    A = spec["structure"]["corners_gps"]["A"]
    B = spec["structure"]["corners_gps"]["B"]

    current_lat, current_lon, _current_rel_alt = get_global_position(master, timeout_s=2.0)
    current_local_ned = get_local_position(master, timeout_s=2.0)

    a_local = current_gps_to_target_local_ned(
        current_lat=current_lat,
        current_lon=current_lon,
        current_local_ned=current_local_ned,
        target_lat=A["lat"],
        target_lon=A["lon"],
        target_alt_m=0.0,
    )

    b_local = current_gps_to_target_local_ned(
        current_lat=current_lat,
        current_lon=current_lon,
        current_local_ned=current_local_ned,
        target_lat=B["lat"],
        target_lon=B["lon"],
        target_alt_m=0.0,
    )

    a_ne = a_local[0], a_local[1]
    b_ne = b_local[0], b_local[1]

    return a_ne, b_ne, current_local_ned


def build_plan_from_gui_spec(master, spec: dict) -> Dict:
    params = load_scan_parameters(spec)

    a_ne, b_ne, current_local_ned = convert_wall_corners_to_local_ne(master, spec)
    current_ne = current_local_ned[0], current_local_ned[1]

    return build_plan_from_ne(
        params=params,
        a_ne=a_ne,
        b_ne=b_ne,
        current_ne=current_ne,
    )


def build_local_test_plan(master) -> Dict:
    _ = master

    params = load_local_scan_parameters()

    a_ne = LOCAL_WALL_A_NE
    b_ne = LOCAL_WALL_B_NE

    try:
        current_local_ned = get_local_position(master, timeout_s=2.0)
        current_ne = current_local_ned[0], current_local_ned[1]
    except Exception:
        current_ne = (0.0, 0.0)

    return build_plan_from_ne(
        params=params,
        a_ne=a_ne,
        b_ne=b_ne,
        current_ne=current_ne,
    )


def build_plan_from_ne(
    params: Dict[str, float],
    a_ne: Vec2,
    b_ne: Vec2,
    current_ne: Vec2,
) -> Dict:
    u_wall, n_side = choose_scan_normal_same_side_as_drone(
        current_ne=current_ne,
        a_ne=a_ne,
        b_ne=b_ne,
        standoff_m=params["standoff_m"],
    )

    left_scan_ne, right_scan_ne = build_scan_endpoints_ne(
        a_ne=a_ne,
        b_ne=b_ne,
        u_wall=u_wall,
        n_side=n_side,
        overshoot_m=params["overshoot_m"],
        standoff_m=params["standoff_m"],
    )

    row_altitudes_m = build_row_altitudes(
        start_alt_m=params["start_alt_m"],
        max_alt_m=params["max_alt_m"],
        lane_spacing_m=params["lane_spacing_m"],
    )

    rows = build_serpentine_rows(
        left_scan_ne=left_scan_ne,
        right_scan_ne=right_scan_ne,
        row_altitudes_m=row_altitudes_m,
    )

    if not rows:
        raise RuntimeError("No scan rows generated.")

    scan_origin_local_ned = rows[0][0]

    return {
        "params": params,
        "a_ne": a_ne,
        "b_ne": b_ne,
        "u_wall": u_wall,
        "n_side": n_side,
        "left_scan_ne": left_scan_ne,
        "right_scan_ne": right_scan_ne,
        "row_altitudes_m": row_altitudes_m,
        "rows": rows,
        "scan_origin_local_ned": scan_origin_local_ned,
    }


# =========================================================
# CORROSION LOGGING
# =========================================================

class CorrosionLoggingState:
    def __init__(self, json_path: str = DEFAULT_DETECTIONS_JSON):
        self.json_path = json_path
        self.saved_detections = []


def save_detection_if_new(
    saved_detections: List[dict],
    detection_ref_ned: Vec3,
    current_local_ned: Vec3,
    commanded_target_local_ned: Vec3,
    scan_origin_local_ned: Vec3,
    threshold_m: float = DETECTION_DEDUP_THRESHOLD_M,
    extra: Optional[dict] = None,
) -> bool:
    """
    Deduplicate using the commanded scan target, not the raw current MAVLink pose.

    This means:
    - if the same corrosion is detected again near a previously SAVED scan target,
      it is ignored
    - if the scan has moved far enough from the last SAVED target,
      it is saved as a new corrosion target
    """

    rel_ned = (
        detection_ref_ned[0] - scan_origin_local_ned[0],
        detection_ref_ned[1] - scan_origin_local_ned[1],
        detection_ref_ned[2] - scan_origin_local_ned[2],
    )

    nearest_dist = None

    for det in saved_detections:
        old = det["rel_ned"]

        dist = math.sqrt(
            (rel_ned[0] - old[0]) ** 2 +
            (rel_ned[1] - old[1]) ** 2 +
            (rel_ned[2] - old[2]) ** 2
        )

        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist

        if dist <= threshold_m:
            print(
                f"[Log] Duplicate detection ignored. "
                f"nearest_saved_dist={dist:.2f} m, threshold={threshold_m:.2f} m"
            )
            return False

    record = {
        "rel_ned": rel_ned,

        # This is the reference used for duplicate filtering.
        "dedup_reference": "commanded_target_ned",

        # The commanded scan target where the frame was checked.
        "commanded_target_ned": {
            "x_north_m": commanded_target_local_ned[0],
            "y_east_m": commanded_target_local_ned[1],
            "z_down_m": commanded_target_local_ned[2],
        },

        # The actual drone position reported by MAVLink at logging time.
        "actual_local_position_ned": {
            "x_north_m": current_local_ned[0],
            "y_east_m": current_local_ned[1],
            "z_down_m": current_local_ned[2],
        },

        "timestamp_s": time.time(),
    }

    if nearest_dist is not None:
        record["nearest_previous_saved_detection_m"] = nearest_dist

    if extra is not None:
        record["extra"] = extra

    saved_detections.append(record)

    return True


def write_corrosion_detections_json(state: CorrosionLoggingState):
    payload = {
        "detection_count": len(state.saved_detections),
        "detections": state.saved_detections,
    }

    with open(state.json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_corrosion_detection_to_json(
    master,
    state: CorrosionLoggingState,
    scan_origin_local_ned: Vec3,
    corroded_area,
    commanded_target_local_ned: Vec3,
    extra: Optional[dict] = None,
    threshold_m: float = DETECTION_DEDUP_THRESHOLD_M,
) -> bool:
    current_local_ned = get_local_position(master, timeout_s=2.0)

    info = {
        "corroded_area": corroded_area,
    }

    if extra is not None:
        info.update(extra)

    is_new = save_detection_if_new(
        saved_detections=state.saved_detections,

        # Important fix:
        # Use commanded target for duplicate filtering.
        detection_ref_ned=commanded_target_local_ned,

        # Still save actual MAVLink pose for debugging.
        current_local_ned=current_local_ned,

        commanded_target_local_ned=commanded_target_local_ned,
        scan_origin_local_ned=scan_origin_local_ned,
        threshold_m=threshold_m,
        extra=info,
    )

    if is_new:
        write_corrosion_detections_json(state)
        print(f"[Log] Saved corrosion detection to {state.json_path}")
        print(f"[Log] Saved target reference: {commanded_target_local_ned}")

    return is_new


def build_corrosion_scan_step_callback(
    plan: Dict,
    logging_state: CorrosionLoggingState,
    detector: VisionCorrosionDetector,
    user_on_scan_step: Optional[Callable[[object, Vec3], None]] = None,
) -> Callable[[object, Vec3], None]:
    scan_origin_local_ned = plan["scan_origin_local_ned"]

    def _callback(master, commanded_target_local_ned: Vec3):
        commanded_target_local_ned = (
            float(commanded_target_local_ned[0]),
            float(commanded_target_local_ned[1]),
            float(commanded_target_local_ned[2]),
        )

        vision_result = detector.process_scan_step(
            master,
            commanded_target_local_ned,
        )

        if vision_result.get("detected", False):
            save_corrosion_detection_to_json(
                master=master,
                state=logging_state,
                scan_origin_local_ned=scan_origin_local_ned,
                corroded_area=vision_result.get("corroded_area"),
                commanded_target_local_ned=commanded_target_local_ned,
                extra=vision_result.get("extra"),
            )

        if user_on_scan_step is not None:
            user_on_scan_step(master, commanded_target_local_ned)

    return _callback


# =========================================================
# MISSION EXECUTION
# =========================================================

def move_to_first_scan_point(
    master,
    plan: Dict,
    yaw_hold_deg: float,
):
    first_scan_point = plan["scan_origin_local_ned"]
    speed_mps = plan["params"]["speed_mps"]

    print(f"[Mission] Moving to first scan point: {first_scan_point}")
    print("[Mission] Vision logging is OFF while moving to first scan point.")

    fly_to_local_point(
        master=master,
        target_xyz=first_scan_point,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
        on_position_command=None,
    )


def execute_single_scan_row(
    master,
    plan: Dict,
    row_start: Vec3,
    row_end: Vec3,
    speed_mps: float,
    yaw_hold_deg: float,
    on_position_command: Optional[Callable[[object, Vec3], None]],
):
    print("[Mission] Going to row start...")
    print("[Mission] Vision logging is OFF while moving to row start.")

    fly_to_local_point(
        master=master,
        target_xyz=row_start,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
        on_position_command=None,
    )

    print("[Mission] Vision logging is OFF during row-start PID correction.")

    run_row_start_correction(
        master=master,
        plan=plan,
        row_start=row_start,
        yaw_hold_deg=yaw_hold_deg,
    )

    print("[Mission] Scanning row...")
    print("[Mission] Vision logging is ON during this row scan.")

    fly_line_local_streamed(
        master=master,
        start_xyz=row_start,
        end_xyz=row_end,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
        on_position_command=on_position_command,
    )

    print("[Mission] Holding row end...")
    print("[Mission] Vision logging is OFF while holding row end.")

    hold_local_target_until_reached(
        master=master,
        target_xyz=row_end,
        yaw_deg=yaw_hold_deg,
        hz=10.0,
        on_position_command=None,
    )

def execute_all_scan_rows(
    master,
    plan: Dict,
    yaw_hold_deg: float,
    on_position_command: Optional[Callable[[object, Vec3], None]],
):
    rows = plan["rows"]
    speed_mps = plan["params"]["speed_mps"]

    for i, (row_start, row_end) in enumerate(rows):
        if _shutdown_requested:
            break

        print(f"\n[Mission] Row {i + 1}/{len(rows)}")
        print(f"  start: {row_start}")
        print(f"  end  : {row_end}")

        execute_single_scan_row(
            master=master,
            plan=plan,
            row_start=row_start,
            row_end=row_end,
            speed_mps=speed_mps,
            yaw_hold_deg=yaw_hold_deg,
            on_position_command=on_position_command,
        )


def execute_wall_scan(
    master,
    spec: Optional[dict],
    detector: VisionCorrosionDetector,
    yaw_hold_deg: float = 0.0,
    detections_json_path: str = DEFAULT_DETECTIONS_JSON,
):
    if spec is None:
        params = load_local_scan_parameters()
        start_alt_m = params["start_alt_m"]
    else:
        params = load_scan_parameters(spec)
        start_alt_m = params["start_alt_m"]

    print(f"[Mission] Takeoff to start altitude: {start_alt_m:.2f} m")

    takeoff(master, start_alt_m)
    wait_until_altitude(
        master,
        start_alt_m,
        tol_m=0.25,
        timeout_s=45.0,
    )

    print("[Mission] Building scan plan...")

    if spec is None:
        plan = build_local_test_plan(master)
    else:
        plan = build_plan_from_gui_spec(master, spec)

    print("[Mission] Scan plan built.")
    print(f"  wall A local NE: {plan['a_ne']}")
    print(f"  wall B local NE: {plan['b_ne']}")
    print(f"  u_wall:          {plan['u_wall']}")
    print(f"  n_side:          {plan['n_side']}")
    print(f"  left_scan_ne:    {plan['left_scan_ne']}")
    print(f"  right_scan_ne:   {plan['right_scan_ne']}")
    print(f"  scan origin:     {plan['scan_origin_local_ned']}")
    print(f"  row count:       {len(plan['rows'])}")

    logging_state = CorrosionLoggingState(
        json_path=detections_json_path,
    )

    on_position_command = build_corrosion_scan_step_callback(
        plan=plan,
        logging_state=logging_state,
        detector=detector,
    )

    move_to_first_scan_point(
        master,
        plan,
        yaw_hold_deg=yaw_hold_deg,
    )

    execute_all_scan_rows(
        master,
        plan,
        yaw_hold_deg=yaw_hold_deg,
        on_position_command=on_position_command,
    )

    print("[Mission] Scan complete.")
    write_corrosion_detections_json(logging_state)

    return {
        "plan": plan,
        "logging_state": logging_state,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    global ULTRASONIC_SER
    import sys

    spec = None

    if len(sys.argv) >= 2:
        spec_path = sys.argv[1]
        print(f"[Main] Loading GUI mission JSON: {spec_path}")
        spec = load_gui_spec(spec_path)
    else:
        print("[Main] No GUI JSON provided. Using local test scan plan.")

    detections_json_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else DEFAULT_DETECTIONS_JSON
    )

    detector = VisionCorrosionDetector()
    ultrasonic_ser = None

    try:
        detector.open()

        if ULTRASONIC_PID_ENABLED:
            try:
                ultrasonic_ser = open_ultrasonic_serial()
                ULTRASONIC_SER = ultrasonic_ser
            except Exception as e:
                ULTRASONIC_SER = None
                print(f"[PID] Could not open ultrasonic serial. Row-start PID disabled for this run. ({e})")

        master = connect_sitl(MAVLINK_URL)

        set_mode(master, "GUIDED")
        arm(master)

        execute_wall_scan(
            master=master,
            spec=spec,
            detector=detector,
            yaw_hold_deg=YAW_HOLD_DEG,
            detections_json_path=detections_json_path,
        )

        print("[Main] Sending RTL...")
        rtl(master)

    finally:
        detector.close()

        if ultrasonic_ser is not None:
            try:
                ultrasonic_ser.close()
                print("[PID] Ultrasonic serial closed.")
            except Exception:
                pass

        ULTRASONIC_SER = None

        print("[Main] Done.")


if __name__ == "__main__":
    main()
