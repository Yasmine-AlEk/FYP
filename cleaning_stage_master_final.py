import argparse
import csv
import json
import math
import socket
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pymavlink import mavutil

try:
    import serial
except Exception:
    serial = None


# =========================================================
# CONNECTION CONFIG
# =========================================================

MAVLINK_URL = "udpin:0.0.0.0:14550"

ESP32_ENABLED = True
ESP32_IP = "172.20.10.14"
ESP32_PORT = 4210
ESP32_TIMEOUT_SEC = 2.0
ESP32_HEARTBEAT_PERIOD_SEC = 1.0
ESP32_START_SEQUENCE_WAIT_S = 2.3

SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 9600
SERIAL_TIMEOUT_S = 0.03


# =========================================================
# CAMERA CONFIG
# =========================================================

CAMERA_SENSOR_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FRAMERATE = 60
CAMERA_WARMUP_FRAMES = 10
CAMERA_WARMUP_SLEEP_S = 0.03

SEGMENT_RESIZE_TO = (640, 480)


# =========================================================
# OUTPUT CONFIG
# =========================================================

OUTPUT_DIR = Path("cleaning_segmentation_outputs")
LOG_DIR = Path("cleaning_mission_logs")


# =========================================================
# CORROSION SEGMENTATION CONFIG
# values from your tuned screenshot
# =========================================================

SEG_HSV_LOWER = np.array([1, 96, 70], dtype=np.uint8)
SEG_HSV_UPPER = np.array([11, 198, 255], dtype=np.uint8)

SEG_USE_ENHANCE = True
SEG_ALPHA = 1.30
SEG_BETA = 20

SEG_USE_CLAHE = True
SEG_CLAHE_CLIP = 2.0

SEG_BLUR = 5
SEG_OPEN_ITER = 1
SEG_CLOSE_ITER = 1
SEG_MEDIAN_BLUR = 5
SEG_MIN_COMPONENT_AREA = 120

SEG_MERGE_GAP_X = 10
SEG_MERGE_GAP_Y = 6
SEG_EXPAND_RECT_PX = 8


# =========================================================
# SPRAY FOOTPRINT CONFIG
# measured from your spray footprint
# =========================================================

SPRAY_FOOTPRINT_AREA_PX = 311
SPRAY_DIAMETER_PX = 19.90
SPRAY_RADIUS_PX = 9.95
SPRAY_OVERLAP_PCT = 30
SPRAY_STEP_PX = max(
    1,
    int(round(SPRAY_DIAMETER_PX * (1.0 - SPRAY_OVERLAP_PCT / 100.0))),
)

SPRAY_EACH_SERPENTINE_POINT = True
MAX_SERPENTINE_POINTS_PER_TARGET = None


# =========================================================
# GIMBAL CONFIG
# =========================================================

GIMBAL_ENABLED = True

GIMBAL_CALIBRATION_JSON = Path.home() / "calibration_grid_outputs" / "gimbal_calibration.json"

GIMBAL_SETTLE_S = 0.70

PAN_SERVO = 10
TILT_SERVO = 9

PAN_MIN_PWM = 1100
PAN_CENTER_PWM = 1500
PAN_MAX_PWM = 1900
PAN_MIN_ANGLE = -40.0
PAN_MAX_ANGLE = 40.0

TILT_MIN_PWM = 1100
TILT_CENTER_PWM = 1500
TILT_MAX_PWM = 1800
TILT_MIN_ANGLE = -30.0
TILT_MAX_ANGLE = 40.0


# =========================================================
# PID CONFIG
# =========================================================

PID_ENABLED = True

# Set False only for dry test if ultrasonic serial is not connected.
PID_REQUIRE_ULTRASONIC = True

TARGET_STANDOFF_M = 2.0

DIST_DEADBAND_M = 0.10
YAW_DIFF_DEADBAND_M = 0.03
HEIGHT_DEADBAND_M = 0.10

PID_CONTROL_HZ = 20.0
PID_STABLE_HOLD_S = 2.0
PID_TIMEOUT_S = 25.0

SENSOR_AVG_WINDOW = 3
SENSOR_MIN_VALID_SAMPLES = 2

KP_DIST = 0.80
KI_DIST = 0.00
KD_DIST = 0.05

KP_YAW = 1.20
KI_YAW = 0.00
KD_YAW = 0.08

KP_HEIGHT = 0.60
KI_HEIGHT = 0.00
KD_HEIGHT = 0.05

MAX_VX_MPS = 0.35
MAX_YAW_RATE_RAD_S = 0.50
MAX_VZ_MPS = 0.35

DIST_SIGN = 1.0
YAW_SIGN = -1.0
HEIGHT_SIGN = -1.0

PID_PRINT_EVERY_N = 5


# =========================================================
# MISSION CONFIG
# =========================================================

FLY_SPEED_MPS = 0.5
HOLD_AT_TARGET_S = 1.0
YAW_HOLD_DEG = 0.0

REQUIRE_CORROSION_BEFORE_SPRAY = True

RTL_BETWEEN_STAGES = True
PROMPT_BETWEEN_STAGES = True

RUST_REMOVER_TOTAL_EXPOSURE_S = 20 * 60
RUST_REMOVER_RESpray_INTERVAL_S = 4 * 60

STAGES = {
    "rust_remover": {
        "name": "rust_remover",
        "label": "Rust Remover",
        "solution": "Citric acid 5% w/v",
        "spray_duration_s": 8.0,
        "rust_repeated_passes": True,
        "total_exposure_s": RUST_REMOVER_TOTAL_EXPOSURE_S,
        "respray_interval_s": RUST_REMOVER_RESpray_INTERVAL_S,
    },
    "water_rinse": {
        "name": "water_rinse",
        "label": "Water Rinse",
        "solution": "Deionized water",
        "spray_duration_s": 60.0,
        "rust_repeated_passes": False,
    },
    "neutralizer": {
        "name": "neutralizer",
        "label": "Neutralizer",
        "solution": "Sodium bicarbonate 1% w/v",
        "spray_duration_s": 120.0,
        "rust_repeated_passes": False,
    },
    "final_water_rinse": {
        "name": "final_water_rinse",
        "label": "Final Water Rinse",
        "solution": "Deionized water",
        "spray_duration_s": 60.0,
        "rust_repeated_passes": False,
    },
    "inhibitor": {
        "name": "inhibitor",
        "label": "Inhibitor",
        "solution": "Sodium benzoate 1% w/v",
        "spray_duration_s": 8.0,
        "rust_repeated_passes": False,
    },
}

ALL_STAGE_ORDER = [
    "rust_remover",
    "water_rinse",
    "neutralizer",
    "final_water_rinse",
    "inhibitor",
]


# =========================================================
# TYPE ALIASES
# =========================================================

Vec3 = Tuple[float, float, float]


# =========================================================
# BASIC HELPERS
# =========================================================

def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_").replace("-", "_").lower()


def v3_dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def apply_deadband(error: float, deadband: float) -> float:
    if abs(error) <= deadband:
        return 0.0
    return error


# =========================================================
# MAVLINK HELPERS
# =========================================================

_T0_MS = None


def now_ms_u32():
    global _T0_MS
    t = time.monotonic()
    if _T0_MS is None:
        _T0_MS = t
    return int((t - _T0_MS) * 1000) & 0xFFFFFFFF


def connect_mavlink():
    print(f"[MAVLink] Connecting using: {MAVLINK_URL}")
    master = mavutil.mavlink_connection(MAVLINK_URL)

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
        raise RuntimeError(f"Mode {mode} not available. Available: {list(mode_map.keys())}")

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_map[mode],
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
        float(target_alt_m),
    )
    print(f"[MAVLink] Taking off to {target_alt_m:.2f} m...")
    time.sleep(1.0)


def rtl(master):
    try:
        set_mode(master, "RTL")
    except Exception:
        pass

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )

    print("[MAVLink] RTL command sent.")


def get_local_position(master, timeout_s: float = 1.0) -> Optional[Vec3]:
    msg = master.recv_match(
        type="LOCAL_POSITION_NED",
        blocking=True,
        timeout=timeout_s,
    )

    if msg is None:
        return None

    return float(msg.x), float(msg.y), float(msg.z)


def wait_until_altitude(
    master,
    target_alt_m: float,
    tolerance_m: float = 0.25,
    timeout_s: float = 45.0,
):
    print("[MAVLink] Waiting until altitude is reached...")

    t0 = time.time()

    while time.time() - t0 < timeout_s:
        pos = get_local_position(master, timeout_s=1.0)

        if pos is None:
            continue

        _x, _y, z_down = pos
        current_alt_m = -z_down

        print(f"[Alt] current={current_alt_m:.2f} m | target={target_alt_m:.2f} m")

        if abs(current_alt_m - target_alt_m) <= tolerance_m:
            print("[MAVLink] Target altitude reached.")
            return True

        time.sleep(0.3)

    print("[MAVLink] Altitude wait timed out.")
    return False


def send_local_position(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)
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
        now_ms_u32(),
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


def send_body_velocity_yawrate(
    master,
    vx: float,
    vy: float,
    vz: float,
    yaw_rate: float,
):
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
        now_ms_u32(),
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


def stop_motion(master):
    send_body_velocity_yawrate(
        master,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        yaw_rate=0.0,
    )


def fly_to_local_point(
    master,
    target_xyz: Vec3,
    yaw_deg: float = 0.0,
    speed_mps: float = FLY_SPEED_MPS,
    hz: float = 10.0,
):
    current = get_local_position(master, timeout_s=2.0)

    if current is None:
        raise RuntimeError("Could not read current local position.")

    sx, sy, sz = current
    ex, ey, ez = target_xyz

    dx = ex - sx
    dy = ey - sy
    dz = ez - sz

    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    total_time = max(1.0, dist / max(0.05, speed_mps))
    steps = max(2, int(total_time * hz))
    dt = 1.0 / hz

    print(f"[Move] Current: {current}")
    print(f"[Move] Target:  {target_xyz}")
    print(f"[Move] Distance: {dist:.2f} m")

    for k in range(steps + 1):
        a = k / steps
        x = sx + a * dx
        y = sy + a * dy
        z = sz + a * dz

        send_local_position(master, x, y, z, yaw_deg)
        time.sleep(dt)

    t_hold = time.time()
    while time.time() - t_hold < HOLD_AT_TARGET_S:
        send_local_position(master, ex, ey, ez, yaw_deg)
        time.sleep(dt)


def ensure_airborne(master, target_alt_m: float):
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, target_alt_m)
    wait_until_altitude(master, target_alt_m)


# =========================================================
# PID CLASSES / ULTRASONIC
# =========================================================

class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: Optional[float] = None,
        integral_limit: Optional[float] = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error: float) -> float:
        now = time.time()

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = clamp(
                self.integral,
                -self.integral_limit,
                self.integral_limit,
            )

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limit is not None:
            output = clamp(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        self.prev_time = now

        return output


class RollingAverage:
    def __init__(self, window_size: int, min_samples: int):
        self.s1 = deque(maxlen=window_size)
        self.s2 = deque(maxlen=window_size)
        self.min_samples = min_samples

    def update(self, sensor1_cm: float, sensor2_cm: float):
        self.s1.append(sensor1_cm)
        self.s2.append(sensor2_cm)

        if len(self.s1) < self.min_samples:
            return None

        avg1 = sum(self.s1) / len(self.s1)
        avg2 = sum(self.s2) / len(self.s2)
        error = avg1 - avg2

        return avg1, avg2, error


def parse_sensor_value(value: str) -> Optional[float]:
    value = value.strip()

    if value in ["", "N/A", "OUT_OF_RANGE"]:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def open_ultrasonic_serial():
    if serial is None:
        print("[Serial] pyserial not available.")
        return None

    try:
        ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT_S,
            write_timeout=SERIAL_TIMEOUT_S,
        )
        time.sleep(2.0)
        ser.reset_input_buffer()
        print(f"[Serial] Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud.")
        return ser
    except Exception as e:
        print(f"[Serial] Could not open ultrasonic serial: {e}")
        return None


def read_ultrasonic_sensors_cm(ser, max_attempts: int = 8) -> Optional[Tuple[float, float, float]]:
    if ser is None:
        return None

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

        sensor1 = parse_sensor_value(parts[0])
        sensor2 = parse_sensor_value(parts[1])
        error = parse_sensor_value(parts[2])

        if sensor1 is None or sensor2 is None:
            continue

        if error is None:
            error = sensor1 - sensor2

        latest_valid = (sensor1, sensor2, error)

    return latest_valid


def stabilize_before_spray(master, ser, target_alt_m: float) -> bool:
    if not PID_ENABLED:
        print("[PID] Disabled. Skipping stabilization.")
        return True

    if PID_REQUIRE_ULTRASONIC and ser is None:
        print("[PID] Ultrasonic serial required but not available. Stabilization failed.")
        return False

    dist_pid = PID(KP_DIST, KI_DIST, KD_DIST, output_limit=MAX_VX_MPS, integral_limit=1.0)
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW, output_limit=MAX_YAW_RATE_RAD_S, integral_limit=1.0)
    height_pid = PID(KP_HEIGHT, KI_HEIGHT, KD_HEIGHT, output_limit=MAX_VZ_MPS, integral_limit=1.0)

    filt = RollingAverage(SENSOR_AVG_WINDOW, SENSOR_MIN_VALID_SAMPLES)

    stable_start = None
    t0 = time.time()
    dt = 1.0 / PID_CONTROL_HZ
    loop_i = 0

    print("[PID] Stabilizing before cleaning.")
    print(f"[PID] Target standoff = {TARGET_STANDOFF_M:.2f} m")
    print(f"[PID] Target altitude = {target_alt_m:.2f} m")

    while time.time() - t0 < PID_TIMEOUT_S:
        loop_i += 1

        pos = get_local_position(master, timeout_s=0.5)
        if pos is None:
            print("[PID] No local position. Sending zero command.")
            stop_motion(master)
            time.sleep(dt)
            continue

        current_alt_m = -pos[2]
        height_error_raw_m = target_alt_m - current_alt_m
        height_error_control_m = apply_deadband(height_error_raw_m, HEIGHT_DEADBAND_M)

        distance_error_raw_m = 0.0
        yaw_error_raw_m = 0.0
        distance_stable = True
        yaw_stable = True
        vx_cmd = 0.0
        yaw_rate_cmd = 0.0

        reading = read_ultrasonic_sensors_cm(ser)

        if reading is not None:
            s1_cm, s2_cm, _err_cm = reading
            filtered = filt.update(s1_cm, s2_cm)

            if filtered is None:
                stop_motion(master)
                time.sleep(dt)
                continue

            avg_s1_cm, avg_s2_cm, avg_err_cm = filtered

            avg_distance_m = ((avg_s1_cm / 100.0) + (avg_s2_cm / 100.0)) / 2.0
            distance_error_raw_m = avg_distance_m - TARGET_STANDOFF_M
            yaw_error_raw_m = avg_err_cm / 100.0

            distance_error_control_m = apply_deadband(distance_error_raw_m, DIST_DEADBAND_M)
            yaw_error_control_m = apply_deadband(yaw_error_raw_m, YAW_DIFF_DEADBAND_M)

            vx_cmd = DIST_SIGN * dist_pid.update(distance_error_control_m)
            yaw_rate_cmd = YAW_SIGN * yaw_pid.update(yaw_error_control_m)

            distance_stable = abs(distance_error_raw_m) <= DIST_DEADBAND_M
            yaw_stable = abs(yaw_error_raw_m) <= YAW_DIFF_DEADBAND_M

        else:
            if PID_REQUIRE_ULTRASONIC:
                print("[PID] No valid ultrasonic reading.")
                stop_motion(master)
                time.sleep(dt)
                continue

        vz_cmd = HEIGHT_SIGN * height_pid.update(height_error_control_m)
        height_stable = abs(height_error_raw_m) <= HEIGHT_DEADBAND_M

        stable = distance_stable and yaw_stable and height_stable

        send_body_velocity_yawrate(
            master,
            vx=vx_cmd,
            vy=0.0,
            vz=vz_cmd,
            yaw_rate=yaw_rate_cmd,
        )

        if loop_i % PID_PRINT_EVERY_N == 0:
            print(
                f"[PID] dist_err={distance_error_raw_m:+.2f}m | "
                f"yaw_err={yaw_error_raw_m:+.2f}m | "
                f"height_err={height_error_raw_m:+.2f}m | "
                f"vx={vx_cmd:+.2f} | "
                f"yaw_rate={yaw_rate_cmd:+.2f} | "
                f"vz={vz_cmd:+.2f} | "
                f"stable={stable}"
            )

        if stable:
            if stable_start is None:
                stable_start = time.time()

            if time.time() - stable_start >= PID_STABLE_HOLD_S:
                print("[PID] Stable. Ready for live capture and cleaning.")
                stop_motion(master)
                return True
        else:
            stable_start = None

        time.sleep(dt)

    stop_motion(master)
    print("[PID] Stabilization timed out.")
    return False


# =========================================================
# CAMERA CAPTURE
# =========================================================

def gstreamer_pipeline(
    sensor_id=CAMERA_SENSOR_ID,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    framerate=CAMERA_FRAMERATE,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )


def capture_live_frame(stage_name: str, pass_id: int, target_id: int) -> Optional[Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = f"{safe_name(stage_name)}_pass_{pass_id:02d}_target_{target_id:03d}_live.jpg"
    out_path = OUTPUT_DIR / filename

    print("[Camera] Opening CSI camera...")
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[Camera] Failed to open camera.")
        return None

    try:
        for _ in range(CAMERA_WARMUP_FRAMES):
            ret, _frame = cap.read()

            if not ret:
                print("[Camera] Failed during warmup.")
                return None

            time.sleep(CAMERA_WARMUP_SLEEP_S)

        ret, frame = cap.read()

        if not ret:
            print("[Camera] Failed to capture frame.")
            return None

        cv2.imwrite(str(out_path), frame)
        print(f"[Camera] Saved live frame: {out_path}")
        return out_path

    finally:
        cap.release()
        print("[Camera] Released.")


# =========================================================
# SEGMENTATION
# =========================================================

def enhance_image(frame_bgr):
    if not SEG_USE_ENHANCE:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return hsv, frame_bgr.copy()

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.convertScaleAbs(v, alpha=SEG_ALPHA, beta=SEG_BETA)

    if SEG_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=SEG_CLAHE_CLIP, tileGridSize=(8, 8))
        v = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v])
    frame_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return hsv_enhanced, frame_enhanced


def segment_corrosion(hsv_enhanced):
    mask = cv2.inRange(hsv_enhanced, SEG_HSV_LOWER, SEG_HSV_UPPER)

    blur = SEG_BLUR
    if blur < 1:
        blur = 1
    if blur % 2 == 0:
        blur += 1

    if blur > 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    if SEG_OPEN_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=SEG_OPEN_ITER)

    if SEG_CLOSE_ITER > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=SEG_CLOSE_ITER)

    median = SEG_MEDIAN_BLUR
    if median < 1:
        median = 1
    if median % 2 == 0:
        median += 1

    if median > 1:
        mask = cv2.medianBlur(mask, median)

    return mask


def get_filtered_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(mask)
    kept = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >= min_area:
            kept.append(cnt)
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask, kept


def contours_to_boxes(contours):
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 999.0

        if aspect >= 2.0:
            orientation = "horizontal"
        elif aspect <= 0.5:
            orientation = "vertical"
        else:
            orientation = "blob"

        boxes.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w - 1),
            "y2": int(y + h - 1),
            "w": int(w),
            "h": int(h),
            "orientation": orientation,
        })

    return boxes


def box_center(box):
    return (
        (box["x1"] + box["x2"]) / 2.0,
        (box["y1"] + box["y2"]) / 2.0,
    )


def horizontal_gap(a, b):
    if a["x2"] < b["x1"]:
        return b["x1"] - a["x2"]
    if b["x2"] < a["x1"]:
        return a["x1"] - b["x2"]
    return 0


def vertical_gap(a, b):
    if a["y2"] < b["y1"]:
        return b["y1"] - a["y2"]
    if b["y2"] < a["y1"]:
        return a["y1"] - b["y2"]
    return 0


def boxes_should_merge(a, b, gap_x, gap_y):
    dx = horizontal_gap(a, b)
    dy = vertical_gap(a, b)

    cxa, cya = box_center(a)
    cxb, cyb = box_center(b)

    center_dx = abs(cxa - cxb)
    center_dy = abs(cya - cyb)

    oa = a["orientation"]
    ob = b["orientation"]

    if oa == "horizontal" and ob == "horizontal":
        same_row = center_dy <= max(a["h"], b["h"])
        return dx <= gap_x and same_row

    if oa == "vertical" and ob == "vertical":
        same_col = center_dx <= max(a["w"], b["w"])
        return dy <= gap_y and same_col

    if oa == "blob" and ob == "blob":
        return dx <= gap_x and dy <= gap_y

    if (oa == "horizontal" and ob == "blob") or (oa == "blob" and ob == "horizontal"):
        same_row = center_dy <= max(a["h"], b["h"]) * 1.2
        return dx <= gap_x and same_row

    if (oa == "vertical" and ob == "blob") or (oa == "blob" and ob == "vertical"):
        same_col = center_dx <= max(a["w"], b["w"]) * 1.2
        return dy <= gap_y and same_col

    return False


def merge_two_boxes(a, b):
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    x2 = max(a["x2"], b["x2"])
    y2 = max(a["y2"], b["y2"])

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    aspect = w / float(h) if h > 0 else 999.0

    if aspect >= 2.0:
        orientation = "horizontal"
    elif aspect <= 0.5:
        orientation = "vertical"
    else:
        orientation = "blob"

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "w": int(w),
        "h": int(h),
        "orientation": orientation,
    }


def merge_nearby_boxes(boxes, gap_x, gap_y):
    if not boxes:
        return []

    current = boxes[:]
    changed = True

    while changed:
        changed = False
        used = [False] * len(current)
        next_boxes = []

        for i in range(len(current)):
            if used[i]:
                continue

            merged = current[i]
            used[i] = True

            local_changed = True

            while local_changed:
                local_changed = False

                for j in range(len(current)):
                    if used[j]:
                        continue

                    if boxes_should_merge(merged, current[j], gap_x, gap_y):
                        merged = merge_two_boxes(merged, current[j])
                        used[j] = True
                        local_changed = True
                        changed = True

            next_boxes.append(merged)

        current = next_boxes

    return current


def expand_boxes(boxes, expand_px, img_w, img_h):
    expanded = []

    boxes = sorted(boxes, key=lambda b: (b["y1"], b["x1"]))

    for i, box in enumerate(boxes):
        x1 = max(0, box["x1"] - expand_px)
        y1 = max(0, box["y1"] - expand_px)
        x2 = min(img_w - 1, box["x2"] + expand_px)
        y2 = min(img_h - 1, box["y2"] + expand_px)

        expanded.append({
            "id": int(i),
            "orientation": box["orientation"],
            "raw_bbox": {
                "x1": int(box["x1"]),
                "y1": int(box["y1"]),
                "x2": int(box["x2"]),
                "y2": int(box["y2"]),
                "w": int(box["w"]),
                "h": int(box["h"]),
            },
            "expanded_bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "w": int(x2 - x1 + 1),
                "h": int(y2 - y1 + 1),
            },
        })

    return expanded


def mask_to_pixels_xy(mask):
    ys, xs = np.where(mask > 0)
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


def mask_to_rle(mask):
    flat = (mask.flatten() > 0).astype(np.uint8)

    if flat.size == 0:
        return []

    rle = []
    current = int(flat[0])
    count = 1

    for value in flat[1:]:
        value = int(value)

        if value == current:
            count += 1
        else:
            rle.append([current, count])
            current = value
            count = 1

    rle.append([current, count])
    return rle


def generate_serpentine_points(mask, rectangles, step_px):
    """
    Top-to-bottom serpentine:
      row 1: left -> right
      row 2: right -> left
      row 3: left -> right
    """
    points = []
    point_id = 0

    for rect in rectangles:
        bbox = rect["expanded_bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        row_index = 0

        for y in range(y1, y2 + 1, step_px):
            if y < 0 or y >= mask.shape[0]:
                continue

            row = mask[y, x1:x2 + 1]
            active_xs = np.where(row > 0)[0]

            if len(active_xs) == 0:
                continue

            active_xs = active_xs + x1

            x_min = int(active_xs.min())
            x_max = int(active_xs.max())

            sampled_xs = list(range(x_min, x_max + 1, step_px))

            if sampled_xs[-1] != x_max:
                sampled_xs.append(x_max)

            if row_index % 2 == 0:
                ordered_xs = sampled_xs
                direction = "left_to_right"
            else:
                ordered_xs = list(reversed(sampled_xs))
                direction = "right_to_left"

            for col_index, x in enumerate(ordered_xs):
                points.append({
                    "point_id": int(point_id),
                    "rect_id": int(rect["id"]),
                    "row_index": int(row_index),
                    "col_index": int(col_index),
                    "direction": direction,
                    "x_px": int(x),
                    "y_px": int(y),
                })
                point_id += 1

            row_index += 1

    return points


def draw_overlay(frame_bgr, mask, contours, rectangles, points):
    overlay = frame_bgr.copy()

    red = np.zeros_like(frame_bgr)
    red[:, :, 2] = mask

    overlay = cv2.addWeighted(overlay, 0.80, red, 0.45, 0)

    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    for rect in rectangles:
        b = rect["expanded_bbox"]
        cv2.rectangle(
            overlay,
            (b["x1"], b["y1"]),
            (b["x2"], b["y2"]),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            overlay,
            f"Rect {rect['id']}",
            (b["x1"], max(20, b["y1"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
        )

    points_by_rect = {}

    for p in points:
        points_by_rect.setdefault(p["rect_id"], []).append(p)

    for _rect_id, pts in points_by_rect.items():
        for i, p in enumerate(pts):
            x = p["x_px"]
            y = p["y_px"]

            cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)

            if i > 0:
                prev = pts[i - 1]
                cv2.line(
                    overlay,
                    (prev["x_px"], prev["y_px"]),
                    (x, y),
                    (255, 0, 255),
                    1,
                )

    coverage = 100.0 * np.sum(mask > 0) / mask.size

    cv2.putText(
        overlay,
        f"Coverage: {coverage:.2f}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        overlay,
        f"Serpentine points: {len(points)}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    return overlay


def process_live_corrosion_image(
    image_path: Path,
    stage_name: str,
    pass_id: int,
    target_id: int,
) -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)

    original = cv2.imread(str(image_path))

    if original is None:
        raise RuntimeError(f"Could not read captured image: {image_path}")

    original_h, original_w = original.shape[:2]

    frame = cv2.resize(original, SEGMENT_RESIZE_TO)
    processed_h, processed_w = frame.shape[:2]

    hsv, enhanced = enhance_image(frame)
    raw_mask = segment_corrosion(hsv)

    filtered_mask, contours = get_filtered_contours(
        raw_mask,
        min_area=SEG_MIN_COMPONENT_AREA,
    )

    boxes = contours_to_boxes(contours)
    merged_boxes = merge_nearby_boxes(
        boxes,
        gap_x=SEG_MERGE_GAP_X,
        gap_y=SEG_MERGE_GAP_Y,
    )

    rectangles = expand_boxes(
        merged_boxes,
        expand_px=SEG_EXPAND_RECT_PX,
        img_w=processed_w,
        img_h=processed_h,
    )

    points = generate_serpentine_points(
        mask=filtered_mask,
        rectangles=rectangles,
        step_px=SPRAY_STEP_PX,
    )

    overlay = draw_overlay(
        enhanced,
        filtered_mask,
        contours,
        rectangles,
        points,
    )

    prefix = f"{safe_name(stage_name)}_pass_{pass_id:02d}_target_{target_id:03d}"

    mask_path = OUTPUT_DIR / f"{prefix}_mask.png"
    overlay_path = OUTPUT_DIR / f"{prefix}_overlay.jpg"
    json_path = OUTPUT_DIR / f"{prefix}_serpentine.json"

    cv2.imwrite(str(mask_path), filtered_mask)
    cv2.imwrite(str(overlay_path), overlay)

    mask_pixels = mask_to_pixels_xy(filtered_mask)

    payload = {
        "detected": len(points) > 0,
        "stage_name": stage_name,
        "pass_id": pass_id,
        "target_id": target_id,
        "source_image": str(image_path),
        "original_image_size": {
            "width": int(original_w),
            "height": int(original_h),
        },
        "processed_image_size": {
            "width": int(processed_w),
            "height": int(processed_h),
        },
        "spray_footprint": {
            "area_px": SPRAY_FOOTPRINT_AREA_PX,
            "equivalent_diameter_px": SPRAY_DIAMETER_PX,
            "radius_px": SPRAY_RADIUS_PX,
            "overlap_pct": SPRAY_OVERLAP_PCT,
            "step_px": SPRAY_STEP_PX,
            "meaning": "step_px is used horizontally and vertically for top-to-bottom serpentine cleaning",
        },
        "segmentation_settings": {
            "hsv_lower": SEG_HSV_LOWER.tolist(),
            "hsv_upper": SEG_HSV_UPPER.tolist(),
            "alpha": SEG_ALPHA,
            "beta": SEG_BETA,
            "clahe_clip": SEG_CLAHE_CLIP,
            "blur": SEG_BLUR,
            "open_iter": SEG_OPEN_ITER,
            "close_iter": SEG_CLOSE_ITER,
            "median_blur": SEG_MEDIAN_BLUR,
            "min_component_area": SEG_MIN_COMPONENT_AREA,
        },
        "mask": {
            "positive_pixel_count": int(len(mask_pixels)),
            "mask_pixels_xy": mask_pixels,
            "rle_row_major": mask_to_rle(filtered_mask),
        },
        "rectangles": rectangles,
        "serpentine_points_px": points,
        "files": {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "json": str(json_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[Segmentation] Saved:")
    print(f"  mask:    {mask_path}")
    print(f"  overlay: {overlay_path}")
    print(f"  json:    {json_path}")
    print(f"[Segmentation] Detected={payload['detected']} | points={len(points)}")

    return payload


# =========================================================
# GIMBAL CALIBRATION + SERVO CONTROL
# =========================================================

_gimbal_calibration_cache = None


def load_gimbal_calibration():
    global _gimbal_calibration_cache

    if _gimbal_calibration_cache is not None:
        return _gimbal_calibration_cache

    path = Path(GIMBAL_CALIBRATION_JSON)

    if not path.exists():
        raise FileNotFoundError(f"Gimbal calibration JSON not found: {path}")

    calib = json.loads(path.read_text(encoding="utf-8"))

    for key in ["theta_pan_center_deg", "theta_pitch_center_deg", "A_inv"]:
        if key not in calib:
            raise KeyError(f"Missing {key} in gimbal calibration file.")

    _gimbal_calibration_cache = calib

    print(f"[Gimbal] Loaded calibration: {path}")
    print(
        f"[Gimbal] center pan={calib['theta_pan_center_deg']} deg | "
        f"center pitch={calib['theta_pitch_center_deg']} deg"
    )

    return calib


def angle_to_pwm(
    angle_deg: float,
    min_pwm: int,
    center_pwm: int,
    max_pwm: int,
    min_angle: float,
    max_angle: float,
) -> int:
    angle_deg = clamp(angle_deg, min_angle, max_angle)

    if angle_deg >= 0.0:
        pwm = center_pwm + (angle_deg / max_angle) * (max_pwm - center_pwm)
    else:
        pwm = center_pwm + (angle_deg / abs(min_angle)) * (center_pwm - min_pwm)

    return int(round(pwm))


def send_servo_pwm(master, servo_num: int, pwm: int):
    print(f"[Gimbal] Servo {servo_num} -> PWM {pwm}")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        int(servo_num),
        int(pwm),
        0, 0, 0, 0, 0,
    )


def convert_processed_pixel_to_original(point: dict, segment: dict) -> Tuple[float, float]:
    x_proc = float(point["x_px"])
    y_proc = float(point["y_px"])

    proc = segment["processed_image_size"]
    orig = segment["original_image_size"]

    proc_w = float(proc["width"])
    proc_h = float(proc["height"])
    orig_w = float(orig["width"])
    orig_h = float(orig["height"])

    x_orig = x_proc * (orig_w / proc_w)
    y_orig = y_proc * (orig_h / proc_h)

    return x_orig, y_orig


def pixel_to_gimbal_command(point: dict, segment: dict) -> dict:
    calib = load_gimbal_calibration()

    x_orig, y_orig = convert_processed_pixel_to_original(point, segment)

    u_c = float(segment["original_image_size"]["width"]) / 2.0
    v_c = float(segment["original_image_size"]["height"]) / 2.0

    du = x_orig - u_c
    dv = y_orig - v_c

    A_inv = calib["A_inv"]

    dpan = float(A_inv[0][0]) * du + float(A_inv[0][1]) * dv
    dpitch = float(A_inv[1][0]) * du + float(A_inv[1][1]) * dv

    pan_center = float(calib["theta_pan_center_deg"])
    pitch_center = float(calib["theta_pitch_center_deg"])

    pan_raw = pan_center + dpan
    pitch_raw = pitch_center + dpitch

    pan_cmd = clamp(pan_raw, PAN_MIN_ANGLE, PAN_MAX_ANGLE)
    pitch_cmd = clamp(pitch_raw, TILT_MIN_ANGLE, TILT_MAX_ANGLE)

    pan_pwm = angle_to_pwm(
        pan_cmd,
        PAN_MIN_PWM,
        PAN_CENTER_PWM,
        PAN_MAX_PWM,
        PAN_MIN_ANGLE,
        PAN_MAX_ANGLE,
    )

    tilt_pwm = angle_to_pwm(
        pitch_cmd,
        TILT_MIN_PWM,
        TILT_CENTER_PWM,
        TILT_MAX_PWM,
        TILT_MIN_ANGLE,
        TILT_MAX_ANGLE,
    )

    return {
        "x_processed_px": float(point["x_px"]),
        "y_processed_px": float(point["y_px"]),
        "x_original_px": x_orig,
        "y_original_px": y_orig,
        "u_center_px": u_c,
        "v_center_px": v_c,
        "delta_u_px": du,
        "delta_v_px": dv,
        "delta_pan_deg": dpan,
        "delta_pitch_deg": dpitch,
        "pan_cmd_deg": pan_cmd,
        "pitch_cmd_deg": pitch_cmd,
        "pan_pwm": pan_pwm,
        "tilt_pwm": tilt_pwm,
    }


def aim_gimbal_to_pixel(master, point: dict, segment: dict) -> bool:
    try:
        cmd = pixel_to_gimbal_command(point, segment)
    except Exception as e:
        print(f"[Gimbal] Could not compute command: {e}")
        return False

    print(
        f"[Gimbal] proc=({cmd['x_processed_px']:.1f},{cmd['y_processed_px']:.1f}) | "
        f"orig=({cmd['x_original_px']:.1f},{cmd['y_original_px']:.1f}) | "
        f"du={cmd['delta_u_px']:+.1f}, dv={cmd['delta_v_px']:+.1f} | "
        f"dpan={cmd['delta_pan_deg']:+.2f}, dpitch={cmd['delta_pitch_deg']:+.2f} | "
        f"pan={cmd['pan_cmd_deg']:+.2f}, pitch={cmd['pitch_cmd_deg']:+.2f}"
    )

    if not GIMBAL_ENABLED:
        print("[Gimbal] Disabled. Not sending servo commands.")
        time.sleep(GIMBAL_SETTLE_S)
        return True

    send_servo_pwm(master, PAN_SERVO, cmd["pan_pwm"])
    send_servo_pwm(master, TILT_SERVO, cmd["tilt_pwm"])

    time.sleep(GIMBAL_SETTLE_S)
    return True


# =========================================================
# ESP32 SPRAY CONTROL
# =========================================================

def send_esp32_command(command: str, expect_reply: bool = True):
    if not ESP32_ENABLED:
        print(f"[ESP32] Disabled. Would send: {command}")
        return True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(ESP32_TIMEOUT_SEC)

    try:
        sock.sendto(command.encode("utf-8"), (ESP32_IP, ESP32_PORT))
        print(f"[ESP32] Sent: {command}")

        if expect_reply:
            data, addr = sock.recvfrom(1024)
            reply = data.decode("utf-8", errors="ignore")
            print(f"[ESP32] Reply from {addr}: {reply}")

        return True

    except socket.timeout:
        if expect_reply:
            print("[ESP32] No reply received.")
        return False

    except Exception as e:
        print(f"[ESP32] Command failed: {e}")
        return False

    finally:
        sock.close()


def sleep_with_esp32_heartbeat(duration_s: float):
    t0 = time.time()
    last_hb = 0.0

    while time.time() - t0 < duration_s:
        now = time.time()

        if now - last_hb >= ESP32_HEARTBEAT_PERIOD_SEC:
            send_esp32_command("hb", expect_reply=False)
            last_hb = now

        time.sleep(0.05)


# =========================================================
# TARGET LOADING + SORTING
# =========================================================

def extract_commanded_target(det: dict) -> Optional[Vec3]:
    target = det.get("commanded_target_ned")

    if isinstance(target, dict):
        return (
            float(target["x_north_m"]),
            float(target["y_east_m"]),
            float(target["z_down_m"]),
        )

    if isinstance(target, list) and len(target) == 3:
        return float(target[0]), float(target[1]), float(target[2])

    extra = det.get("extra", {})
    target = extra.get("commanded_target_ned")

    if isinstance(target, list) and len(target) == 3:
        return float(target[0]), float(target[1]), float(target[2])

    return None


def load_targets(json_path: Path) -> List[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    detections = payload.get("detections", [])

    targets = []

    for i, det in enumerate(detections):
        ned = extract_commanded_target(det)

        if ned is None:
            print(f"[Targets] Skipping detection {i}: no commanded target.")
            continue

        targets.append({
            "id": i,
            "ned": ned,
            "raw": det,
        })

    print(f"[Targets] Loaded {len(targets)} targets.")
    return targets


def sort_targets_nearest_neighbor(targets: List[dict]) -> List[dict]:
    if not targets:
        return []

    remaining = targets[:]
    ordered = []

    current = (0.0, 0.0, 0.0)

    while remaining:
        best_idx = min(
            range(len(remaining)),
            key=lambda i: v3_dist(current, remaining[i]["ned"]),
        )

        nxt = remaining.pop(best_idx)
        ordered.append(nxt)
        current = nxt["ned"]

    print("[Targets] Sorted by nearest-neighbor path.")
    return ordered


# =========================================================
# SPRAY LOGIC
# =========================================================

def group_points_by_rectangle(points: List[dict]):
    groups = []

    rect_ids = []
    for p in points:
        rid = p["rect_id"]
        if rid not in rect_ids:
            rect_ids.append(rid)

    for rid in rect_ids:
        pts = [p for p in points if p["rect_id"] == rid]
        groups.append((rid, pts))

    return groups


def spray_segment(master, stage: dict, target: dict, segment: dict) -> bool:
    if REQUIRE_CORROSION_BEFORE_SPRAY and not segment.get("detected", False):
        print("[Spray] No corrosion detected. Skipping spray.")
        return False

    points = segment.get("serpentine_points_px", [])

    if not points:
        print("[Spray] No serpentine points. Skipping spray.")
        return False

    if MAX_SERPENTINE_POINTS_PER_TARGET is not None:
        points = points[:MAX_SERPENTINE_POINTS_PER_TARGET]

    total_points = len(points)
    dwell_per_point_s = max(0.05, float(stage["spray_duration_s"]) / max(1, total_points))

    print(
        f"[Spray] Target #{target['id']} | stage={stage['label']} | "
        f"points={total_points} | dwell={dwell_per_point_s:.2f}s"
    )

    groups = group_points_by_rectangle(points)
    sprayed_any = False

    for rect_id, rect_points in groups:
        if not rect_points:
            continue

        print(f"[Spray] Rectangle {rect_id}: {len(rect_points)} points")
        print("[Spray] Sequence: aim first -> wait -> s -> wait valve -> move points -> x")

        first = rect_points[0]

        if not aim_gimbal_to_pixel(master, first, segment):
            print(f"[Spray] Could not aim to first point of rectangle {rect_id}. Skipping.")
            continue

        send_esp32_command("s", expect_reply=True)
        sleep_with_esp32_heartbeat(ESP32_START_SEQUENCE_WAIT_S)

        try:
            for i, p in enumerate(rect_points):
                if i == 0:
                    print("[Spray] First point already aimed. Spraying.")
                else:
                    if not aim_gimbal_to_pixel(master, p, segment):
                        print("[Spray] Aim failed for point. Continuing.")
                        continue

                print(
                    f"[Spray] rect={rect_id} | point={p['point_id']} | "
                    f"row={p['row_index']} | {p['direction']} | "
                    f"x={p['x_px']} y={p['y_px']}"
                )

                sleep_with_esp32_heartbeat(dwell_per_point_s)
                sprayed_any = True

        finally:
            send_esp32_command("x", expect_reply=True)
            print(f"[Spray] Rectangle {rect_id} finished. Pump/valve OFF.")

    return sprayed_any


# =========================================================
# LOGGING
# =========================================================

class MissionLogger:
    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        stamp = now_string()

        self.json_path = LOG_DIR / f"cleaning_log_{stamp}.json"
        self.csv_path = LOG_DIR / f"cleaning_log_{stamp}.csv"

        self.records = []

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "stage",
                "pass_id",
                "target_id",
                "target_x",
                "target_y",
                "target_z",
                "pid_stable",
                "corrosion_detected",
                "serpentine_points",
                "sprayed",
                "image",
                "segmentation_json",
            ])

    def add(self, record: dict):
        self.records.append(record)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "count": len(self.records),
                    "records": self.records,
                },
                f,
                indent=2,
            )

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            target = record["target_ned"]

            writer.writerow([
                record["timestamp"],
                record["stage"],
                record["pass_id"],
                record["target_id"],
                target[0],
                target[1],
                target[2],
                int(record["pid_stable"]),
                int(record["corrosion_detected"]),
                record["serpentine_points"],
                int(record["sprayed"]),
                record.get("image", ""),
                record.get("segmentation_json", ""),
            ])

        print(f"[Log] Updated {self.json_path}")
        print(f"[Log] Updated {self.csv_path}")


# =========================================================
# TARGET EXECUTION
# =========================================================

def handle_target(master, ser, stage: dict, pass_id: int, target: dict, logger: MissionLogger):
    target_ned = target["ned"]
    target_alt_m = -target_ned[2]

    print("\n" + "=" * 70)
    print(f"[Target] Stage={stage['label']} | pass={pass_id} | target #{target['id']}")
    print(f"[Target] NED={target_ned}")
    print("=" * 70)

    fly_to_local_point(
        master,
        target_ned,
        yaw_deg=YAW_HOLD_DEG,
        speed_mps=FLY_SPEED_MPS,
    )

    pid_stable = stabilize_before_spray(
        master,
        ser,
        target_alt_m=target_alt_m,
    )

    image_path = None
    segment = None
    sprayed = False
    corrosion_detected = False
    points_count = 0
    seg_json = ""

    if pid_stable:
        image_path = capture_live_frame(
            stage_name=stage["name"],
            pass_id=pass_id,
            target_id=target["id"],
        )

        if image_path is not None:
            segment = process_live_corrosion_image(
                image_path=image_path,
                stage_name=stage["name"],
                pass_id=pass_id,
                target_id=target["id"],
            )

            corrosion_detected = bool(segment.get("detected", False))
            points_count = len(segment.get("serpentine_points_px", []))
            seg_json = segment["files"]["json"]

            if corrosion_detected:
                sprayed = spray_segment(master, stage, target, segment)
            else:
                print("[Target] Segmentation found no corrosion. No spray.")

    record = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage["name"],
        "pass_id": pass_id,
        "target_id": target["id"],
        "target_ned": list(target_ned),
        "pid_stable": pid_stable,
        "corrosion_detected": corrosion_detected,
        "serpentine_points": points_count,
        "sprayed": sprayed,
        "image": str(image_path) if image_path is not None else "",
        "segmentation_json": seg_json,
    }

    logger.add(record)


def run_targets_once(master, ser, targets: List[dict], stage: dict, pass_id: int, logger: MissionLogger):
    for target in targets:
        handle_target(master, ser, stage, pass_id, target, logger)


# =========================================================
# STAGE EXECUTION
# =========================================================

def run_rust_remover_stage(master, ser, targets: List[dict], stage: dict, logger: MissionLogger):
    print("\n" + "#" * 70)
    print("[Stage] Rust remover repeated-pass logic")
    print("#" * 70)

    exposure_start = time.time()
    pass_id = 1

    while True:
        first_alt = max(1.0, -targets[0]["ned"][2])
        ensure_airborne(master, first_alt)

        print(f"[Stage] Rust remover pass {pass_id}")
        run_targets_once(master, ser, targets, stage, pass_id, logger)

        elapsed = time.time() - exposure_start
        remaining = float(stage["total_exposure_s"]) - elapsed

        print(f"[Stage] Rust remover elapsed={elapsed:.1f}s | remaining={remaining:.1f}s")

        if remaining <= 0:
            print("[Stage] Rust remover total exposure complete.")
            break

        wait_s = min(float(stage["respray_interval_s"]), remaining)

        print(f"[Stage] RTL during rust remover wait interval: {wait_s:.1f}s")
        rtl(master)

        t0 = time.time()
        while time.time() - t0 < wait_s:
            left = wait_s - (time.time() - t0)
            print(f"[Stage] Waiting before next rust remover pass: {left:.0f}s remaining")
            time.sleep(min(30.0, max(1.0, left)))

        pass_id += 1

    rtl(master)


def run_regular_stage(master, ser, targets: List[dict], stage: dict, logger: MissionLogger):
    first_alt = max(1.0, -targets[0]["ned"][2])
    ensure_airborne(master, first_alt)

    run_targets_once(master, ser, targets, stage, pass_id=1, logger=logger)

    if RTL_BETWEEN_STAGES:
        rtl(master)


def run_stage(master, ser, targets: List[dict], stage_name: str, logger: MissionLogger):
    stage = STAGES[stage_name]

    print("\n" + "#" * 70)
    print(f"[Stage] Starting: {stage['label']}")
    print(f"[Stage] Solution: {stage['solution']}")
    print("#" * 70)

    if stage.get("rust_repeated_passes", False):
        run_rust_remover_stage(master, ser, targets, stage, logger)
    else:
        run_regular_stage(master, ser, targets, stage, logger)

    print(f"[Stage] Finished: {stage['label']}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "detections_json",
        help="Path to corrosion_detections.json from scanning stage",
    )

    parser.add_argument(
        "stage",
        choices=["all"] + ALL_STAGE_ORDER,
        help="Which cleaning stage to run",
    )

    parser.add_argument(
        "--disable-esp32",
        action="store_true",
        help="Dry run without pump/valve commands",
    )

    parser.add_argument(
        "--disable-gimbal",
        action="store_true",
        help="Dry run without servo commands",
    )

    parser.add_argument(
        "--disable-pid",
        action="store_true",
        help="Skip PID stabilization",
    )

    parser.add_argument(
        "--no-ultrasonic-required",
        action="store_true",
        help="Allow height-only PID if ultrasonic serial is unavailable",
    )

    args = parser.parse_args()

    global ESP32_ENABLED
    global GIMBAL_ENABLED
    global PID_ENABLED
    global PID_REQUIRE_ULTRASONIC

    if args.disable_esp32:
        ESP32_ENABLED = False

    if args.disable_gimbal:
        GIMBAL_ENABLED = False

    if args.disable_pid:
        PID_ENABLED = False

    if args.no_ultrasonic_required:
        PID_REQUIRE_ULTRASONIC = False

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    detections_path = Path(args.detections_json)

    if not detections_path.exists():
        raise FileNotFoundError(f"Detections JSON not found: {detections_path}")

    targets = load_targets(detections_path)
    targets = sort_targets_nearest_neighbor(targets)

    if not targets:
        raise RuntimeError("No valid targets found.")

    ser = open_ultrasonic_serial()

    try:
        master = connect_mavlink()
        logger = MissionLogger()

        if args.stage == "all":
            stage_sequence = ALL_STAGE_ORDER
        else:
            stage_sequence = [args.stage]

        for idx, stage_name in enumerate(stage_sequence):
            run_stage(master, ser, targets, stage_name, logger)

            if args.stage == "all" and PROMPT_BETWEEN_STAGES and idx < len(stage_sequence) - 1:
                next_stage = STAGES[stage_sequence[idx + 1]]
                print("\n" + "=" * 70)
                print(f"[User] Finished {STAGES[stage_name]['label']}.")
                print(f"[User] Please change solution to: {next_stage['solution']}")
                print(f"[User] Next stage: {next_stage['label']}")
                input("[User] Press ENTER when ready to continue...")
                print("=" * 70)

        print("[Main] Cleaning mission complete.")
        rtl(master)

    except KeyboardInterrupt:
        print("\n[Main] Stopped by user.")
        try:
            send_esp32_command("x", expect_reply=True)
            rtl(master)
        except Exception:
            pass

    finally:
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

        print("[Main] Done.")


if __name__ == "__main__":
    main()