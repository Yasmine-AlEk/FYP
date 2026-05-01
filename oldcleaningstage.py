import argparse
import csv
import json
import math
import socket
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import serial
from pymavlink import mavutil


# =========================================================
# TYPE ALIASES
# =========================================================

Vec3 = Tuple[float, float, float]


# =========================================================
# CONNECTION CONFIG
# =========================================================

# Jetson receives MAVLink from WSL/MAVProxy on port 14550.
MAVLINK_URL = "udpin:0.0.0.0:14550"

# Arduino ultrasonic serial port.
SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 9600
SERIAL_TIMEOUT_S = 0.02
SENSOR_LOST_TIMEOUT_S = 0.50

# ESP32 pump/solenoid UDP.
ESP32_ENABLED = True
ESP32_IP = "172.20.10.14"   # change if your ESP32 IP changes
ESP32_PORT = 4210
ESP32_TIMEOUT_S = 2.0


# =========================================================
# CLEANING BEHAVIOR CONFIG
# =========================================================

# True = short timings for demo/testing.
# False = closer to real treatment timings.
DEMO_TIMING = True

# If True, the system will not spray unless distance/yaw/height PID becomes stable.
REQUIRE_STABLE_BEFORE_SPRAY = True

# If True, use the scan-time saved mask/regions as a placeholder for cleaning segmentation.
# The real final system should replace this with live segmentation at the target.
USE_SCAN_REGIONS_FOR_DEMO = True

# Gimbal/nozzle control is left as a hook until calibration matrix is connected.
GIMBAL_ENABLED = False

# Return to launch after each stage.
RTL_AFTER_EACH_STAGE = True

# During rust remover repeated passes, return home between passes or just wait in place.
RETURN_HOME_BETWEEN_RUST_PASSES = True

# Local yaw while visiting targets.
YAW_HOLD_DEG = 0.0

# Movement tuning.
NAV_SPEED_MPS = 0.5
NAV_HZ = 10.0
TARGET_HOLD_S = 1.0

# Output logs.
CLEANING_RUN_LOG_JSON = "cleaning_run_log.json"
TARGET_VISIT_LOG_CSV = "cleaning_target_visits.csv"
PID_LOG_CSV = "cleaning_pid_log.csv"


# =========================================================
# STAGE CONFIG
# =========================================================

if DEMO_TIMING:
    CLEANING_STAGES = {
        "rust_remover": {
            "label": "1) Rust Remover",
            "solution": "Citric acid 5% w/v",
            "spray_duration_s": 3.0,
            "total_exposure_s": 40.0,
            "respray_interval_s": 10.0,
            "repeated_passes": True,
        },
        "water_rinse": {
            "label": "2) Water Rinse",
            "solution": "Deionized (DI) water",
            "spray_duration_s": 3.0,
            "repeated_passes": False,
        },
        "neutralizer": {
            "label": "3) Neutralizer",
            "solution": "Sodium bicarbonate (NaHCO3) 1% w/v",
            "spray_duration_s": 3.0,
            "repeated_passes": False,
        },
        "final_water_rinse": {
            "label": "4) Final Water Rinse",
            "solution": "Deionized (DI) water",
            "spray_duration_s": 3.0,
            "repeated_passes": False,
        },
        "inhibitor": {
            "label": "5) Inhibitor",
            "solution": "Sodium benzoate 1% w/v",
            "spray_duration_s": 3.0,
            "repeated_passes": False,
        },
    }
else:
    CLEANING_STAGES = {
        "rust_remover": {
            "label": "1) Rust Remover",
            "solution": "Citric acid 5% w/v",
            "spray_duration_s": 8.0,
            "total_exposure_s": 20.0 * 60.0,
            "respray_interval_s": 4.0 * 60.0,
            "repeated_passes": True,
        },
        "water_rinse": {
            "label": "2) Water Rinse",
            "solution": "Deionized (DI) water",
            "spray_duration_s": 60.0,
            "repeated_passes": False,
        },
        "neutralizer": {
            "label": "3) Neutralizer",
            "solution": "Sodium bicarbonate (NaHCO3) 1% w/v",
            "spray_duration_s": 120.0,
            "repeated_passes": False,
        },
        "final_water_rinse": {
            "label": "4) Final Water Rinse",
            "solution": "Deionized (DI) water",
            "spray_duration_s": 60.0,
            "repeated_passes": False,
        },
        "inhibitor": {
            "label": "5) Inhibitor",
            "solution": "Sodium benzoate 1% w/v",
            "spray_duration_s": 8.0,
            "repeated_passes": False,
        },
    }

STAGE_ORDER = [
    "rust_remover",
    "water_rinse",
    "neutralizer",
    "final_water_rinse",
    "inhibitor",
]


# =========================================================
# PID CONFIG
# =========================================================

TARGET_STANDOFF_M = 2.0
DIST_DEADBAND_M = 0.10
YAW_DIFF_DEADBAND_M = 0.03
HEIGHT_DEADBAND_M = 0.10

PID_HZ = 20.0
PID_MAX_TIME_S = 8.0
PID_STABLE_HOLD_TIME_S = 1.0
PID_PRINT_EVERY_N = 1

SENSOR_AVG_WINDOW = 3
SENSOR_MIN_VALID_SAMPLES = 2

KP_DIST, KI_DIST, KD_DIST = 0.80, 0.00, 0.05
KP_YAW, KI_YAW, KD_YAW = 1.20, 0.00, 0.08
KP_HEIGHT, KI_HEIGHT, KD_HEIGHT = 0.60, 0.00, 0.05

MAX_VX_MPS = 0.35
MAX_YAW_RATE_RAD_S = 0.50
MAX_VZ_MPS = 0.30

# Flip signs if correction direction is wrong.
DIST_SIGN = 1.0
YAW_SIGN = -1.0
HEIGHT_SIGN = 1.0


# =========================================================
# PID CLASS
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

        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt

        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))

        self.prev_error = error
        self.prev_time = now
        return output


# =========================================================
# ULTRASONIC HELPERS
# =========================================================

def parse_value(value: str) -> Optional[float]:
    value = value.strip()
    if value in ["OUT_OF_RANGE", "N/A", ""]:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def open_ultrasonic_serial() -> Optional[serial.Serial]:
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


def read_ultrasonic_sensors_cm(ser, max_attempts: int = 5) -> Optional[Tuple[float, float, float]]:
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

        sensor1 = parse_value(parts[0])
        sensor2 = parse_value(parts[1])
        error = parse_value(parts[2])

        if sensor1 is None or sensor2 is None:
            continue
        if error is None:
            error = sensor1 - sensor2

        latest_valid = (sensor1, sensor2, error)

    return latest_valid


class UltrasonicRollingAverage:
    def __init__(self, window_size: int, min_samples: int):
        self.s1_buffer = deque(maxlen=window_size)
        self.s2_buffer = deque(maxlen=window_size)
        self.min_samples = min_samples

    def update(self, sensor1_cm: float, sensor2_cm: float):
        self.s1_buffer.append(sensor1_cm)
        self.s2_buffer.append(sensor2_cm)

        if len(self.s1_buffer) < self.min_samples:
            return None

        avg_s1_cm = sum(self.s1_buffer) / len(self.s1_buffer)
        avg_s2_cm = sum(self.s2_buffer) / len(self.s2_buffer)
        avg_error_cm = avg_s1_cm - avg_s2_cm

        return avg_s1_cm, avg_s2_cm, avg_error_cm


# =========================================================
# MAVLINK HELPERS
# =========================================================

_T0_MS = None


def now_ms_u32() -> int:
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
    print(f"[MAVLink] Connected: system={master.target_system}, component={master.target_component}")
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
        target_alt_m,
    )
    print(f"[MAVLink] Taking off to {target_alt_m:.2f} m...")
    time.sleep(1.0)


def get_local_position(master, timeout_s: float = 1.0) -> Optional[Vec3]:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)
    if msg is None:
        return None
    return float(msg.x), float(msg.y), float(msg.z)


def wait_until_altitude(master, target_alt_m: float, tolerance_m: float = 0.25, timeout_s: float = 45.0) -> bool:
    print("[MAVLink] Waiting for altitude...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pos = get_local_position(master, timeout_s=1.0)
        if pos is None:
            continue
        alt_m = -pos[2]
        print(f"[Alt] current={alt_m:.2f}m | target={target_alt_m:.2f}m")
        if abs(alt_m - target_alt_m) <= tolerance_m:
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


def send_body_velocity_yawrate(master, vx: float, vy: float, vz: float, yaw_rate: float):
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
    send_body_velocity_yawrate(master, 0.0, 0.0, 0.0, 0.0)


def rtl(master):
    try:
        set_mode(master, "RTL")
    except Exception:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )
        print("[MAVLink] RTL command sent.")


def hold_local_point(master, target_xyz: Vec3, duration_s: float):
    t0 = time.time()
    dt = 1.0 / NAV_HZ
    while time.time() - t0 < duration_s:
        send_local_position(master, target_xyz[0], target_xyz[1], target_xyz[2], YAW_HOLD_DEG)
        time.sleep(dt)


def fly_line_local(master, start: Vec3, end: Vec3, speed_mps: float = NAV_SPEED_MPS):
    sx, sy, sz = start
    ex, ey, ez = end
    dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)
    total_time = max(1.0, dist / max(speed_mps, 0.05))
    steps = max(2, int(total_time * NAV_HZ))
    dt = 1.0 / NAV_HZ

    for k in range(steps + 1):
        a = k / steps
        x = sx + a * (ex - sx)
        y = sy + a * (ey - sy)
        z = sz + a * (ez - sz)
        send_local_position(master, x, y, z, YAW_HOLD_DEG)
        time.sleep(dt)


def fly_to_local_point(master, target_xyz: Vec3):
    current = get_local_position(master, timeout_s=2.0)
    if current is None:
        print("[NAV] No current position. Holding target command directly.")
        hold_local_point(master, target_xyz, duration_s=3.0)
        return

    print(f"[NAV] Flying to target: {target_xyz}")
    fly_line_local(master, current, target_xyz, NAV_SPEED_MPS)
    hold_local_point(master, target_xyz, TARGET_HOLD_S)


# =========================================================
# TARGET EXTRACTION + SORTING
# =========================================================

def distance3(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def load_detection_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_corrosion_targets(payload: dict) -> List[dict]:
    detections = payload.get("detections", [])
    targets = []

    for idx, det in enumerate(detections):
        extra = det.get("extra", {}) or {}
        target_xyz = None
        source = None

        cmd_dict = det.get("commanded_target_ned")
        if isinstance(cmd_dict, dict):
            target_xyz = (
                float(cmd_dict["x_north_m"]),
                float(cmd_dict["y_east_m"]),
                float(cmd_dict["z_down_m"]),
            )
            source = "commanded_target_ned"

        if target_xyz is None:
            cmd_list = extra.get("commanded_target_ned")
            if isinstance(cmd_list, list) and len(cmd_list) == 3:
                target_xyz = (float(cmd_list[0]), float(cmd_list[1]), float(cmd_list[2]))
                source = "extra.commanded_target_ned"

        if target_xyz is None:
            actual = det.get("actual_local_position_ned")
            if isinstance(actual, dict):
                target_xyz = (
                    float(actual["x_north_m"]),
                    float(actual["y_east_m"]),
                    float(actual["z_down_m"]),
                )
                source = "actual_local_position_ned"

        if target_xyz is None:
            print(f"[Targets] Skipping detection #{idx}: no valid target position.")
            continue

        targets.append({
            "id": idx,
            "target_xyz": target_xyz,
            "target_source": source,
            "timestamp_s": det.get("timestamp_s"),
            "corrosion_percent": extra.get("corrosion_percent"),
            "severity_grade": extra.get("severity_grade"),
            "regions": extra.get("regions", []),
            "vision_files": extra.get("vision_files"),
            "raw_detection": det,
            "extra": extra,
        })

    if not targets:
        raise RuntimeError("No valid cleaning targets found.")

    print(f"[Targets] Extracted {len(targets)} targets:")
    for t in targets:
        print(f"  #{t['id']} {t['target_xyz']} | source={t['target_source']} | severity={t['severity_grade']}")

    return targets


def sort_targets_nearest_neighbor(targets: List[dict], start_xyz: Vec3) -> List[dict]:
    remaining = targets.copy()
    ordered = []
    current = start_xyz

    while remaining:
        nearest = min(remaining, key=lambda t: distance3(current, t["target_xyz"]))
        ordered.append(nearest)
        remaining.remove(nearest)
        current = nearest["target_xyz"]

    return ordered


# =========================================================
# PID STABILITY AT EACH TARGET
# =========================================================

def apply_deadband(error: float, deadband: float) -> float:
    return 0.0 if abs(error) <= deadband else error


def run_target_pid_stabilization(master, ser, target_xyz: Vec3) -> bool:
    if ser is None:
        print("[PID] No ultrasonic serial. Cannot verify distance/yaw stability.")
        return False

    desired_alt_m = -target_xyz[2]

    dist_pid = PID(KP_DIST, KI_DIST, KD_DIST, output_limit=MAX_VX_MPS, integral_limit=1.0)
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW, output_limit=MAX_YAW_RATE_RAD_S, integral_limit=1.0)
    height_pid = PID(KP_HEIGHT, KI_HEIGHT, KD_HEIGHT, output_limit=MAX_VZ_MPS, integral_limit=1.0)

    sensor_filter = UltrasonicRollingAverage(SENSOR_AVG_WINDOW, SENSOR_MIN_VALID_SAMPLES)

    dt = 1.0 / PID_HZ
    t0 = time.time()
    stable_start = None
    last_sensor_time = time.time()
    loop_count = 0

    csv_exists = Path(PID_LOG_CSV).exists()
    with open(PID_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "timestamp_s", "target_x", "target_y", "target_z",
                "avg_distance_m", "distance_error_m", "vx_cmd_mps",
                "yaw_error_m", "yaw_rate_cmd_rad_s",
                "current_alt_m", "alt_ref_m", "alt_error_m", "vz_cmd_mps",
                "stable",
            ])

        print("[PID] Stabilizing before spray...")

        while time.time() - t0 < PID_MAX_TIME_S:
            loop_count += 1

            reading = read_ultrasonic_sensors_cm(ser)
            if reading is None:
                if time.time() - last_sensor_time > SENSOR_LOST_TIMEOUT_S:
                    print("[PID] Sensor timeout. Zeroing correction.")
                    stop_motion(master)
                time.sleep(0.005)
                continue

            last_sensor_time = time.time()
            raw_s1_cm, raw_s2_cm, _raw_err_cm = reading
            filtered = sensor_filter.update(raw_s1_cm, raw_s2_cm)

            if filtered is None:
                print(f"[PID] Averaging sensors... s1={raw_s1_cm:.1f}cm | s2={raw_s2_cm:.1f}cm")
                stop_motion(master)
                time.sleep(dt)
                continue

            s1_cm, s2_cm, yaw_error_cm = filtered
            s1_m = s1_cm / 100.0
            s2_m = s2_cm / 100.0
            avg_distance_m = (s1_m + s2_m) / 2.0

            distance_error_m = avg_distance_m - TARGET_STANDOFF_M
            yaw_error_m = yaw_error_cm / 100.0

            dist_ctrl = apply_deadband(distance_error_m, DIST_DEADBAND_M)
            yaw_ctrl = apply_deadband(yaw_error_m, YAW_DIFF_DEADBAND_M)

            vx = DIST_SIGN * dist_pid.update(dist_ctrl)
            yaw_rate = YAW_SIGN * yaw_pid.update(yaw_ctrl)

            pos = get_local_position(master, timeout_s=0.2)
            if pos is None:
                current_alt_m = None
                alt_error_m = 0.0
                vz = 0.0
                height_stable = False
            else:
                current_alt_m = -pos[2]
                alt_error_m = current_alt_m - desired_alt_m
                alt_ctrl = apply_deadband(alt_error_m, HEIGHT_DEADBAND_M)
                vz = HEIGHT_SIGN * height_pid.update(alt_ctrl)
                height_stable = abs(alt_error_m) <= HEIGHT_DEADBAND_M

            distance_stable = abs(distance_error_m) <= DIST_DEADBAND_M
            yaw_stable = abs(yaw_error_m) <= YAW_DIFF_DEADBAND_M
            stable = distance_stable and yaw_stable and height_stable

            if stable:
                if stable_start is None:
                    stable_start = time.time()
            else:
                stable_start = None

            send_body_velocity_yawrate(master, vx=vx, vy=0.0, vz=vz, yaw_rate=yaw_rate)

            writer.writerow([
                round(time.time(), 3),
                round(target_xyz[0], 4), round(target_xyz[1], 4), round(target_xyz[2], 4),
                round(avg_distance_m, 4), round(distance_error_m, 4), round(vx, 4),
                round(yaw_error_m, 4), round(yaw_rate, 4),
                round(current_alt_m, 4) if current_alt_m is not None else None,
                round(desired_alt_m, 4), round(alt_error_m, 4), round(vz, 4),
                int(stable),
            ])
            f.flush()

            if loop_count % PID_PRINT_EVERY_N == 0:
                alt_text = f"{current_alt_m:.2f}m" if current_alt_m is not None else "N/A"
                print(
                    f"[PID] dist={avg_distance_m:.2f}m | dist_err={distance_error_m:+.2f}m | vx={vx:+.2f}m/s || "
                    f"yaw_err={yaw_error_m:+.2f}m | yaw_rate={yaw_rate:+.2f}rad/s || "
                    f"alt={alt_text} | alt_ref={desired_alt_m:.2f}m | alt_err={alt_error_m:+.2f}m | vz={vz:+.2f}m/s | "
                    f"stable={stable}"
                )

            if stable_start is not None and time.time() - stable_start >= PID_STABLE_HOLD_TIME_S:
                stop_motion(master)
                print("[PID] Stable. Ready to spray.")
                return True

            time.sleep(dt)

    stop_motion(master)
    print("[PID] Stability timeout. Not stable.")
    return False


# =========================================================
# SEGMENTATION / GIMBAL / SPRAY HOOKS
# =========================================================

def run_cleaning_segmentation(target: dict) -> Optional[dict]:
    """
    Placeholder for live cleaning segmentation.

    Current implementation uses saved scan-time regions for demo.
    Replace this with live camera segmentation at the cleaning target.
    """
    regions = target.get("regions", []) or []

    if not regions:
        print("[Segment] No saved regions available.")
        return None

    largest = max(regions, key=lambda r: r.get("area_pixels", 0))
    centroid = largest.get("centroid", {})

    result = {
        "source": "saved_scan_regions_demo",
        "bbox": largest.get("bbox"),
        "centroid_px": (centroid.get("x"), centroid.get("y")),
        "area_pixels": largest.get("area_pixels"),
        "dominant_severity": largest.get("dominant_severity"),
    }

    print(
        f"[Segment] Using saved scan region: centroid={result['centroid_px']} | "
        f"area_px={result['area_pixels']} | severity={result['dominant_severity']}"
    )
    return result


def aim_gimbal_to_segment(segment: Optional[dict]) -> bool:
    if segment is None:
        print("[Gimbal] No segment to aim at.")
        return False

    if not GIMBAL_ENABLED:
        print("[Gimbal] Placeholder: would aim hose to mask/centroid now.")
        return True

    # TODO: connect calibration matrix + servo_angle_control.py here.
    print("[Gimbal] TODO: calibrated gimbal control not connected yet.")
    return False


def send_esp32_command(command: str) -> Optional[str]:
    if not ESP32_ENABLED:
        print(f"[ESP32] Disabled. Would send: {command}")
        return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(ESP32_TIMEOUT_S)

    try:
        sock.sendto(command.encode("utf-8"), (ESP32_IP, ESP32_PORT))
        try:
            data, addr = sock.recvfrom(1024)
            reply = data.decode("utf-8", errors="ignore")
            print(f"[ESP32] {command} -> reply from {addr}: {reply}")
            return reply
        except socket.timeout:
            print(f"[ESP32] {command} sent. No reply.")
            return None
    finally:
        sock.close()


def spray_target(stage: dict, target: dict, segment: Optional[dict]) -> bool:
    print(f"[Spray] Stage: {stage['label']} | Solution: {stage['solution']}")
    print(f"[Spray] Duration: {stage['spray_duration_s']:.1f}s")

    aimed = aim_gimbal_to_segment(segment)
    if not aimed:
        print("[Spray] Could not aim gimbal. Skipping spray.")
        return False

    send_esp32_command("s")
    time.sleep(float(stage["spray_duration_s"]))
    send_esp32_command("x")

    print(f"[Spray] Done target #{target['id']}")
    return True


# =========================================================
# LOGGING
# =========================================================

def append_run_log(event: dict):
    path = Path(CLEANING_RUN_LOG_JSON)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {"events": []}
    else:
        payload = {"events": []}

    payload["events"].append(event)
    path.write_text(json.dumps(payload, indent=2))


def append_visit_csv(row: dict):
    path = Path(TARGET_VISIT_LOG_CSV)
    exists = path.exists()

    fieldnames = [
        "timestamp_s", "stage_key", "stage_label", "pass_number", "target_id",
        "target_x", "target_y", "target_z", "pid_stable", "sprayed", "corrosion_percent", "severity_grade",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================
# CLEANING MISSION LOGIC
# =========================================================

def ensure_guided_airborne(master, target_alt_m: float):
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, target_alt_m)
    wait_until_altitude(master, target_alt_m)


def visit_and_treat_targets_once(
    master,
    ser,
    targets: List[dict],
    stage_key: str,
    stage: dict,
    pass_number: int,
):
    current = get_local_position(master, timeout_s=2.0)
    if current is None:
        current = (0.0, 0.0, 0.0)

    ordered_targets = sort_targets_nearest_neighbor(targets, current)

    print(f"[Stage] Visiting {len(ordered_targets)} targets using nearest-neighbor order.")

    for target in ordered_targets:
        target_xyz = target["target_xyz"]

        print("\n" + "=" * 70)
        print(f"[Target] Stage={stage['label']} | Pass={pass_number} | Target #{target['id']} -> {target_xyz}")
        print("=" * 70)

        fly_to_local_point(master, target_xyz)

        pid_stable = run_target_pid_stabilization(master, ser, target_xyz)

        if REQUIRE_STABLE_BEFORE_SPRAY and not pid_stable:
            print("[Target] PID not stable. Skipping spray for this target.")
            sprayed = False
            segment = None
        else:
            segment = run_cleaning_segmentation(target)
            sprayed = spray_target(stage, target, segment)

        event = {
            "timestamp_s": time.time(),
            "stage_key": stage_key,
            "stage_label": stage["label"],
            "solution": stage["solution"],
            "pass_number": pass_number,
            "target_id": target["id"],
            "target_xyz": target_xyz,
            "pid_stable": pid_stable,
            "segment": segment,
            "sprayed": sprayed,
            "corrosion_percent": target.get("corrosion_percent"),
            "severity_grade": target.get("severity_grade"),
        }
        append_run_log(event)

        append_visit_csv({
            "timestamp_s": round(event["timestamp_s"], 3),
            "stage_key": stage_key,
            "stage_label": stage["label"],
            "pass_number": pass_number,
            "target_id": target["id"],
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "pid_stable": int(pid_stable),
            "sprayed": int(sprayed),
            "corrosion_percent": target.get("corrosion_percent"),
            "severity_grade": target.get("severity_grade"),
        })


def run_rust_remover_stage(master, ser, targets: List[dict], stage_key: str, stage: dict):
    print("\n" + "#" * 80)
    print(f"[Stage] Starting {stage['label']} | {stage['solution']}")
    print(f"[Stage] Rust remover total exposure: {stage['total_exposure_s']:.1f}s")
    print(f"[Stage] Respray interval: {stage['respray_interval_s']:.1f}s")
    print("#" * 80)

    stage_start = time.time()
    pass_number = 1

    while True:
        elapsed = time.time() - stage_start
        if elapsed >= stage["total_exposure_s"]:
            break

        print(f"\n[Stage] Rust remover pass {pass_number} | elapsed={elapsed:.1f}s")
        visit_and_treat_targets_once(master, ser, targets, stage_key, stage, pass_number)

        elapsed = time.time() - stage_start
        remaining = stage["total_exposure_s"] - elapsed
        if remaining <= 0:
            break

        wait_s = min(stage["respray_interval_s"], remaining)

        if RETURN_HOME_BETWEEN_RUST_PASSES:
            print("[Stage] Returning home between rust remover passes.")
            rtl(master)
            print(f"[Stage] Waiting {wait_s:.1f}s before next rust remover pass.")
            time.sleep(wait_s)
            first_alt = -targets[0]["target_xyz"][2]
            ensure_guided_airborne(master, first_alt)
        else:
            print(f"[Stage] Waiting {wait_s:.1f}s before next rust remover pass.")
            time.sleep(wait_s)

        pass_number += 1

    print("[Stage] Rust remover exposure complete.")


def run_single_pass_stage(master, ser, targets: List[dict], stage_key: str, stage: dict):
    print("\n" + "#" * 80)
    print(f"[Stage] Starting {stage['label']} | {stage['solution']}")
    print("#" * 80)

    visit_and_treat_targets_once(master, ser, targets, stage_key, stage, pass_number=1)


def run_stage(master, ser, targets: List[dict], stage_key: str):
    stage = CLEANING_STAGES[stage_key]

    first_alt = -targets[0]["target_xyz"][2]
    ensure_guided_airborne(master, first_alt)

    if stage.get("repeated_passes", False):
        run_rust_remover_stage(master, ser, targets, stage_key, stage)
    else:
        run_single_pass_stage(master, ser, targets, stage_key, stage)

    if RTL_AFTER_EACH_STAGE:
        print(f"[Stage] {stage['label']} complete. Sending RTL.")
        rtl(master)


def run_all_stages(master, ser, targets: List[dict]):
    for idx, stage_key in enumerate(STAGE_ORDER):
        stage = CLEANING_STAGES[stage_key]
        print("\n" + "=" * 80)
        print(f"Next stage: {stage['label']}")
        print(f"Solution required: {stage['solution']}")
        print("=" * 80)

        if idx > 0:
            input("Change/refill the solution, then press ENTER to continue...")

        run_stage(master, ser, targets, stage_key)

    print("\n[Mission] All cleaning stages complete.")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detections_json", help="Path to corrosion_detections.json")
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        help="Stage key or 'all'. Options: all, " + ", ".join(STAGE_ORDER),
    )
    args = parser.parse_args()

    if args.stage != "all" and args.stage not in CLEANING_STAGES:
        raise ValueError(f"Unknown stage {args.stage}. Use all or one of: {STAGE_ORDER}")

    payload = load_detection_json(args.detections_json)
    targets = extract_corrosion_targets(payload)

    ser = None
    master = None

    try:
        ser = open_ultrasonic_serial()
        master = connect_mavlink()

        if args.stage == "all":
            run_all_stages(master, ser, targets)
        else:
            run_stage(master, ser, targets, args.stage)

    except KeyboardInterrupt:
        print("\n[Main] Stopped by user.")

    finally:
        if master is not None:
            try:
                stop_motion(master)
            except Exception:
                pass

        if ser is not None:
            try:
                ser.close()
                print("[Serial] Closed ultrasonic serial.")
            except Exception:
                pass

        print("[Main] Done.")


if __name__ == "__main__":
    main()