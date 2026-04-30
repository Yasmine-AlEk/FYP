import csv
import math
import serial
import time
from collections import deque
from typing import Optional, Tuple

from pymavlink import mavutil


# =========================================================
# CONNECTION CONFIG
# =========================================================

# Jetson receives MAVLink from WSL/MAVProxy on port 14550.
MAVLINK_URL = "udpin:0.0.0.0:14550"

# Arduino ultrasonic serial port.
SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 9600


# =========================================================
# TEST CONFIG
# =========================================================

TARGET_ALT_M = 2.0

# Desired wall standoff distance.
TARGET_STANDOFF_M = 2.0

# If distance error is less than or equal to 10 cm, ignore distance correction.
DIST_DEADBAND_M = 0.10

# Make yaw less conservative.
# If the ultrasonic difference is more than 3 cm, correct yaw.
YAW_DIFF_DEADBAND_M = 0.03

CONTROL_HZ = 20.0

# Print every control loop.
# CONTROL_HZ = 10, so this prints 10 times per second.
PRINT_EVERY_N = 1

# None means run forever until CTRL+C.
MAX_TEST_TIME_S = None

# Keep running even when stable.
STOP_WHEN_STABLE = False
STABLE_HOLD_TIME_S = 2.0

LOG_CSV = "pid_wall_align_log.csv"

# Print every N control loops.
# CONTROL_HZ = 10, so PRINT_EVERY_N = 5 means print about 2 times per second.
PRINT_EVERY_N = 1
SERIAL_TIMEOUT_S = 0.02
SENSOR_LOST_TIMEOUT_S = 0.50

# =========================================================
# PID GAINS
# Start gentle. Tune later.
# =========================================================

KP_DIST = 0.80
KI_DIST = 0.00
KD_DIST = 0.05

KP_YAW = 1.20
KI_YAW = 0.00
KD_YAW = 0.08

MAX_YAW_RATE_RAD_S = 0.50

MAX_VX_MPS = 0.35
MAX_YAW_RATE_RAD_S = 0.50


# =========================================================
# SENSOR FILTER CONFIG
# =========================================================

# Number of ultrasonic readings to average.
# At 10 Hz, window=5 gives around 0.5 s smoothing.
SENSOR_AVG_WINDOW = 3

# Wait until we have at least this many valid samples before controlling.
SENSOR_MIN_VALID_SAMPLES = 2


# =========================================================
# SIGN CONFIG
# =========================================================
# If the drone is too far from the wall and moves farther away, flip DIST_SIGN.
# If yaw correction makes the ultrasonic difference worse, flip YAW_SIGN.
# Use -1.0 if direction is wrong.

DIST_SIGN = 1.0
YAW_SIGN = -1.0


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

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = max(
                -self.integral_limit,
                min(self.integral_limit, self.integral),
            )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))

        self.prev_error = error
        self.prev_time = now

        return output


# =========================================================
# SERIAL / ULTRASONIC HELPERS
# =========================================================

def parse_value(value: str) -> Optional[float]:
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

    print(f"[Serial] Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud.")
    print(f"[Serial] Timeout: {SERIAL_TIMEOUT_S}s")
    return ser


def read_ultrasonic_sensors_cm(
    ser,
    max_attempts: int = 5,
) -> Optional[Tuple[float, float, float]]:
    """
    Fast reader.

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
    def __init__(self, window_size: int = 5, min_samples: int = 3):
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

        # Recompute yaw error from filtered values.
        # Positive means sensor1 is farther than sensor2.
        avg_error_cm = avg_s1_cm - avg_s2_cm

        return avg_s1_cm, avg_s2_cm, avg_error_cm


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
    print(f"[MAVLink] Connecting using {MAVLINK_URL}")
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
        raise RuntimeError(
            f"Mode {mode} not available. Available modes: {list(mode_map.keys())}"
        )

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


def get_local_position(master, timeout_s: float = 1.0):
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
    print("[MAVLink] Waiting until takeoff altitude is reached...")

    t0 = time.time()

    while time.time() - t0 < timeout_s:
        pos = get_local_position(master, timeout_s=1.0)

        if pos is None:
            continue

        _x, _y, z_down = pos
        alt_m = -z_down

        print(f"[Alt] current={alt_m:.2f} m | target={target_alt_m:.2f} m")

        if abs(alt_m - target_alt_m) <= tolerance_m:
            print("[MAVLink] Target altitude reached.")
            return True

        time.sleep(0.3)

    print("[MAVLink] Altitude wait timed out.")
    return False


def send_body_velocity_yawrate(
    master,
    vx: float,
    vy: float,
    vz: float,
    yaw_rate: float,
):
    """
    BODY_NED frame:
        +vx = forward
        +vy = right
        +vz = down
        +yaw_rate = yaw rotation

    This test uses:
        vx       -> distance/standoff correction
        yaw_rate -> perpendicular/yaw correction
        vz = 0  -> no altitude correction in this test
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


# =========================================================
# CONTROL LOOP
# =========================================================

def apply_deadband(error: float, deadband: float) -> float:
    # Ignore errors inside or exactly equal to the deadband.
    if abs(error) <= deadband:
        return 0.0
    return error


def run_pid_test(master, ser):
    dist_pid = PID(
        KP_DIST,
        KI_DIST,
        KD_DIST,
        output_limit=MAX_VX_MPS,
        integral_limit=1.0,
    )

    yaw_pid = PID(
        KP_YAW,
        KI_YAW,
        KD_YAW,
        output_limit=MAX_YAW_RATE_RAD_S,
        integral_limit=1.0,
    )

    sensor_filter = UltrasonicRollingAverage(
        window_size=SENSOR_AVG_WINDOW,
        min_samples=SENSOR_MIN_VALID_SAMPLES,
    )

    dt = 1.0 / CONTROL_HZ
    stable_start_time = None
    t0 = time.time()
    last_sensor_time = time.time()
    loop_count = 0

    print("\n[PID] Starting wall alignment PID test.")
    print(f"[PID] Target standoff: {TARGET_STANDOFF_M:.2f} m")
    print(f"[PID] Distance deadband: ±{DIST_DEADBAND_M:.2f} m")
    print(f"[PID] Yaw ultrasonic-difference deadband: ±{YAW_DIFF_DEADBAND_M:.2f} m")
    print(f"[PID] Sensor average window: {SENSOR_AVG_WINDOW} samples")
    print(f"[PID] Stop when stable: {STOP_WHEN_STABLE}")
    print(f"[PID] Max test time: {MAX_TEST_TIME_S}")
    print(f"[PID] CSV log: {LOG_CSV}\n")

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "avg_distance_m",
            "distance_error_m",
            "vx_cmd_mps",
            "yaw_error_m",
            "yaw_rate_cmd_rad_s",
            "stable",
        ])

        while True:
            now = time.time()
            elapsed = now - t0
            loop_count += 1

            if MAX_TEST_TIME_S is not None and elapsed > MAX_TEST_TIME_S:
                print("[PID] Max test time reached.")
                break

            reading = read_ultrasonic_sensors_cm(ser)

            if reading is None:
                if time.time() - last_sensor_time > SENSOR_LOST_TIMEOUT_S:
                    print("[PID] Sensor timeout. Sending zero command.")
                    stop_motion(master)

                time.sleep(0.005)
                continue

            last_sensor_time = time.time()
            loop_count += 1

            raw_sensor1_cm, raw_sensor2_cm, raw_error_cm = reading

            filtered = sensor_filter.update(raw_sensor1_cm, raw_sensor2_cm)

            if filtered is None:
                print(
                    f"[PID] Collecting sensor average... "
                    f"raw_s1={raw_sensor1_cm:.2f}cm | "
                    f"raw_s2={raw_sensor2_cm:.2f}cm"
                )
                stop_motion(master)
                time.sleep(dt)
                continue

            sensor1_cm, sensor2_cm, error_cm = filtered

            sensor1_m = sensor1_cm / 100.0
            sensor2_m = sensor2_cm / 100.0

            avg_distance_m = (sensor1_m + sensor2_m) / 2.0

            # Positive means average distance is larger than target.
            # Negative means drone is too close to the wall.
            distance_error_raw_m = avg_distance_m - TARGET_STANDOFF_M

            # Positive/negative meaning depends on which ultrasonic is sensor 1.
            yaw_diff_raw_m = error_cm / 100.0

            distance_error_control_m = apply_deadband(
                distance_error_raw_m,
                DIST_DEADBAND_M,
            )

            yaw_diff_control_m = apply_deadband(
                yaw_diff_raw_m,
                YAW_DIFF_DEADBAND_M,
            )

            vx_cmd = DIST_SIGN * dist_pid.update(distance_error_control_m)
            yaw_rate_cmd = YAW_SIGN * yaw_pid.update(yaw_diff_control_m)

            distance_stable = abs(distance_error_raw_m) <= DIST_DEADBAND_M
            yaw_stable = abs(yaw_diff_raw_m) <= YAW_DIFF_DEADBAND_M
            stable = distance_stable and yaw_stable

            if stable:
                if stable_start_time is None:
                    stable_start_time = now
            else:
                stable_start_time = None

            send_body_velocity_yawrate(
                master,
                vx=vx_cmd,
                vy=0.0,
                vz=0.0,
                yaw_rate=yaw_rate_cmd,
            )

            writer.writerow([
                round(elapsed, 3),
                round(avg_distance_m, 4),
                round(distance_error_raw_m, 4),
                round(vx_cmd, 4),
                round(yaw_diff_raw_m, 4),
                round(yaw_rate_cmd, 4),
                int(stable),
            ])

            f.flush()

            if loop_count % PRINT_EVERY_N == 0:
                print(
                    f"[PID] "
                    f"dist={avg_distance_m:.2f}m | "
                    f"dist_err={distance_error_raw_m:+.2f}m | "
                    f"vx={vx_cmd:+.2f}m/s || "
                    f"yaw_err={yaw_diff_raw_m:+.2f}m | "
                    f"yaw_rate={yaw_rate_cmd:+.2f}rad/s | "
                    f"stable={stable}"
                )

            if STOP_WHEN_STABLE and stable_start_time is not None:
                stable_duration = now - stable_start_time

                if stable_duration >= STABLE_HOLD_TIME_S:
                    print(
                        f"[PID] Stable for {STABLE_HOLD_TIME_S:.1f}s. "
                        f"Stopping PID test."
                    )
                    break

            time.sleep(dt)

    stop_motion(master)
    print("[PID] PID test finished. Drone command set to zero.")


# =========================================================
# MAIN
# =========================================================

def main():
    ser = None
    master = None

    try:
        ser = open_ultrasonic_serial()

        master = connect_mavlink()

        set_mode(master, "GUIDED")
        arm(master)

        takeoff(master, TARGET_ALT_M)
        wait_until_altitude(master, TARGET_ALT_M)

        run_pid_test(master, ser)

    except KeyboardInterrupt:
        print("\n[Main] Stopped by user.")

    finally:
        if master is not None:
            try:
                stop_motion(master)
            except Exception:
                pass

        if ser is not None:
            ser.close()

        print("[Main] Done.")


if __name__ == "__main__":
    main()