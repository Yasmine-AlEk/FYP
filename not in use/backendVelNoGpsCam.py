import json
import math
import time
import os
from datetime import datetime

from pymavlink import mavutil

# Camera
from picamera2 import Picamera2


# -----------------------------
# Camera helpers
# -----------------------------

def setup_camera(width: int = 1280, height: int = 720) -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (width, height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)
    return picam2


def script_folder() -> str:
    # saves to the folder where this .py file exists
    return os.path.dirname(os.path.abspath(__file__))


def save_frame(picam2: Picamera2, out_dir: str, frame_idx: int) -> str:
    frame = picam2.capture_array()

    # Use timestamp in filename to avoid overwriting
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{frame_idx:06d}_{ts}.png"
    path = os.path.join(out_dir, filename)

    # Picamera2 capture_array gives a numpy array. We can save with OpenCV if available,
    # but to keep it minimal, use PIL (usually installed) or fallback to OpenCV if you prefer.
    try:
        from PIL import Image
        Image.fromarray(frame).save(path)
    except Exception:
        # Fallback to opencv if PIL isn't available
        import cv2
        cv2.imwrite(path, frame)

    return path


# -----------------------------
# Connection + basic commands
# -----------------------------

def connect_sitl(url: str = "udp:127.0.0.1:14550"):
    master = mavutil.mavlink_connection(url)
    master.wait_heartbeat()
    print(f"Connected: system {master.target_system}, component {master.target_component}")
    return master


def set_mode(master, mode: str):
    mode_map = master.mode_mapping()
    if mode not in mode_map:
        raise RuntimeError(f"Mode {mode} not available. Available: {list(mode_map.keys())}")
    mode_id = mode_map[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    time.sleep(1)


def arm(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    print("Arming...")
    time.sleep(2)


def takeoff(master, target_alt_m: float):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        target_alt_m
    )
    print(f"Taking off to {target_alt_m:.1f}m...")
    time.sleep(1)


def goto_position_target_global_int(master, lat_deg: float, lon_deg: float, alt_rel_m: float):
    # uses GPS internally, but we do NOT read any position back
    type_mask = 0b0000111111111000  # enable only position
    master.mav.set_position_target_global_int_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        type_mask,
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        alt_rel_m,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )


# -----------------------------
# Velocity in BODY_NED (no position reads)
# -----------------------------

def send_velocity_body_ned(master, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
    """
    BODY_NED:
      vx forward (m/s)
      vy right   (m/s)
      vz down    (m/s)   negative is up
      yaw_rate rad/s
    """
    msg = master.mav.set_position_target_local_ned_encode(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000011111000111,  # ignore pos/accel, enable vel + yaw_rate
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )
    master.mav.send(msg)


def stop_velocity(master,
                  picam2: Picamera2,
                  out_dir: str,
                  frame_state: dict,
                  hz: float = 10.0,
                  stop_time_s: float = 0.5,
                  capture_period_s: float = 0.5):
    """
    Sends zero velocity for a short time and still captures frames every capture_period_s.
    frame_state = {"idx": int, "next_capture_t": float}
    """
    dt = 1.0 / hz
    steps = max(1, int(stop_time_s * hz))
    for _ in range(steps):
        send_velocity_body_ned(master, 0.0, 0.0, 0.0, yaw_rate=0.0)

        now = time.time()
        if now >= frame_state["next_capture_t"]:
            save_frame(picam2, out_dir, frame_state["idx"])
            frame_state["idx"] += 1
            frame_state["next_capture_t"] = now + capture_period_s

        time.sleep(dt)


def run_velocity_for(master,
                     vx: float, vy: float, vz: float,
                     duration_s: float,
                     picam2: Picamera2,
                     out_dir: str,
                     frame_state: dict,
                     hz: float = 10.0,
                     capture_period_s: float = 0.5):
    """
    Streams velocity for duration_s and captures images every capture_period_s.
    """
    dt = 1.0 / hz
    steps = max(1, int(duration_s * hz))

    for _ in range(steps):
        send_velocity_body_ned(master, vx, vy, vz, yaw_rate=0.0)

        now = time.time()
        if now >= frame_state["next_capture_t"]:
            save_frame(picam2, out_dir, frame_state["idx"])
            frame_state["idx"] += 1
            frame_state["next_capture_t"] = now + capture_period_s

        time.sleep(dt)


# -----------------------------
# Optional yaw control
# -----------------------------

def set_yaw_heading(master, yaw_deg: float, yaw_rate_dps: float = 30.0, relative: bool = False):
    direction = 1
    is_relative = 1 if relative else 0
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        float(yaw_deg),
        float(yaw_rate_dps),
        float(direction),
        float(is_relative),
        0, 0, 0
    )


# -----------------------------
# Geometry helpers
# -----------------------------

def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wall_heading_deg_from_gps(A_lat, A_lon, B_lat, B_lon) -> float:
    lat0 = math.radians(A_lat)
    dlat = math.radians(B_lat - A_lat)
    dlon = math.radians(B_lon - A_lon)
    x_east = dlon * math.cos(lat0)
    y_north = dlat
    bearing = math.degrees(math.atan2(x_east, y_north))
    if bearing < 0:
        bearing += 360.0
    return bearing


def wall_length_m_from_gps(A_lat, A_lon, B_lat, B_lon) -> float:
    R = 6378137.0
    lat0 = math.radians(A_lat)
    dlat = math.radians(B_lat - A_lat)
    dlon = math.radians(B_lon - A_lon)
    north = dlat * R
    east = dlon * R * math.cos(lat0)
    return math.sqrt(north * north + east * east)


# -----------------------------
# Open-loop wall scan + camera
# -----------------------------

def execute_wall_scan_open_loop(master, spec: dict, picam2: Picamera2):
    out_dir = script_folder()
    print(f"Saving images to: {out_dir}")

    # state for image capture timing
    frame_state = {
        "idx": 0,
        "next_capture_t": time.time()  # capture immediately on first loop
    }
    capture_period_s = 0.5  # every 0.5 seconds

    A = spec["structure"]["corners_gps"]["A"]
    B = spec["structure"]["corners_gps"]["B"]

    wall_total_h = float(spec["structure"]["dimensions_m"]["height"])
    start_alt_m = float(spec["scan_settings"]["start_alt_m"])
    speed = float(spec["scan_settings"]["speed_mps"])
    standoff = float(spec["scan_settings"]["standoff_m"])
    lane_spacing = float(spec["scan_settings"]["lane_spacing_m"])

    scan_span = wall_total_h - start_alt_m
    if scan_span <= 0:
        raise ValueError("wall total height must be greater than start altitude")

    extend_m = 2.0
    hz = 10.0

    # Move near A (GPS setpoint, no reads)
    print("Going near corner A (GPS setpoint, no position reads)...")
    goto_position_target_global_int(master, A["lat"], A["lon"], start_alt_m)

    # While waiting, still capture
    wait_s = 10.0
    t0 = time.time()
    while time.time() - t0 < wait_s:
        now = time.time()
        if now >= frame_state["next_capture_t"]:
            save_frame(picam2, out_dir, frame_state["idx"])
            frame_state["idx"] += 1
            frame_state["next_capture_t"] = now + capture_period_s
        time.sleep(0.05)

    # Face along wall
    heading_deg = wall_heading_deg_from_gps(A["lat"], A["lon"], B["lat"], B["lon"])
    print(f"Setting yaw to wall heading ~ {heading_deg:.1f} deg")
    set_yaw_heading(master, heading_deg, yaw_rate_dps=45.0, relative=False)
    time.sleep(2)

    # Apply standoff (right)
    side_speed = min(max(speed * 0.6, 0.2), 1.5)
    t_standoff = abs(standoff) / side_speed if standoff > 1e-3 else 0.0

    print(f"Applying standoff: {standoff:.2f} m at {side_speed:.2f} m/s (time {t_standoff:.2f}s)")
    if t_standoff > 0:
        run_velocity_for(master, 0.0, +side_speed, 0.0, t_standoff,
                         picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)
        stop_velocity(master, picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)

    # Row timing from GPS corners (math only)
    wall_len = wall_length_m_from_gps(A["lat"], A["lon"], B["lat"], B["lon"])
    row_dist = wall_len + 2.0 * extend_m
    if row_dist < 0.5:
        raise ValueError("Wall length too small from GPS corners.")

    t_row = row_dist / max(speed, 0.05)

    # Climb timing
    climb_speed = 0.5
    t_lane = lane_spacing / climb_speed if lane_spacing > 1e-3 else 0.0

    num_rows = int(math.ceil(scan_span / lane_spacing)) + 1

    print(f"Wall length ~ {wall_len:.2f} m, row distance (with overshoot) ~ {row_dist:.2f} m")
    print(f"Rows: {num_rows}, row time ~ {t_row:.2f}s, climb time per lane ~ {t_lane:.2f}s")
    print("Capturing frames every 0.5s during all motion.")

    for i in range(num_rows):
        print(f"Row {i+1}/{num_rows} (open-loop).")

        vx = +speed if (i % 2 == 0) else -speed

        # fly row + capture
        run_velocity_for(master, vx, 0.0, 0.0, t_row,
                         picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)
        stop_velocity(master, picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)

        # climb + capture
        if i != num_rows - 1 and t_lane > 0:
            run_velocity_for(master, 0.0, 0.0, -climb_speed, t_lane,
                             picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)
            stop_velocity(master, picam2, out_dir, frame_state, hz=hz, capture_period_s=capture_period_s)

    print("Scan complete (open-loop velocity + camera).")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python3 scan_backend_openloop_vel_cam.py <path_to_gui_exported_json>")
        sys.exit(1)

    spec_path = sys.argv[1]
    spec = load_gui_spec(spec_path)

    start_alt_m = float(spec["scan_settings"]["start_alt_m"])

    # Camera init
    picam2 = setup_camera(1280, 720)

    master = connect_sitl("udp:127.0.0.1:14550")
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, start_alt_m)
    time.sleep(6)

    execute_wall_scan_open_loop(master, spec, picam2)