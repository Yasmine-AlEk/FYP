import json
import math
import time
from typing import Tuple

from pymavlink import mavutil


# -----------------------------
# MAVLink helpers
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
        0, 0, 0, 0, 0, 0, 0, target_alt_m
    )
    print(f"Taking off to {target_alt_m:.1f}m...")
    time.sleep(1)


def goto_position_target_global_int(master, lat_deg: float, lon_deg: float, alt_rel_m: float):
    type_mask = 0b0000111111111000  # position enabled
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


def goto_position_target_local_ned(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)
    type_mask = 0b0000111111111000  # position + yaw

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        yaw_rad, 0
    )


def get_global_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No GLOBAL_POSITION_INT received.")
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    rel_alt = msg.relative_alt / 1000.0
    return lat, lon, rel_alt


def get_local_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No LOCAL_POSITION_NED received.")
    return float(msg.x), float(msg.y), float(msg.z)


def wait_seconds_sending_local(master, target_xyz: Tuple[float, float, float], yaw_deg: float, seconds: float, hz: float = 10.0):
    dt = 1.0 / hz
    steps = int(seconds * hz)
    x, y, z = target_xyz
    for _ in range(steps):
        goto_position_target_local_ned(master, x, y, z, yaw_deg=yaw_deg)
        time.sleep(dt)


# -----------------------------
# Math helpers
# -----------------------------
def latlon_to_ned_m(lat0, lon0, lat, lon) -> Tuple[float, float]:
    R = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = dlat * R
    east = dlon * R * math.cos(math.radians(lat0))
    return north, east


def unit(vx, vy) -> Tuple[float, float]:
    n = math.sqrt(vx * vx + vy * vy)
    if n < 1e-9:
        raise ValueError("Zero-length direction vector (A and B too close).")
    return vx / n, vy / n


def yaw_from_vector_ned(vx_north, vy_east) -> float:
    return (math.degrees(math.atan2(vy_east, vx_north)) + 360.0) % 360.0


# -----------------------------
# IO + Mission
# -----------------------------
def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_wall_scan(master, spec: dict):
    A = spec["structure"]["corners_gps"]["A"]
    B = spec["structure"]["corners_gps"]["B"]

    width = float(spec["structure"]["dimensions_m"]["width"])   # computed by GUI
    height = float(spec["structure"]["dimensions_m"]["height"])

    start_alt_m = float(spec["scan_settings"]["start_alt_m"])
    speed = float(spec["scan_settings"]["speed_mps"])
    standoff = float(spec["scan_settings"]["standoff_m"])
    lane_spacing = float(spec["scan_settings"]["lane_spacing_m"])

    # 1) Setup flight
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, start_alt_m)
    time.sleep(8)

    # 2) Go near Corner A (global) at start altitude
    print("Going to Corner A (GPS)...")
    goto_position_target_global_int(master, A["lat"], A["lon"], start_alt_m)
    time.sleep(12)

    # 3) Define software origin at current local position
    origin_local = get_local_position(master)
    lat0, lon0, _ = get_global_position(master)

    # 4) Wall direction from A->B in N/E meters
    a_n, a_e = latlon_to_ned_m(lat0, lon0, A["lat"], A["lon"])
    b_n, b_e = latlon_to_ned_m(lat0, lon0, B["lat"], B["lon"])
    ab_n = b_n - a_n
    ab_e = b_e - a_e
    u_w_n, u_w_e = unit(ab_n, ab_e)

    # Left normal (enforces: drone on left side of A->B)
    u_p_n =  u_w_e
    u_p_e = -u_w_n

    yaw_face_wall = yaw_from_vector_ned(-u_p_n, -u_p_e)

    # 5) Plan lanes
    num_lanes = int(math.ceil(width / lane_spacing)) + 1

    # NED z down: bottom at start altitude, top is higher (more negative)
    z_bottom = -start_alt_m
    z_top = z_bottom - height

    ox, oy, oz = origin_local

    def local_target_from_offsets(off_n, off_e, target_z):
        return (ox + off_n, oy + off_e, target_z)

    # 6) Execute zig-zag lanes
    for i in range(num_lanes):
        s = min(i * lane_spacing, width)

        lane_n = s * u_w_n
        lane_e = s * u_w_e

        # Apply LEFT-side standoff
        lane_n += standoff * u_p_n
        lane_e += standoff * u_p_e

        if i % 2 == 0:
            start_z, end_z = z_bottom, z_top
        else:
            start_z, end_z = z_top, z_bottom

        start_xyz = local_target_from_offsets(lane_n, lane_e, start_z)
        end_xyz = local_target_from_offsets(lane_n, lane_e, end_z)

        print(f"Lane {i+1}/{num_lanes} | width={width:.2f}m | offsetN={lane_n:.2f} offsetE={lane_e:.2f} yaw={yaw_face_wall:.1f}")

        wait_seconds_sending_local(master, start_xyz, yaw_face_wall, seconds=6)

        vertical_dist = abs(end_z - start_z)
        t_move = max(6.0, vertical_dist / max(speed, 0.05))
        wait_seconds_sending_local(master, end_xyz, yaw_face_wall, seconds=t_move)

    print("Scan complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 scan_backend.py <path_to_gui_exported_json>")
        sys.exit(1)

    spec_path = sys.argv[1]
    spec = load_gui_spec(spec_path)

    master = connect_sitl("udp:127.0.0.1:14550")
    execute_wall_scan(master, spec)
