import json
import math
import time
from typing import Tuple

from pymavlink import mavutil


# -----------------------------
# Raspberry Pi ↔ Pixhawk UART connection
# -----------------------------
DEVICE = "/dev/serial0"   # Pi UART (GPIO serial)
BAUD   = 115200           # must match Pixhawk SERIALx_BAUD


def connect_fc(device=DEVICE, baud=BAUD):
    print(f"[MAV] Connecting to {device} at {baud} baud...")
    master = mavutil.mavlink_connection(
        device,
        baud=baud,
        source_system=255  # companion computer id
    )

    print("[MAV] Waiting for heartbeat from FC...")
    hb = master.wait_heartbeat()
    sysid = hb.get_srcSystem()
    compid = hb.get_srcComponent()
    print(f"[MAV] Heartbeat from system {sysid}, component {compid}")

    # make sure messages go to the right target
    master.target_system = sysid
    master.target_component = compid
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


def goto_position_target_local_ned(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)
    type_mask = 0b0000111111111000  # enable position only
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


# -----------------------------
# VELOCITY CONTROL HELPERS
# -----------------------------

def send_velocity_local_ned(master, vx: float, vy: float, vz: float, yaw_rate: float = 0.0):
    """
    Stream a velocity setpoint in LOCAL_NED frame.
    vx: north m/s
    vy: east  m/s
    vz: down  m/s  (negative = up)
    yaw_rate: rad/s
    """
    type_mask = 0b0000011111000111  # ignore pos + accel, enable vel + yaw_rate
    msg = master.mav.set_position_target_local_ned_encode(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,          # position ignored
        vx, vy, vz,        # velocity used
        0, 0, 0,           # accel ignored
        0, yaw_rate        # yaw ignored, yaw_rate used
    )
    master.mav.send(msg)


def fly_line_local_velocity(
    master,
    start_xyz: Tuple[float, float, float],
    end_xyz: Tuple[float, float, float],
    speed_mps: float,
    hz: float = 10.0,
    min_time_s: float = 0.5,
    stop_time_s: float = 0.5,
):
    """
    Fly a straight segment by streaming constant LOCAL_NED velocity.
    Keeps it simple: no position interpolation, just a velocity vector for T seconds.
    """
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz

    dx = ex - sx
    dy = ey - sy
    dz = ez - sz

    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-3:
        # still send a short stop so the vehicle doesn't keep moving
        for _ in range(max(1, int(stop_time_s * hz))):
            send_velocity_local_ned(master, 0.0, 0.0, 0.0, yaw_rate=0.0)
            time.sleep(1.0 / hz)
        return

    speed = max(float(speed_mps), 0.05)
    T = max(min_time_s, dist / speed)

    # unit direction
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    vx = ux * speed
    vy = uy * speed
    vz = uz * speed

    dt = 1.0 / hz
    steps = max(1, int(T * hz))

    # stream velocity for the duration
    for _ in range(steps):
        send_velocity_local_ned(master, vx, vy, vz, yaw_rate=0.0)
        time.sleep(dt)

    # stop at the end (important!)
    stop_steps = max(1, int(stop_time_s * hz))
    for _ in range(stop_steps):
        send_velocity_local_ned(master, 0.0, 0.0, 0.0, yaw_rate=0.0)
        time.sleep(dt)


# -----------------------------
# REST OF YOUR MISSION LOGIC
# -----------------------------

def latlon_to_ned_m(lat0, lon0, lat, lon) -> Tuple[float, float]:
    R = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = dlat * R
    east = dlon * R * math.cos(math.radians(lat0))
    return north, east


def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_wall_scan(master, spec: dict):
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

    print("going to corner a (gps)...")
    goto_position_target_global_int(master, A["lat"], A["lon"], start_alt_m)
    time.sleep(10)

    origin_local = get_local_position(master)
    lat0, lon0, _ = get_global_position(master)
    ox, oy, oz = origin_local

    a_n, a_e = latlon_to_ned_m(lat0, lon0, A["lat"], A["lon"])
    b_n, b_e = latlon_to_ned_m(lat0, lon0, B["lat"], B["lon"])

    ab_n = b_n - a_n
    ab_e = b_e - a_e
    ab_len = math.sqrt(ab_n * ab_n + ab_e * ab_e)
    if ab_len < 1e-3:
        raise ValueError("a and b are too close to define a wall direction")

    u_w_n = ab_n / ab_len
    u_w_e = ab_e / ab_len

    u_p_n = u_w_e
    u_p_e = -u_w_n

    def local_from_ne(off_n: float, off_e: float, alt_m: float) -> Tuple[float, float, float]:
        return (ox + off_n, oy + off_e, -alt_m)

    num_rows = int(math.ceil(scan_span / lane_spacing)) + 1

    for i in range(num_rows):
        climb = min(i * lane_spacing, scan_span)
        alt_row = start_alt_m + climb
        if alt_row > wall_total_h:
            alt_row = wall_total_h

        a_ext_n = a_n - extend_m * u_w_n
        a_ext_e = a_e - extend_m * u_w_e
        b_ext_n = b_n + extend_m * u_w_n
        b_ext_e = b_e + extend_m * u_w_e

        leftA_n = a_ext_n + standoff * u_p_n
        leftA_e = a_ext_e + standoff * u_p_e
        rightB_n = b_ext_n + standoff * u_p_n
        rightB_e = b_ext_e + standoff * u_p_e

        if i % 2 == 0:
            p0_n, p0_e = leftA_n, leftA_e
            p1_n, p1_e = rightB_n, rightB_e
        else:
            p0_n, p0_e = rightB_n, rightB_e
            p1_n, p1_e = leftA_n, leftA_e

        p0 = local_from_ne(p0_n, p0_e, alt_row)
        p1 = local_from_ne(p1_n, p1_e, alt_row)

        print(f"row {i+1}/{num_rows} alt={alt_row:.2f} speed={speed:.2f} extend=2.00 standoff={standoff:.2f}")

        # optional: "snap" near the row start using position setpoint, then do velocity pass
        goto_position_target_local_ned(master, p0[0], p0[1], p0[2], yaw_deg=0.0)
        time.sleep(2.0)

        # velocity-based row traversal
        fly_line_local_velocity(master, p0, p1, speed_mps=speed, hz=10.0)

    print("scan complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python3 scan_backend.py <path_to_gui_exported_json>")
        sys.exit(1)

    spec_path = sys.argv[1]
    spec = load_gui_spec(spec_path)

    # FIX: start_alt_m must come from spec
    start_alt_m = float(spec["scan_settings"]["start_alt_m"])

    master = connect_fc()  # UART connection instead of SITL UDP
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, start_alt_m)
    time.sleep(6)

    execute_wall_scan(master, spec)