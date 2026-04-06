import json
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

from pymavlink import mavutil


# =========================================================
# config
# =========================================================

# row-start correction config
ROW_START_CORRECTION_HZ = 10.0
ROW_START_CORRECTION_TIME_S = 1.0

# placeholder PID gains / limits for future tuning
KP_WALL_DIST, KI_WALL_DIST, KD_WALL_DIST = 0.0, 0.0, 0.0
KP_YAW, KI_YAW, KD_YAW = 0.0, 0.0, 0.0
KP_HEIGHT, KI_HEIGHT, KD_HEIGHT = 0.0, 0.0, 0.0

MAX_VX_CORR = 0.5
MAX_YAW_RATE_CORR = 0.4
MAX_VZ_CORR = 0.4

# JSON logging
DEFAULT_DETECTIONS_JSON = "corrosion_detections.json"
DETECTION_DEDUP_THRESHOLD_M = 0.5


# =========================================================
# type aliases
# =========================================================

Vec2 = Tuple[float, float]  # (north, east)
Vec3 = Tuple[float, float, float]  # (north, east, down) in local NED


# =========================================================
# low-level mavlink functions
# =========================================================

# connects to the autopilot over mavlink and waits for a heartbeat
def connect_sitl(url: str = "udp:127.0.0.1:14550"):
    master = mavutil.mavlink_connection(url)
    master.wait_heartbeat()
    print(f"Connected: system {master.target_system}, component {master.target_component}")
    return master


# sets the flight mode using the autopilot mode mapping
def set_mode(master, mode: str):
    mode_map = master.mode_mapping()
    if mode not in mode_map:
        raise RuntimeError(f"Mode {mode} not available. Available: {list(mode_map.keys())}")
    mode_id = mode_map[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    time.sleep(1.0)


# arms the vehicle
def arm(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0,
    )
    print("Arming...")
    time.sleep(2.0)


# commands a takeoff to a target relative altitude
def takeoff(master, target_alt_m: float):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, target_alt_m,
    )
    print(f"Taking off to {target_alt_m:.2f} m...")
    time.sleep(1.0)


# sends a global gps setpoint
def goto_position_target_global_int(master, lat_deg: float, lon_deg: float, alt_rel_m: float):
    type_mask = 0b0000111111111000  # enable position only
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
        0, 0,
    )


# sends a local NED position setpoint
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
        yaw_rad, 0,
    )


_T0_MS = None


def _now_ms_u32():
    global _T0_MS
    t = time.monotonic()
    if _T0_MS is None:
        _T0_MS = t
    return int((t - _T0_MS) * 1000) & 0xFFFFFFFF


# sends a body-frame velocity + yaw-rate command
def send_body_velocity_yawrate(master, vx: float, vy: float, vz: float, yaw_rate: float):
    """
    BODY_NED: +x forward, +y right, +z down; yaw_rate in rad/s.
    This is used only during the short row-start correction stage.
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
        0, 0, 0,
        float(vx), float(vy), float(vz),
        0, 0, 0,
        0.0,
        float(yaw_rate),
    )


# reads current global position
def get_global_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No GLOBAL_POSITION_INT received.")
    lat = msg.lat / 1e7
    lon = msg.lon / 1e7
    rel_alt = msg.relative_alt / 1000.0
    return lat, lon, rel_alt


# reads current local position in LOCAL_NED
def get_local_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No LOCAL_POSITION_NED received.")
    return float(msg.x), float(msg.y), float(msg.z)


# loads the json file exported by the gui
def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# small vector helpers
# =========================================================


def v2_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def v2_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def v2_scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def v2_norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def v2_unit(a: Vec2) -> Vec2:
    n = v2_norm(a)
    if n < 1e-6:
        raise ValueError("Zero-length vector.")
    return (a[0] / n, a[1] / n)


def v2_midpoint(a: Vec2, b: Vec2) -> Vec2:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def v3_dist(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# =========================================================
# gps/local conversion helpers
# =========================================================

# converts gps lat/lon to local north/east offsets in meters around a reference lat/lon
def latlon_to_ne_m(lat0: float, lon0: float, lat: float, lon: float) -> Vec2:
    R = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = dlat * R
    east = dlon * R * math.cos(math.radians(lat0))
    return north, east


# converts a gps target to a LOCAL_NED target using the current gps/local pose as anchor
def current_gps_to_target_local_ned(
    current_lat: float,
    current_lon: float,
    current_local_ned: Vec3,
    target_lat: float,
    target_lon: float,
    target_alt_m: float,
) -> Vec3:
    dn, de = latlon_to_ne_m(current_lat, current_lon, target_lat, target_lon)
    cx, cy, _cz = current_local_ned
    return (cx + dn, cy + de, -target_alt_m)


# =========================================================
# PID helper
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

        de = 0.0 if self.prev is None else (e - self.prev)

        self.ei += e * dt
        if self.i_limit is not None:
            self.ei = max(-self.i_limit, min(self.i_limit, self.ei))

        u = self.kp * e + self.ki * self.ei + self.kd * (de / dt if dt > 0 else 0.0)

        if self.out_limit is not None:
            u = max(-self.out_limit, min(self.out_limit, u))

        self.prev = e
        self.tprev = now
        return u


# =========================================================
# correction placeholders
# =========================================================


def build_row_start_pid_controllers() -> Dict[str, PID]:
    return {
        "wall_distance": PID(KP_WALL_DIST, KI_WALL_DIST, KD_WALL_DIST, out_limit=MAX_VX_CORR),
        "yaw_rate": PID(KP_YAW, KI_YAW, KD_YAW, out_limit=MAX_YAW_RATE_CORR),
        "height": PID(KP_HEIGHT, KI_HEIGHT, KD_HEIGHT, out_limit=MAX_VZ_CORR),
    }


# placeholder for wall distance correction at the start of each row
def wall_distance_pid(
    master,
    pid: PID,
    desired_standoff_m: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_standoff_m, row_start_local_ned, plan)
    return 0.0


# placeholder for yaw-rate correction at the start of each row
def yawrate_pid(
    master,
    pid: PID,
    desired_yaw_deg: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_yaw_deg, row_start_local_ned, plan)
    return 0.0


# placeholder for height correction at the start of each row
def height_pid(
    master,
    pid: PID,
    desired_alt_m: float,
    row_start_local_ned: Vec3,
    plan: Dict,
) -> float:
    _ = (master, pid, desired_alt_m, row_start_local_ned, plan)
    return 0.0


# runs the three placeholder PID corrections before each row begins
def run_row_start_correction(
    master,
    plan: Dict,
    row_start: Vec3,
    yaw_hold_deg: float,
    duration_s: float = ROW_START_CORRECTION_TIME_S,
    hz: float = ROW_START_CORRECTION_HZ,
):
    print("Running row-start correction...")

    controllers = build_row_start_pid_controllers()
    desired_alt_m = -row_start[2]
    desired_standoff_m = float(plan["params"]["standoff_m"])

    dt = 1.0 / max(hz, 1e-6)
    steps = max(1, int(duration_s * hz))

    for _ in range(steps):
        vx = wall_distance_pid(
            master=master,
            pid=controllers["wall_distance"],
            desired_standoff_m=desired_standoff_m,
            row_start_local_ned=row_start,
            plan=plan,
        )

        yaw_rate = yawrate_pid(
            master=master,
            pid=controllers["yaw_rate"],
            desired_yaw_deg=yaw_hold_deg,
            row_start_local_ned=row_start,
            plan=plan,
        )

        vz = height_pid(
            master=master,
            pid=controllers["height"],
            desired_alt_m=desired_alt_m,
            row_start_local_ned=row_start,
            plan=plan,
        )

        send_body_velocity_yawrate(master, vx=vx, vy=0.0, vz=vz, yaw_rate=yaw_rate)
        time.sleep(dt)

    send_body_velocity_yawrate(master, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)


# =========================================================
# wait / hold helpers
# =========================================================

# waits until the drone reaches the target altitude within tolerance
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


# keeps sending the same local target until it is reached
def hold_local_target_until_reached(
    master,
    target_xyz: Vec3,
    yaw_deg: float = 0.0,
    xy_tol_m: float = 0.40,
    z_tol_m: float = 0.25,
    timeout_s: float = 60.0,
    hz: float = 10.0,
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

        cx, cy, cz = get_local_position(master, timeout_s=1.0)
        err_xy = math.hypot(cx - target_xyz[0], cy - target_xyz[1])
        err_z = abs(cz - target_xyz[2])

        if err_xy <= xy_tol_m and err_z <= z_tol_m:
            return

        time.sleep(dt)

    raise TimeoutError(f"Did not reach local target {target_xyz} in time.")


# =========================================================
# movement functions
# =========================================================

# streams interpolated local NED setpoints along a straight line
def fly_line_local_streamed(
    master,
    start_xyz: Vec3,
    end_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = 10.0,
    min_time_s: float = 0.5,
    on_scan_step: Optional[Callable[[object, Vec3], None]] = None,
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
        a = k / steps
        x = sx + a * dx
        y = sy + a * dy
        z = sz + a * dz

        goto_position_target_local_ned(master, x, y, z, yaw_deg=yaw_deg)

        if on_scan_step is not None:
            on_scan_step(master, (x, y, z))

        time.sleep(dt)


# moves from the current pose to a requested local NED target
def fly_to_local_point(
    master,
    target_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = 10.0,
):
    start_xyz = get_local_position(master, timeout_s=2.0)

    fly_line_local_streamed(
        master=master,
        start_xyz=start_xyz,
        end_xyz=target_xyz,
        yaw_deg=yaw_deg,
        speed_mps=speed_mps,
        hz=hz,
        on_scan_step=None,
    )

    hold_local_target_until_reached(
        master=master,
        target_xyz=target_xyz,
        yaw_deg=yaw_deg,
        hz=hz,
    )


# =========================================================
# scan geometry functions
# =========================================================

# loads scan parameters from the gui spec
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


# computes the unit vector along the wall from A to B
def compute_wall_unit_vector(a_ne: Vec2, b_ne: Vec2) -> Vec2:
    wall_vec = v2_sub(b_ne, a_ne)
    if v2_norm(wall_vec) < 1e-6:
        raise ValueError("A and B are too close to define a wall direction.")
    return v2_unit(wall_vec)


# computes the two possible perpendicular unit vectors to the wall
def compute_candidate_wall_normals(u_wall: Vec2) -> Tuple[Vec2, Vec2]:
    n1 = (-u_wall[1], u_wall[0])
    n2 = (u_wall[1], -u_wall[0])
    return n1, n2


# chooses the wall side that is closer to the drone's current position
def choose_scan_normal_same_side_as_drone(
    current_ne: Vec2,
    a_ne: Vec2,
    b_ne: Vec2,
    standoff_m: float,
) -> Tuple[Vec2, Vec2]:
    u_wall = compute_wall_unit_vector(a_ne, b_ne)
    n1, n2 = compute_candidate_wall_normals(u_wall)

    wall_mid = v2_midpoint(a_ne, b_ne)

    # candidate scan midpoints, one on each side of the wall
    candidate_mid_1 = v2_add(wall_mid, v2_scale(n1, standoff_m))
    candidate_mid_2 = v2_add(wall_mid, v2_scale(n2, standoff_m))

    d1 = v2_norm(v2_sub(current_ne, candidate_mid_1))
    d2 = v2_norm(v2_sub(current_ne, candidate_mid_2))

    # choose the scan side closer to the current drone position
    n_side = n1 if d1 <= d2 else n2
    return u_wall, n_side


# builds the horizontal scan endpoints for one row
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


# builds the row altitudes from start_alt to max_alt inclusive
def build_row_altitudes(start_alt_m: float, max_alt_m: float, lane_spacing_m: float) -> List[float]:
    alts = []
    z = start_alt_m

    while z <= max_alt_m + 1e-6:
        alts.append(round(z, 6))
        z += lane_spacing_m

    if len(alts) == 0:
        raise RuntimeError("No row altitudes generated.")

    return alts


# converts a north/east point and altitude into local NED
def ne_alt_to_local_ned(ne: Vec2, alt_m: float) -> Vec3:
    return (ne[0], ne[1], -alt_m)


# builds all serpentine rows
def build_serpentine_rows(
    left_scan_ne: Vec2,
    right_scan_ne: Vec2,
    row_altitudes_m: List[float],
) -> List[Tuple[Vec3, Vec3]]:
    rows: List[Tuple[Vec3, Vec3]] = []

    for i, alt_m in enumerate(row_altitudes_m):
        left_pt = ne_alt_to_local_ned(left_scan_ne, alt_m)
        right_pt = ne_alt_to_local_ned(right_scan_ne, alt_m)

        if i % 2 == 0:
            rows.append((left_pt, right_pt))
        else:
            rows.append((right_pt, left_pt))

    return rows


# converts wall gps corners into local north/east coordinates around the current pose
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

    a_ne = (a_local[0], a_local[1])
    b_ne = (b_local[0], b_local[1])

    return a_ne, b_ne, current_local_ned


# builds the full scan plan
def build_wall_scan_plan(master, spec: dict) -> Dict:
    params = load_scan_parameters(spec)

    a_ne, b_ne, current_local_ned = convert_wall_corners_to_local_ne(master, spec)
    current_ne = (current_local_ned[0], current_local_ned[1])

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

    if len(rows) == 0:
        raise RuntimeError("No scan rows generated.")

    scan_origin_local_ned = rows[0][0]  # actual first scan point

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
# corrosion detection / logging helpers
# =========================================================


class CorrosionLoggingState:
    def __init__(self, json_path: str = DEFAULT_DETECTIONS_JSON):
        self.json_path = json_path
        self.saved_detections: List[dict] = []


# saves a detection only if it is not within threshold_m of an existing saved detection
def save_detection_if_new(
    saved_detections: List[dict],
    current_local_ned: Vec3,
    scan_origin_local_ned: Vec3,
    threshold_m: float = DETECTION_DEDUP_THRESHOLD_M,
    extra: Optional[dict] = None,
) -> bool:
    rel_ned = (
        current_local_ned[0] - scan_origin_local_ned[0],
        current_local_ned[1] - scan_origin_local_ned[1],
        current_local_ned[2] - scan_origin_local_ned[2],
    )

    for det in saved_detections:
        old = det["rel_ned"]
        dist = math.sqrt(
            (rel_ned[0] - old[0]) ** 2 +
            (rel_ned[1] - old[1]) ** 2 +
            (rel_ned[2] - old[2]) ** 2
        )
        if dist <= threshold_m:
            return False

    record = {
        "rel_ned": rel_ned,
        "abs_local_ned": current_local_ned,
        "timestamp_s": time.time(),
    }

    if extra is not None:
        record["extra"] = extra

    saved_detections.append(record)
    return True


# writes all logged detections to disk
def write_corrosion_detections_json(state: CorrosionLoggingState):
    payload = {
        "detection_count": len(state.saved_detections),
        "detections": state.saved_detections,
    }
    with open(state.json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# placeholder for future transformer-based corrosion vision
def run_vision_inference(master, commanded_target_local_ned: Vec3) -> Dict:
    _ = (master, commanded_target_local_ned)
    return {
        "detected": False,
        "corroded_area": None,
        "extra": {},
    }


# stores current local position + corroded area into the json file if a new detection is found
def save_corrosion_detection_to_json(
    master,
    state: CorrosionLoggingState,
    scan_origin_local_ned: Vec3,
    corroded_area,
    extra: Optional[dict] = None,
    threshold_m: float = DETECTION_DEDUP_THRESHOLD_M,
) -> bool:
    current_local_ned = get_local_position(master, timeout_s=2.0)

    info = {"corroded_area": corroded_area}
    if extra is not None:
        info.update(extra)

    is_new = save_detection_if_new(
        saved_detections=state.saved_detections,
        current_local_ned=current_local_ned,
        scan_origin_local_ned=scan_origin_local_ned,
        threshold_m=threshold_m,
        extra=info,
    )

    if is_new:
        write_corrosion_detections_json(state)
        print(f"Saved corrosion detection to {state.json_path}")

    return is_new


# default no-op hook for any extra user logic
def noop_on_scan_step(master, commanded_target_local_ned: Vec3):
    _ = (master, commanded_target_local_ned)
    return


# builds the row-step callback that runs vision and logs corrosion detections
def build_corrosion_scan_step_callback(
    plan: Dict,
    logging_state: CorrosionLoggingState,
    user_on_scan_step: Optional[Callable[[object, Vec3], None]] = None,
) -> Callable[[object, Vec3], None]:
    scan_origin_local_ned = plan["scan_origin_local_ned"]

    def _callback(master, commanded_target_local_ned: Vec3):
        vision_result = run_vision_inference(master, commanded_target_local_ned)

        if vision_result.get("detected", False):
            save_corrosion_detection_to_json(
                master=master,
                state=logging_state,
                scan_origin_local_ned=scan_origin_local_ned,
                corroded_area=vision_result.get("corroded_area"),
                extra=vision_result.get("extra"),
            )

        if user_on_scan_step is not None:
            user_on_scan_step(master, commanded_target_local_ned)

    return _callback


# =========================================================
# mission execution functions
# =========================================================

# moves to the first scan point
def move_to_first_scan_point(master, plan: Dict, yaw_hold_deg: float):
    first_scan_point = plan["scan_origin_local_ned"]
    speed_mps = plan["params"]["speed_mps"]

    print(f"Moving to first scan point: {first_scan_point}")
    fly_to_local_point(
        master=master,
        target_xyz=first_scan_point,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
    )


# executes one row of the scan
def execute_single_scan_row(
    master,
    plan: Dict,
    row_start: Vec3,
    row_end: Vec3,
    speed_mps: float,
    yaw_hold_deg: float,
    on_scan_step: Optional[Callable[[object, Vec3], None]],
):
    # explicitly go to the row start
    fly_to_local_point(
        master=master,
        target_xyz=row_start,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
    )

    # short correction stage at the start of the row
    run_row_start_correction(
        master=master,
        plan=plan,
        row_start=row_start,
        yaw_hold_deg=yaw_hold_deg,
    )

    # stream setpoints along the row while running scan-step logic
    fly_line_local_streamed(
        master=master,
        start_xyz=row_start,
        end_xyz=row_end,
        yaw_deg=yaw_hold_deg,
        speed_mps=speed_mps,
        hz=10.0,
        on_scan_step=on_scan_step,
    )

    # hold final endpoint until reached
    hold_local_target_until_reached(
        master=master,
        target_xyz=row_end,
        yaw_deg=yaw_hold_deg,
        hz=10.0,
    )


# executes all rows
def execute_all_scan_rows(
    master,
    plan: Dict,
    yaw_hold_deg: float,
    on_scan_step: Optional[Callable[[object, Vec3], None]],
):
    rows = plan["rows"]
    speed_mps = plan["params"]["speed_mps"]

    for i, (row_start, row_end) in enumerate(rows):
        print(f"Row {i+1}/{len(rows)}")
        print(f"  start: {row_start}")
        print(f"  end  : {row_end}")

        execute_single_scan_row(
            master=master,
            plan=plan,
            row_start=row_start,
            row_end=row_end,
            speed_mps=speed_mps,
            yaw_hold_deg=yaw_hold_deg,
            on_scan_step=on_scan_step,
        )


# top-level wall scan function
def execute_wall_scan(
    master,
    spec: dict,
    yaw_hold_deg: float = 0.0,
    user_on_scan_step: Optional[Callable[[object, Vec3], None]] = None,
    detections_json_path: str = DEFAULT_DETECTIONS_JSON,
) -> Dict:
    params = load_scan_parameters(spec)
    start_alt_m = params["start_alt_m"]

    print(f"Takeoff to start altitude: {start_alt_m:.2f} m")
    takeoff(master, start_alt_m)
    wait_until_altitude(master, start_alt_m, tol_m=0.25, timeout_s=45.0)

    print("Building scan plan...")
    plan = build_wall_scan_plan(master, spec)

    print("Scan plan built.")
    print(f"  wall A local NE: {plan['a_ne']}")
    print(f"  wall B local NE: {plan['b_ne']}")
    print(f"  u_wall:         {plan['u_wall']}")
    print(f"  n_side:         {plan['n_side']}")
    print(f"  left_scan_ne:   {plan['left_scan_ne']}")
    print(f"  right_scan_ne:  {plan['right_scan_ne']}")
    print(f"  scan origin:    {plan['scan_origin_local_ned']}")
    print(f"  row count:      {len(plan['rows'])}")

    logging_state = CorrosionLoggingState(json_path=detections_json_path)
    on_scan_step = build_corrosion_scan_step_callback(
        plan=plan,
        logging_state=logging_state,
        user_on_scan_step=user_on_scan_step,
    )

    move_to_first_scan_point(master, plan, yaw_hold_deg=yaw_hold_deg)
    execute_all_scan_rows(master, plan, yaw_hold_deg=yaw_hold_deg, on_scan_step=on_scan_step)

    print("Scan complete.")
    write_corrosion_detections_json(logging_state)

    return {
        "plan": plan,
        "logging_state": logging_state,
    }


# =========================================================
# main
# =========================================================


def main():
    import sys

    if len(sys.argv) < 2:
        print("usage: python3 scan_backend.py <path_to_gui_exported_json> [detections_json_path]")
        return

    spec_path = sys.argv[1]
    detections_json_path = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_DETECTIONS_JSON

    spec = load_gui_spec(spec_path)

    master = connect_sitl("udp:127.0.0.1:14550")
    set_mode(master, "GUIDED")
    arm(master)

    execute_wall_scan(
        master=master,
        spec=spec,
        yaw_hold_deg=0.0,
        user_on_scan_step=noop_on_scan_step,
        detections_json_path=detections_json_path,
    )


if __name__ == "__main__":
    main()
