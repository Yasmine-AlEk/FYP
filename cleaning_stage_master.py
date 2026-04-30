import json
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

from pymavlink import mavutil


# =========================================================
# config
# =========================================================

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]  # (north, east, down)

DEFAULT_DETECTIONS_JSON = "corrosion_detections.json"
DEFAULT_CONNECTION_URL = "udpin:0.0.0.0:14550"

# navigation / settling
NAV_HZ = 10.0
TARGET_XY_TOL_M = 0.40
TARGET_Z_TOL_M = 0.25
TARGET_REACH_TIMEOUT_S = 60.0
TAKEOFF_TIMEOUT_S = 45.0
MIN_LINE_TIME_S = 0.5
MIN_SPEED_MPS = 0.05

# per-point work behavior
POINT_SETTLE_S = 1.0
ACTION_SAFETY_HOLD_S = 0.5

# stage definitions for flights 2-5
STAGE_CONFIGS: Dict[str, Dict] = {
    "flight2_detergent_wash": {
        "label": "Flight 2 - Detergent Wash",
        "action_type": "detergent_wash",
        "spray_duration_s": 3.0,
        "dwell_after_action_s": 1.0,
    },
    "flight3_rinse": {
        "label": "Flight 3 - Rinse",
        "action_type": "rinse",
        "spray_duration_s": 3.0,
        "dwell_after_action_s": 1.0,
    },
    "flight4_chemical_treatment": {
        "label": "Flight 4 - Chemical Treatment",
        "action_type": "chemical_treatment",
        "spray_duration_s": 4.0,
        "dwell_after_action_s": 2.0,
    },
    "flight5_final_rinse_protective_coating": {
        "label": "Flight 5 - Final Rinse and Protective Coating",
        "action_type": "final_rinse_protective_coating",
        "spray_duration_s": 4.0,
        "dwell_after_action_s": 1.0,
    },
}


# =========================================================
# low-level mavlink helpers
# =========================================================

def connect_sitl(url: str = DEFAULT_CONNECTION_URL):
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
        mode_id,
    )
    time.sleep(1.0)


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


def rtl(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )
    print("RTL command sent.")


def goto_position_target_local_ned(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)
    type_mask = 0b0000111111111000  # position + yaw enabled
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


def get_local_position(master, timeout_s: float = 2.0) -> Vec3:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No LOCAL_POSITION_NED received.")
    return float(msg.x), float(msg.y), float(msg.z)


# =========================================================
# file / data helpers
# =========================================================

def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_detections_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cleaning_parameters(spec: dict) -> Dict[str, float]:
    scan_settings = spec["scan_settings"]
    start_alt_m = float(scan_settings["start_alt_m"])
    speed_mps = float(scan_settings["speed_mps"])
    return {
        "start_alt_m": start_alt_m,
        "speed_mps": speed_mps,
    }


def extract_corrosion_targets_from_json(detections_payload: dict) -> List[dict]:
    """
    Extract cleaning targets from corrosion_detections.json.

    Priority:
    1. top-level commanded_target_ned dictionary
    2. extra.commanded_target_ned list
    3. actual_local_position_ned dictionary
    4. legacy abs_local_ned list

    For cleaning, commanded_target_ned is preferred because it represents the
    planned scan location where corrosion was detected.
    """

    detections = detections_payload.get("detections", [])
    targets: List[dict] = []

    for idx, det in enumerate(detections):
        extra = det.get("extra", {}) or {}

        target_xyz = None
        target_source = None

        # New preferred format: top-level commanded_target_ned dictionary
        cmd_dict = det.get("commanded_target_ned")
        if isinstance(cmd_dict, dict):
            try:
                target_xyz = (
                    float(cmd_dict["x_north_m"]),
                    float(cmd_dict["y_east_m"]),
                    float(cmd_dict["z_down_m"]),
                )
                target_source = "commanded_target_ned"
            except KeyError:
                target_xyz = None

        # Fallback: extra.commanded_target_ned list
        if target_xyz is None:
            cmd_list = extra.get("commanded_target_ned")
            if isinstance(cmd_list, list) and len(cmd_list) == 3:
                target_xyz = (
                    float(cmd_list[0]),
                    float(cmd_list[1]),
                    float(cmd_list[2]),
                )
                target_source = "extra.commanded_target_ned"

        # Fallback: actual_local_position_ned dictionary
        if target_xyz is None:
            actual_dict = det.get("actual_local_position_ned")
            if isinstance(actual_dict, dict):
                try:
                    target_xyz = (
                        float(actual_dict["x_north_m"]),
                        float(actual_dict["y_east_m"]),
                        float(actual_dict["z_down_m"]),
                    )
                    target_source = "actual_local_position_ned"
                except KeyError:
                    target_xyz = None

        # Legacy fallback: abs_local_ned list
        if target_xyz is None:
            abs_local = det.get("abs_local_ned")
            if isinstance(abs_local, list) and len(abs_local) == 3:
                target_xyz = (
                    float(abs_local[0]),
                    float(abs_local[1]),
                    float(abs_local[2]),
                )
                target_source = "abs_local_ned"

        if target_xyz is None:
            print(f"Skipping detection #{idx}: no valid target position found.")
            continue

        target = {
            "id": idx,
            "target_xyz": target_xyz,
            "target_source": target_source,
            "rel_ned": tuple(det.get("rel_ned", [0.0, 0.0, 0.0])),
            "timestamp_s": det.get("timestamp_s"),
            "corroded_area": extra.get("corroded_area"),
            "corrosion_percent": extra.get("corrosion_percent"),
            "severity_grade": extra.get("severity_grade"),
            "vision_files": extra.get("vision_files"),
            "extra": extra,
        }

        targets.append(target)

    if not targets:
        raise RuntimeError("No valid corrosion targets found in detections JSON.")

    print(f"Extracted {len(targets)} cleaning targets.")
    for t in targets:
        print(
            f"  target #{t['id']} from {t['target_source']} -> "
            f"{t['target_xyz']} | severity={t.get('severity_grade')}"
        )

    return targets


# =========================================================
# small geometry helpers
# =========================================================

def v3_dist(a: Vec3, b: Vec3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def sort_targets_nearest_neighbor(start_xyz: Vec3, targets: List[dict]) -> List[dict]:
    remaining = targets[:]
    ordered: List[dict] = []
    cursor = start_xyz

    while remaining:
        best_i = min(range(len(remaining)), key=lambda i: v3_dist(cursor, remaining[i]["target_xyz"]))
        best = remaining.pop(best_i)
        ordered.append(best)
        cursor = best["target_xyz"]

    return ordered


# =========================================================
# movement helpers
# =========================================================

def wait_until_altitude(
    master,
    target_alt_m: float,
    tol_m: float = TARGET_Z_TOL_M,
    timeout_s: float = TAKEOFF_TIMEOUT_S,
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
    xy_tol_m: float = TARGET_XY_TOL_M,
    z_tol_m: float = TARGET_Z_TOL_M,
    timeout_s: float = TARGET_REACH_TIMEOUT_S,
    hz: float = NAV_HZ,
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


def fly_line_local_streamed(
    master,
    start_xyz: Vec3,
    end_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = NAV_HZ,
    min_time_s: float = MIN_LINE_TIME_S,
):
    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz

    dx = ex - sx
    dy = ey - sy
    dz = ez - sz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    speed = max(float(speed_mps), MIN_SPEED_MPS)
    total_time = max(min_time_s, dist / speed)

    steps = max(2, int(total_time * hz))
    dt = 1.0 / hz

    for k in range(steps + 1):
        a = k / steps
        x = sx + a * dx
        y = sy + a * dy
        z = sz + a * dz
        goto_position_target_local_ned(master, x, y, z, yaw_deg=yaw_deg)
        time.sleep(dt)


def fly_to_local_point(
    master,
    target_xyz: Vec3,
    yaw_deg: float,
    speed_mps: float,
    hz: float = NAV_HZ,
):
    start_xyz = get_local_position(master, timeout_s=2.0)
    fly_line_local_streamed(
        master=master,
        start_xyz=start_xyz,
        end_xyz=target_xyz,
        yaw_deg=yaw_deg,
        speed_mps=speed_mps,
        hz=hz,
    )
    hold_local_target_until_reached(
        master=master,
        target_xyz=target_xyz,
        yaw_deg=yaw_deg,
        hz=hz,
    )


# =========================================================
# cleaning-stage action hooks
# =========================================================

def perform_stage_action(
    master,
    stage_cfg: Dict,
    target: dict,
    action_callback: Optional[Callable[[object, Dict, dict], None]] = None,
):
    """
    Placeholder for the real cleaning command.

    Later, replace or augment this with your actual pump/solenoid trigger,
    ESP32 serial command, detergent valve command, or nozzle routine.
    """
    print(
        f"Applying {stage_cfg['label']} at target #{target['id']} "
        f"for {stage_cfg['spray_duration_s']:.1f} s"
    )

    if action_callback is not None:
        action_callback(master, stage_cfg, target)
    else:
        # placeholder timing so the mission structure is complete
        time.sleep(stage_cfg["spray_duration_s"])

    time.sleep(stage_cfg.get("dwell_after_action_s", 0.0))


# =========================================================
# mission execution
# =========================================================

def visit_corrosion_targets(
    master,
    targets: List[dict],
    speed_mps: float,
    yaw_hold_deg: float,
    stage_cfg: Dict,
    action_callback: Optional[Callable[[object, Dict, dict], None]] = None,
):
    for i, target in enumerate(targets, start=1):
        xyz = target["target_xyz"]
        print(f"Target {i}/{len(targets)} -> local NED {xyz}")

        fly_to_local_point(
            master=master,
            target_xyz=xyz,
            yaw_deg=yaw_hold_deg,
            speed_mps=speed_mps,
            hz=NAV_HZ,
        )

        # short settle before starting spray / treatment
        time.sleep(POINT_SETTLE_S)

        perform_stage_action(
            master=master,
            stage_cfg=stage_cfg,
            target=target,
            action_callback=action_callback,
        )

        # small hold after the action to avoid abrupt motion
        time.sleep(ACTION_SAFETY_HOLD_S)


def execute_cleaning_stage(
    master,
    spec: dict,
    detections_json_path: str,
    stage_key: str,
    yaw_hold_deg: float = 0.0,
    action_callback: Optional[Callable[[object, Dict, dict], None]] = None,
    reorder_targets: bool = True,
):
    if stage_key not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage '{stage_key}'. Available: {list(STAGE_CONFIGS.keys())}")

    stage_cfg = STAGE_CONFIGS[stage_key]
    params = load_cleaning_parameters(spec)

    detections_payload = load_detections_json(detections_json_path)
    targets = extract_corrosion_targets_from_json(detections_payload)

    print(f"Loaded {len(targets)} corrosion targets from {detections_json_path}")
    print(f"Executing {stage_cfg['label']}")

    takeoff(master, params["start_alt_m"])
    wait_until_altitude(master, params["start_alt_m"], tol_m=TARGET_Z_TOL_M, timeout_s=TAKEOFF_TIMEOUT_S)

    start_xyz = get_local_position(master, timeout_s=2.0)
    if reorder_targets:
        targets = sort_targets_nearest_neighbor(start_xyz, targets)

    visit_corrosion_targets(
        master=master,
        targets=targets,
        speed_mps=params["speed_mps"],
        yaw_hold_deg=yaw_hold_deg,
        stage_cfg=stage_cfg,
        action_callback=action_callback,
    )

    print(f"Stage complete: {stage_cfg['label']}")
    rtl(master)

    return {
        "stage_key": stage_key,
        "stage_cfg": stage_cfg,
        "target_count": len(targets),
    }


# =========================================================
# default user action callback
# =========================================================

def noop_stage_action_callback(master, stage_cfg: Dict, target: dict):
    _ = (master, stage_cfg, target)
    return


# =========================================================
# main
# =========================================================

def main():
    import sys

    if len(sys.argv) < 4:
        print(
            "usage: python3 cleaning_stage_master.py "
            "<path_to_gui_exported_json> <detections_json_path> <stage_key>"
        )
        print(f"available stages: {list(STAGE_CONFIGS.keys())}")
        return

    spec_path = sys.argv[1]
    detections_json_path = sys.argv[2]
    stage_key = sys.argv[3]

    spec = load_gui_spec(spec_path)

    master = connect_sitl(DEFAULT_CONNECTION_URL)
    set_mode(master, "GUIDED")
    arm(master)

    execute_cleaning_stage(
        master=master,
        spec=spec,
        detections_json_path=detections_json_path,
        stage_key=stage_key,
        yaw_hold_deg=0.0,
        action_callback=noop_stage_action_callback,
        reorder_targets=True,
    )


if __name__ == "__main__":
    main()
