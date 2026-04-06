import math
import time
from typing import Tuple

import numpy as np
import airsim
from pymavlink import mavutil


# -----------------------------
# MAVLink helpers (same style as your backend)
# -----------------------------
def connect_sitl(url: str = "udp:127.0.0.1:14550"):
    master = mavutil.mavlink_connection(url)
    master.wait_heartbeat()
    print(f"Connected MAVLink: system {master.target_system}, component {master.target_component}")
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
    time.sleep(8)


def get_local_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No LOCAL_POSITION_NED received.")
    return float(msg.x), float(msg.y), float(msg.z)


def get_yaw_rad(master, timeout_s: float = 2.0) -> float:
    """
    ATTITUDE yaw is in radians. In ArduPilot SITL this is typically available.
    """
    msg = master.recv_match(type="ATTITUDE", blocking=True, timeout=timeout_s)
    if msg is None:
        raise TimeoutError("No ATTITUDE received.")
    return float(msg.yaw)


def send_velocity_body(master, vx_fwd: float, vy_right: float, vz_down: float, yaw_deg: float = None):
    """
    Send velocity command in BODY_NED frame.
    vx_fwd: forward (m/s)
    vy_right: right (m/s)
    vz_down: down (m/s)
    Optionally set yaw (deg) if you want; otherwise yaw ignored.
    """
    if yaw_deg is None:
        # ignore yaw + yaw_rate
        type_mask = 0b0000001110000111
        yaw = 0.0
    else:
        # include yaw, ignore yaw_rate
        type_mask = 0b0000001100000111
        yaw = math.radians(yaw_deg)

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,              # x,y,z ignored
        vx_fwd, vy_right, vz_down,
        0, 0, 0,              # ax,ay,az ignored
        yaw,
        0
    )


# -----------------------------
# AirSim depth (read-only sensor path)
# -----------------------------
def connect_airsim_for_depth():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected AirSim API (for depth only).")
    return client


def depth_distance_estimate_m(client, camera_name: str = "0", patch_half: int = 20) -> float:
    """
    Robust distance estimate using DepthPerspective:
    - take center patch
    - median as robust estimator
    """
    resp = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True)
    ])[0]

    if resp.width == 0 or resp.height == 0:
        return float("nan")

    depth = np.array(resp.image_data_float, dtype=np.float32).reshape(resp.height, resp.width)

    h, w = depth.shape
    cx, cy = w // 2, h // 2
    patch = depth[cy - patch_half: cy + patch_half, cx - patch_half: cx + patch_half]

    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > 0.2) & (patch < 50.0)]
    if patch.size == 0:
        return float("nan")

    return float(np.median(patch))


# -----------------------------
# PID
# -----------------------------
class PID:
    def __init__(self, kp: float, ki: float, kd: float, i_clamp: float = 2.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.i_clamp = float(i_clamp)
        self.i = 0.0
        self.prev_e = 0.0
        self.has_prev = False

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.has_prev = False

    def step(self, e: float, dt: float) -> float:
        self.i += e * dt
        self.i = max(-self.i_clamp, min(self.i_clamp, self.i))

        if self.has_prev and dt > 1e-6:
            de = (e - self.prev_e) / dt
        else:
            de = 0.0

        self.prev_e = e
        self.has_prev = True

        return self.kp * e + self.ki * self.i + self.kd * de


# -----------------------------
# Mission: standoff PID while moving forward ~2m
# -----------------------------
def execute_standoff_pid_mavlink(
    master,
    airsim_client,
    target_dist_m: float = 2.0,
    forward_goal_m: float = 2.0,
    start_alt_m: float = 3.0,
    camera_name: str = "0",
    hz: float = 10.0,
):
    """
    Keeps the drone at ~target_dist_m from the wall using depth camera + PID.
    If dist > target -> forward; if dist < target -> backward.
    Uses MAVLink velocity control (BODY_NED).

    Threshold logic:
      - deadband: ignores tiny noise around 0 error
      - trigger_thresh: PID only runs if |error| >= trigger_thresh, otherwise hold (vx=0)
    """

    # Tune these gains in sim
    pid = PID(kp=0.8, ki=0.05, kd=0.2, i_clamp=2.0)

    # Velocity clamp (m/s)
    vx_min, vx_max = -0.8, 0.8

    # Thresholds (meters)
    deadband = 0.02          # tiny noise ignore
    trigger_thresh = 0.10    # PID only runs outside this band (10 cm)
    hold_vx = 0.0            # command when close enough

    # Setup
    set_mode(master, "GUIDED")
    arm(master)
    takeoff(master, start_alt_m)

    # Reference pose for progress measurement (optional "stop after ~2m forward")
    x0, y0, z0 = get_local_position(master)
    yaw0 = get_yaw_rad(master)
    fwd_unit = (math.cos(yaw0), math.sin(yaw0))  # in N/E

    print("Starting standoff PID (MAVLink control, AirSim depth)...")
    print(f"Target distance: {target_dist_m:.2f} m | Forward goal: {forward_goal_m:.2f} m")
    print(f"deadband={deadband:.2f}m | trigger_thresh={trigger_thresh:.2f}m")

    dt = 1.0 / hz
    last_t = time.time()

    while True:
        now = time.time()
        dt_real = now - last_t
        last_t = now
        if dt_real <= 0:
            dt_real = dt

        # Forward progress from local position projected on initial yaw
        x, y, z = get_local_position(master)
        dx = x - x0
        dy = y - y0
        forward_progress = dx * fwd_unit[0] + dy * fwd_unit[1]

        # Depth-based distance estimate
        dist = depth_distance_estimate_m(airsim_client, camera_name=camera_name, patch_half=20)
        if not math.isfinite(dist):
            # stop if depth invalid
            send_velocity_body(master, 0.0, 0.0, 0.0)
            print("Depth invalid -> holding...")
            time.sleep(dt)
            continue

        # Error: + => too far (go forward), - => too close (go backward)
        e = dist - target_dist_m

        # 1) deadband: ignore tiny noise
        if abs(e) < deadband:
            e = 0.0

        # 2) trigger threshold: only run PID if error is large enough
        if abs(e) < trigger_thresh:
            vx_cmd = hold_vx
            pid.reset()  # prevents integral windup/drift while close enough
        else:
            vx_cmd = pid.step(e, dt_real)
            vx_cmd = max(vx_min, min(vx_max, vx_cmd))

        # Command forward/back in body frame
        send_velocity_body(master, vx_cmd, 0.0, 0.0)

        print(
            f"dist={dist:.2f}m | err={e:+.2f} | vx={vx_cmd:+.2f}m/s | "
            f"fwd={forward_progress:.2f}/{forward_goal_m:.2f}m"
        )

        # Stop condition: reached goal AND close to desired standoff
        if forward_progress >= forward_goal_m and abs(dist - target_dist_m) < 0.15:
            break

        time.sleep(dt)

    # Stop at end
    for _ in range(int(1.0 * hz)):
        send_velocity_body(master, 0.0, 0.0, 0.0)
        time.sleep(dt)

    print("Done. Stopped at target distance.")


if __name__ == "__main__":
    # MAVLink motion control (AirSim MAVLink port)
    master = connect_sitl("udp:127.0.0.1:14550")

    # AirSim depth sensor only (read-only)
    airsim_client = connect_airsim_for_depth()

    execute_standoff_pid_mavlink(
        master,
        airsim_client,
        target_dist_m=2.0,
        forward_goal_m=2.0,
        start_alt_m=3.0,
        camera_name="0",
        hz=10.0,
    )
