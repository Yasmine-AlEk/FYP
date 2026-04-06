import json  # read the gui json mission spec
import math  # basic math for distances and angles
import time  # sleeps between commands
from typing import Tuple  # type hints for tuples

from pymavlink import mavutil  # mavlink connection and message helpers


# connects to the autopilot over mavlink and waits for a heartbeat
def connect_sitl(url: str = "udp:127.0.0.1:14550"):
    master = mavutil.mavlink_connection(url)  # open mavlink link to the autopilot
    master.wait_heartbeat()  # wait until autopilot is alive
    print(f"Connected: system {master.target_system}, component {master.target_component}")  # show ids
    return master  # return mavlink handle


# sets the flight mode (guided) using the autopilot mode mapping
def set_mode(master, mode: str):
    mode_map = master.mode_mapping()  # get available modes from autopilot
    if mode not in mode_map:  # check requested mode exists
        raise RuntimeError(f"Mode {mode} not available. Available: {list(mode_map.keys())}")  # fail early
    mode_id = mode_map[mode]  # get numeric mode id
    master.mav.set_mode_send(  # send mode change command
        master.target_system,  # target vehicle
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,  # use custom mode field
        mode_id  # mode to set
    )
    time.sleep(1)  # give the flight stack time to switch


# arms the vehicle motors so it can take off and fly
def arm(master):
    master.mav.command_long_send(  # send arm command
        master.target_system, master.target_component,  # target ids
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # arm/disarm command
        0, 1, 0, 0, 0, 0, 0, 0  # arm=1 then unused params
    )
    print("Arming...")  # log intent
    time.sleep(2)  # wait for arming to take effect


# commands a takeoff to a target relative altitude
def takeoff(master, target_alt_m: float):
    master.mav.command_long_send(  # send takeoff command
        master.target_system,  # target vehicle
        master.target_component,  # target component
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,  # takeoff command
        0, 0, 0, 0, 0, 0, 0, target_alt_m  # altitude in last param for many stacks
    )
    print(f"Taking off to {target_alt_m:.1f}m...")  # log target altitude
    time.sleep(1)  # brief delay after command


# sends a global gps setpoint (lat lon relative altitude) for guided navigation
def goto_position_target_global_int(master, lat_deg: float, lon_deg: float, alt_rel_m: float):
    type_mask = 0b0000111111111000  # enable only position fields, ignore the rest
    master.mav.set_position_target_global_int_send(  # send global position setpoint
        0,  # time boot ms (0 ok for streaming)
        master.target_system,  # target system
        master.target_component,  # target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,  # lat/lon + relative altitude
        type_mask,  # what fields are used
        int(lat_deg * 1e7),  # lat as int
        int(lon_deg * 1e7),  # lon as int
        alt_rel_m,  # relative altitude (meters)
        0, 0, 0,  # velocity unused
        0, 0, 0,  # accel unused
        0, 0  # yaw/yaw rate unused
    )


# sends a local ned setpoint (x north y east z down) plus a yaw angle
def goto_position_target_local_ned(master, x: float, y: float, z: float, yaw_deg: float):
    yaw_rad = math.radians(yaw_deg)  # convert yaw to radians for mavlink
    type_mask = 0b0000111111111000  # enable position fields, ignore velocities

    master.mav.set_position_target_local_ned_send(  # send local ned setpoint
        0,  # time boot ms
        master.target_system,  # target system
        master.target_component,  # target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # local north east down frame
        type_mask,  # what fields are used
        x, y, z,  # local position
        0, 0, 0,  # velocity unused
        0, 0, 0,  # accel unused
        yaw_rad, 0  # yaw set, yaw rate unused
    )


# reads one global position message and returns lat lon and relative altitude
def get_global_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout_s)  # wait for gps/alt msg
    if msg is None:  # timeout case
        raise TimeoutError("No GLOBAL_POSITION_INT received.")  # stop mission if no data
    lat = msg.lat / 1e7  # degrees
    lon = msg.lon / 1e7  # degrees
    rel_alt = msg.relative_alt / 1000.0  # meters
    return lat, lon, rel_alt  # return current global position


# reads one local position message and returns x y z in local ned
def get_local_position(master, timeout_s: float = 2.0) -> Tuple[float, float, float]:
    msg = master.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=timeout_s)  # wait for local pose msg
    if msg is None:  # timeout case
        raise TimeoutError("No LOCAL_POSITION_NED received.")  # stop mission if no data
    return float(msg.x), float(msg.y), float(msg.z)  # return local ned x y z


# flies a straight line in local ned by streaming interpolated setpoints at a fixed rate
def fly_line_local(
    master,
    start_xyz: Tuple[float, float, float],
    end_xyz: Tuple[float, float, float],
    yaw_deg: float,
    speed_mps: float,
    hz: float = 10.0,
    min_time_s: float = 0.5,
):
    sx, sy, sz = start_xyz  # start point
    ex, ey, ez = end_xyz  # end point

    dx = ex - sx  # delta x
    dy = ey - sy  # delta y
    dz = ez - sz  # delta z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)  # line length

    speed = max(float(speed_mps), 0.05)  # clamp speed so time is not infinite
    T = max(min_time_s, dist / speed)  # planned travel time

    steps = max(2, int(T * hz))  # how many setpoints to send
    dt = 1.0 / hz  # time between setpoints

    for k in range(steps + 1):  # stream setpoints along the line
        a = k / steps  # interpolation factor
        x = sx + a * dx  # interpolated x
        y = sy + a * dy  # interpolated y
        z = sz + a * dz  # interpolated z
        goto_position_target_local_ned(master, x, y, z, yaw_deg=yaw_deg)  # send setpoint
        time.sleep(dt)  # wait for next setpoint





# converts two gps points into local north/east meter offsets around a reference lat lon
def latlon_to_ned_m(lat0, lon0, lat, lon) -> Tuple[float, float]:
    R = 6378137.0  # earth radius used for local approximation
    dlat = math.radians(lat - lat0)  # delta lat in radians
    dlon = math.radians(lon - lon0)  # delta lon in radians
    north = dlat * R  # meters north
    east = dlon * R * math.cos(math.radians(lat0))  # meters east
    return north, east  # return local offsets

# loads the json file exported by the gui into a python dictionary
def load_gui_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:  # open exported gui json
        return json.load(f)  # parse json into dict

# runs the wall scan with 2m overshoot before a and 2m overshoot after b, then zigzag per row
def execute_wall_scan(master, spec: dict):
    A = spec["structure"]["corners_gps"]["A"]  # read corner a gps
    B = spec["structure"]["corners_gps"]["B"]  # read corner b gps

    wall_total_h = float(spec["structure"]["dimensions_m"]["height"])  # treat height as total wall height
    start_alt_m = float(spec["scan_settings"]["start_alt_m"])  # start altitude
    speed = float(spec["scan_settings"]["speed_mps"])  # scan speed
    standoff = float(spec["scan_settings"]["standoff_m"])  # distance from wall line (kept)
    lane_spacing = float(spec["scan_settings"]["lane_spacing_m"])  # vertical row spacing

    scan_span = wall_total_h - start_alt_m  # how much vertical distance we will actually scan
    if scan_span <= 0:  # if start is above or equal to wall top there is nothing to scan
        raise ValueError("wall total height must be greater than start altitude")  # stop mission early

    extend_m = 2.0  # overshoot distance along the wall at both ends

    takeoff(master, start_alt_m)  # take off to start altitude
    time.sleep(6)  # wait to stabilize after takeoff

    print("going to corner a (gps)...")  # log start move
    goto_position_target_global_int(master, A["lat"], A["lon"], start_alt_m)  # go near a in global frame
    time.sleep(10)  # wait to reach the area

    origin_local = get_local_position(master)  # reads current local ned position to treat as a local origin anchor
    lat0, lon0, _ = get_global_position(master)  # reads current global lat lon to use as reference for meter conversion
    ox, oy, oz = origin_local  # stores the origin local coordinates for later offsetting

    a_n, a_e = latlon_to_ned_m(lat0, lon0, A["lat"], A["lon"])  # converts corner a gps into local north/east meters
    b_n, b_e = latlon_to_ned_m(lat0, lon0, B["lat"], B["lon"])  # converts corner b gps into local north/east meters

    ab_n = b_n - a_n  # computes the north component of the wall direction vector from a to b
    ab_e = b_e - a_e  # computes the east component of the wall direction vector from a to b
    ab_len = math.sqrt(ab_n * ab_n + ab_e * ab_e)  # computes the wall length in meters

    if ab_len < 1e-3:  # avoid division by zero
        raise ValueError("a and b are too close to define a wall direction")  # invalid mission

    u_w_n = ab_n / ab_len  # makes a unit direction vector along the wall (north component)
    u_w_e = ab_e / ab_len  # makes a unit direction vector along the wall (east component)

    u_p_n = u_w_e  # computes a perpendicular direction (left normal) north component
    u_p_e = -u_w_n  # computes a perpendicular direction (left normal) east component

    yaw_hold = 0.0  # sets a constant yaw command for all rows (no yaw planning in this version)

    def local_from_ne(off_n: float, off_e: float, alt_m: float) -> Tuple[float, float, float]:
        return (ox + off_n, oy + off_e, -alt_m)  # converts north/east offsets + altitude into local ned x/y/z setpoint

    num_rows = int(math.ceil(scan_span / lane_spacing)) + 1  # computes rows from the scan span, not total height

    for i in range(num_rows):  # iterates over each row index
        climb = min(i * lane_spacing, scan_span)  # climb within the scan span
        alt_row = start_alt_m + climb  # altitude for this row
        if alt_row > wall_total_h:  # final clamp for safety
            alt_row = wall_total_h  # never command above the wall top

        a_ext_n = a_n - extend_m * u_w_n  # computes the point 2m before a along the wall direction
        a_ext_e = a_e - extend_m * u_w_e  # computes the point 2m before a along the wall direction
        b_ext_n = b_n + extend_m * u_w_n  # computes the point 2m after b along the wall direction
        b_ext_e = b_e + extend_m * u_w_e  # computes the point 2m after b along the wall direction

        leftA_n = a_ext_n + standoff * u_p_n  # shifts the start endpoint away from the wall by standoff distance
        leftA_e = a_ext_e + standoff * u_p_e  # shifts the start endpoint away from the wall by standoff distance
        rightB_n = b_ext_n + standoff * u_p_n  # shifts the end endpoint away from the wall by standoff distance
        rightB_e = b_ext_e + standoff * u_p_e  # shifts the end endpoint away from the wall by standoff distance

        if i % 2 == 0:  # checks row parity to choose direction
            p0_n, p0_e = leftA_n, leftA_e  # even rows start on the a side (before a)
            p1_n, p1_e = rightB_n, rightB_e  # even rows end on the b side (after b)
        else:
            p0_n, p0_e = rightB_n, rightB_e  # odd rows start on the b side (after b)
            p1_n, p1_e = leftA_n, leftA_e  # odd rows end on the a side (before a)

        p0 = local_from_ne(p0_n, p0_e, alt_row)  # converts start point to local ned setpoint at the row altitude
        p1 = local_from_ne(p1_n, p1_e, alt_row)  # converts end point to local ned setpoint at the row altitude

        print(f"row {i+1}/{num_rows} alt={alt_row:.2f} speed={speed:.2f} extend=2.00 standoff={standoff:.2f}")  # prints row plan

        fly_line_local(master, p0, p1, yaw_hold, speed_mps=speed, hz=10.0)  # flies the row by streaming setpoints

    print("scan complete.")  # prints when all rows are done


if __name__ == "__main__":
    import sys  # read command line args

    if len(sys.argv) < 2:  # require a json file path
        print("usage: python3 scan_backend.py <path_to_gui_exported_json>")  # show usage
        sys.exit(1)  # exit with error

    spec_path = sys.argv[1]  # take json path from args
    spec = load_gui_spec(spec_path)  # load gui spec

    master = connect_sitl("udp:127.0.0.1:14550")  # connect to mavlink udp
    set_mode(master, "GUIDED")  # switch to guided control
    arm(master)  # arm motors

    execute_wall_scan(master, spec)  # run the scan mission