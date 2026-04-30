
from __future__ import annotations

from pymavlink import mavutil
import time

PORT = "/dev/ttyTHS1"
BAUD = 115200

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

COMMAND_DELAY_S = 0.05
SETTLE_TIME_S = 0.30


def connect_pixhawk():
    print(f"Connecting to Pixhawk on {PORT}...")
    master = mavutil.mavlink_connection(PORT, baud=BAUD)
    hb = master.wait_heartbeat(timeout=15)
    if hb is None:
        raise RuntimeError("No heartbeat received")
    print("Connected to Pixhawk")
    return master


def angle_to_pwm(angle_deg, min_pwm, center_pwm, max_pwm, min_angle_deg, max_angle_deg):
    angle_deg = max(min_angle_deg, min(max_angle_deg, angle_deg))

    if angle_deg >= 0:
        pwm = center_pwm + (angle_deg / max_angle_deg) * (max_pwm - center_pwm)
    else:
        pwm = center_pwm + (angle_deg / abs(min_angle_deg)) * (center_pwm - min_pwm)

    return int(round(pwm))


def send_servo(master, servo_num, pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_num,
        pwm,
        0, 0, 0, 0, 0,
    )


def set_pan_tilt(master, pan_deg, tilt_deg):
    pan_pwm = angle_to_pwm(
        pan_deg,
        PAN_MIN_PWM, PAN_CENTER_PWM, PAN_MAX_PWM,
        PAN_MIN_ANGLE, PAN_MAX_ANGLE,
    )
    tilt_pwm = angle_to_pwm(
        tilt_deg,
        TILT_MIN_PWM, TILT_CENTER_PWM, TILT_MAX_PWM,
        TILT_MIN_ANGLE, TILT_MAX_ANGLE,
    )

    print(f"PAN {pan_deg:.2f} deg -> {pan_pwm} PWM")
    print(f"TILT {tilt_deg:.2f} deg -> {tilt_pwm} PWM")

    send_servo(master, PAN_SERVO, pan_pwm)
    time.sleep(COMMAND_DELAY_S)
    send_servo(master, TILT_SERVO, tilt_pwm)
    time.sleep(SETTLE_TIME_S)


def main():
    master = connect_pixhawk()

    while True:
        try:
            pan = float(input("Enter PAN angle (-40 to 40): "))
            tilt = float(input("Enter TILT angle (-30 to 40): "))
            set_pan_tilt(master, pan, tilt)
        except ValueError:
            print("Invalid input, please enter numbers")
        except KeyboardInterrupt:
            print("
Exiting...")
            break


if __name__ == "__main__":
    main()
