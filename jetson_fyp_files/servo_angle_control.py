from pymavlink import mavutil
import time

# ===== CONNECTION =====
PORT = "/dev/ttyTHS1"
BAUD = 115200

PAN_SERVO = 10
TILT_SERVO = 9

# ===== CALIBRATION =====
PAN_MIN_PWM = 1100
PAN_CENTER_PWM = 1500
PAN_MAX_PWM = 1900
PAN_MAX_ANGLE = 40   # degrees

TILT_MIN_PWM = 1100
TILT_CENTER_PWM = 1500
TILT_MAX_PWM = 1800
TILT_MIN_ANGLE = -30
TILT_MAX_ANGLE = 40

# ===== FUNCTIONS =====

def connect_pixhawk():
    print(f"Connecting to Pixhawk on {PORT}...")
    master = mavutil.mavlink_connection(PORT, baud=BAUD)

    hb = master.wait_heartbeat(timeout=15)
    if hb is None:
        raise Exception("No heartbeat received")

    print("Connected to Pixhawk")
    return master


def angle_to_pwm(angle, min_pwm, center_pwm, max_pwm, min_angle, max_angle):
    # clamp angle
    angle = max(min_angle, min(max_angle, angle))

    if angle >= 0:
        pwm = center_pwm + (angle / max_angle) * (max_pwm - center_pwm)
    else:
        pwm = center_pwm + (angle / abs(min_angle)) * (center_pwm - min_pwm)

    return int(pwm)


def send_servo(master, servo_num, pwm):
    print(f"Servo {servo_num} → PWM {pwm}")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_num,
        pwm,
        0, 0, 0, 0, 0
    )
    time.sleep(1)


# ===== MAIN =====

def main():
    master = connect_pixhawk()

    while True:
        try:
            pan_angle = float(input("\nEnter PAN angle (-40 to 40): "))
            tilt_angle = float(input("Enter TILT angle (-30 to 40): "))

            pan_pwm = angle_to_pwm(
                pan_angle,
                PAN_MIN_PWM, PAN_CENTER_PWM, PAN_MAX_PWM,
                -PAN_MAX_ANGLE, PAN_MAX_ANGLE
            )

            tilt_pwm = angle_to_pwm(
                tilt_angle,
                TILT_MIN_PWM, TILT_CENTER_PWM, TILT_MAX_PWM,
                TILT_MIN_ANGLE, TILT_MAX_ANGLE
            )

            send_servo(master, PAN_SERVO, pan_pwm)
            send_servo(master, TILT_SERVO, tilt_pwm)

        except ValueError:
            print("Invalid input, please enter numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
