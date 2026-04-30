
from servo_interface import PixhawkServoController


def main() -> None:
    servo = PixhawkServoController()

    while True:
        try:
            pan_angle = float(input("Enter PAN angle in deg (-40 to 40): "))
            tilt_angle = float(input("Enter TILT angle in deg (-30 to 40): "))
            servo.set_pan_tilt(pan_angle, tilt_angle)
        except ValueError:
            print("Invalid input. Please enter numeric angles.")
        except KeyboardInterrupt:
            print("
Exiting manual servo test.")
            break


if __name__ == "__main__":
    main()
