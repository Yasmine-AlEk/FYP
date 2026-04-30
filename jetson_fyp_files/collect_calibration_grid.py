import csv
import time
from pathlib import Path

import cv2

from servo_angle_control import (
    connect_pixhawk,
    angle_to_pwm,
    send_servo,
    PAN_MIN_PWM,
    PAN_CENTER_PWM,
    PAN_MAX_PWM,
    PAN_MAX_ANGLE,
    TILT_MIN_PWM,
    TILT_CENTER_PWM,
    TILT_MAX_PWM,
    TILT_MIN_ANGLE,
    TILT_MAX_ANGLE,
    PAN_SERVO,
    TILT_SERVO,
)

from laser_center import (
    capture_frame,
    detect_laser_dot,
    compute_center_error,
    make_overlay,
)


# =========================================================
# CENTER-ALIGNED ANGLES YOU FOUND
# =========================================================
THETA_PAN_CENTER_DEG = 8.0
THETA_PITCH_CENTER_DEG = 5.0

# small offsets around the center
PAN_OFFSETS_DEG = [-2, -1, 0, 1, 2]
PITCH_OFFSETS_DEG = [-2, -1, 0, 1, 2]

# extra wait after both servos move
EXTRA_SETTLE_TIME_S = 0.2

# output folder
OUTPUT_DIR = Path("calibration_grid_outputs")
CSV_PATH = OUTPUT_DIR / "calibration_samples.csv"


# =========================================================
# Servo helper
# =========================================================
def set_pan_tilt(master, pan_angle_deg, tilt_angle_deg):
    pan_pwm = angle_to_pwm(
        pan_angle_deg,
        PAN_MIN_PWM, PAN_CENTER_PWM, PAN_MAX_PWM,
        -PAN_MAX_ANGLE, PAN_MAX_ANGLE,
    )

    tilt_pwm = angle_to_pwm(
        tilt_angle_deg,
        TILT_MIN_PWM, TILT_CENTER_PWM, TILT_MAX_PWM,
        TILT_MIN_ANGLE, TILT_MAX_ANGLE,
    )

    print(f"PAN  {pan_angle_deg:.2f} deg -> PWM {pan_pwm}")
    print(f"TILT {tilt_angle_deg:.2f} deg -> PWM {tilt_pwm}")

    send_servo(master, PAN_SERVO, pan_pwm)
    send_servo(master, TILT_SERVO, tilt_pwm)
    time.sleep(EXTRA_SETTLE_TIME_S)


# =========================================================
# Main collection routine
# =========================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # save run info
    run_info_path = OUTPUT_DIR / "run_info.txt"
    run_info_path.write_text(
        "\n".join([
            f"theta_pan_center_deg={THETA_PAN_CENTER_DEG}",
            f"theta_pitch_center_deg={THETA_PITCH_CENTER_DEG}",
            f"pan_offsets_deg={PAN_OFFSETS_DEG}",
            f"pitch_offsets_deg={PITCH_OFFSETS_DEG}",
        ])
    )

    master = connect_pixhawk()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx",
            "theta_pan_deg",
            "theta_pitch_deg",
            "delta_theta_pan_deg",
            "delta_theta_pitch_deg",
            "u_c",
            "v_c",
            "u_i",
            "v_i",
            "delta_u_i",
            "delta_v_i",
            "detected",
            "raw_image",
            "mask_image",
            "overlay_image",
        ])

        sample_idx = 0

        for dpan in PAN_OFFSETS_DEG:
            for dpitch in PITCH_OFFSETS_DEG:
                theta_pan_i = THETA_PAN_CENTER_DEG + dpan
                theta_pitch_i = THETA_PITCH_CENTER_DEG + dpitch

                print("\n==================================================")
                print(f"Sample {sample_idx:02d}")
                print(f"theta_pan_i   = {theta_pan_i:.2f} deg")
                print(f"theta_pitch_i = {theta_pitch_i:.2f} deg")
                print(f"delta_pan     = {dpan:+.2f} deg")
                print(f"delta_pitch   = {dpitch:+.2f} deg")
                print("==================================================")

                # move servos
                set_pan_tilt(master, theta_pan_i, theta_pitch_i)

                # capture frame
                frame = capture_frame()

                raw_name = f"raw_{sample_idx:02d}_pan_{theta_pan_i:.1f}_pitch_{theta_pitch_i:.1f}.jpg"
                raw_path = OUTPUT_DIR / raw_name
                cv2.imwrite(str(raw_path), frame)

                # detect laser
                dot_uv, mask, crop_y = detect_laser_dot(frame)

                mask_name = f"mask_{sample_idx:02d}_pan_{theta_pan_i:.1f}_pitch_{theta_pitch_i:.1f}.png"
                mask_path = OUTPUT_DIR / mask_name
                cv2.imwrite(str(mask_path), mask)

                if dot_uv is None:
                    print("[WARN] Laser not detected for this sample")

                    writer.writerow([
                        sample_idx,
                        theta_pan_i,
                        theta_pitch_i,
                        dpan,
                        dpitch,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        0,
                        raw_name,
                        mask_name,
                        "",
                    ])

                    sample_idx += 1
                    continue

                # compute center and error
                (u_c, v_c), (e_u, e_v) = compute_center_error(frame, dot_uv)
                u_i, v_i = dot_uv

                # IMPORTANT:
                # methodology uses Delta_u_i = u_i - u_c, Delta_v_i = v_i - v_c
                delta_u_i = u_i - u_c
                delta_v_i = v_i - v_c

                overlay = make_overlay(frame, dot_uv, crop_y)
                overlay_name = f"overlay_{sample_idx:02d}_pan_{theta_pan_i:.1f}_pitch_{theta_pitch_i:.1f}.jpg"
                overlay_path = OUTPUT_DIR / overlay_name
                cv2.imwrite(str(overlay_path), overlay)

                print(f"Image center: (u_c, v_c) = ({u_c:.1f}, {v_c:.1f})")
                print(f"Laser dot:    (u_i, v_i) = ({u_i:.1f}, {v_i:.1f})")
                print(f"Center error: e_u = {e_u:.1f}, e_v = {e_v:.1f}")
                print(f"Pixel shifts: Delta_u_i = {delta_u_i:.1f}, Delta_v_i = {delta_v_i:.1f}")

                writer.writerow([
                    sample_idx,
                    theta_pan_i,
                    theta_pitch_i,
                    dpan,
                    dpitch,
                    u_c,
                    v_c,
                    u_i,
                    v_i,
                    delta_u_i,
                    delta_v_i,
                    1,
                    raw_name,
                    mask_name,
                    overlay_name,
                ])

                sample_idx += 1

    print("\nDone.")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
