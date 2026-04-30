"""Minimal end-to-end demo wiring.

Replace the placeholder parts with:
- your real camera capture pipeline
- your camera capture pipeline
- your corrosion segmentation model
- your valve/pump trigger function
"""

from __future__ import annotations

import cv2
import numpy as np

from servo_interface import PixhawkServoController, PixhawkServoConfig, spray_action_stub
from calibration_types import GimbalCalibration
from section1_center_alignment import detect_red_laser_dot, get_image_center
from section2_calibration_matrix import collect_calibration_samples, fit_calibration_matrix, print_calibration_summary
from section3_coverage_kernel import SprayFootprint, draw_cleaning_points
from section4_cleaning_serpentine import run_cleaning_pass


# -----------------------------------------------------------------------------
# CAMERA GRABBER
# -----------------------------------------------------------------------------
class OpenCVCamera:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read frame from camera")
        return frame

    def close(self) -> None:
        self.cap.release()


# -----------------------------------------------------------------------------
# PLACEHOLDER CORROSION SEGMENTER
# -----------------------------------------------------------------------------
def dummy_corrosion_segmenter(frame_bgr: np.ndarray) -> np.ndarray:
    """Replace this with your transformer segmentation model.

    This dummy version thresholds dark regions just to keep the pipeline runnable.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    return mask


# -----------------------------------------------------------------------------
# PLACEHOLDER SPRAY ACTION
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------------
def example_fit_calibration() -> GimbalCalibration:
    cam = OpenCVCamera(0)
    servo = PixhawkServoController()
    try:
        frame = cam.grab()
        u_c, v_c = get_image_center(frame)

        # Replace these with the real center-aligned angles you found experimentally.
        theta_pan_center_deg = 0.0
        theta_pitch_center_deg = 0.0

        samples = collect_calibration_samples(
            frame_grabber=cam.grab,
            servo_controller=servo,
            image_center_uv=(u_c, v_c),
            theta_pan_center_deg=theta_pan_center_deg,
            theta_pitch_center_deg=theta_pitch_center_deg,
            pan_offsets_deg=(-2, -1, 0, 1, 2),
            pitch_offsets_deg=(-2, -1, 0, 1, 2),
        )

        calib = fit_calibration_matrix(
            samples=samples,
            image_center_uv=(u_c, v_c),
            theta_pan_center_deg=theta_pan_center_deg,
            theta_pitch_center_deg=theta_pitch_center_deg,
        )
        print_calibration_summary(calib)
        return calib
    finally:
        cam.close()


def example_run_cleaning(calib: GimbalCalibration) -> None:
    cam = OpenCVCamera(0)
    servo = PixhawkServoController()

    # Replace these with values estimated from the spray-footprint experiment.
    footprint = SprayFootprint(
        area_px=314.0,
        radius_px=10.0,
        diameter_px=20.0,
        width_px=20.0,
        height_px=20.0,
        radius_u_px=10.0,
        radius_v_px=10.0,
    )

    try:
        points = run_cleaning_pass(
            frame_grabber=cam.grab,
            segment_corrosion_mask=dummy_corrosion_segmenter,
            servo_controller=servo,
            spray_action=spray_action_stub,
            calib=calib,
            footprint=footprint,
        )

        frame = cam.grab()
        vis = draw_cleaning_points(frame, points, footprint.radius_px)
        cv2.imwrite("cleaning_points_preview.png", vis)
        print("[INFO] Saved preview to cleaning_points_preview.png")
    finally:
        cam.close()


if __name__ == "__main__":
    calibration = example_fit_calibration()
    example_run_cleaning(calibration)
