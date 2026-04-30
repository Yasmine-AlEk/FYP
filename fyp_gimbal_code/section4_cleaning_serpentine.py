"""Section 4: image-based cleaning loop using the inverse calibration matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import time
import cv2
import numpy as np

from calibration_types import GimbalCalibration
from section3_coverage_kernel import SprayFootprint, generate_cleaning_points_from_mask


FrameGrabber = Callable[[], np.ndarray]
MaskInferencer = Callable[[np.ndarray], np.ndarray]
SprayAction = Callable[[float], None]


@dataclass
class CleaningRunConfig:
    dwell_time_s: float = 0.2
    servo_settle_time_s: float = 0.25
    preview: bool = False


def target_angles_from_pixel(
    u_t: float,
    v_t: float,
    calib: GimbalCalibration,
) -> tuple[float, float, float, float]:
    """Convert a desired target pixel into pan/pitch commands.

    Returns:
        delta_u, delta_v, theta_pan_target_deg, theta_pitch_target_deg
    """
    delta_u = float(u_t - calib.image_center_u)
    delta_v = float(v_t - calib.image_center_v)

    delta_theta = calib.A_inv_np @ np.array([delta_u, delta_v], dtype=float)
    delta_theta_pan = float(delta_theta[0])
    delta_theta_pitch = float(delta_theta[1])

    theta_pan_target = calib.theta_pan_center_deg + delta_theta_pan
    theta_pitch_target = calib.theta_pitch_center_deg + delta_theta_pitch
    return delta_u, delta_v, theta_pan_target, theta_pitch_target


def run_cleaning_pass(
    frame_grabber: FrameGrabber,
    segment_corrosion_mask: MaskInferencer,
    servo_controller,
    spray_action: SprayAction,
    calib: GimbalCalibration,
    footprint: SprayFootprint,
    config: CleaningRunConfig | None = None,
) -> list[tuple[int, int]]:
    """Run one serpentine cleaning pass over the current corrosion mask."""
    if config is None:
        config = CleaningRunConfig()

    frame = frame_grabber()
    corrosion_mask = segment_corrosion_mask(frame)
    points = generate_cleaning_points_from_mask(corrosion_mask, footprint)

    print(f"[INFO] Generated {len(points)} cleaning points.")

    for idx, (u_t, v_t) in enumerate(points):
        delta_u, delta_v, pan_cmd, pitch_cmd = target_angles_from_pixel(u_t, v_t, calib)
        print(
            f"[PT {idx:03d}] target=({u_t},{v_t}), "
            f"delta_uv=({delta_u:.1f},{delta_v:.1f}), "
            f"pan={pan_cmd:.2f} deg, pitch={pitch_cmd:.2f} deg"
        )

        servo_controller.set_pan_tilt(pan_cmd, pitch_cmd)
        time.sleep(config.servo_settle_time_s)
        spray_action(config.dwell_time_s)

    return points
