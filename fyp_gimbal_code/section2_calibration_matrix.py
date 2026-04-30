"""Section 2: fit the 2x2 calibration matrix.

This file assumes you already know the center-aligned pan/pitch angles.
It collects samples around that center and estimates:

    [Delta_u] = A [Delta_theta_pan]
    [Delta_v]     [Delta_theta_pitch]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import itertools
import time
import cv2
import numpy as np

from section1_center_alignment import detect_red_laser_dot
from calibration_types import GimbalCalibration


FrameGrabber = Callable[[], np.ndarray]


@dataclass
class CalibrationSample:
    theta_pan_deg: float
    theta_pitch_deg: float
    delta_theta_pan_deg: float
    delta_theta_pitch_deg: float
    u: float
    v: float
    delta_u: float
    delta_v: float


def collect_calibration_samples(
    frame_grabber: FrameGrabber,
    servo_controller,
    image_center_uv: tuple[float, float],
    theta_pan_center_deg: float,
    theta_pitch_center_deg: float,
    pan_offsets_deg: Sequence[float] = (-2, -1, 0, 1, 2),
    pitch_offsets_deg: Sequence[float] = (-2, -1, 0, 1, 2),
    settle_time_s: float = 0.4,
) -> list[CalibrationSample]:
    """Collect a grid of laser-dot samples around the center alignment.

    Example offsets:
        pan   = -2, -1, 0, +1, +2 deg
        pitch = -2, -1, 0, +1, +2 deg
    """
    u_c, v_c = image_center_uv
    samples: list[CalibrationSample] = []

    for dpan, dpitch in itertools.product(pan_offsets_deg, pitch_offsets_deg):
        theta_pan = theta_pan_center_deg + dpan
        theta_pitch = theta_pitch_center_deg + dpitch

        servo_controller.set_pan_tilt(theta_pan, theta_pitch)
        time.sleep(settle_time_s)
        frame = frame_grabber()

        result = detect_red_laser_dot(frame)
        if result is None:
            print(f"[WARN] No laser dot found at pan={theta_pan:.2f}, pitch={theta_pitch:.2f}")
            continue

        u, v, _ = result
        samples.append(
            CalibrationSample(
                theta_pan_deg=theta_pan,
                theta_pitch_deg=theta_pitch,
                delta_theta_pan_deg=dpan,
                delta_theta_pitch_deg=dpitch,
                u=u,
                v=v,
                delta_u=u - u_c,
                delta_v=v - v_c,
            )
        )
        print(f"[OK] pan={theta_pan:.2f}, pitch={theta_pitch:.2f}, u={u:.1f}, v={v:.1f}")

    return samples


def fit_calibration_matrix(
    samples: Iterable[CalibrationSample],
    image_center_uv: tuple[float, float],
    theta_pan_center_deg: float,
    theta_pitch_center_deg: float,
) -> GimbalCalibration:
    """Estimate A using least squares.

    Model:
        [Delta_u] = A [Delta_theta_pan]
        [Delta_v]     [Delta_theta_pitch]
    """
    sample_list = list(samples)
    if len(sample_list) < 4:
        raise ValueError("Need at least 4 valid calibration samples to fit a 2x2 matrix.")

    # X contains angle deltas. Y contains pixel deltas.
    X = np.array([[s.delta_theta_pan_deg, s.delta_theta_pitch_deg] for s in sample_list], dtype=float)
    Y = np.array([[s.delta_u, s.delta_v] for s in sample_list], dtype=float)

    # Solve X * B ~= Y. Here B = A^T.
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = B.T

    if np.linalg.det(A) == 0:
        raise ValueError("Calibration matrix is singular. Try collecting more diverse samples.")

    A_inv = np.linalg.inv(A)
    u_c, v_c = image_center_uv
    return GimbalCalibration(
        image_center_u=u_c,
        image_center_v=v_c,
        theta_pan_center_deg=theta_pan_center_deg,
        theta_pitch_center_deg=theta_pitch_center_deg,
        A=A.tolist(),
        A_inv=A_inv.tolist(),
    )


def print_calibration_summary(calib: GimbalCalibration) -> None:
    A = np.array(calib.A)
    A_inv = np.array(calib.A_inv)
    print("\n=== Calibration summary ===")
    print(f"Image center: ({calib.image_center_u:.2f}, {calib.image_center_v:.2f})")
    print(f"Center angles: pan={calib.theta_pan_center_deg:.3f} deg, pitch={calib.theta_pitch_center_deg:.3f} deg")
    print("A =")
    print(A)
    print("A_inv =")
    print(A_inv)
