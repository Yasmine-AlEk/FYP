"""Section 1: hose-camera center alignment.

This file helps you:
1) detect the laser dot in the image
2) compute the error from the image center
3) store the pan/pitch angles that place the laser on the camera center
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class LaserDetectConfig:
    lower_red_1: tuple[int, int, int] = (0, 120, 180)
    upper_red_1: tuple[int, int, int] = (12, 255, 255)
    lower_red_2: tuple[int, int, int] = (165, 120, 180)
    upper_red_2: tuple[int, int, int] = (179, 255, 255)
    min_area_px: int = 4
    blur_kernel: int = 5


@dataclass
class CenterAlignmentResult:
    image_center: Tuple[float, float]
    laser_dot: Tuple[float, float]
    error_uv: Tuple[float, float]
    theta_pan_center_deg: float
    theta_pitch_center_deg: float


def get_image_center(frame: np.ndarray) -> Tuple[float, float]:
    h, w = frame.shape[:2]
    return (w / 2.0, h / 2.0)


def detect_red_laser_dot(
    frame_bgr: np.ndarray,
    cfg: LaserDetectConfig | None = None,
) -> Optional[Tuple[float, float, np.ndarray]]:
    """Detect a red laser dot and return (u, v, debug_mask)."""
    if cfg is None:
        cfg = LaserDetectConfig()

    blurred = cv2.GaussianBlur(frame_bgr, (cfg.blur_kernel, cfg.blur_kernel), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array(cfg.lower_red_1), np.array(cfg.upper_red_1))
    mask2 = cv2.inRange(hsv, np.array(cfg.lower_red_2), np.array(cfg.upper_red_2))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None

    best_idx = -1
    best_score = -1.0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < cfg.min_area_px:
            continue

        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        roi = hsv[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        brightness_score = float(np.mean(roi[:, :, 2]))
        score = area + 0.02 * brightness_score
        if score > best_score:
            best_idx = idx
            best_score = score

    if best_idx == -1:
        return None

    u, v = centroids[best_idx]
    return float(u), float(v), mask


def compute_center_error(frame_bgr: np.ndarray, dot_uv: Tuple[float, float]) -> Tuple[float, float]:
    u_c, v_c = get_image_center(frame_bgr)
    u_0, v_0 = dot_uv
    e_u0 = u_c - u_0
    e_v0 = v_c - v_0
    return e_u0, e_v0


def make_debug_overlay(frame_bgr: np.ndarray, dot_uv: Tuple[float, float]) -> np.ndarray:
    overlay = frame_bgr.copy()
    u_c, v_c = get_image_center(frame_bgr)
    u_0, v_0 = dot_uv

    cv2.drawMarker(overlay, (int(round(u_c)), int(round(v_c))), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.circle(overlay, (int(round(u_0)), int(round(v_0))), 6, (0, 0, 255), 2)
    cv2.line(overlay, (int(round(u_c)), int(round(v_c))), (int(round(u_0)), int(round(v_0))), (255, 255, 0), 2)
    return overlay


def build_center_alignment_result(
    frame_bgr: np.ndarray,
    dot_uv: Tuple[float, float],
    theta_pan_center_deg: float,
    theta_pitch_center_deg: float,
) -> CenterAlignmentResult:
    u_c, v_c = get_image_center(frame_bgr)
    e_u0, e_v0 = compute_center_error(frame_bgr, dot_uv)
    return CenterAlignmentResult(
        image_center=(u_c, v_c),
        laser_dot=dot_uv,
        error_uv=(e_u0, e_v0),
        theta_pan_center_deg=theta_pan_center_deg,
        theta_pitch_center_deg=theta_pitch_center_deg,
    )
