"""Section 3: corrosion coverage and spray footprint.

This file helps you:
1) estimate the image-space spray footprint
2) compute an equivalent circular or elliptical kernel
3) generate cleaning points with overlap
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np


@dataclass
class SprayFootprint:
    area_px: float
    diameter_px: float
    radius_px: float
    width_px: float
    height_px: float
    radius_u_px: float
    radius_v_px: float


@dataclass
class CleaningGridConfig:
    overlap_ratio: float = 0.25
    use_bbox: bool = True
    min_mask_overlap_px: int = 1


def clean_binary_mask(mask_u8: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_spray_footprint(before_bgr: np.ndarray, after_bgr: np.ndarray, thresh: int = 25) -> np.ndarray:
    """Simple before/after difference-based footprint segmentation.

    Adjust this if your liquid color, lighting, or wall texture changes.
    """
    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(after_gray, before_gray)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return clean_binary_mask(mask)


def estimate_spray_footprint(footprint_mask_u8: np.ndarray) -> SprayFootprint:
    ys, xs = np.where(footprint_mask_u8 > 0)
    if len(xs) == 0:
        raise ValueError("Spray footprint mask is empty.")

    area_px = float(len(xs))
    radius_px = float(np.sqrt(area_px / np.pi))
    diameter_px = 2.0 * radius_px

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width_px = x_max - x_min + 1.0
    height_px = y_max - y_min + 1.0

    radius_u_px = width_px / 2.0
    radius_v_px = height_px / 2.0

    return SprayFootprint(
        area_px=area_px,
        diameter_px=diameter_px,
        radius_px=radius_px,
        width_px=width_px,
        height_px=height_px,
        radius_u_px=radius_u_px,
        radius_v_px=radius_v_px,
    )


def step_size_from_overlap(diameter_px: float, overlap_ratio: float) -> float:
    return diameter_px * (1.0 - overlap_ratio)


def circular_kernel_mask(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    kernel = (x * x + y * y) <= radius_px * radius_px
    return (kernel.astype(np.uint8) * 255)


def generate_cleaning_points_from_mask(
    corrosion_mask_u8: np.ndarray,
    footprint: SprayFootprint,
    cfg: CleaningGridConfig | None = None,
) -> list[tuple[int, int]]:
    """Generate kernel-based serpentine points.

    Strategy:
    - compute a step from the spray diameter and overlap
    - scan rows in serpentine order
    - keep points whose spray kernel overlaps the corrosion mask
    """
    if cfg is None:
        cfg = CleaningGridConfig()

    mask = clean_binary_mask(corrosion_mask_u8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    step_x = max(1, int(round(step_size_from_overlap(footprint.width_px, cfg.overlap_ratio))))
    step_y = max(1, int(round(step_size_from_overlap(footprint.height_px, cfg.overlap_ratio))))

    rad_x = int(np.ceil(footprint.radius_u_px))
    rad_y = int(np.ceil(footprint.radius_v_px))

    points: list[tuple[int, int]] = []
    row_idx = 0
    for v in range(y_min, y_max + 1, step_y):
        x_range = range(x_min, x_max + 1, step_x)
        if row_idx % 2 == 1:
            x_range = reversed(list(x_range))

        for u in x_range:
            x0 = max(0, u - rad_x)
            x1 = min(mask.shape[1], u + rad_x + 1)
            y0 = max(0, v - rad_y)
            y1 = min(mask.shape[0], v + rad_y + 1)
            local = mask[y0:y1, x0:x1]
            if int(np.count_nonzero(local)) >= cfg.min_mask_overlap_px:
                points.append((u, v))
        row_idx += 1

    return points


def draw_cleaning_points(frame_bgr: np.ndarray, points: Iterable[tuple[int, int]], radius_px: float) -> np.ndarray:
    out = frame_bgr.copy()
    r = int(round(radius_px))
    for idx, (u, v) in enumerate(points):
        cv2.circle(out, (int(u), int(v)), r, (0, 255, 255), 1)
        cv2.circle(out, (int(u), int(v)), 2, (0, 0, 255), -1)
        cv2.putText(out, str(idx), (int(u) + 4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out
