import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# DEFAULT CONFIG
# =========================================================

# --------------------------------------------------
# Fixed tuned values
# --------------------------------------------------
IMG_PATH = "report.jpeg"
RESIZE_TO = (640, 480)

HSV_LOWER = np.array([1, 96, 70], dtype=np.uint8)
HSV_UPPER = np.array([11, 198, 255], dtype=np.uint8)

USE_ENHANCE = True
ALPHA = 1.30
BETA = 20

USE_CLAHE = True
CLAHE_CLIP = 2.0

BLUR = 5
OPEN_ITER = 1
CLOSE_ITER = 1
MEDIAN_BLUR = 5

MIN_COMPONENT_AREA = 120

MERGE_GAP_X = 10
MERGE_GAP_Y = 6

EXPAND_RECT_PX = 8

OVERLAP_PCT = 30

# Measured spray footprint
SPRAY_DIAMETER_PX = 19.90
STEP_PX = max(1, int(round(SPRAY_DIAMETER_PX * (1.0 - OVERLAP_PCT / 100.0))))

SAVE_OVERLAY_NAME = "overlay_result.jpg"
SAVE_MASK_NAME = "corrosion_mask.png"
SAVE_JSON_NAME = "corrosion_rectangles.json"


# =========================================================
# GUI TRACKBAR HELPERS
# =========================================================

def nothing(_):
    pass


def create_controls(defaults):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 520, 900)

    cv2.createTrackbar("L_H", "Controls", defaults["l_h"], 179, nothing)
    cv2.createTrackbar("L_S", "Controls", defaults["l_s"], 255, nothing)
    cv2.createTrackbar("L_V", "Controls", defaults["l_v"], 255, nothing)

    cv2.createTrackbar("U_H", "Controls", defaults["u_h"], 179, nothing)
    cv2.createTrackbar("U_S", "Controls", defaults["u_s"], 255, nothing)
    cv2.createTrackbar("U_V", "Controls", defaults["u_v"], 255, nothing)

    cv2.createTrackbar("UseEnhance", "Controls", defaults["use_enhance"], 1, nothing)
    cv2.createTrackbar("Alpha_x100", "Controls", defaults["alpha_x100"], 250, nothing)
    cv2.createTrackbar("Beta", "Controls", defaults["beta"], 100, nothing)
    cv2.createTrackbar("UseCLAHE", "Controls", defaults["use_clahe"], 1, nothing)
    cv2.createTrackbar("CLAHE_clip_x10", "Controls", defaults["clahe_clip_x10"], 80, nothing)

    cv2.createTrackbar("Blur", "Controls", defaults["blur"], 31, nothing)
    cv2.createTrackbar("OpenIter", "Controls", defaults["open_iter"], 10, nothing)
    cv2.createTrackbar("CloseIter", "Controls", defaults["close_iter"], 10, nothing)
    cv2.createTrackbar("MedianBlur", "Controls", defaults["median_blur"], 31, nothing)
    cv2.createTrackbar("MinArea", "Controls", defaults["min_area"], 5000, nothing)

    cv2.createTrackbar("MergeGapX", "Controls", defaults["merge_gap_x"], 100, nothing)
    cv2.createTrackbar("MergeGapY", "Controls", defaults["merge_gap_y"], 100, nothing)
    cv2.createTrackbar("ExpandRect", "Controls", defaults["expand_rect_px"], 100, nothing)

    cv2.createTrackbar("OverlapPct", "Controls", int(OVERLAP * 100), 80, nothing)


def get_control_values():
    vals = {
        "l_h": cv2.getTrackbarPos("L_H", "Controls"),
        "l_s": cv2.getTrackbarPos("L_S", "Controls"),
        "l_v": cv2.getTrackbarPos("L_V", "Controls"),

        "u_h": cv2.getTrackbarPos("U_H", "Controls"),
        "u_s": cv2.getTrackbarPos("U_S", "Controls"),
        "u_v": cv2.getTrackbarPos("U_V", "Controls"),

        "use_enhance": cv2.getTrackbarPos("UseEnhance", "Controls"),
        "alpha_x100": cv2.getTrackbarPos("Alpha_x100", "Controls"),
        "beta": cv2.getTrackbarPos("Beta", "Controls"),
        "use_clahe": cv2.getTrackbarPos("UseCLAHE", "Controls"),
        "clahe_clip_x10": cv2.getTrackbarPos("CLAHE_clip_x10", "Controls"),

        "blur": cv2.getTrackbarPos("Blur", "Controls"),
        "open_iter": cv2.getTrackbarPos("OpenIter", "Controls"),
        "close_iter": cv2.getTrackbarPos("CloseIter", "Controls"),
        "median_blur": cv2.getTrackbarPos("MedianBlur", "Controls"),
        "min_area": cv2.getTrackbarPos("MinArea", "Controls"),

        "merge_gap_x": cv2.getTrackbarPos("MergeGapX", "Controls"),
        "merge_gap_y": cv2.getTrackbarPos("MergeGapY", "Controls"),
        "expand_rect_px": cv2.getTrackbarPos("ExpandRect", "Controls"),

        "overlap_pct": cv2.getTrackbarPos("OverlapPct", "Controls"),
    }

    if vals["blur"] < 1:
        vals["blur"] = 1
    if vals["blur"] % 2 == 0:
        vals["blur"] += 1

    if vals["median_blur"] < 1:
        vals["median_blur"] = 1
    if vals["median_blur"] % 2 == 0:
        vals["median_blur"] += 1

    if vals["alpha_x100"] < 1:
        vals["alpha_x100"] = 1

    if vals["clahe_clip_x10"] < 1:
        vals["clahe_clip_x10"] = 1

    return vals


# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

def enhance_image(frame_bgr, vals):
    if vals["use_enhance"] == 0:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return hsv, frame_bgr.copy()

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    alpha = vals["alpha_x100"] / 100.0
    beta = vals["beta"]

    v = cv2.convertScaleAbs(v, alpha=alpha, beta=beta)

    if vals["use_clahe"] == 1:
        clip = vals["clahe_clip_x10"] / 10.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        v = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v])
    frame_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return hsv_enhanced, frame_enhanced


# =========================================================
# SEGMENTATION
# =========================================================

def segment_corrosion(hsv_enhanced, vals):
    lower = np.array([vals["l_h"], vals["l_s"], vals["l_v"]], dtype=np.uint8)
    upper = np.array([vals["u_h"], vals["u_s"], vals["u_v"]], dtype=np.uint8)

    mask = cv2.inRange(hsv_enhanced, lower, upper)

    if vals["blur"] > 1:
        mask = cv2.GaussianBlur(mask, (vals["blur"], vals["blur"]), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    if vals["open_iter"] > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=vals["open_iter"])

    if vals["close_iter"] > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=vals["close_iter"])

    if vals["median_blur"] > 1:
        mask = cv2.medianBlur(mask, vals["median_blur"])

    return mask


def get_filtered_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(mask)
    kept_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            kept_contours.append(cnt)
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask, kept_contours


# =========================================================
# BOX HELPERS
# =========================================================

def contours_to_boxes(contours):
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 999.0

        if aspect >= 2.0:
            orientation = "horizontal"
        elif aspect <= 0.5:
            orientation = "vertical"
        else:
            orientation = "blob"

        boxes.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w - 1),
            "y2": int(y + h - 1),
            "w": int(w),
            "h": int(h),
            "orientation": orientation,
        })

    return boxes


def box_center(box):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return cx, cy


def horizontal_gap(a, b):
    if a["x2"] < b["x1"]:
        return b["x1"] - a["x2"]
    if b["x2"] < a["x1"]:
        return a["x1"] - b["x2"]
    return 0


def vertical_gap(a, b):
    if a["y2"] < b["y1"]:
        return b["y1"] - a["y2"]
    if b["y2"] < a["y1"]:
        return a["y1"] - b["y2"]
    return 0


def boxes_should_merge(a, b, gap_x, gap_y):
    dx = horizontal_gap(a, b)
    dy = vertical_gap(a, b)

    cxa, cya = box_center(a)
    cxb, cyb = box_center(b)

    center_dx = abs(cxa - cxb)
    center_dy = abs(cya - cyb)

    oa = a["orientation"]
    ob = b["orientation"]

    if oa == "horizontal" and ob == "horizontal":
        same_row = center_dy <= max(a["h"], b["h"])
        return dx <= gap_x and same_row

    if oa == "vertical" and ob == "vertical":
        same_col = center_dx <= max(a["w"], b["w"])
        return dy <= gap_y and same_col

    if oa == "blob" and ob == "blob":
        return dx <= gap_x and dy <= gap_y

    if (oa == "horizontal" and ob == "blob") or (oa == "blob" and ob == "horizontal"):
        same_row = center_dy <= max(a["h"], b["h"]) * 1.2
        return dx <= gap_x and same_row

    if (oa == "vertical" and ob == "blob") or (oa == "blob" and ob == "vertical"):
        same_col = center_dx <= max(a["w"], b["w"]) * 1.2
        return dy <= gap_y and same_col

    return False


def merge_two_boxes(a, b):
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    x2 = max(a["x2"], b["x2"])
    y2 = max(a["y2"], b["y2"])

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    aspect = w / float(h) if h > 0 else 999.0

    if aspect >= 2.0:
        orientation = "horizontal"
    elif aspect <= 0.5:
        orientation = "vertical"
    else:
        orientation = "blob"

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "w": int(w),
        "h": int(h),
        "orientation": orientation,
    }


def merge_nearby_boxes(boxes, gap_x, gap_y):
    if not boxes:
        return []

    current_boxes = boxes[:]
    changed_global = True

    while changed_global:
        changed_global = False
        used = [False] * len(current_boxes)
        next_boxes = []

        for i in range(len(current_boxes)):
            if used[i]:
                continue

            merged_box = current_boxes[i]
            used[i] = True

            changed_local = True
            while changed_local:
                changed_local = False

                for j in range(len(current_boxes)):
                    if used[j]:
                        continue

                    if boxes_should_merge(merged_box, current_boxes[j], gap_x, gap_y):
                        merged_box = merge_two_boxes(merged_box, current_boxes[j])
                        used[j] = True
                        changed_local = True
                        changed_global = True

            next_boxes.append(merged_box)

        current_boxes = next_boxes

    return current_boxes


def expand_boxes(boxes, expand_px, img_w, img_h):
    expanded = []

    for i, box in enumerate(boxes):
        x1 = max(0, box["x1"] - expand_px)
        y1 = max(0, box["y1"] - expand_px)
        x2 = min(img_w - 1, box["x2"] + expand_px)
        y2 = min(img_h - 1, box["y2"] + expand_px)

        expanded.append({
            "id": int(i),
            "orientation": box["orientation"],
            "raw_bbox": {
                "x1": int(box["x1"]),
                "y1": int(box["y1"]),
                "x2": int(box["x2"]),
                "y2": int(box["y2"]),
                "w": int(box["w"]),
                "h": int(box["h"]),
            },
            "expanded_bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "w": int(x2 - x1 + 1),
                "h": int(y2 - y1 + 1),
            },
            "corners": {
                "top_left": [int(x1), int(y1)],
                "top_right": [int(x2), int(y1)],
                "bottom_right": [int(x2), int(y2)],
                "bottom_left": [int(x1), int(y2)],
            },
        })

    return expanded


# =========================================================
# MASK JSON ENCODING
# =========================================================

def mask_to_pixels_xy(mask):
    ys, xs = np.where(mask > 0)
    return [[int(x), int(y)] for x, y in zip(xs, ys)]


def mask_to_rle(mask):
    flat = (mask.flatten() > 0).astype(np.uint8)

    if flat.size == 0:
        return []

    rle = []
    current = int(flat[0])
    count = 1

    for value in flat[1:]:
        value = int(value)
        if value == current:
            count += 1
        else:
            rle.append([current, count])
            current = value
            count = 1

    rle.append([current, count])
    return rle


# =========================================================
# SERPENTINE POINT GENERATION
# =========================================================

def make_axis_values(start, end, step):
    start = int(round(start))
    end = int(round(end))
    step = max(1, int(round(step)))

    values = list(range(start, end + 1, step))

    if not values:
        values = [start]

    if values[-1] != end:
        values.append(end)

    return values


def circular_kernel_intersects_mask(mask, x, y, radius_px):
    h, w = mask.shape[:2]
    r = int(math.ceil(radius_px))

    x0 = max(0, x - r)
    x1 = min(w - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(h - 1, y + r)

    roi = mask[y0:y1 + 1, x0:x1 + 1]

    if roi.size == 0:
        return False

    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2

    return np.any((roi > 0) & circle)


def generate_serpentine_points(mask, rectangles, step_px, spray_radius_px):
    points = []
    point_id = 0

    for rect in rectangles:
        bbox = rect["expanded_bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        y_values = make_axis_values(y1, y2, step_px)
        x_values_base = make_axis_values(x1, x2, step_px)

        for row_idx, y in enumerate(y_values):
            if row_idx % 2 == 0:
                x_values = x_values_base
                direction = "left_to_right"
            else:
                x_values = list(reversed(x_values_base))
                direction = "right_to_left"

            row_has_point = False

            for col_idx, x in enumerate(x_values):
                x = int(x)
                y = int(y)

                if circular_kernel_intersects_mask(mask, x, y, spray_radius_px):
                    points.append({
                        "point_id": int(point_id),
                        "rect_id": int(rect["id"]),
                        "row_index": int(row_idx),
                        "col_index": int(col_idx),
                        "direction": direction,
                        "x_px": int(x),
                        "y_px": int(y),
                    })
                    point_id += 1
                    row_has_point = True

            # Safety: if the row crosses corrosion but no circular point was accepted,
            # add the row midpoint of active mask pixels.
            row_mask = mask[y, x1:x2 + 1] if 0 <= y < mask.shape[0] else None
            if row_mask is not None and np.any(row_mask > 0) and not row_has_point:
                active_xs = np.where(row_mask > 0)[0] + x1
                x_mid = int(round(float(np.mean(active_xs))))

                points.append({
                    "point_id": int(point_id),
                    "rect_id": int(rect["id"]),
                    "row_index": int(row_idx),
                    "col_index": -1,
                    "direction": direction,
                    "x_px": int(x_mid),
                    "y_px": int(y),
                    "fallback_row_midpoint": True,
                })
                point_id += 1

    return points


# =========================================================
# DRAW OVERLAY
# =========================================================

def draw_overlay(frame_enhanced, corrosion_mask, kept_contours, rectangles, serpentine_points):
    overlay = frame_enhanced.copy()

    mask_colored = np.zeros_like(frame_enhanced)
    mask_colored[:, :, 2] = corrosion_mask
    overlay = cv2.addWeighted(overlay, 0.80, mask_colored, 0.45, 0)

    cv2.drawContours(overlay, kept_contours, -1, (0, 255, 255), 2)

    for rect in rectangles:
        x1 = rect["expanded_bbox"]["x1"]
        y1 = rect["expanded_bbox"]["y1"]
        x2 = rect["expanded_bbox"]["x2"]
        y2 = rect["expanded_bbox"]["y2"]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            overlay,
            f"Rect {rect['id']}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw serpentine points and connection lines per rectangle
    points_by_rect = {}
    for p in serpentine_points:
        points_by_rect.setdefault(p["rect_id"], []).append(p)

    for rect_id, pts in points_by_rect.items():
        for i, p in enumerate(pts):
            x = p["x_px"]
            y = p["y_px"]

            cv2.circle(overlay, (x, y), 3, (255, 0, 255), -1)

            if i > 0:
                prev = pts[i - 1]
                cv2.line(
                    overlay,
                    (prev["x_px"], prev["y_px"]),
                    (x, y),
                    (255, 0, 255),
                    1,
                )

    corrosion_ratio = np.sum(corrosion_mask > 0) / corrosion_mask.size * 100.0

    cv2.putText(
        overlay,
        f"Corrosion coverage: {corrosion_ratio:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        f"Serpentine points: {len(serpentine_points)}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return overlay


# =========================================================
# FULL PIPELINE
# =========================================================

def process_image(frame, vals):
    hsv_enhanced, frame_enhanced = enhance_image(frame, vals)

    raw_mask = segment_corrosion(hsv_enhanced, vals)

    filtered_mask, kept_contours = get_filtered_contours(
        raw_mask,
        min_area=vals["min_area"],
    )

    boxes = contours_to_boxes(kept_contours)

    merged_boxes = merge_nearby_boxes(
        boxes,
        gap_x=vals["merge_gap_x"],
        gap_y=vals["merge_gap_y"],
    )

    img_h, img_w = filtered_mask.shape[:2]

    rectangles = expand_boxes(
        merged_boxes,
        expand_px=vals["expand_rect_px"],
        img_w=img_w,
        img_h=img_h,
    )

    overlap = vals["overlap_pct"] / 100.0
    step_px = SPRAY_DIAMETER_PX * (1.0 - overlap)

    serpentine_points = generate_serpentine_points(
        mask=filtered_mask,
        rectangles=rectangles,
        step_px=step_px,
        spray_radius_px=SPRAY_RADIUS_PX,
    )

    overlay = draw_overlay(
        frame_enhanced=frame_enhanced,
        corrosion_mask=filtered_mask,
        kept_contours=kept_contours,
        rectangles=rectangles,
        serpentine_points=serpentine_points,
    )

    return {
        "raw_mask": raw_mask,
        "filtered_mask": filtered_mask,
        "frame_enhanced": frame_enhanced,
        "overlay": overlay,
        "rectangles": rectangles,
        "serpentine_points": serpentine_points,
        "step_px": step_px,
        "overlap": overlap,
    }


def build_output_json(result, vals, image_shape):
    h, w = image_shape[:2]
    mask = result["filtered_mask"]

    mask_pixels = mask_to_pixels_xy(mask)
    mask_rle = mask_to_rle(mask)

    payload = {
        "image_size": {
            "width": int(w),
            "height": int(h),
        },

        "spray_footprint": {
            "area_px": 311,
            "equivalent_diameter_px": SPRAY_DIAMETER_PX,
            "radius_px": SPRAY_RADIUS_PX,
            "overlap": result["overlap"],
            "step_px": result["step_px"],
            "step_px_rounded": int(round(result["step_px"])),
            "meaning": "serpentine points use this spacing in both x direction and y row spacing",
        },

        "threshold_settings": vals,

        "mask": {
            "encoding": "positive_pixels_xy_and_rle_row_major",
            "positive_pixel_count": int(len(mask_pixels)),
            "mask_pixels_xy": mask_pixels,
            "rle_row_major": mask_rle,
        },

        "rectangles": result["rectangles"],

        "serpentine_points_px": result["serpentine_points"],

        "serpentine_note": (
            "Rows are spaced by step_px. Points along each row are also spaced by step_px. "
            "Direction alternates between left_to_right and right_to_left."
        ),
    }

    return payload


def save_outputs(result, vals, frame):
    cv2.imwrite(SAVE_MASK_NAME, result["filtered_mask"])
    cv2.imwrite(SAVE_OVERLAY_NAME, result["overlay"])

    payload = build_output_json(result, vals, frame.shape)

    with open(SAVE_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n[SAVED]")
    print(f" - {SAVE_MASK_NAME}")
    print(f" - {SAVE_OVERLAY_NAME}")
    print(f" - {SAVE_JSON_NAME}")

    print("\n[RESULT]")
    print(f"Corrosion pixels: {payload['mask']['positive_pixel_count']}")
    print(f"Rectangles: {len(payload['rectangles'])}")
    print(f"Serpentine points: {len(payload['serpentine_points_px'])}")
    print(f"Spray diameter px: {SPRAY_DIAMETER_PX:.2f}")
    print(f"Step px: {result['step_px']:.2f}")


def print_settings(vals, result):
    print("\nCurrent settings:")
    print(f"HSV lower = ({vals['l_h']}, {vals['l_s']}, {vals['l_v']})")
    print(f"HSV upper = ({vals['u_h']}, {vals['u_s']}, {vals['u_v']})")
    print(f"UseEnhance = {vals['use_enhance']}")
    print(f"Alpha = {vals['alpha_x100'] / 100.0:.2f}")
    print(f"Beta = {vals['beta']}")
    print(f"UseCLAHE = {vals['use_clahe']}")
    print(f"CLAHE clip = {vals['clahe_clip_x10'] / 10.0:.1f}")
    print(f"Blur = {vals['blur']}")
    print(f"OpenIter = {vals['open_iter']}")
    print(f"CloseIter = {vals['close_iter']}")
    print(f"MedianBlur = {vals['median_blur']}")
    print(f"MinArea = {vals['min_area']}")
    print(f"MergeGapX = {vals['merge_gap_x']}")
    print(f"MergeGapY = {vals['merge_gap_y']}")
    print(f"ExpandRect = {vals['expand_rect_px']}")
    print(f"Overlap = {vals['overlap_pct']}%")
    print(f"Step_px = {result['step_px']:.2f}")
    print(f"Rectangles = {len(result['rectangles'])}")
    print(f"Serpentine points = {len(result['serpentine_points'])}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default=DEFAULT_IMG_PATH, help="Input corrosion image")
    parser.add_argument("--width", type=int, default=DEFAULT_RESIZE_TO[0])
    parser.add_argument("--height", type=int, default=DEFAULT_RESIZE_TO[1])
    parser.add_argument("--no-gui", action="store_true", help="Run once and save outputs without GUI")

    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Error: image not found: {image_path}")
        return

    if args.width > 0 and args.height > 0:
        frame = cv2.resize(frame, (args.width, args.height))

    defaults = {
        "l_h": DEFAULT_HSV_LOWER[0],
        "l_s": DEFAULT_HSV_LOWER[1],
        "l_v": DEFAULT_HSV_LOWER[2],
        "u_h": DEFAULT_HSV_UPPER[0],
        "u_s": DEFAULT_HSV_UPPER[1],
        "u_v": DEFAULT_HSV_UPPER[2],

        "use_enhance": DEFAULT_USE_ENHANCE,
        "alpha_x100": DEFAULT_ALPHA_X100,
        "beta": DEFAULT_BETA,
        "use_clahe": DEFAULT_USE_CLAHE,
        "clahe_clip_x10": DEFAULT_CLAHE_CLIP_X10,

        "blur": DEFAULT_BLUR,
        "open_iter": DEFAULT_OPEN_ITER,
        "close_iter": DEFAULT_CLOSE_ITER,
        "median_blur": DEFAULT_MEDIAN_BLUR,
        "min_area": DEFAULT_MIN_COMPONENT_AREA,

        "merge_gap_x": DEFAULT_MERGE_GAP_X,
        "merge_gap_y": DEFAULT_MERGE_GAP_Y,
        "expand_rect_px": DEFAULT_EXPAND_RECT_PX,
        "overlap_pct": int(OVERLAP * 100),
    }

    if args.no_gui:
        vals = defaults
        result = process_image(frame, vals)
        save_outputs(result, vals, frame)
        return

    create_controls(defaults)

    print("\nControls:")
    print("  p = print current settings")
    print("  s = save mask + overlay + JSON")
    print("  q = quit")
    print("\nTune HSV/enhancement first, then press s.\n")

    while True:
        vals = get_control_values()
        result = process_image(frame, vals)

        cv2.imshow("Original", frame)
        cv2.imshow("Raw Mask", result["raw_mask"])
        cv2.imshow("Filtered Mask", result["filtered_mask"])
        cv2.imshow("Corrosion Overlay + Serpentine", result["overlay"])

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            print_settings(vals, result)

        if key == ord("s"):
            save_outputs(result, vals, frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()