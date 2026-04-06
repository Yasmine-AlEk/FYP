import cv2
import numpy as np
import json

# --------------------------------------------------
# Configuration
# --------------------------------------------------
IMG_PATH = "report.jpeg"   # <-- change this to your image path
RESIZE_TO = (640, 480)

# Fixed HSV thresholds
HSV_LOWER = np.array([0, 75, 70], dtype=np.uint8)
HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)

# Cleaning
CLEAN_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 120

# Merge thresholds
# Horizontal objects merge mainly using MERGE_GAP_X
# Vertical objects merge mainly using MERGE_GAP_Y
MERGE_GAP_X = 10
MERGE_GAP_Y = 6

# Rectangle expansion
EXPAND_RECT_PX = 2

# Save names
SAVE_OVERLAY_NAME = "overlay_result.jpg"
SAVE_MASK_NAME = "corrosion_mask.png"
SAVE_JSON_NAME = "corrosion_rectangles.json"


# --------------------------------------------------
# Image enhancement
# --------------------------------------------------
def enhance_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # brighten value channel
    v = cv2.convertScaleAbs(v, alpha=1.3, beta=20)

    # local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v])
    frame_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return hsv_enhanced, frame_enhanced


# --------------------------------------------------
# Segment corrosion
# --------------------------------------------------
def segment_corrosion(hsv_enhanced):
    mask = cv2.inRange(hsv_enhanced, HSV_LOWER, HSV_UPPER)

    kernel = np.ones((CLEAN_KERNEL_SIZE, CLEAN_KERNEL_SIZE), np.uint8)

    # clean small noise and fill small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)

    return mask


# --------------------------------------------------
# Keep only large enough contours
# --------------------------------------------------
def get_filtered_contours(mask, min_area=MIN_COMPONENT_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(mask)
    kept_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            kept_contours.append(cnt)
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filtered_mask, kept_contours


# --------------------------------------------------
# Convert contours to bounding boxes + orientation
# --------------------------------------------------
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
            "x1": x,
            "y1": y,
            "x2": x + w - 1,
            "y2": y + h - 1,
            "w": w,
            "h": h,
            "orientation": orientation
        })

    return boxes


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------
def box_center(box):
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return cx, cy


def horizontal_gap(a, b):
    if a["x2"] < b["x1"]:
        return b["x1"] - a["x2"]
    elif b["x2"] < a["x1"]:
        return a["x1"] - b["x2"]
    return 0


def vertical_gap(a, b):
    if a["y2"] < b["y1"]:
        return b["y1"] - a["y2"]
    elif b["y2"] < a["y1"]:
        return a["y1"] - b["y2"]
    return 0


# --------------------------------------------------
# Smart merge rule
# --------------------------------------------------
def boxes_should_merge(a, b, gap_x, gap_y):
    dx = horizontal_gap(a, b)
    dy = vertical_gap(a, b)

    cxa, cya = box_center(a)
    cxb, cyb = box_center(b)

    center_dx = abs(cxa - cxb)
    center_dy = abs(cya - cyb)

    oa = a["orientation"]
    ob = b["orientation"]

    # horizontal + horizontal
    if oa == "horizontal" and ob == "horizontal":
        same_row = center_dy <= max(a["h"], b["h"])
        return dx <= gap_x and same_row

    # vertical + vertical
    if oa == "vertical" and ob == "vertical":
        same_col = center_dx <= max(a["w"], b["w"])
        return dy <= gap_y and same_col

    # blob + blob
    if oa == "blob" and ob == "blob":
        return dx <= gap_x and dy <= gap_y

    # horizontal + blob
    if (oa == "horizontal" and ob == "blob") or (oa == "blob" and ob == "horizontal"):
        same_row = center_dy <= max(a["h"], b["h"]) * 1.2
        return dx <= gap_x and same_row

    # vertical + blob
    if (oa == "vertical" and ob == "blob") or (oa == "blob" and ob == "vertical"):
        same_col = center_dx <= max(a["w"], b["w"]) * 1.2
        return dy <= gap_y and same_col

    # vertical + horizontal should never merge
    return False


# --------------------------------------------------
# Merge boxes
# --------------------------------------------------
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
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": w,
        "h": h,
        "orientation": orientation
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


# --------------------------------------------------
# Expand boxes slightly
# --------------------------------------------------
def expand_boxes(boxes, expand_px, img_w, img_h):
    expanded = []

    for i, box in enumerate(boxes):
        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        ex1 = max(0, x1 - expand_px)
        ey1 = max(0, y1 - expand_px)
        ex2 = min(img_w - 1, x2 + expand_px)
        ey2 = min(img_h - 1, y2 + expand_px)

        expanded.append({
            "id": i,
            "orientation": box["orientation"],
            "raw_bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "w": int(x2 - x1 + 1),
                "h": int(y2 - y1 + 1)
            },
            "expanded_bbox": {
                "x1": int(ex1),
                "y1": int(ey1),
                "x2": int(ex2),
                "y2": int(ey2),
                "w": int(ex2 - ex1 + 1),
                "h": int(ey2 - ey1 + 1)
            },
            "corners": {
                "top_left": [int(ex1), int(ey1)],
                "top_right": [int(ex2), int(ey1)],
                "bottom_right": [int(ex2), int(ey2)],
                "bottom_left": [int(ex1), int(ey2)]
            }
        })

    return expanded


# --------------------------------------------------
# Draw overlay
# --------------------------------------------------
def draw_overlay(frame_enhanced, corrosion_mask, kept_contours, rectangles):
    overlay = frame_enhanced.copy()

    # red corrosion mask
    mask_colored = np.zeros_like(frame_enhanced)
    mask_colored[:, :, 2] = corrosion_mask
    overlay = cv2.addWeighted(overlay, 0.80, mask_colored, 0.45, 0)

    # yellow boundaries
    cv2.drawContours(overlay, kept_contours, -1, (0, 255, 255), 2)

    # draw rectangles and corners
    for rect in rectangles:
        x1 = rect["expanded_bbox"]["x1"]
        y1 = rect["expanded_bbox"]["y1"]
        x2 = rect["expanded_bbox"]["x2"]
        y2 = rect["expanded_bbox"]["y2"]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        corners = rect["corners"]
        pts = [
            ("TL", tuple(corners["top_left"])),
            ("TR", tuple(corners["top_right"])),
            ("BR", tuple(corners["bottom_right"])),
            ("BL", tuple(corners["bottom_left"]))
        ]

        for label, pt in pts:
            cv2.circle(overlay, pt, 4, (255, 0, 255), -1)
            cv2.putText(
                overlay,
                label,
                (pt[0] + 4, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.putText(
            overlay,
            f"Rect {rect['id']}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    corrosion_ratio = np.sum(corrosion_mask > 0) / corrosion_mask.size * 100
    cv2.putText(
        overlay,
        f"Corrosion coverage: {corrosion_ratio:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return overlay


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        print("Error: image not found.")
        return

    frame = cv2.resize(frame, RESIZE_TO)

    hsv_enhanced, frame_enhanced = enhance_image(frame)
    raw_mask = segment_corrosion(hsv_enhanced)

    filtered_mask, kept_contours = get_filtered_contours(raw_mask, MIN_COMPONENT_AREA)
    boxes = contours_to_boxes(kept_contours)
    merged_boxes = merge_nearby_boxes(boxes, MERGE_GAP_X, MERGE_GAP_Y)

    img_h, img_w = filtered_mask.shape
    rectangles = expand_boxes(merged_boxes, EXPAND_RECT_PX, img_w, img_h)

    overlay = draw_overlay(frame_enhanced, filtered_mask, kept_contours, rectangles)

    print("\nDetected corrosion rectangles:")
    if not rectangles:
        print("No valid corrosion regions found.")
    else:
        for rect in rectangles:
            print(json.dumps(rect, indent=2))

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Corrosion Overlay", cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow("Original", frame)
        cv2.imshow("Corrosion Overlay", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(SAVE_OVERLAY_NAME, overlay)
            cv2.imwrite(SAVE_MASK_NAME, filtered_mask)

            with open(SAVE_JSON_NAME, "w") as f:
                json.dump(rectangles, f, indent=2)

            print("\nSaved:")
            print(f" - {SAVE_OVERLAY_NAME}")
            print(f" - {SAVE_MASK_NAME}")
            print(f" - {SAVE_JSON_NAME}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()