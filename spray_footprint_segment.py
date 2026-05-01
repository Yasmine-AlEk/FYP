import cv2
import numpy as np
import argparse
import math
import json
from pathlib import Path


# =========================================================
# DEFAULT VALUES FROM YOUR SCREENSHOT + FIXES
# =========================================================

DEFAULTS = {
    "l_h": 7,
    "l_s": 73,
    "l_v": 0,

    "u_h": 14,
    "u_s": 91,
    "u_v": 122,

    "blur": 7,
    "open_iter": 1,
    "close_iter": 2,
    "erode_iter": 0,
    "dilate_iter": 1,

    # fixed: your screenshot had MinArea=2000, which rejected the blob
    "min_area": 50,
    "max_area": 120000,

    "crop_top_pct": 0,
    "crop_bottom_pct": 18,
    "crop_left_pct": 22,
    "crop_right_pct": 28,
}


def nothing(_):
    pass


def create_controls():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 800)

    cv2.createTrackbar("L_H", "Controls", DEFAULTS["l_h"], 179, nothing)
    cv2.createTrackbar("L_S", "Controls", DEFAULTS["l_s"], 255, nothing)
    cv2.createTrackbar("L_V", "Controls", DEFAULTS["l_v"], 255, nothing)

    cv2.createTrackbar("U_H", "Controls", DEFAULTS["u_h"], 179, nothing)
    cv2.createTrackbar("U_S", "Controls", DEFAULTS["u_s"], 255, nothing)
    cv2.createTrackbar("U_V", "Controls", DEFAULTS["u_v"], 255, nothing)

    cv2.createTrackbar("Blur", "Controls", DEFAULTS["blur"], 31, nothing)
    cv2.createTrackbar("OpenIter", "Controls", DEFAULTS["open_iter"], 10, nothing)
    cv2.createTrackbar("CloseIter", "Controls", DEFAULTS["close_iter"], 10, nothing)
    cv2.createTrackbar("ErodeIter", "Controls", DEFAULTS["erode_iter"], 10, nothing)
    cv2.createTrackbar("DilateIter", "Controls", DEFAULTS["dilate_iter"], 10, nothing)

    cv2.createTrackbar("MinArea", "Controls", DEFAULTS["min_area"], 100000, nothing)
    cv2.createTrackbar("MaxArea", "Controls", DEFAULTS["max_area"], 300000, nothing)

    cv2.createTrackbar("CropTopPct", "Controls", DEFAULTS["crop_top_pct"], 60, nothing)
    cv2.createTrackbar("CropBottomPct", "Controls", DEFAULTS["crop_bottom_pct"], 60, nothing)
    cv2.createTrackbar("CropLeftPct", "Controls", DEFAULTS["crop_left_pct"], 70, nothing)
    cv2.createTrackbar("CropRightPct", "Controls", DEFAULTS["crop_right_pct"], 70, nothing)


def get_values():
    vals = {
        "l_h": cv2.getTrackbarPos("L_H", "Controls"),
        "l_s": cv2.getTrackbarPos("L_S", "Controls"),
        "l_v": cv2.getTrackbarPos("L_V", "Controls"),

        "u_h": cv2.getTrackbarPos("U_H", "Controls"),
        "u_s": cv2.getTrackbarPos("U_S", "Controls"),
        "u_v": cv2.getTrackbarPos("U_V", "Controls"),

        "blur": cv2.getTrackbarPos("Blur", "Controls"),
        "open_iter": cv2.getTrackbarPos("OpenIter", "Controls"),
        "close_iter": cv2.getTrackbarPos("CloseIter", "Controls"),
        "erode_iter": cv2.getTrackbarPos("ErodeIter", "Controls"),
        "dilate_iter": cv2.getTrackbarPos("DilateIter", "Controls"),

        "min_area": cv2.getTrackbarPos("MinArea", "Controls"),
        "max_area": cv2.getTrackbarPos("MaxArea", "Controls"),

        "crop_top_pct": cv2.getTrackbarPos("CropTopPct", "Controls"),
        "crop_bottom_pct": cv2.getTrackbarPos("CropBottomPct", "Controls"),
        "crop_left_pct": cv2.getTrackbarPos("CropLeftPct", "Controls"),
        "crop_right_pct": cv2.getTrackbarPos("CropRightPct", "Controls"),
    }

    if vals["blur"] < 1:
        vals["blur"] = 1

    if vals["blur"] % 2 == 0:
        vals["blur"] += 1

    if vals["max_area"] <= vals["min_area"]:
        vals["max_area"] = vals["min_area"] + 1

    return vals


def detect_spray_footprint(image_bgr, vals):
    h, w = image_bgr.shape[:2]

    # =====================================================
    # ROI crop
    # =====================================================

    x0 = int(w * vals["crop_left_pct"] / 100.0)
    x1 = int(w * (1.0 - vals["crop_right_pct"] / 100.0))
    y0 = int(h * vals["crop_top_pct"] / 100.0)
    y1 = int(h * (1.0 - vals["crop_bottom_pct"] / 100.0))

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 255

    # =====================================================
    # HSV threshold
    # =====================================================

    blurred = cv2.GaussianBlur(image_bgr, (vals["blur"], vals["blur"]), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array([vals["l_h"], vals["l_s"], vals["l_v"]], dtype=np.uint8)
    upper = np.array([vals["u_h"], vals["u_s"], vals["u_v"]], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, roi_mask)

    # =====================================================
    # Morphology
    # =====================================================

    kernel = np.ones((3, 3), np.uint8)

    if vals["open_iter"] > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=vals["open_iter"],
        )

    if vals["close_iter"] > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=vals["close_iter"],
        )

    if vals["erode_iter"] > 0:
        mask = cv2.erode(mask, kernel, iterations=vals["erode_iter"])

    if vals["dilate_iter"] > 0:
        mask = cv2.dilate(mask, kernel, iterations=vals["dilate_iter"])

    # =====================================================
    # Connected component selection
    # =====================================================

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    overlay = image_bgr.copy()

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1 - 1, y1 - 1),
        (255, 0, 255),
        2,
    )

    cv2.putText(
        overlay,
        "ROI",
        (x0 + 5, max(30, y0 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )

    best_idx = -1
    best_area = 0
    largest_raw_area = 0

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        largest_raw_area = max(largest_raw_area, area)

        if area < vals["min_area"]:
            continue

        if area > vals["max_area"]:
            continue

        if area > best_area:
            best_area = area
            best_idx = idx

    selected_mask = np.zeros_like(mask)
    result = None

    if best_idx != -1:
        selected_mask[labels == best_idx] = 255

        area_px = int(np.count_nonzero(selected_mask))
        eq_diameter_px = math.sqrt((4.0 * area_px) / math.pi)

        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

        u, v = centroids[best_idx]
        u = float(u)
        v = float(v)

        contours, _ = cv2.findContours(
            selected_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + bw, y + bh),
            (255, 255, 0),
            2,
        )

        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            6,
            (0, 0, 255),
            -1,
        )

        # draw equivalent diameter circle
        radius = eq_diameter_px / 2.0
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            int(round(radius)),
            (0, 165, 255),
            2,
        )

        cv2.putText(
            overlay,
            f"Area = {area_px} px",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            f"d_eq = {eq_diameter_px:.1f} px",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            f"center = ({u:.1f}, {v:.1f})",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        result = {
            "area_px": area_px,
            "equivalent_diameter_px": eq_diameter_px,
            "radius_px": radius,
            "centroid_u": u,
            "centroid_v": v,
            "bbox": {
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
            },
            "roi": {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            },
            "settings": vals,
        }

    else:
        cv2.putText(
            overlay,
            f"Spray footprint not detected | largest area={largest_raw_area}px",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return mask, selected_mask, overlay, result


def save_outputs(mask, selected_mask, overlay, result):
    cv2.imwrite("spray_mask_all.png", mask)
    cv2.imwrite("spray_mask_selected.png", selected_mask)
    cv2.imwrite("spray_overlay.jpg", overlay)

    if result is not None:
        with open("spray_footprint_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\n[SAVED]")
        print("spray_mask_all.png")
        print("spray_mask_selected.png")
        print("spray_overlay.jpg")
        print("spray_footprint_result.json")
        print()
        print(f"Area_px = {result['area_px']}")
        print(f"Equivalent diameter d_eq_px = {result['equivalent_diameter_px']:.2f}")
        print(f"Radius_px = {result['radius_px']:.2f}")

    else:
        print("\n[SAVED]")
        print("spray_mask_all.png")
        print("spray_mask_selected.png")
        print("spray_overlay.jpg")
        print("No footprint result saved because no component was detected.")


def print_result(result):
    if result is None:
        print("\nNo spray footprint detected.")
        return

    print("\nSpray footprint measurement:")
    print(f"  Area_px: {result['area_px']}")
    print(f"  Equivalent diameter_px: {result['equivalent_diameter_px']:.2f}")
    print(f"  Radius_px: {result['radius_px']:.2f}")
    print(f"  Centroid: ({result['centroid_u']:.1f}, {result['centroid_v']:.1f})")
    print(
        f"  BBox: x={result['bbox']['x']}, y={result['bbox']['y']}, "
        f"w={result['bbox']['w']}, h={result['bbox']['h']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input spray image")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    create_controls()

    last_result = None

    print("\nControls:")
    print("  p = print current spray diameter")
    print("  s = save mask, overlay, and JSON result")
    print("  q = quit\n")

    while True:
        vals = get_values()

        mask, selected_mask, overlay, result = detect_spray_footprint(image, vals)
        last_result = result

        cv2.imshow("Original", image)
        cv2.imshow("Threshold Mask", mask)
        cv2.imshow("Selected Footprint", selected_mask)
        cv2.imshow("Overlay", overlay)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            print_result(last_result)

        if key == ord("s"):
            save_outputs(mask, selected_mask, overlay, last_result)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()