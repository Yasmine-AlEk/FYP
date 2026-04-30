import cv2
import numpy as np
import argparse
from pathlib import Path


def get_image_center(image):
    h, w = image.shape[:2]
    return w / 2.0, h / 2.0


def detect_red_laser_dot(image_bgr):
    # blur a little to reduce noise
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # red wraps around HSV hue, so use two ranges
    lower_red_1 = np.array([0, 120, 180], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([165, 120, 180], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None, mask

    best_idx = -1
    best_score = -1.0

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 4:
            continue

        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]

        roi = hsv[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        brightness_score = float(np.mean(roi[:, :, 2]))

        # prefer small, bright spots
        score = area + 0.02 * brightness_score

        if score > best_score:
            best_idx = idx
            best_score = score

    if best_idx == -1:
        return None, mask

    u, v = centroids[best_idx]
    return (float(u), float(v)), mask


def compute_center_error(image_bgr, dot_uv):
    u_c, v_c = get_image_center(image_bgr)
    u, v = dot_uv

    e_u = u_c - u
    e_v = v_c - v

    return (u_c, v_c), (e_u, e_v)


def make_overlay(image_bgr, dot_uv):
    overlay = image_bgr.copy()

    u_c, v_c = get_image_center(image_bgr)
    u, v = dot_uv

    # image center
    cv2.drawMarker(
        overlay,
        (int(round(u_c)), int(round(v_c))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    # laser dot
    cv2.circle(overlay, (int(round(u)), int(round(v))), 8, (0, 0, 255), 2)

    # line from center to laser
    cv2.line(
        overlay,
        (int(round(u_c)), int(round(v_c))),
        (int(round(u)), int(round(v))),
        (255, 255, 0),
        2
    )

    # text
    e_u = u_c - u
    e_v = v_c - v
    cv2.putText(
        overlay,
        f"laser=({u:.1f}, {v:.1f})",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        overlay,
        f"error=(e_u={e_u:.1f}, e_v={e_v:.1f})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask_out", default="laser_mask.png", help="Output mask image")
    parser.add_argument("--overlay_out", default="laser_overlay.jpg", help="Output overlay image")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    dot_uv, mask = detect_red_laser_dot(image)

    cv2.imwrite(args.mask_out, mask)

    if dot_uv is None:
        print("Laser not detected")
        print(f"Saved mask to: {args.mask_out}")
        return

    (u_c, v_c), (e_u, e_v) = compute_center_error(image, dot_uv)
    overlay = make_overlay(image, dot_uv)
    cv2.imwrite(args.overlay_out, overlay)

    u, v = dot_uv

    print("Laser detected")
    print(f"Image center: (u_c, v_c) = ({u_c:.1f}, {v_c:.1f})")
    print(f"Laser dot:    (u, v)     = ({u:.1f}, {v:.1f})")
    print(f"Center error: e_u = {e_u:.1f}, e_v = {e_v:.1f}")
    print(f"Saved mask to: {args.mask_out}")
    print(f"Saved overlay to: {args.overlay_out}")


if __name__ == "__main__":
    main()