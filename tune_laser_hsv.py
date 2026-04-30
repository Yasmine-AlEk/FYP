import cv2
import numpy as np
import argparse


def nothing(_):
    pass


def create_controls():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 700)

    # red range 1
    cv2.createTrackbar("L1_H", "Controls", 0, 179, nothing)
    cv2.createTrackbar("L1_S", "Controls", 120, 255, nothing)
    cv2.createTrackbar("L1_V", "Controls", 180, 255, nothing)
    cv2.createTrackbar("U1_H", "Controls", 12, 179, nothing)
    cv2.createTrackbar("U1_S", "Controls", 255, 255, nothing)
    cv2.createTrackbar("U1_V", "Controls", 255, 255, nothing)

    # red range 2
    cv2.createTrackbar("L2_H", "Controls", 165, 179, nothing)
    cv2.createTrackbar("L2_S", "Controls", 120, 255, nothing)
    cv2.createTrackbar("L2_V", "Controls", 180, 255, nothing)
    cv2.createTrackbar("U2_H", "Controls", 179, 179, nothing)
    cv2.createTrackbar("U2_S", "Controls", 255, 255, nothing)
    cv2.createTrackbar("U2_V", "Controls", 255, 255, nothing)

    # extra tuning
    cv2.createTrackbar("Blur", "Controls", 5, 31, nothing)          # odd only
    cv2.createTrackbar("MinArea", "Controls", 4, 500, nothing)
    cv2.createTrackbar("MaxArea", "Controls", 300, 3000, nothing)
    cv2.createTrackbar("Dilate", "Controls", 1, 10, nothing)
    cv2.createTrackbar("CropBottomPct", "Controls", 18, 50, nothing)  # ignore floor reflection region


def get_trackbar_values():
    vals = {
        "l1_h": cv2.getTrackbarPos("L1_H", "Controls"),
        "l1_s": cv2.getTrackbarPos("L1_S", "Controls"),
        "l1_v": cv2.getTrackbarPos("L1_V", "Controls"),
        "u1_h": cv2.getTrackbarPos("U1_H", "Controls"),
        "u1_s": cv2.getTrackbarPos("U1_S", "Controls"),
        "u1_v": cv2.getTrackbarPos("U1_V", "Controls"),

        "l2_h": cv2.getTrackbarPos("L2_H", "Controls"),
        "l2_s": cv2.getTrackbarPos("L2_S", "Controls"),
        "l2_v": cv2.getTrackbarPos("L2_V", "Controls"),
        "u2_h": cv2.getTrackbarPos("U2_H", "Controls"),
        "u2_s": cv2.getTrackbarPos("U2_S", "Controls"),
        "u2_v": cv2.getTrackbarPos("U2_V", "Controls"),

        "blur": cv2.getTrackbarPos("Blur", "Controls"),
        "min_area": cv2.getTrackbarPos("MinArea", "Controls"),
        "max_area": cv2.getTrackbarPos("MaxArea", "Controls"),
        "dilate": cv2.getTrackbarPos("Dilate", "Controls"),
        "crop_bottom_pct": cv2.getTrackbarPos("CropBottomPct", "Controls"),
    }

    if vals["blur"] < 1:
        vals["blur"] = 1
    if vals["blur"] % 2 == 0:
        vals["blur"] += 1

    if vals["max_area"] < vals["min_area"]:
        vals["max_area"] = vals["min_area"] + 1

    return vals


def detect_laser(image_bgr, vals):
    h, w = image_bgr.shape[:2]
    crop_bottom_pct = vals["crop_bottom_pct"]
    crop_y = int(h * (1.0 - crop_bottom_pct / 100.0))

    work = image_bgr.copy()

    # ignore bottom strip to avoid floor reflection
    if crop_y < h:
        work[crop_y:, :] = 0

    blurred = cv2.GaussianBlur(work, (vals["blur"], vals["blur"]), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower1 = np.array([vals["l1_h"], vals["l1_s"], vals["l1_v"]], dtype=np.uint8)
    upper1 = np.array([vals["u1_h"], vals["u1_s"], vals["u1_v"]], dtype=np.uint8)

    lower2 = np.array([vals["l2_h"], vals["l2_s"], vals["l2_v"]], dtype=np.uint8)
    upper2 = np.array([vals["u2_h"], vals["u2_s"], vals["u2_v"]], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if vals["dilate"] > 0:
        mask = cv2.dilate(mask, kernel, iterations=vals["dilate"])

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    overlay = image_bgr.copy()
    u_c, v_c = w / 2.0, h / 2.0

    # draw image center
    cv2.drawMarker(
        overlay,
        (int(round(u_c)), int(round(v_c))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    # draw crop line
    cv2.line(overlay, (0, crop_y), (w - 1, crop_y), (255, 0, 255), 2)
    cv2.putText(
        overlay,
        f"Ignored bottom: {crop_bottom_pct}%",
        (20, max(25, crop_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    best_idx = -1
    best_score = -1.0

    hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < vals["min_area"] or area > vals["max_area"]:
            continue

        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        ww = stats[idx, cv2.CC_STAT_WIDTH]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]

        roi = hsv_full[y:y + hh, x:x + ww]
        if roi.size == 0:
            continue

        brightness = float(np.mean(roi[:, :, 2]))

        # prefer bright, compact blobs
        compact_penalty = abs(area - 20.0)
        score = brightness - 0.6 * compact_penalty

        if score > best_score:
            best_score = score
            best_idx = idx

    result = None

    if best_idx != -1:
        u, v = centroids[best_idx]
        u = float(u)
        v = float(v)

        e_u = u_c - u
        e_v = v_c - v

        result = {
            "u": u,
            "v": v,
            "u_c": u_c,
            "v_c": v_c,
            "e_u": e_u,
            "e_v": e_v,
        }

        cv2.circle(overlay, (int(round(u)), int(round(v))), 8, (0, 0, 255), 2)
        cv2.line(
            overlay,
            (int(round(u_c)), int(round(v_c))),
            (int(round(u)), int(round(v))),
            (255, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            f"laser=({u:.1f}, {v:.1f})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            overlay,
            f"error=(e_u={e_u:.1f}, e_v={e_v:.1f})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    else:
        cv2.putText(
            overlay,
            "Laser not detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return mask, overlay, result


def print_values(vals):
    print("\nCurrent HSV settings:")
    print(f"Range 1 lower = ({vals['l1_h']}, {vals['l1_s']}, {vals['l1_v']})")
    print(f"Range 1 upper = ({vals['u1_h']}, {vals['u1_s']}, {vals['u1_v']})")
    print(f"Range 2 lower = ({vals['l2_h']}, {vals['l2_s']}, {vals['l2_v']})")
    print(f"Range 2 upper = ({vals['u2_h']}, {vals['u2_s']}, {vals['u2_v']})")
    print(f"Blur = {vals['blur']}")
    print(f"MinArea = {vals['min_area']}")
    print(f"MaxArea = {vals['max_area']}")
    print(f"Dilate = {vals['dilate']}")
    print(f"CropBottomPct = {vals['crop_bottom_pct']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to read image: {args.image}")
        return

    create_controls()

    while True:
        vals = get_trackbar_values()
        mask, overlay, result = detect_laser(image, vals)

        cv2.imshow("Original", image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Overlay", overlay)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            print_values(vals)
            if result is not None:
                print(f"Detected laser: ({result['u']:.1f}, {result['v']:.1f})")
                print(f"Image center:   ({result['u_c']:.1f}, {result['v_c']:.1f})")
                print(f"Error:          e_u={result['e_u']:.1f}, e_v={result['e_v']:.1f}")
            else:
                print("Laser not detected")
        elif key == ord("s"):
            cv2.imwrite("laser_mask_tuned.png", mask)
            cv2.imwrite("laser_overlay_tuned.jpg", overlay)
            print("Saved: laser_mask_tuned.png")
            print("Saved: laser_overlay_tuned.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()