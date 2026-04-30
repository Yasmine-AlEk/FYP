import cv2
import numpy as np
import time


# =========================================================
# CAMERA SETTINGS
# =========================================================
SENSOR_ID = 0
WIDTH = 1280
HEIGHT = 720
FRAMERATE = 60
WARMUP_FRAMES = 10


# =========================================================
# TUNED DETECTION VALUES YOU GAVE
# =========================================================
L1_H = 0
L1_S = 0
L1_V = 183

U1_H = 179
U1_S = 255
U1_V = 255

L2_H = 179
L2_S = 255
L2_V = 255

U2_H = 179
U2_S = 255
U2_V = 255

BLUR = 5
MIN_AREA = 4
MAX_AREA = 300
DILATE_ITERS = 1
CROP_BOTTOM_PCT = 18


# =========================================================
# GStreamer camera pipeline
# =========================================================
def gstreamer_pipeline(sensor_id=0, width=1280, height=720, framerate=60):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )


# =========================================================
# Capture one frame from Jetson camera
# =========================================================
def capture_frame():
    cap = cv2.VideoCapture(
        gstreamer_pipeline(SENSOR_ID, WIDTH, HEIGHT, FRAMERATE),
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    frame = None

    # warmup
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed during warmup")
        time.sleep(0.03)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame")

    return frame


# =========================================================
# Utility
# =========================================================
def get_image_center(image):
    h, w = image.shape[:2]
    return w / 2.0, h / 2.0


# =========================================================
# Laser detection using your tuned values
# =========================================================
def detect_laser_dot(image_bgr):
    h, w = image_bgr.shape[:2]

    # ignore bottom strip to avoid floor reflection
    crop_y = int(h * (1.0 - CROP_BOTTOM_PCT / 100.0))
    work = image_bgr.copy()
    work[crop_y:, :] = 0

    blur_k = BLUR if BLUR % 2 == 1 else BLUR + 1
    blurred = cv2.GaussianBlur(work, (blur_k, blur_k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower1 = np.array([L1_H, L1_S, L1_V], dtype=np.uint8)
    upper1 = np.array([U1_H, U1_S, U1_V], dtype=np.uint8)

    lower2 = np.array([L2_H, L2_S, L2_V], dtype=np.uint8)
    upper2 = np.array([U2_H, U2_S, U2_V], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if DILATE_ITERS > 0:
        mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERS)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None, mask, crop_y

    best_idx = -1
    best_score = -1.0

    # score blobs: prefer bright + compact + within area range
    hsv_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        ww = stats[idx, cv2.CC_STAT_WIDTH]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]

        roi = hsv_full[y:y + hh, x:x + ww]
        if roi.size == 0:
            continue

        brightness = float(np.mean(roi[:, :, 2]))

        # prefer small bright blobs, around laser-spot size
        compact_penalty = abs(area - 20.0)
        score = brightness - 0.6 * compact_penalty

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx == -1:
        return None, mask, crop_y

    u, v = centroids[best_idx]
    return (float(u), float(v)), mask, crop_y


# =========================================================
# Compute center error
# e_u = u_c - u
# e_v = v_c - v
# =========================================================
def compute_center_error(image_bgr, dot_uv):
    u_c, v_c = get_image_center(image_bgr)
    u, v = dot_uv

    e_u = u_c - u
    e_v = v_c - v

    return (u_c, v_c), (e_u, e_v)


# =========================================================
# Make overlay image
# =========================================================
def make_overlay(image_bgr, dot_uv, crop_y):
    overlay = image_bgr.copy()

    u_c, v_c = get_image_center(image_bgr)
    u, v = dot_uv

    # center marker
    cv2.drawMarker(
        overlay,
        (int(round(u_c)), int(round(v_c))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    # laser marker
    cv2.circle(overlay, (int(round(u)), int(round(v))), 8, (0, 0, 255), 2)

    # line from center to laser
    cv2.line(
        overlay,
        (int(round(u_c)), int(round(v_c))),
        (int(round(u)), int(round(v))),
        (255, 255, 0),
        2
    )

    # crop line
    cv2.line(
        overlay,
        (0, crop_y),
        (overlay.shape[1] - 1, crop_y),
        (255, 0, 255),
        2
    )

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

    cv2.putText(
        overlay,
        f"Ignored bottom: {CROP_BOTTOM_PCT}%",
        (20, max(25, crop_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    return overlay


# =========================================================
# Main
# =========================================================
def main():
    print("Capturing frame from Jetson camera...")
    frame = capture_frame()
    cv2.imwrite("captured_frame.jpg", frame)
    print("Saved: captured_frame.jpg")

    print("Detecting laser dot...")
    dot_uv, mask, crop_y = detect_laser_dot(frame)

    cv2.imwrite("laser_mask.png", mask)
    print("Saved: laser_mask.png")

    if dot_uv is None:
        print("Laser not detected")
        return

    (u_c, v_c), (e_u, e_v) = compute_center_error(frame, dot_uv)
    overlay = make_overlay(frame, dot_uv, crop_y)
    cv2.imwrite("laser_overlay.jpg", overlay)
    print("Saved: laser_overlay.jpg")

    u, v = dot_uv

    print("\nLaser detected")
    print(f"Image center: (u_c, v_c) = ({u_c:.1f}, {v_c:.1f})")
    print(f"Laser dot:    (u, v)     = ({u:.1f}, {v:.1f})")
    print(f"Center error: e_u = {e_u:.1f}, e_v = {e_v:.1f}")


if __name__ == "__main__":
    main()
