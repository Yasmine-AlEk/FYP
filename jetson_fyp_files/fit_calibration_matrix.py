import csv
import json
import numpy as np


CSV_PATH = "calibration_grid_outputs/calibration_samples.csv"
OUTPUT_JSON = "calibration_grid_outputs/gimbal_calibration.json"

THETA_PAN_CENTER_DEG = 8.0
THETA_PITCH_CENTER_DEG = 5.0


def main():
    samples = []

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["detected"]) != 1:
                continue

            dpan = float(row["delta_theta_pan_deg"])
            dpitch = float(row["delta_theta_pitch_deg"])
            du = float(row["delta_u_i"])
            dv = float(row["delta_v_i"])

            samples.append([dpan, dpitch, du, dv])

    if len(samples) < 4:
        raise RuntimeError("Not enough valid samples to fit calibration matrix")

    samples = np.array(samples, dtype=float)

    # X = angle offsets
    X = samples[:, 0:2]   # [Delta_theta_pan, Delta_theta_pitch]

    # Y = pixel offsets
    Y = samples[:, 2:4]   # [Delta_u, Delta_v]

    # Solve X * B ~= Y
    # Then A = B^T
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = B.T

    detA = np.linalg.det(A)
    if abs(detA) < 1e-9:
        raise RuntimeError("Calibration matrix is singular or nearly singular")

    A_inv = np.linalg.inv(A)

    print("\n=== Fitted calibration ===")
    print("A =")
    print(A)
    print("\nA_inv =")
    print(A_inv)

    print("\nExpanded equations:")
    print(f"Delta_u = {A[0,0]:.6f}*Delta_theta_pan + {A[0,1]:.6f}*Delta_theta_pitch")
    print(f"Delta_v = {A[1,0]:.6f}*Delta_theta_pan + {A[1,1]:.6f}*Delta_theta_pitch")

    print("\nInverse control equations:")
    print(f"Delta_theta_pan   = {A_inv[0,0]:.6f}*Delta_u + {A_inv[0,1]:.6f}*Delta_v")
    print(f"Delta_theta_pitch = {A_inv[1,0]:.6f}*Delta_u + {A_inv[1,1]:.6f}*Delta_v")

    data = {
        "theta_pan_center_deg": THETA_PAN_CENTER_DEG,
        "theta_pitch_center_deg": THETA_PITCH_CENTER_DEG,
        "A": A.tolist(),
        "A_inv": A_inv.tolist(),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved calibration JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
