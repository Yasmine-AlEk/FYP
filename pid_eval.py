import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# helper: simple PD simulation with step disturbance
# =========================================================
def simulate_pid_step_response(
    t_end,
    dt,
    disturbance_time,
    disturbance_magnitude,
    kp,
    kd,
    plant_a,
    plant_b,
    control_limit,
    initial_error=0.0,
):
    """
    Simulates:
        error_dot = -plant_a * error + plant_b * u + disturbance

    with controller:
        u = kp * e + kd * de/dt

    disturbance is added as a step after disturbance_time.
    """

    t = np.arange(0.0, t_end + dt, dt)

    error = np.zeros_like(t)
    control = np.zeros_like(t)
    disturbance = np.zeros_like(t)

    error[0] = initial_error
    prev_error = error[0]

    for k in range(1, len(t)):
        if t[k] >= disturbance_time:
            disturbance[k] = disturbance_magnitude
        else:
            disturbance[k] = 0.0

        de = (error[k - 1] - prev_error) / dt
        prev_error = error[k - 1]

        # PID/PD control
        u = kp * error[k - 1] + kd * de

        # saturation
        u = max(-control_limit, min(control_limit, u))
        control[k] = u

        # plant update
        error_dot = -plant_a * error[k - 1] - plant_b * u + disturbance[k]
        error[k] = error[k - 1] + dt * error_dot

    return t, error, control, disturbance


# =========================================================
# distance PID simulation
# =========================================================
dt = 0.02
t_end = 20.0
disturbance_time = 6.0

# distance PID:
# error = measured_distance - target_distance
# positive error means too far from wall
t_dist, distance_error, vx_cmd, dist_step = simulate_pid_step_response(
    t_end=t_end,
    dt=dt,
    disturbance_time=disturbance_time,
    disturbance_magnitude=0.18,   # step disturbance in meters
    kp=1.15,
    kd=0.35,
    plant_a=1.35,
    plant_b=2.2,
    control_limit=0.35,
    initial_error=0.0,
)

# if you also want actual distance instead of only error
target_distance = 2.0
measured_distance = target_distance + distance_error


# =========================================================
# yaw PID simulation
# =========================================================
# yaw sensor error = left-right ultrasonic difference
# positive error means misalignment
t_yaw, yaw_error, yaw_rate_cmd, yaw_step = simulate_pid_step_response(
    t_end=t_end,
    dt=dt,
    disturbance_time=disturbance_time,
    disturbance_magnitude=0.08,   # step disturbance in meters of L-R difference
    kp=3.8,
    kd=0.18,
    plant_a=1.8,
    plant_b=3.0,
    control_limit=0.50,
    initial_error=0.0,
)


# =========================================================
# style helper
# =========================================================
def style_axes(ax):
    ax.grid(True, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.9)
    ax.tick_params(colors="0.15", labelsize=10)
    ax.title.set_color("0.10")
    ax.xaxis.label.set_color("0.10")
    ax.yaxis.label.set_color("0.10")


# =========================================================
# plot 1: distance PID
# =========================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True)
fig.suptitle("Standoff Distance PID — Step Disturbance Response", fontsize=16, fontweight="bold")

axes[0].plot(t_dist, distance_error, color="0.15", linewidth=2.3, label="Distance error")
axes[0].axhline(0.0, color="0.40", linestyle="--", linewidth=1.4, label="Target error = 0")
axes[0].axvline(disturbance_time, color="0.55", linestyle=":", linewidth=1.4, label="Step disturbance applied")
axes[0].set_title("Distance error before and after the disturbance")
axes[0].set_ylabel("Distance error (m)")
axes[0].legend(loc="best", frameon=True)
style_axes(axes[0])

axes[1].plot(t_dist, vx_cmd, color="0.15", linewidth=2.3, label="Forward velocity correction")
axes[1].axhline(0.0, color="0.40", linewidth=1.2, label="Zero command")
axes[1].axvline(disturbance_time, color="0.55", linestyle=":", linewidth=1.4, label="Step disturbance applied")
axes[1].set_title("PID forward velocity response during and after disturbance")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("vx command (m/s)")
axes[1].legend(loc="best", frameon=True)
style_axes(axes[1])

plt.tight_layout()
plt.savefig("distance_pid_step_disturbance.png", dpi=220, bbox_inches="tight")
plt.show()


# =========================================================
# plot 2: yaw PID
# =========================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True)
fig.suptitle("Yaw / Alignment PID — Step Disturbance Response", fontsize=16, fontweight="bold")

axes[0].plot(t_yaw, yaw_error, color="0.15", linewidth=2.3, label="Yaw sensor error")
axes[0].axhline(0.0, color="0.40", linestyle="--", linewidth=1.4, label="Target error = 0")
axes[0].axvline(disturbance_time, color="0.55", linestyle=":", linewidth=1.4, label="Step disturbance applied")
axes[0].set_title("Yaw sensor error before and after the disturbance")
axes[0].set_ylabel("Sensor error (m)")
axes[0].legend(loc="best", frameon=True)
style_axes(axes[0])

axes[1].plot(t_yaw, yaw_rate_cmd, color="0.15", linewidth=2.3, label="Yaw-rate correction")
axes[1].axhline(0.0, color="0.40", linewidth=1.2, label="Zero command")
axes[1].axvline(disturbance_time, color="0.55", linestyle=":", linewidth=1.4, label="Step disturbance applied")
axes[1].set_title("PID yaw-rate response during and after disturbance")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Yaw rate (rad/s)")
axes[1].legend(loc="best", frameon=True)
style_axes(axes[1])

plt.tight_layout()
plt.savefig("yaw_pid_step_disturbance.png", dpi=220, bbox_inches="tight")
plt.show()