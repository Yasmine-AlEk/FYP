
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from pymavlink import mavutil


@dataclass
class PixhawkServoConfig:
    port: str = "/dev/ttyTHS1"
    baud: int = 115200

    pan_servo: int = 10
    tilt_servo: int = 9

    pan_min_pwm: int = 1100
    pan_center_pwm: int = 1500
    pan_max_pwm: int = 1900
    pan_min_angle_deg: float = -40.0
    pan_max_angle_deg: float = 40.0

    tilt_min_pwm: int = 1100
    tilt_center_pwm: int = 1500
    tilt_max_pwm: int = 1800
    tilt_min_angle_deg: float = -30.0
    tilt_max_angle_deg: float = 40.0

    command_delay_s: float = 0.05
    settle_time_s: float = 0.30
    heartbeat_timeout_s: float = 15.0


class PixhawkServoController:
    """Reusable pan/tilt servo controller for the gimbal pipeline.

    This is the cleaned-up version of your teammates' script:
    - keeps the same MAVLink + PWM logic
    - removes the interactive input loop
    - exposes set_pan_tilt(pan_deg, pitch_deg)
    - can be plugged directly into the calibration / cleaning pipeline
    """

    def __init__(self, config: Optional[PixhawkServoConfig] = None):
        self.cfg = config or PixhawkServoConfig()
        self.master = self._connect_pixhawk()

    def _connect_pixhawk(self):
        print(f"[INFO] Connecting to Pixhawk on {self.cfg.port} @ {self.cfg.baud}...")
        master = mavutil.mavlink_connection(self.cfg.port, baud=self.cfg.baud)
        hb = master.wait_heartbeat(timeout=self.cfg.heartbeat_timeout_s)
        if hb is None:
            raise RuntimeError("No heartbeat received from Pixhawk")
        print("[INFO] Connected to Pixhawk")
        return master

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def angle_to_pwm(
        self,
        angle_deg: float,
        min_pwm: int,
        center_pwm: int,
        max_pwm: int,
        min_angle_deg: float,
        max_angle_deg: float,
    ) -> int:
        angle_deg = self._clamp(angle_deg, min_angle_deg, max_angle_deg)

        if angle_deg >= 0:
            pwm = center_pwm + (angle_deg / max_angle_deg) * (max_pwm - center_pwm)
        else:
            pwm = center_pwm + (angle_deg / abs(min_angle_deg)) * (center_pwm - min_pwm)

        return int(round(pwm))

    def _send_servo_pwm(self, servo_num: int, pwm: int) -> None:
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_num,
            pwm,
            0, 0, 0, 0, 0,
        )

    def set_pan_tilt(self, pan_deg: float, pitch_deg: float) -> None:
        """Set both gimbal angles in degrees."""
        pan_pwm = self.angle_to_pwm(
            angle_deg=pan_deg,
            min_pwm=self.cfg.pan_min_pwm,
            center_pwm=self.cfg.pan_center_pwm,
            max_pwm=self.cfg.pan_max_pwm,
            min_angle_deg=self.cfg.pan_min_angle_deg,
            max_angle_deg=self.cfg.pan_max_angle_deg,
        )
        tilt_pwm = self.angle_to_pwm(
            angle_deg=pitch_deg,
            min_pwm=self.cfg.tilt_min_pwm,
            center_pwm=self.cfg.tilt_center_pwm,
            max_pwm=self.cfg.tilt_max_pwm,
            min_angle_deg=self.cfg.tilt_min_angle_deg,
            max_angle_deg=self.cfg.tilt_max_angle_deg,
        )

        print(
            f"[SERVO] pan={pan_deg:.2f} deg -> {pan_pwm} PWM | "
            f"pitch={pitch_deg:.2f} deg -> {tilt_pwm} PWM"
        )

        self._send_servo_pwm(self.cfg.pan_servo, pan_pwm)
        time.sleep(self.cfg.command_delay_s)
        self._send_servo_pwm(self.cfg.tilt_servo, tilt_pwm)
        time.sleep(self.cfg.settle_time_s)

    def center(self) -> None:
        self.set_pan_tilt(0.0, 0.0)


def spray_action_stub(dwell_time_s: float) -> None:
    """Replace this with your real pump/solenoid trigger function."""
    print(f"[SPRAY] ON for {dwell_time_s:.2f} s")
    time.sleep(dwell_time_s)
