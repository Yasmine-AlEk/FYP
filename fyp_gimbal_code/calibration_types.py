from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class GimbalCalibration:
    image_center_u: float
    image_center_v: float
    theta_pan_center_deg: float
    theta_pitch_center_deg: float
    A: list[list[float]]
    A_inv: list[list[float]]

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_json(path: str) -> "GimbalCalibration":
        data = json.loads(Path(path).read_text())
        return GimbalCalibration(**data)

    @property
    def A_np(self) -> np.ndarray:
        return np.array(self.A, dtype=float)

    @property
    def A_inv_np(self) -> np.ndarray:
        return np.array(self.A_inv, dtype=float)
