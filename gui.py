import json
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

APP_TITLE = "Corrosion Drone — Wall Scan GUI"

# defaults agreed on for the wall-scan mission
DEFAULT_START_ALT_M = 2.0
DEFAULT_MAX_ALT_M = 6.0
DEFAULT_SPEED_MPS = 1.0
DEFAULT_STANDOFF_M = 2.0
DEFAULT_LANE_SPACING_M = 0.5
DEFAULT_OVERSHOOT_M = 2.0

# hard limit requested by you
MAX_ALLOWED_ALT_M = 6.0


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def validate_lat_lon(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class WallScanGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x700")
        self.minsize(920, 640)

        self._build_style()
        self._build_layout()

        for v in (self.var_lat1, self.var_lon1, self.var_lat2, self.var_lon2):
            v.trace_add("write", lambda *_: self.update_width_display())

        self.update_width_display()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def _build_layout(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(main, text="Wall Inputs", padding=12)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        form.columnconfigure(1, weight=1)

        out = ttk.LabelFrame(main, text="Output (JSON Spec Preview)", padding=12)
        out.grid(row=0, column=1, sticky="nsew")
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)

        # variables
        self.var_wall_name = tk.StringVar(value="Structure A")

        # explicit left/right semantics
        self.var_lat1 = tk.StringVar(value="")
        self.var_lon1 = tk.StringVar(value="")
        self.var_lat2 = tk.StringVar(value="")
        self.var_lon2 = tk.StringVar(value="")

        self.var_width_m = tk.StringVar(value="—")

        self.var_start_alt = tk.StringVar(value=f"{DEFAULT_START_ALT_M:.1f}")
        self.var_max_alt = tk.StringVar(value=f"{DEFAULT_MAX_ALT_M:.1f}")
        self.var_speed = tk.StringVar(value=f"{DEFAULT_SPEED_MPS:.1f}")
        self.var_standoff = tk.StringVar(value=f"{DEFAULT_STANDOFF_M:.1f}")
        self.var_lane = tk.StringVar(value=f"{DEFAULT_LANE_SPACING_M:.1f}")

        row = 0

        ttk.Label(form, text="Wall Name").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_wall_name).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Separator(form).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        row += 1
        ttk.Label(
            form,
            text="Corner A Latitude (LEFT corner)"
        ).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_lat1).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(
            form,
            text="Corner A Longitude (LEFT corner)"
        ).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_lon1).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(
            form,
            text="Corner B Latitude (RIGHT corner)"
        ).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_lat2).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(
            form,
            text="Corner B Longitude (RIGHT corner)"
        ).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_lon2).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Separator(form).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        row += 1
        ttk.Label(form, text="Wall Width (m) [computed from A→B]").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_width_m, state="readonly").grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Separator(form).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        row += 1
        ttk.Label(form, text="Start Altitude (m) [default 2.0]").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_start_alt).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(form, text=f"Max Altitude (m) [must be ≤ {MAX_ALLOWED_ALT_M:.1f}]").grid(
            row=row, column=0, sticky="w", pady=4
        )
        ttk.Entry(form, textvariable=self.var_max_alt).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Separator(form).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        row += 1
        ttk.Label(form, text="Scan Speed (m/s) [default 1.0]").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_speed).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(form, text="Standoff Distance (m)").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_standoff).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(form, text="Lane Spacing (m)").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.var_lane).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        note = (
            "Notes:\n"
            "- Corner A must be the LEFT wall corner.\n"
            "- Corner B must be the RIGHT wall corner.\n"
            "- Backend uses LOCAL_NED.\n"
            "- Scan side is chosen automatically from the drone's current side of the wall.\n"
            "- Overshoot is fixed to 2.0 m."
        )
        ttk.Label(form, text=note, justify="left").grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))

        btns = ttk.Frame(form)
        btns.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)

        ttk.Button(btns, text="Preview JSON", command=self.on_preview).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(btns, text="Export JSON...", command=self.on_export).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(btns, text="Clear", command=self.on_clear).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self.txt = tk.Text(out, wrap="none", font=("Consolas", 10))
        self.txt.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(out, orient="vertical", command=self.txt.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=yscroll.set)

        xscroll = ttk.Scrollbar(out, orient="horizontal", command=self.txt.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.txt.configure(xscrollcommand=xscroll.set)

    def _get_corners(self):
        lat1s = self.var_lat1.get().strip()
        lon1s = self.var_lon1.get().strip()
        lat2s = self.var_lat2.get().strip()
        lon2s = self.var_lon2.get().strip()

        if "" in (lat1s, lon1s, lat2s, lon2s):
            raise ValueError("Please fill Corner A (LEFT) and Corner B (RIGHT) latitude/longitude.")

        for label, val in (
            ("Corner A Latitude (LEFT)", lat1s),
            ("Corner A Longitude (LEFT)", lon1s),
            ("Corner B Latitude (RIGHT)", lat2s),
            ("Corner B Longitude (RIGHT)", lon2s),
        ):
            if not is_float(val):
                raise ValueError(f"{label} must be a number.")

        lat1 = float(lat1s)
        lon1 = float(lon1s)
        lat2 = float(lat2s)
        lon2 = float(lon2s)

        if not (validate_lat_lon(lat1, lon1) and validate_lat_lon(lat2, lon2)):
            raise ValueError("Latitude must be in [-90, 90] and Longitude must be in [-180, 180].")

        if lat1 == lat2 and lon1 == lon2:
            raise ValueError("Corner A and Corner B cannot be the same point.")

        return lat1, lon1, lat2, lon2

    def compute_width_m(self) -> float:
        lat1, lon1, lat2, lon2 = self._get_corners()
        return haversine_m(lat1, lon1, lat2, lon2)

    def update_width_display(self):
        try:
            w = self.compute_width_m()
            self.var_width_m.set(f"{w:.2f}")
        except Exception:
            self.var_width_m.set("—")

    def collect_spec(self):
        lat1, lon1, lat2, lon2 = self._get_corners()
        width = haversine_m(lat1, lon1, lat2, lon2)

        if width <= 0:
            raise ValueError("Computed wall width is invalid.")

        start_alt_s = self.var_start_alt.get().strip()
        max_alt_s = self.var_max_alt.get().strip()
        speed_s = self.var_speed.get().strip()
        standoff_s = self.var_standoff.get().strip()
        lane_s = self.var_lane.get().strip()

        if start_alt_s == "":
            raise ValueError("Missing required field: start_alt_m")
        if max_alt_s == "":
            raise ValueError("Missing required field: max_alt_m")

        for label, val in (
            ("Start Altitude", start_alt_s),
            ("Max Altitude", max_alt_s),
            ("Scan Speed", speed_s),
            ("Standoff Distance", standoff_s),
            ("Lane Spacing", lane_s),
        ):
            if val == "":
                raise ValueError(f"Missing required field: {label}")
            if not is_float(val):
                raise ValueError(f"{label} must be a number.")

        start_alt = float(start_alt_s)
        max_alt = float(max_alt_s)
        speed = float(speed_s)
        standoff = float(standoff_s)
        lane = float(lane_s)

        for label, value in (
            ("Start Altitude", start_alt),
            ("Max Altitude", max_alt),
            ("Scan Speed", speed),
            ("Standoff Distance", standoff),
            ("Lane Spacing", lane),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be a finite number.")

        if start_alt < 0.0:
            raise ValueError("Start Altitude must be >= 0.")
        if max_alt <= 0.0:
            raise ValueError("Max Altitude must be > 0.")
        if max_alt > MAX_ALLOWED_ALT_M:
            raise ValueError(f"Max Altitude cannot exceed {MAX_ALLOWED_ALT_M:.1f} m.")
        if start_alt >= max_alt:
            raise ValueError("Start Altitude must be less than Max Altitude.")
        if speed <= 0.0:
            raise ValueError("Scan Speed must be > 0.")
        if standoff <= 0.0:
            raise ValueError("Standoff Distance must be > 0.")
        if lane <= 0.0:
            raise ValueError("Lane Spacing must be > 0.")

        scan_height = max_alt - start_alt

        spec = {
            "project": "corrosion_detection_then_cleaning",
            "structure": {
                "name": self.var_wall_name.get().strip() or "Unnamed Structure",
                "type": "vertical_wall",
                "corners_gps": {
                    "A": {"lat": lat1, "lon": lon1},  # LEFT corner
                    "B": {"lat": lat2, "lon": lon2},  # RIGHT corner
                },
                "dimensions_m": {
                    "width": width,
                    "scan_height": scan_height,
                },
            },
            "scan_settings": {
                "start_alt_m": start_alt,
                "max_alt_m": max_alt,
                "speed_mps": speed,
                "standoff_m": standoff,
                "lane_spacing_m": lane,
                "overshoot_m": DEFAULT_OVERSHOOT_M,
            },
            "assumptions": {
                "corner_A_is_left_corner": True,
                "corner_B_is_right_corner": True,
                "width_is_computed_from_A_to_B": True,
                "scan_side_selection": "automatic_from_current_drone_side",
                "mavlink_frame": "LOCAL_NED",
            },
        }

        return spec

    def show_json(self, spec: dict):
        pretty = json.dumps(spec, indent=2)
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", pretty)

    def on_preview(self):
        try:
            spec = self.collect_spec()
            self.show_json(spec)
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def on_export(self):
        try:
            spec = self.collect_spec()
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        path = filedialog.asksaveasfilename(
            title="Save JSON Spec",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)

            self.show_json(spec)
            messagebox.showinfo("Saved", f"Saved JSON spec to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def on_clear(self):
        self.var_wall_name.set("Structure A")

        self.var_lat1.set("")
        self.var_lon1.set("")
        self.var_lat2.set("")
        self.var_lon2.set("")

        self.var_width_m.set("—")

        self.var_start_alt.set(f"{DEFAULT_START_ALT_M:.1f}")
        self.var_max_alt.set(f"{DEFAULT_MAX_ALT_M:.1f}")
        self.var_speed.set(f"{DEFAULT_SPEED_MPS:.1f}")
        self.var_standoff.set(f"{DEFAULT_STANDOFF_M:.1f}")
        self.var_lane.set(f"{DEFAULT_LANE_SPACING_M:.1f}")

        self.txt.delete("1.0", "end")


if __name__ == "__main__":
    app = WallScanGUI()
    app.mainloop()