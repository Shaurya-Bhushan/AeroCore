from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

MU_EARTH_M3_S2 = 3.986004418e14
R_EARTH_M = 6378137.0


@dataclass(frozen=True)
class TLEElements:
    name: str
    line1: str
    line2: str
    epoch_year: int
    epoch_day_of_year: float
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float

    @property
    def mean_motion_rad_s(self) -> float:
        return self.mean_motion_rev_day * (2.0 * math.pi) / 86400.0

    @property
    def semi_major_axis_m(self) -> float:
        n = max(1e-9, self.mean_motion_rad_s)
        return (MU_EARTH_M3_S2 / (n**2)) ** (1.0 / 3.0)

    @property
    def altitude_m(self) -> float:
        return self.semi_major_axis_m - R_EARTH_M


def _parse_epoch_year(two_digit_year: int) -> int:
    return 2000 + two_digit_year if two_digit_year < 57 else 1900 + two_digit_year


def parse_tle_lines(lines: Sequence[str], name: str | None = None) -> TLEElements:
    cleaned = [str(x).rstrip() for x in lines if str(x).strip()]
    if len(cleaned) < 2:
        raise ValueError("TLE requires at least two non-empty lines")

    if cleaned[0].startswith("1 ") and cleaned[1].startswith("2 "):
        line1 = cleaned[0]
        line2 = cleaned[1]
        sat_name = name or "TLE_OBJECT"
    else:
        if len(cleaned) < 3:
            raise ValueError("TLE with name line requires 3 lines: name, line1, line2")
        sat_name = name or cleaned[0].strip()
        line1 = cleaned[1]
        line2 = cleaned[2]

    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("Invalid TLE format: expected line1 starting with '1 ' and line2 with '2 '")

    epoch_year_2d = int(line1[18:20])
    epoch_day = float(line1[20:32])
    inclination_deg = float(line2[8:16])
    raan_deg = float(line2[17:25])
    ecc = float("0." + line2[26:33].strip())
    arg_perigee_deg = float(line2[34:42])
    mean_anomaly_deg = float(line2[43:51])
    mean_motion_rev_day = float(line2[52:63])

    return TLEElements(
        name=sat_name,
        line1=line1,
        line2=line2,
        epoch_year=_parse_epoch_year(epoch_year_2d),
        epoch_day_of_year=epoch_day,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        eccentricity=ecc,
        arg_perigee_deg=arg_perigee_deg,
        mean_anomaly_deg=mean_anomaly_deg,
        mean_motion_rev_day=mean_motion_rev_day,
    )


def load_tle_file(path: str | Path) -> TLEElements:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TLE file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    return parse_tle_lines(lines)


def tle_to_spacecraft_config_overrides(tle: TLEElements) -> Dict[str, Any]:
    return {
        "tle_name": tle.name,
        "altitude_m": float(tle.altitude_m),
        "inclination_deg": float(tle.inclination_deg),
        "raan_deg": float(tle.raan_deg),
        "eccentricity": float(tle.eccentricity),
        "arg_perigee_deg": float(tle.arg_perigee_deg),
        "mean_anomaly0_deg": float(tle.mean_anomaly_deg),
        "epoch_year": int(tle.epoch_year),
        "epoch_day_of_year": float(tle.epoch_day_of_year),
        "tle_mean_motion_rev_day": float(tle.mean_motion_rev_day),
    }


def tle_file_to_config_json(tle_path: str | Path) -> str:
    tle = load_tle_file(tle_path)
    payload = {
        "tle": asdict(tle),
        "spacecraft_overrides": tle_to_spacecraft_config_overrides(tle),
    }
    return json.dumps(payload, indent=2)
