from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.core import MissionTask, Transition

MU_EARTH_M3_S2 = 3.986004418e14
R_EARTH_M = 6378137.0
OMEGA_EARTH_RAD_S = 7.2921159e-5
J2_EARTH = 1.08262668e-3
SEA_LEVEL_DENSITY_KG_M3 = 1.225

_OPPORTUNITY_CACHE: Dict[str, Dict[str, MissionTask]] = {}
_TRANSITION_CACHE: Dict[tuple[Any, ...], Dict[tuple[str, str], Transition]] = {}
_EPHEMERIS_CACHE: Dict[str, "OrbitEphemeris"] = {}


def clear_spacecraft_model_caches() -> None:
    _OPPORTUNITY_CACHE.clear()
    _TRANSITION_CACHE.clear()
    _EPHEMERIS_CACHE.clear()
    _julian_day_from_year_day.cache_clear()
    _sun_vector_components_from_jd_second.cache_clear()


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def vector_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    dot = float(np.clip(np.dot(_unit(av), _unit(bv)), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def latlon_to_ecef(
    lat_deg: float,
    lon_deg: float,
    radius_m: float = R_EARTH_M,
    altitude_m: float = 0.0,
) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    clat = math.cos(lat)
    slat = math.sin(lat)

    # Default to WGS-84 ellipsoid when using Earth radius.
    if abs(float(radius_m) - R_EARTH_M) < 1e-6:
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)
        N = a / math.sqrt(max(1e-12, 1.0 - e2 * slat * slat))
        alt = float(altitude_m)
        x = (N + alt) * clat * math.cos(lon)
        y = (N + alt) * clat * math.sin(lon)
        z = (N * (1.0 - e2) + alt) * slat
        return np.array([x, y, z], dtype=float)

    sphere_r = float(radius_m) + float(altitude_m)
    return np.array([sphere_r * clat * math.cos(lon), sphere_r * clat * math.sin(lon), sphere_r * slat], dtype=float)


def ecef_to_eci(r_ecef: np.ndarray, t_s: float, theta0_rad: float = 0.0) -> np.ndarray:
    theta = theta0_rad + OMEGA_EARTH_RAD_S * t_s
    c = math.cos(theta)
    s = math.sin(theta)
    x = c * r_ecef[0] - s * r_ecef[1]
    y = s * r_ecef[0] + c * r_ecef[1]
    z = r_ecef[2]
    return np.array([x, y, z], dtype=float)


def eci_to_ecef(r_eci: np.ndarray, t_s: float, theta0_rad: float = 0.0) -> np.ndarray:
    theta = theta0_rad + OMEGA_EARTH_RAD_S * t_s
    c = math.cos(theta)
    s = math.sin(theta)
    x = c * r_eci[0] + s * r_eci[1]
    y = -s * r_eci[0] + c * r_eci[1]
    z = r_eci[2]
    return np.array([x, y, z], dtype=float)


@dataclass
class OrbitConfig:
    altitude_m: float
    inclination_deg: float
    raan_deg: float
    arg_lat0_deg: float
    epoch_theta0_deg: float
    epoch_julian_day: float
    horizon_s: float
    j2_enabled: bool = True
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_anomaly0_deg: Optional[float] = None
    drag_enabled: bool = False
    ballistic_coefficient_kg_m2: float = 120.0
    atmosphere_ref_density_kg_m3: float = 5.0e-12
    atmosphere_ref_altitude_m: float = 550000.0
    atmosphere_scale_height_m: float = 70000.0
    atmosphere_corotation_factor: float = 1.0
    propagation_step_s: float = 30.0
    propagation_backend: str = "rk4"

    @property
    def semi_major_m(self) -> float:
        return R_EARTH_M + self.altitude_m

    @property
    def mean_motion_rad_s(self) -> float:
        return math.sqrt(MU_EARTH_M3_S2 / (self.semi_major_m ** 3))

    @property
    def orbit_period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def inclination_rad(self) -> float:
        return math.radians(self.inclination_deg)

    @property
    def raan_rad(self) -> float:
        return math.radians(self.raan_deg)

    @property
    def arg_lat0_rad(self) -> float:
        return math.radians(self.arg_lat0_deg)

    @property
    def epoch_theta0_rad(self) -> float:
        return math.radians(self.epoch_theta0_deg)

    @property
    def raan_drift_rad_s(self) -> float:
        # First-order secular J2 node precession model.
        a = self.semi_major_m
        e = max(0.0, min(0.95, self.eccentricity))
        n = self.mean_motion_rad_s
        denom = max(1e-6, (1.0 - e * e) ** 2)
        return -1.5 * J2_EARTH * ((R_EARTH_M / a) ** 2) * n * math.cos(self.inclination_rad) / denom

    @property
    def arg_perigee_drift_rad_s(self) -> float:
        # First-order secular J2 argument-of-perigee precession.
        a = self.semi_major_m
        e = max(0.0, min(0.95, self.eccentricity))
        n = self.mean_motion_rad_s
        denom = max(1e-6, (1.0 - e * e) ** 2)
        return (
            0.75
            * J2_EARTH
            * ((R_EARTH_M / a) ** 2)
            * n
            * (5.0 * (math.cos(self.inclination_rad) ** 2) - 1.0)
            / denom
        )


@dataclass
class OrbitEphemeris:
    orbit: OrbitConfig
    times_s: np.ndarray
    positions_eci_m: np.ndarray
    velocities_eci_mps: np.ndarray
    step_s: float


def _orbit_signature(orbit: OrbitConfig) -> str:
    return json.dumps(
        {
            "altitude_m": float(orbit.altitude_m),
            "inclination_deg": float(orbit.inclination_deg),
            "raan_deg": float(orbit.raan_deg),
            "arg_lat0_deg": float(orbit.arg_lat0_deg),
            "epoch_theta0_deg": float(orbit.epoch_theta0_deg),
            "epoch_julian_day": float(orbit.epoch_julian_day),
            "horizon_s": float(orbit.horizon_s),
            "j2_enabled": bool(orbit.j2_enabled),
            "eccentricity": float(orbit.eccentricity),
            "arg_perigee_deg": float(orbit.arg_perigee_deg),
            "mean_anomaly0_deg": orbit.mean_anomaly0_deg,
            "drag_enabled": bool(orbit.drag_enabled),
            "ballistic_coefficient_kg_m2": float(orbit.ballistic_coefficient_kg_m2),
            "atmosphere_ref_density_kg_m3": float(orbit.atmosphere_ref_density_kg_m3),
            "atmosphere_ref_altitude_m": float(orbit.atmosphere_ref_altitude_m),
            "atmosphere_scale_height_m": float(orbit.atmosphere_scale_height_m),
            "atmosphere_corotation_factor": float(orbit.atmosphere_corotation_factor),
            "propagation_step_s": float(orbit.propagation_step_s),
            "propagation_backend": str(orbit.propagation_backend),
        },
        sort_keys=True,
    )


def _solve_kepler_eccentric_anomaly(mean_anomaly_rad: float, e: float) -> float:
    m = (mean_anomaly_rad + math.pi) % (2.0 * math.pi) - math.pi
    if e < 1e-10:
        return m
    E = m
    for _ in range(10):
        f = E - e * math.sin(E) - m
        fp = 1.0 - e * math.cos(E)
        dE = -f / max(1e-12, fp)
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def _rotation_r3(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_r1(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _elements_to_eci_state(orbit: OrbitConfig) -> tuple[np.ndarray, np.ndarray]:
    a = max(R_EARTH_M + 120000.0, float(orbit.semi_major_m))
    e = max(0.0, min(0.2, float(orbit.eccentricity)))
    inc = float(orbit.inclination_rad)
    raan = float(orbit.raan_rad)
    arg_per = math.radians(float(orbit.arg_perigee_deg))

    if orbit.mean_anomaly0_deg is not None:
        m0 = math.radians(float(orbit.mean_anomaly0_deg))
    else:
        # Preserve prior behavior for circular-orbit configs that specify argument of latitude.
        m0 = math.radians(float(orbit.arg_lat0_deg - orbit.arg_perigee_deg))

    E = _solve_kepler_eccentric_anomaly(m0, e)
    cos_E = math.cos(E)
    sin_E = math.sin(E)
    sqrt_one_minus_e2 = math.sqrt(max(1e-12, 1.0 - e * e))
    true_anom = math.atan2(sqrt_one_minus_e2 * sin_E, cos_E - e)
    p = a * (1.0 - e * e)
    r_mag = p / max(1e-9, 1.0 + e * math.cos(true_anom))

    r_pqw = np.array([r_mag * math.cos(true_anom), r_mag * math.sin(true_anom), 0.0], dtype=float)
    v_pqw = np.array(
        [
            -math.sqrt(MU_EARTH_M3_S2 / p) * math.sin(true_anom),
            math.sqrt(MU_EARTH_M3_S2 / p) * (e + math.cos(true_anom)),
            0.0,
        ],
        dtype=float,
    )

    rot = _rotation_r3(raan) @ _rotation_r1(inc) @ _rotation_r3(arg_per)
    return rot @ r_pqw, rot @ v_pqw


def _atmospheric_density_kg_m3(altitude_m: float, orbit: OrbitConfig) -> float:
    if altitude_m <= 0.0:
        return SEA_LEVEL_DENSITY_KG_M3
    rho_ref = max(1e-15, float(orbit.atmosphere_ref_density_kg_m3))
    h_ref = float(orbit.atmosphere_ref_altitude_m)
    H = max(1000.0, float(orbit.atmosphere_scale_height_m))
    exponent = -(float(altitude_m) - h_ref) / H
    exponent = max(-80.0, min(20.0, exponent))
    return float(rho_ref * math.exp(exponent))


def _orbital_acceleration_eci(r_eci_m: np.ndarray, v_eci_mps: np.ndarray, orbit: OrbitConfig) -> np.ndarray:
    r = np.asarray(r_eci_m, dtype=float)
    v = np.asarray(v_eci_mps, dtype=float)
    r_norm = max(1.0, _norm(r))
    a = (-MU_EARTH_M3_S2 / (r_norm ** 3)) * r

    if orbit.j2_enabled:
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        r2 = max(1.0, x * x + y * y + z * z)
        r5 = max(1.0, r2 ** 2.5)
        z2 = z * z
        f = 1.5 * J2_EARTH * MU_EARTH_M3_S2 * (R_EARTH_M ** 2) / r5
        a += np.array(
            [
                f * x * (5.0 * z2 / r2 - 1.0),
                f * y * (5.0 * z2 / r2 - 1.0),
                f * z * (5.0 * z2 / r2 - 3.0),
            ],
            dtype=float,
        )

    if orbit.drag_enabled:
        beta = max(1.0, float(orbit.ballistic_coefficient_kg_m2))
        alt = r_norm - R_EARTH_M
        rho = _atmospheric_density_kg_m3(alt, orbit)
        omega_vec = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S * float(orbit.atmosphere_corotation_factor)], dtype=float)
        v_atm = np.cross(omega_vec, r)
        v_rel = v - v_atm
        v_rel_n = _norm(v_rel)
        if v_rel_n > 1e-6:
            a_drag = -(0.5 * rho / beta) * v_rel_n * v_rel
            a += a_drag

    return a


def _rk4_orbit_step(r_eci_m: np.ndarray, v_eci_mps: np.ndarray, dt_s: float, orbit: OrbitConfig) -> tuple[np.ndarray, np.ndarray]:
    dt = float(dt_s)
    if dt <= 0.0:
        return np.asarray(r_eci_m, dtype=float), np.asarray(v_eci_mps, dtype=float)

    def f(state: np.ndarray) -> np.ndarray:
        r = state[:3]
        v = state[3:]
        a = _orbital_acceleration_eci(r, v, orbit)
        return np.hstack((v, a))

    y0 = np.hstack((np.asarray(r_eci_m, dtype=float), np.asarray(v_eci_mps, dtype=float)))
    k1 = f(y0)
    k2 = f(y0 + 0.5 * dt * k1)
    k3 = f(y0 + 0.5 * dt * k2)
    k4 = f(y0 + dt * k3)
    y = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y[:3], y[3:]


@lru_cache(maxsize=512)
def _julian_day_from_year_day(year: int, day_of_year: float, utc_hour: float) -> float:
    day_of_year = float(round(day_of_year, 6))
    utc_hour = float(round(utc_hour, 6))
    doy_int = int(max(1, min(366, math.floor(day_of_year))))
    doy_frac = float(day_of_year) - float(doy_int)
    hour = float(utc_hour) + 24.0 * doy_frac
    base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy_int - 1, hours=hour)
    unix_s = base.timestamp()
    return unix_s / 86400.0 + 2440587.5


@lru_cache(maxsize=16384)
def _sun_vector_components_from_jd_second(jd_second: int) -> tuple[float, float, float]:
    julian_day = float(jd_second) / 86400.0
    # Low-order solar ephemeris sufficient for eclipse proxy over week-scale planning.
    n = julian_day - 2451545.0
    mean_long_deg = (280.460 + 0.9856474 * n) % 360.0
    mean_anom_deg = (357.528 + 0.9856003 * n) % 360.0
    mean_anom = math.radians(mean_anom_deg)
    ecliptic_long = math.radians(
        mean_long_deg + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2.0 * mean_anom)
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)

    sun = np.array(
        [
            math.cos(ecliptic_long),
            math.cos(obliquity) * math.sin(ecliptic_long),
            math.sin(obliquity) * math.sin(ecliptic_long),
        ],
        dtype=float,
    )
    unit = _unit(sun)
    return float(unit[0]), float(unit[1]), float(unit[2])


def _sun_vector_eci_unit(julian_day: float) -> np.ndarray:
    jd_second = int(round(float(julian_day) * 86400.0))
    x, y, z = _sun_vector_components_from_jd_second(jd_second)
    return np.array([x, y, z], dtype=float)


@dataclass
class SpacecraftResourceConfig:
    battery_capacity_wh: float
    battery_min_wh: float
    battery_initial_wh: float
    data_buffer_capacity_mb: float
    slew_rate_deg_s: float
    obs_power_w: float
    downlink_power_w: float
    housekeeping_power_w: float
    solar_charge_w: float
    solar_capture_efficiency: float
    downlink_rate_mb_s: float
    max_ops_per_orbit: int
    min_downlink_elevation_deg: float
    max_observation_off_nadir_deg: float
    min_operation_gap_s: float


@dataclass
class SpacecraftState:
    time_s: float
    battery_wh: float
    solar_scale: float
    data_buffer_mb: float
    science_buffer_value: float
    delivered_science: float
    ops_this_orbit: int
    last_orbit_index: int
    pointing_vec_eci: tuple[float, float, float]
    last_op_end_s: float
    total_energy_used_wh: float
    total_time_s: float
    executed_observations: int
    executed_downlinks: int
    target_visits: Dict[str, int]
    repeat_value_decay: float
    repeat_value_floor_fraction: float

    @classmethod
    def from_mapping(
        cls,
        state: Dict[str, Any],
        *,
        resources: SpacecraftResourceConfig,
    ) -> "SpacecraftState":
        pv = state.get("pointing_vec_eci", (-1.0, 0.0, 0.0))
        return cls(
            time_s=float(state.get("time_s", 0.0)),
            battery_wh=float(state.get("battery_wh", resources.battery_initial_wh)),
            solar_scale=float(state.get("solar_scale", 1.0)),
            data_buffer_mb=float(state.get("data_buffer_mb", 0.0)),
            science_buffer_value=float(state.get("science_buffer_value", 0.0)),
            delivered_science=float(state.get("delivered_science", 0.0)),
            ops_this_orbit=int(state.get("ops_this_orbit", 0)),
            last_orbit_index=int(state.get("last_orbit_index", 0)),
            pointing_vec_eci=(float(pv[0]), float(pv[1]), float(pv[2])),
            last_op_end_s=float(state.get("last_op_end_s", -1e12)),
            total_energy_used_wh=float(state.get("total_energy_used_wh", 0.0)),
            total_time_s=float(state.get("total_time_s", 0.0)),
            executed_observations=int(state.get("executed_observations", 0)),
            executed_downlinks=int(state.get("executed_downlinks", 0)),
            target_visits={str(k): int(v) for k, v in dict(state.get("target_visits", {})).items()},
            repeat_value_decay=float(state.get("repeat_value_decay", 0.75)),
            repeat_value_floor_fraction=float(state.get("repeat_value_floor_fraction", 0.35)),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "time_s": float(self.time_s),
            "battery_wh": float(self.battery_wh),
            "solar_scale": float(self.solar_scale),
            "data_buffer_mb": float(self.data_buffer_mb),
            "science_buffer_value": float(self.science_buffer_value),
            "delivered_science": float(self.delivered_science),
            "ops_this_orbit": int(self.ops_this_orbit),
            "last_orbit_index": int(self.last_orbit_index),
            "pointing_vec_eci": tuple(float(x) for x in self.pointing_vec_eci),
            "last_op_end_s": float(self.last_op_end_s),
            "total_energy_used_wh": float(self.total_energy_used_wh),
            "total_time_s": float(self.total_time_s),
            "executed_observations": int(self.executed_observations),
            "executed_downlinks": int(self.executed_downlinks),
            "target_visits": {str(k): int(v) for k, v in self.target_visits.items()},
            "repeat_value_decay": float(self.repeat_value_decay),
            "repeat_value_floor_fraction": float(self.repeat_value_floor_fraction),
        }


@dataclass
class VisibilityMetrics:
    visible: bool
    elevation_deg: float
    off_nadir_deg: float


def orbit_from_config(spacecraft_cfg: Dict[str, Any], horizon_s: float) -> OrbitConfig:
    tle_overrides: Dict[str, Any] = {}
    tle_payload = spacecraft_cfg.get("tle")
    tle_file = spacecraft_cfg.get("tle_file")
    if tle_payload is not None or tle_file is not None:
        from src.spacecraft.tle_ingest import load_tle_file, parse_tle_lines, tle_to_spacecraft_config_overrides

        try:
            if tle_file is not None:
                tle = load_tle_file(str(tle_file))
            elif isinstance(tle_payload, dict):
                if "line1" in tle_payload and "line2" in tle_payload:
                    lines = [str(tle_payload.get("name", "")).strip(), str(tle_payload["line1"]), str(tle_payload["line2"])]
                elif "lines" in tle_payload:
                    lines = list(tle_payload["lines"])
                else:
                    lines = []
                tle = parse_tle_lines(lines, name=str(tle_payload.get("name", "TLE_OBJECT")))
            else:
                tle = parse_tle_lines(list(tle_payload))
            tle_overrides = tle_to_spacecraft_config_overrides(tle)
        except Exception:
            tle_overrides = {}

    cfg_merged = dict(spacecraft_cfg)
    cfg_merged.update(tle_overrides)

    if "altitude_m" in cfg_merged:
        altitude_m = float(cfg_merged["altitude_m"])
    else:
        period_s = float(cfg_merged.get("orbit_period_s", 5760.0))
        radius_m = (MU_EARTH_M3_S2 * (period_s / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
        altitude_m = radius_m - R_EARTH_M

    arg_lat0_deg = float(cfg_merged.get("arg_lat0_deg", math.degrees(float(cfg_merged.get("initial_phase_rad", 0.0)))))
    raan_deg = float(cfg_merged.get("raan_deg", cfg_merged.get("initial_lon_deg", 0.0)))
    epoch_julian_day = cfg_merged.get("epoch_julian_day")
    if epoch_julian_day is None:
        epoch_julian_day = _julian_day_from_year_day(
            year=int(cfg_merged.get("epoch_year", 2026)),
            day_of_year=float(cfg_merged.get("epoch_day_of_year", 54.0)),
            utc_hour=float(cfg_merged.get("epoch_utc_hour", 0.0)),
        )

    return OrbitConfig(
        altitude_m=altitude_m,
        inclination_deg=float(cfg_merged.get("inclination_deg", 97.6)),
        raan_deg=raan_deg,
        arg_lat0_deg=arg_lat0_deg,
        epoch_theta0_deg=float(cfg_merged.get("epoch_theta0_deg", 0.0)),
        epoch_julian_day=float(epoch_julian_day),
        horizon_s=horizon_s,
        j2_enabled=bool(cfg_merged.get("j2_enabled", True)),
        eccentricity=float(cfg_merged.get("eccentricity", 0.0)),
        arg_perigee_deg=float(cfg_merged.get("arg_perigee_deg", 0.0)),
        mean_anomaly0_deg=cfg_merged.get("mean_anomaly0_deg"),
        drag_enabled=bool(cfg_merged.get("drag_enabled", True)),
        ballistic_coefficient_kg_m2=float(cfg_merged.get("ballistic_coefficient_kg_m2", 120.0)),
        atmosphere_ref_density_kg_m3=float(cfg_merged.get("atmosphere_ref_density_kg_m3", 5.0e-12)),
        atmosphere_ref_altitude_m=float(cfg_merged.get("atmosphere_ref_altitude_m", 550000.0)),
        atmosphere_scale_height_m=float(cfg_merged.get("atmosphere_scale_height_m", 70000.0)),
        atmosphere_corotation_factor=float(cfg_merged.get("atmosphere_corotation_factor", 1.0)),
        propagation_step_s=float(cfg_merged.get("propagation_step_s", 30.0)),
        propagation_backend=str(cfg_merged.get("propagation_backend", "rk4")).lower(),
    )


def _build_orbit_ephemeris(orbit: OrbitConfig) -> OrbitEphemeris:
    key = _orbit_signature(orbit)
    cached = _EPHEMERIS_CACHE.get(key)
    if cached is not None:
        return cached

    step_s = max(5.0, float(orbit.propagation_step_s))
    n_steps = int(math.ceil(max(0.0, float(orbit.horizon_s)) / step_s)) + 2
    times = np.linspace(0.0, step_s * (n_steps - 1), num=n_steps, dtype=float)
    positions = np.zeros((n_steps, 3), dtype=float)
    velocities = np.zeros((n_steps, 3), dtype=float)

    r, v = _elements_to_eci_state(orbit)
    positions[0, :] = r
    velocities[0, :] = v

    backend = str(getattr(orbit, "propagation_backend", "rk4")).lower()
    use_solve_ivp = backend == "solve_ivp" or (backend == "auto" and n_steps >= 25000)
    solve_ivp_ok = False

    if use_solve_ivp:
        try:
            from scipy.integrate import solve_ivp  # type: ignore

            def dyn(_t: float, y: np.ndarray) -> np.ndarray:
                rr = y[:3]
                vv = y[3:]
                aa = _orbital_acceleration_eci(rr, vv, orbit)
                return np.hstack((vv, aa))

            y0 = np.hstack((r, v))
            sol = solve_ivp(
                dyn,
                (float(times[0]), float(times[-1])),
                y0,
                t_eval=times,
                method="DOP853",
                rtol=1e-8,
                atol=1e-9,
            )
            if bool(sol.success) and sol.y.shape[1] == len(times):
                positions[:, :] = sol.y[:3, :].T
                velocities[:, :] = sol.y[3:, :].T
                solve_ivp_ok = True
        except Exception:
            solve_ivp_ok = False

    if not solve_ivp_ok:
        r_step = np.array(r, dtype=float)
        v_step = np.array(v, dtype=float)
        for i in range(1, n_steps):
            r_step, v_step = _rk4_orbit_step(r_step, v_step, step_s, orbit)
            positions[i, :] = r_step
            velocities[i, :] = v_step

    eph = OrbitEphemeris(
        orbit=orbit,
        times_s=times,
        positions_eci_m=positions,
        velocities_eci_mps=velocities,
        step_s=step_s,
    )
    _EPHEMERIS_CACHE[key] = eph
    return eph


def _sample_orbit_state(t_s: float, orbit: OrbitConfig, ephemeris: OrbitEphemeris | None = None) -> tuple[np.ndarray, np.ndarray]:
    eph = ephemeris or _build_orbit_ephemeris(orbit)
    t = float(max(0.0, min(t_s, float(eph.times_s[-1]))))

    idx = int(min(len(eph.times_s) - 2, max(0, math.floor(t / max(1e-6, eph.step_s)))))
    t0 = float(eph.times_s[idx])
    t1 = float(eph.times_s[idx + 1])
    alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)

    r = (1.0 - alpha) * eph.positions_eci_m[idx] + alpha * eph.positions_eci_m[idx + 1]
    v = (1.0 - alpha) * eph.velocities_eci_mps[idx] + alpha * eph.velocities_eci_mps[idx + 1]
    return r.astype(float), v.astype(float)


def satellite_state_eci(t_s: float, orbit: OrbitConfig, ephemeris: OrbitEphemeris | None = None) -> tuple[np.ndarray, np.ndarray]:
    return _sample_orbit_state(t_s, orbit, ephemeris=ephemeris)


def satellite_position_eci(t_s: float, orbit: OrbitConfig, ephemeris: OrbitEphemeris | None = None) -> np.ndarray:
    r, _ = satellite_state_eci(t_s, orbit, ephemeris=ephemeris)
    return r


def satellite_velocity_eci(t_s: float, orbit: OrbitConfig, ephemeris: OrbitEphemeris | None = None) -> np.ndarray:
    _, v = satellite_state_eci(t_s, orbit, ephemeris=ephemeris)
    return v


def sunlit_indicator(t_s: float, orbit: OrbitConfig, ephemeris: OrbitEphemeris | None = None) -> float:
    sat_eci = satellite_position_eci(t_s, orbit, ephemeris=ephemeris)
    sun_jd = orbit.epoch_julian_day + (t_s / 86400.0)
    sun_hat = _sun_vector_eci_unit(sun_jd)
    behind_earth = float(np.dot(sat_eci, sun_hat)) < 0.0
    perpendicular = _norm(np.cross(sat_eci, sun_hat))
    eclipsed = behind_earth and perpendicular < R_EARTH_M
    return 0.0 if eclipsed else 1.0


def integrate_battery_change_wh(
    t0_s: float,
    dt_s: float,
    orbit: OrbitConfig,
    solar_w: float,
    load_w: float,
    n_samples: int = 8,
    ephemeris: OrbitEphemeris | None = None,
) -> float:
    if dt_s <= 0.0:
        return 0.0
    sample_dt = dt_s / n_samples
    delta = 0.0
    for i in range(n_samples):
        t = t0_s + (i + 0.5) * sample_dt
        net_w = sunlit_indicator(t, orbit, ephemeris=ephemeris) * solar_w - load_w
        delta += net_w * (sample_dt / 3600.0)
    return delta


def ground_elevation_deg(sat_ecef: np.ndarray, station_ecef: np.ndarray) -> float:
    rho = sat_ecef - station_ecef
    rho_n = _unit(rho)
    up = _unit(station_ecef)
    dot = float(np.clip(np.dot(rho_n, up), -1.0, 1.0))
    return math.degrees(math.asin(dot))


def line_of_sight_clear_to_surface(sat_eci: np.ndarray, target_eci: np.ndarray, body_radius_m: float = R_EARTH_M) -> bool:
    """
    Check whether the segment from spacecraft to a surface point intersects the body interior
    before the endpoint at the surface point.
    """
    sat = np.asarray(sat_eci, dtype=float)
    tgt = np.asarray(target_eci, dtype=float)
    d = tgt - sat
    dd = float(np.dot(d, d))
    if dd <= 1e-12:
        return False
    t_closest = float(-np.dot(sat, d) / dd)
    t_clamped = min(1.0, max(0.0, t_closest))
    closest = sat + t_clamped * d
    closest_r = _norm(closest)

    # Endpoint is on/near the Earth surface; we only reject true penetrations.
    if t_clamped >= 0.999:
        return True
    return closest_r >= body_radius_m - 1e-3


def observation_visibility(
    sat_eci: np.ndarray,
    target_ecef: np.ndarray,
    t_s: float,
    orbit: OrbitConfig,
    max_off_nadir_deg: float,
) -> VisibilityMetrics:
    target_eci = ecef_to_eci(target_ecef, t_s, orbit.epoch_theta0_rad)
    los_eci = target_eci - sat_eci
    los_hat = _unit(los_eci)

    nadir_hat = -_unit(sat_eci)
    off_nadir_deg = vector_angle_deg(nadir_hat, los_hat)

    # Target-side elevation check: spacecraft must be above local horizon at target.
    target_to_sat = sat_eci - target_eci
    elevation_deg = math.degrees(math.asin(float(np.clip(np.dot(_unit(target_to_sat), _unit(target_eci)), -1.0, 1.0))))
    los_clear = line_of_sight_clear_to_surface(sat_eci, target_eci, R_EARTH_M)
    visible = elevation_deg > 0.0 and los_clear and off_nadir_deg <= max_off_nadir_deg
    return VisibilityMetrics(visible=visible, elevation_deg=elevation_deg, off_nadir_deg=off_nadir_deg)


def downlink_visibility(
    sat_eci: np.ndarray,
    station_ecef: np.ndarray,
    t_s: float,
    orbit: OrbitConfig,
    min_elev_deg: float,
) -> VisibilityMetrics:
    sat_ecef = eci_to_ecef(sat_eci, t_s, orbit.epoch_theta0_rad)
    elev = ground_elevation_deg(sat_ecef, station_ecef)

    station_eci = ecef_to_eci(station_ecef, t_s, orbit.epoch_theta0_rad)
    los_eci = station_eci - sat_eci
    nadir_hat = -_unit(sat_eci)
    off_nadir_deg = vector_angle_deg(nadir_hat, _unit(los_eci))
    los_clear = line_of_sight_clear_to_surface(sat_eci, station_eci, R_EARTH_M)

    return VisibilityMetrics(visible=los_clear and elev >= min_elev_deg, elevation_deg=elev, off_nadir_deg=off_nadir_deg)


def _collect_windows(
    times: np.ndarray,
    mask: np.ndarray,
    min_duration_s: float,
) -> List[tuple[float, float, float, float]]:
    windows: List[tuple[float, float, float, float]] = []
    if mask.size == 0:
        return windows

    start_idx: int | None = None
    for i, ok in enumerate(mask):
        if ok and start_idx is None:
            start_idx = i
        if (not ok) and start_idx is not None:
            start_t = float(times[start_idx])
            end_t = float(times[i - 1])
            if end_t - start_t >= min_duration_s:
                windows.append((start_t, end_t, float(start_idx), float(i - 1)))
            start_idx = None

    if start_idx is not None:
        start_t = float(times[start_idx])
        end_t = float(times[-1])
        if end_t - start_t >= min_duration_s:
            windows.append((start_t, end_t, float(start_idx), float(len(times) - 1)))

    return windows


def _opportunity_signature(spacecraft_cfg: Dict[str, Any]) -> str:
    signature_obj = {
        "mission_days": int(spacecraft_cfg.get("mission_days", 7)),
        "altitude_m": float(spacecraft_cfg.get("altitude_m", 0.0)),
        "orbit_period_s": float(spacecraft_cfg.get("orbit_period_s", 5760.0)),
        "inclination_deg": float(spacecraft_cfg.get("inclination_deg", 97.6)),
        "raan_deg": float(spacecraft_cfg.get("raan_deg", spacecraft_cfg.get("initial_lon_deg", 0.0))),
        "arg_lat0_deg": float(spacecraft_cfg.get("arg_lat0_deg", math.degrees(float(spacecraft_cfg.get("initial_phase_rad", 0.0))))),
        "eccentricity": float(spacecraft_cfg.get("eccentricity", 0.0)),
        "arg_perigee_deg": float(spacecraft_cfg.get("arg_perigee_deg", 0.0)),
        "mean_anomaly0_deg": float(spacecraft_cfg.get("mean_anomaly0_deg", 0.0)),
        "epoch_theta0_deg": float(spacecraft_cfg.get("epoch_theta0_deg", 0.0)),
        "epoch_julian_day": spacecraft_cfg.get("epoch_julian_day"),
        "epoch_year": int(spacecraft_cfg.get("epoch_year", 2026)),
        "epoch_day_of_year": float(spacecraft_cfg.get("epoch_day_of_year", 54.0)),
        "epoch_utc_hour": float(spacecraft_cfg.get("epoch_utc_hour", 0.0)),
        "j2_enabled": bool(spacecraft_cfg.get("j2_enabled", True)),
        "drag_enabled": bool(spacecraft_cfg.get("drag_enabled", False)),
        "ballistic_coefficient_kg_m2": float(spacecraft_cfg.get("ballistic_coefficient_kg_m2", 120.0)),
        "atmosphere_ref_density_kg_m3": float(spacecraft_cfg.get("atmosphere_ref_density_kg_m3", 5.0e-12)),
        "atmosphere_ref_altitude_m": float(spacecraft_cfg.get("atmosphere_ref_altitude_m", 550000.0)),
        "atmosphere_scale_height_m": float(spacecraft_cfg.get("atmosphere_scale_height_m", 70000.0)),
        "atmosphere_corotation_factor": float(spacecraft_cfg.get("atmosphere_corotation_factor", 1.0)),
        "propagation_step_s": float(spacecraft_cfg.get("propagation_step_s", 30.0)),
        "propagation_backend": str(spacecraft_cfg.get("propagation_backend", "rk4")).lower(),
        "opportunity_step_s": float(spacecraft_cfg.get("opportunity_step_s", 60.0)),
        "observation_duration_s": float(spacecraft_cfg.get("observation_duration_s", 120.0)),
        "downlink_duration_s": float(spacecraft_cfg.get("downlink_duration_s", 300.0)),
        "max_obs_windows": int(spacecraft_cfg.get("max_obs_windows", 500)),
        "max_dl_windows": int(spacecraft_cfg.get("max_dl_windows", 500)),
        "min_downlink_elevation_deg": float(spacecraft_cfg.get("min_downlink_elevation_deg", 10.0)),
        "max_observation_off_nadir_deg": float(spacecraft_cfg.get("max_observation_off_nadir_deg", 30.0)),
        "targets": spacecraft_cfg.get("targets", []),
        "ground_stations": spacecraft_cfg.get("ground_stations", []),
    }
    return json.dumps(signature_obj, sort_keys=True)


def generate_opportunity_tasks(spacecraft_cfg: Dict[str, Any]) -> Dict[str, MissionTask]:
    cache_key = _opportunity_signature(spacecraft_cfg)
    cached = _OPPORTUNITY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    mission_days = int(spacecraft_cfg.get("mission_days", 7))
    horizon_s = float(mission_days * 24 * 3600)
    orbit = orbit_from_config(spacecraft_cfg, horizon_s)
    ephemeris = _build_orbit_ephemeris(orbit)

    resources = SpacecraftResourceConfig(
        battery_capacity_wh=float(spacecraft_cfg.get("battery_capacity_wh", 240.0)),
        battery_min_wh=float(spacecraft_cfg.get("battery_min_wh", 40.0)),
        battery_initial_wh=float(spacecraft_cfg.get("battery_initial_wh", 190.0)),
        data_buffer_capacity_mb=float(spacecraft_cfg.get("data_buffer_capacity_mb", 2000.0)),
        slew_rate_deg_s=float(spacecraft_cfg.get("slew_rate_deg_s", 1.2)),
        obs_power_w=float(spacecraft_cfg.get("obs_power_w", 32.0)),
        downlink_power_w=float(spacecraft_cfg.get("downlink_power_w", 28.0)),
        housekeeping_power_w=float(spacecraft_cfg.get("housekeeping_power_w", 10.0)),
        solar_charge_w=float(spacecraft_cfg.get("solar_charge_w", 38.0)),
        solar_capture_efficiency=float(spacecraft_cfg.get("solar_capture_efficiency", 0.62)),
        downlink_rate_mb_s=float(spacecraft_cfg.get("downlink_rate_mb_s", 2.0)),
        max_ops_per_orbit=int(spacecraft_cfg.get("max_ops_per_orbit", 4)),
        min_downlink_elevation_deg=float(spacecraft_cfg.get("min_downlink_elevation_deg", 10.0)),
        max_observation_off_nadir_deg=float(spacecraft_cfg.get("max_observation_off_nadir_deg", 30.0)),
        min_operation_gap_s=float(spacecraft_cfg.get("min_operation_gap_s", 60.0)),
    )

    dt = float(spacecraft_cfg.get("opportunity_step_s", 60.0))
    times = np.arange(0.0, horizon_s + dt, dt)

    sat_eci_samples = [satellite_position_eci(float(t), orbit, ephemeris=ephemeris) for t in times]

    tasks: Dict[str, MissionTask] = {}
    tasks["SC_START"] = MissionTask(
        task_id="SC_START",
        domain="spacecraft",
        task_type="start",
        window_start_s=0.0,
        window_end_s=horizon_s,
        duration_s=0.0,
        value=0.0,
        required=False,
        metadata={"pointing_type": "nadir"},
    )

    obs_duration = float(spacecraft_cfg.get("observation_duration_s", 120.0))
    dl_duration = float(spacecraft_cfg.get("downlink_duration_s", 300.0))
    max_obs_windows = int(spacecraft_cfg.get("max_obs_windows", 500))
    max_dl_windows = int(spacecraft_cfg.get("max_dl_windows", 500))

    obs_created = 0
    for target in spacecraft_cfg["targets"]:
        if obs_created >= max_obs_windows:
            break

        lat = float(target["lat_deg"])
        lon = float(target["lon_deg"])
        target_ecef = latlon_to_ecef(lat, lon)

        vis_mask: List[bool] = []
        off_nadir_track: List[float] = []
        for idx, t in enumerate(times):
            vis = observation_visibility(
                sat_eci=sat_eci_samples[idx],
                target_ecef=target_ecef,
                t_s=float(t),
                orbit=orbit,
                max_off_nadir_deg=resources.max_observation_off_nadir_deg,
            )
            vis_mask.append(vis.visible)
            off_nadir_track.append(vis.off_nadir_deg)

        windows = _collect_windows(times, np.array(vis_mask, dtype=bool), min_duration_s=obs_duration)
        for window_idx, (w_start, w_end, i_start, i_end) in enumerate(windows):
            if obs_created >= max_obs_windows:
                break
            i0 = int(i_start)
            i1 = int(i_end)
            min_off = float(np.min(off_nadir_track[i0 : i1 + 1]))
            max_off = float(np.max(off_nadir_track[i0 : i1 + 1]))
            task_id = f"OBS_{target['id']}_{window_idx}"
            tasks[task_id] = MissionTask(
                task_id=task_id,
                domain="spacecraft",
                task_type="observation",
                window_start_s=w_start,
                window_end_s=w_end,
                duration_s=obs_duration,
                value=float(target.get("value", 1.0)),
                required=False,
                metadata={
                    "target_id": target["id"],
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "data_mb": float(target.get("data_mb", 120.0)),
                    "min_off_nadir_deg": min_off,
                    "max_off_nadir_deg": max_off,
                    "pointing_type": "ground_fixed",
                },
            )
            obs_created += 1

    dl_created = 0
    for station in spacecraft_cfg["ground_stations"]:
        if dl_created >= max_dl_windows:
            break

        lat = float(station["lat_deg"])
        lon = float(station["lon_deg"])
        station_ecef = latlon_to_ecef(lat, lon)

        vis_mask: List[bool] = []
        elev_track: List[float] = []
        for idx, t in enumerate(times):
            vis = downlink_visibility(
                sat_eci=sat_eci_samples[idx],
                station_ecef=station_ecef,
                t_s=float(t),
                orbit=orbit,
                min_elev_deg=resources.min_downlink_elevation_deg,
            )
            vis_mask.append(vis.visible)
            elev_track.append(vis.elevation_deg)

        windows = _collect_windows(times, np.array(vis_mask, dtype=bool), min_duration_s=dl_duration)
        for window_idx, (w_start, w_end, i_start, i_end) in enumerate(windows):
            if dl_created >= max_dl_windows:
                break
            i0 = int(i_start)
            i1 = int(i_end)
            min_elev = float(np.min(elev_track[i0 : i1 + 1]))
            max_elev = float(np.max(elev_track[i0 : i1 + 1]))

            task_id = f"DL_{station['id']}_{window_idx}"
            tasks[task_id] = MissionTask(
                task_id=task_id,
                domain="spacecraft",
                task_type="downlink",
                window_start_s=w_start,
                window_end_s=w_end,
                duration_s=dl_duration,
                value=float(station.get("value", 0.0)),
                required=False,
                metadata={
                    "station_id": station["id"],
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "min_elevation_deg": min_elev,
                    "max_elevation_deg": max_elev,
                    "pointing_type": "ground_fixed",
                },
            )
            dl_created += 1

    tasks["SC_END"] = MissionTask(
        task_id="SC_END",
        domain="spacecraft",
        task_type="end",
        window_start_s=horizon_s,
        window_end_s=horizon_s,
        duration_s=0.0,
        value=0.0,
        required=False,
        metadata={"pointing_type": "nadir"},
    )

    _OPPORTUNITY_CACHE[cache_key] = tasks
    return tasks


def build_spacecraft_transitions(
    tasks: Dict[str, MissionTask],
    resources: SpacecraftResourceConfig,
    orbit: OrbitConfig | None = None,
) -> Dict[tuple[str, str], Transition]:
    task_sig = tuple(
        sorted(
            (
                task_id,
                task.task_type,
                round(float(task.window_start_s), 1),
                round(float(task.window_end_s), 1),
                round(float(task.metadata.get("lat_deg", 0.0)), 4),
                round(float(task.metadata.get("lon_deg", 0.0)), 4),
            )
            for task_id, task in tasks.items()
        )
    )
    orbit_sig: tuple[Any, ...] | None = None
    if orbit is not None:
        orbit_sig = (
            float(orbit.altitude_m),
            float(orbit.inclination_deg),
            float(orbit.raan_deg),
            float(orbit.arg_lat0_deg),
            float(orbit.epoch_theta0_deg),
            bool(orbit.j2_enabled),
            float(orbit.eccentricity),
            float(orbit.arg_perigee_deg),
            float(orbit.mean_anomaly0_deg),
            bool(orbit.drag_enabled),
            float(orbit.ballistic_coefficient_kg_m2),
            float(orbit.atmosphere_ref_density_kg_m3),
            float(orbit.atmosphere_ref_altitude_m),
            float(orbit.atmosphere_scale_height_m),
            float(orbit.atmosphere_corotation_factor),
            float(orbit.propagation_step_s),
            str(orbit.propagation_backend),
        )
    cache_key = (json.dumps(task_sig), round(float(resources.slew_rate_deg_s), 6), orbit_sig)
    cached = _TRANSITION_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    transitions: Dict[tuple[str, str], Transition] = {}
    task_ids = list(tasks.keys())
    ephemeris = _build_orbit_ephemeris(orbit) if orbit is not None else None

    for from_id in task_ids:
        for to_id in task_ids:
            if from_id == to_id:
                continue
            from_task = tasks[from_id]
            to_task = tasks[to_id]

            if orbit is not None:
                from_ref_s = float(from_task.window_start_s) + 0.5 * float(from_task.duration_s)
                to_ref_s = max(from_ref_s, float(to_task.window_start_s) + 0.5 * float(to_task.duration_s))

                sat_from = satellite_position_eci(from_ref_s, orbit, ephemeris=ephemeris)
                sat_to = satellite_position_eci(to_ref_s, orbit, ephemeris=ephemeris)

                if from_task.task_type in {"start", "end"}:
                    from_vec = -_unit(sat_from)
                elif ("lat_deg" in from_task.metadata and "lon_deg" in from_task.metadata):
                    from_ecef = latlon_to_ecef(float(from_task.metadata["lat_deg"]), float(from_task.metadata["lon_deg"]))
                    from_eci = ecef_to_eci(from_ecef, from_ref_s, orbit.epoch_theta0_rad)
                    from_vec = _unit(from_eci - sat_from)
                else:
                    from_vec = -_unit(sat_from)

                if to_task.task_type in {"start", "end"}:
                    to_vec = -_unit(sat_to)
                elif ("lat_deg" in to_task.metadata and "lon_deg" in to_task.metadata):
                    to_ecef = latlon_to_ecef(float(to_task.metadata["lat_deg"]), float(to_task.metadata["lon_deg"]))
                    to_eci = ecef_to_eci(to_ecef, to_ref_s, orbit.epoch_theta0_rad)
                    to_vec = _unit(to_eci - sat_to)
                else:
                    to_vec = -_unit(sat_to)

                nominal_slew_deg = vector_angle_deg(from_vec, to_vec)
            elif from_task.task_type in {"start", "end"} or to_task.task_type in {"start", "end"}:
                nominal_slew_deg = 0.0
            elif ("lat_deg" in from_task.metadata and "lat_deg" in to_task.metadata):
                a = latlon_to_ecef(float(from_task.metadata["lat_deg"]), float(from_task.metadata["lon_deg"]))
                b = latlon_to_ecef(float(to_task.metadata["lat_deg"]), float(to_task.metadata["lon_deg"]))
                nominal_slew_deg = vector_angle_deg(a, b)
            else:
                nominal_slew_deg = 0.0

            transitions[(from_id, to_id)] = Transition(
                from_task_id=from_id,
                to_task_id=to_id,
                travel_time_s=nominal_slew_deg / max(1e-6, resources.slew_rate_deg_s),
                energy_cost_wh=0.0,
                feasible=True,
                metadata={"slew_angle_deg_nominal": nominal_slew_deg},
            )

    _TRANSITION_CACHE[cache_key] = transitions
    return transitions


def build_initial_state(spacecraft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = SpacecraftState(
        time_s=0.0,
        battery_wh=float(spacecraft_cfg["battery_initial_wh"]),
        solar_scale=float(spacecraft_cfg.get("solar_scale", 1.0)),
        data_buffer_mb=0.0,
        science_buffer_value=0.0,
        delivered_science=0.0,
        ops_this_orbit=0,
        last_orbit_index=0,
        pointing_vec_eci=(-1.0, 0.0, 0.0),
        last_op_end_s=-1e12,
        total_energy_used_wh=0.0,
        total_time_s=0.0,
        executed_observations=0,
        executed_downlinks=0,
        target_visits={},
        repeat_value_decay=float(spacecraft_cfg.get("repeat_value_decay", 0.75)),
        repeat_value_floor_fraction=float(spacecraft_cfg.get("repeat_value_floor_fraction", 0.35)),
    )
    return state.to_mapping()


def _task_pointing_vector_and_visibility(
    to_task: MissionTask,
    sat_eci: np.ndarray,
    t_s: float,
    orbit: OrbitConfig,
    resources: SpacecraftResourceConfig,
) -> tuple[np.ndarray, VisibilityMetrics]:
    if to_task.task_type in {"start", "end"}:
        nadir = -_unit(sat_eci)
        return nadir, VisibilityMetrics(visible=True, elevation_deg=0.0, off_nadir_deg=0.0)

    if "lat_deg" not in to_task.metadata or "lon_deg" not in to_task.metadata:
        nadir = -_unit(sat_eci)
        return nadir, VisibilityMetrics(visible=False, elevation_deg=0.0, off_nadir_deg=0.0)

    target_ecef = latlon_to_ecef(float(to_task.metadata["lat_deg"]), float(to_task.metadata["lon_deg"]))
    target_eci = ecef_to_eci(target_ecef, t_s, orbit.epoch_theta0_rad)
    los_eci = _unit(target_eci - sat_eci)

    if to_task.task_type == "observation":
        vis = observation_visibility(
            sat_eci=sat_eci,
            target_ecef=target_ecef,
            t_s=t_s,
            orbit=orbit,
            max_off_nadir_deg=resources.max_observation_off_nadir_deg,
        )
        return los_eci, vis

    if to_task.task_type == "downlink":
        vis = downlink_visibility(
            sat_eci=sat_eci,
            station_ecef=target_ecef,
            t_s=t_s,
            orbit=orbit,
            min_elev_deg=resources.min_downlink_elevation_deg,
        )
        return los_eci, vis

    return los_eci, VisibilityMetrics(visible=True, elevation_deg=0.0, off_nadir_deg=0.0)


def simulate_spacecraft_step(
    state: Dict[str, Any],
    from_task: MissionTask,
    to_task: MissionTask,
    transition: Transition,
    orbit: OrbitConfig,
    resources: SpacecraftResourceConfig,
) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    typed_state = SpacecraftState.from_mapping(state, resources=resources)
    now = float(typed_state.time_s)
    prev_vec = np.asarray(typed_state.pointing_vec_eci, dtype=float)
    ephemeris = _build_orbit_ephemeris(orbit)

    control_slew_scale = float(transition.metadata.get("control_slew_scale", 1.0))
    control_slew_scale = float(np.clip(control_slew_scale, 0.6, 1.4))
    control_obs_power_scale = float(transition.metadata.get("control_obs_power_scale", 1.0))
    control_obs_power_scale = float(np.clip(control_obs_power_scale, 0.7, 1.4))
    control_downlink_power_scale = float(transition.metadata.get("control_downlink_power_scale", 1.0))
    control_downlink_power_scale = float(np.clip(control_downlink_power_scale, 0.7, 1.5))
    control_downlink_rate_scale = float(transition.metadata.get("control_downlink_rate_scale", 1.0))
    control_downlink_rate_scale = float(np.clip(control_downlink_rate_scale, 0.6, 1.6))
    effective_slew_rate = resources.slew_rate_deg_s * control_slew_scale

    guess_start = max(now, float(to_task.window_start_s))
    slew_angle_deg = 0.0
    slew_time_s = 0.0
    to_vec = prev_vec
    vis_start = VisibilityMetrics(visible=True, elevation_deg=0.0, off_nadir_deg=0.0)

    if to_task.task_type == "end":
        # Terminal closure should remain feasible even when no time remains for a final repoint.
        to_vec = prev_vec
        slew_angle_deg = 0.0
        slew_time_s = 0.0
        guess_start = max(now, float(to_task.window_start_s))
    else:
        for _ in range(4):
            sat_eci_guess = satellite_position_eci(guess_start, orbit, ephemeris=ephemeris)
            to_vec, vis_start = _task_pointing_vector_and_visibility(to_task, sat_eci_guess, guess_start, orbit, resources)

            if to_task.task_type in {"observation", "downlink"} and not vis_start.visible:
                return None

            slew_angle_deg = vector_angle_deg(prev_vec, to_vec)
            slew_time_s = slew_angle_deg / max(1e-6, effective_slew_rate)
            new_start = max(now + slew_time_s, float(to_task.window_start_s))
            if abs(new_start - guess_start) <= 1e-3:
                guess_start = new_start
                break
            guess_start = new_start

    if to_task.task_type == "end":
        horizon_start = float(to_task.window_start_s)
        horizon_end = float(to_task.window_end_s)
        task_start_s = min(max(guess_start, horizon_start), horizon_end)
        task_end_s = horizon_end
    else:
        task_start_s = guess_start
        task_end_s = task_start_s + float(to_task.duration_s)
        if task_end_s > float(to_task.window_end_s):
            return None

    if to_task.task_type in {"observation", "downlink"}:
        mid_t = 0.5 * (task_start_s + task_end_s)
        sat_mid = satellite_position_eci(mid_t, orbit, ephemeris=ephemeris)
        _, vis_mid = _task_pointing_vector_and_visibility(to_task, sat_mid, mid_t, orbit, resources)
        if not vis_mid.visible:
            return None

        last_op_end = float(typed_state.last_op_end_s)
        if task_start_s - last_op_end < resources.min_operation_gap_s:
            return None

    idle_dt_s = max(0.0, task_start_s - now)
    solar_scale = float(typed_state.solar_scale)
    solar_w = resources.solar_charge_w * resources.solar_capture_efficiency * solar_scale
    idle_delta_wh = integrate_battery_change_wh(
        t0_s=now,
        dt_s=idle_dt_s,
        orbit=orbit,
        solar_w=solar_w,
        load_w=resources.housekeeping_power_w,
        ephemeris=ephemeris,
    )
    battery_after_idle = float(typed_state.battery_wh) + idle_delta_wh
    battery_after_idle = min(resources.battery_capacity_wh, battery_after_idle)

    task_load_w = resources.housekeeping_power_w
    if to_task.task_type == "observation":
        task_load_w += resources.obs_power_w * control_obs_power_scale
    elif to_task.task_type == "downlink":
        task_load_w += resources.downlink_power_w * control_downlink_power_scale

    task_delta_wh = integrate_battery_change_wh(
        t0_s=task_start_s,
        dt_s=float(to_task.duration_s),
        orbit=orbit,
        solar_w=solar_w,
        load_w=task_load_w,
        ephemeris=ephemeris,
    )
    battery_after = battery_after_idle + task_delta_wh
    battery_after = min(resources.battery_capacity_wh, battery_after)

    data_before = float(typed_state.data_buffer_mb)
    science_before = float(typed_state.science_buffer_value)
    delivered_before = float(typed_state.delivered_science)

    data_after = data_before
    science_after = science_before
    delivered_after = delivered_before
    downlinked_mb = 0.0
    delivered_gain = 0.0

    target_visits = dict(typed_state.target_visits)

    effective_observation_value = 0.0
    observation_repeat_count = 0
    if to_task.task_type == "observation":
        target_id = str(to_task.metadata.get("target_id", to_task.task_id))
        visits = int(target_visits.get(target_id, 0))
        observation_repeat_count = visits
        value_decay = float(np.clip(typed_state.repeat_value_decay, 0.0, 1.0))
        value_floor = float(np.clip(typed_state.repeat_value_floor_fraction, 0.0, 1.0))
        effective_observation_value = float(to_task.value) * (value_decay ** visits)
        effective_observation_value = max(effective_observation_value, float(to_task.value) * value_floor)

        data_after += float(to_task.metadata.get("data_mb", 120.0))
        science_after += effective_observation_value
        target_visits[target_id] = visits + 1

    if to_task.task_type == "downlink" and data_before > 0.0:
        capacity_mb = resources.downlink_rate_mb_s * control_downlink_rate_scale * float(to_task.duration_s)
        downlinked_mb = min(data_before, capacity_mb)
        fraction = downlinked_mb / data_before if data_before > 1e-6 else 0.0
        delivered_gain = science_before * fraction
        data_after = data_before - downlinked_mb
        science_after = max(0.0, science_before - delivered_gain)
        delivered_after = delivered_before + delivered_gain

    orbit_idx = int(task_start_s // orbit.orbit_period_s)
    previous_orbit = int(typed_state.last_orbit_index)
    ops_count = int(typed_state.ops_this_orbit)
    if orbit_idx != previous_orbit:
        ops_count = 0

    if to_task.task_type in {"observation", "downlink"}:
        ops_count += 1

    if battery_after < resources.battery_min_wh - 1e-9:
        return None
    if data_after > resources.data_buffer_capacity_mb + 1e-9:
        return None
    if ops_count > resources.max_ops_per_orbit:
        return None

    gross_use_wh = (
        resources.housekeeping_power_w * (idle_dt_s / 3600.0)
        + task_load_w * (float(to_task.duration_s) / 3600.0)
    )

    next_state = SpacecraftState(
        time_s=float(task_end_s),
        battery_wh=float(battery_after),
        solar_scale=float(solar_scale),
        data_buffer_mb=float(data_after),
        science_buffer_value=float(science_after),
        delivered_science=float(delivered_after),
        ops_this_orbit=int(ops_count),
        last_orbit_index=int(orbit_idx),
        pointing_vec_eci=tuple(float(x) for x in to_vec.tolist()),
        last_op_end_s=float(task_end_s if to_task.task_type in {"observation", "downlink"} else typed_state.last_op_end_s),
        total_energy_used_wh=float(typed_state.total_energy_used_wh + gross_use_wh),
        total_time_s=float(task_end_s),
        executed_observations=int(typed_state.executed_observations + (1 if to_task.task_type == "observation" else 0)),
        executed_downlinks=int(typed_state.executed_downlinks + (1 if to_task.task_type == "downlink" else 0)),
        target_visits=target_visits,
        repeat_value_decay=float(typed_state.repeat_value_decay),
        repeat_value_floor_fraction=float(typed_state.repeat_value_floor_fraction),
    )
    state_after = dict(state)
    state_after.update(next_state.to_mapping())

    step_meta: Dict[str, Any] = {
        "action_type": str(to_task.task_type),
        "task_start_s": task_start_s,
        "task_end_s": task_end_s,
        "transition_time_s": task_start_s - now,
        "task_duration_s": float(to_task.duration_s),
        "slew_angle_deg": slew_angle_deg,
        "slew_time_s": slew_time_s,
        "effective_slew_rate_deg_s": effective_slew_rate,
        "control_slew_scale": control_slew_scale,
        "control_obs_power_scale": control_obs_power_scale,
        "control_downlink_power_scale": control_downlink_power_scale,
        "control_downlink_rate_scale": control_downlink_rate_scale,
        "idle_time_s": idle_dt_s,
        "battery_delta_idle_wh": idle_delta_wh,
        "battery_delta_task_wh": task_delta_wh,
        "downlinked_mb": downlinked_mb,
        "delivered_gain": delivered_gain,
        "effective_observation_value": effective_observation_value,
        "target_obs_repeat_count": observation_repeat_count,
        "data_buffer_before_mb": data_before,
        "data_buffer_after_mb": data_after,
        "science_buffer_before": science_before,
        "science_buffer_after": science_after,
        "orbit_index": orbit_idx,
        "visibility_elevation_deg": vis_start.elevation_deg,
        "visibility_off_nadir_deg": vis_start.off_nadir_deg,
    }

    return state_after, step_meta
