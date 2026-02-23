from __future__ import annotations

from typing import Any, Dict, List


class ConfigValidationError(ValueError):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(cfg: Dict[str, Any], key: str, path: str, errors: List[str]) -> Any:
    if key not in cfg:
        errors.append(f"Missing key: {path}.{key}")
        return None
    return cfg[key]


def _validate_aircraft(cfg: Dict[str, Any], errors: List[str]) -> None:
    required_keys = [
        "mission_horizon_s",
        "base",
        "waypoints",
        "no_fly_zones",
        "cruise_speed_mps",
        "cruise_power_w",
        "max_turn_rate_deg_s",
        "battery_capacity_wh",
        "reserve_energy_wh",
    ]
    for key in required_keys:
        _require(cfg, key, "aircraft", errors)

    if not _is_number(cfg.get("mission_horizon_s")) or float(cfg.get("mission_horizon_s", 0.0)) <= 0.0:
        errors.append("aircraft.mission_horizon_s must be > 0")

    base = cfg.get("base", {})
    if not isinstance(base, dict):
        errors.append("aircraft.base must be a mapping")
    else:
        for k in ["x_m", "y_m", "alt_m"]:
            if not _is_number(base.get(k)):
                errors.append(f"aircraft.base.{k} must be numeric")

    waypoints = cfg.get("waypoints", [])
    if not isinstance(waypoints, list) or len(waypoints) == 0:
        errors.append("aircraft.waypoints must be a non-empty list")
    else:
        seen = set()
        for idx, wp in enumerate(waypoints):
            if not isinstance(wp, dict):
                errors.append(f"aircraft.waypoints[{idx}] must be a mapping")
                continue
            wp_id = wp.get("id")
            if wp_id in seen:
                errors.append(f"aircraft.waypoints duplicate id: {wp_id}")
            seen.add(wp_id)
            for k in ["x_m", "y_m"]:
                if not _is_number(wp.get(k)):
                    errors.append(f"aircraft.waypoints[{idx}].{k} must be numeric")

    no_fly = cfg.get("no_fly_zones", [])
    if not isinstance(no_fly, list):
        errors.append("aircraft.no_fly_zones must be a list")
    else:
        for i, poly in enumerate(no_fly):
            if not isinstance(poly, list) or len(poly) < 3:
                errors.append(f"aircraft.no_fly_zones[{i}] must have at least 3 vertices")
                continue
            for j, point in enumerate(poly):
                if not isinstance(point, list) or len(point) != 2:
                    errors.append(f"aircraft.no_fly_zones[{i}][{j}] must be [x,y]")
                    continue
                if not _is_number(point[0]) or not _is_number(point[1]):
                    errors.append(f"aircraft.no_fly_zones[{i}][{j}] coordinates must be numeric")

    capacity = float(cfg.get("battery_capacity_wh", 0.0)) if _is_number(cfg.get("battery_capacity_wh")) else -1.0
    reserve = float(cfg.get("reserve_energy_wh", 0.0)) if _is_number(cfg.get("reserve_energy_wh")) else -1.0
    if capacity <= 0.0:
        errors.append("aircraft.battery_capacity_wh must be > 0")
    if reserve < 0.0:
        errors.append("aircraft.reserve_energy_wh must be >= 0")
    if capacity > 0.0 and reserve >= capacity:
        errors.append("aircraft.reserve_energy_wh must be less than aircraft.battery_capacity_wh")

    for k in ["cruise_speed_mps", "cruise_power_w", "max_turn_rate_deg_s", "max_bank_angle_deg", "max_climb_rate_mps"]:
        if k in cfg and (not _is_number(cfg[k]) or float(cfg[k]) <= 0.0):
            errors.append(f"aircraft.{k} must be > 0")

    aero = cfg.get("aerodynamics", {})
    if aero and not isinstance(aero, dict):
        errors.append("aircraft.aerodynamics must be a mapping")
    elif isinstance(aero, dict):
        positive_keys = [
            "mass_kg",
            "wing_area_m2",
            "cd0",
            "aspect_ratio",
            "oswald_efficiency",
            "propulsive_efficiency",
            "air_density_kg_m3",
            "cl_max",
            "stall_margin",
            "max_propulsion_power_w",
            "prop_power_altitude_scale_m",
        ]
        for key in positive_keys:
            if key in aero and (not _is_number(aero.get(key)) or float(aero.get(key)) <= 0.0):
                errors.append(f"aircraft.aerodynamics.{key} must be > 0")

    planner = cfg.get("planner", {})
    if planner and not isinstance(planner, dict):
        errors.append("aircraft.planner must be a mapping")
    elif isinstance(planner, dict):
        prune_keys = [
            "prune_time_bucket_s",
            "prune_energy_bucket_wh",
            "prune_battery_bucket_wh",
            "prune_data_buffer_bucket_mb",
            "prune_science_bucket",
        ]
        for key in prune_keys:
            if key in planner and (not _is_number(planner.get(key)) or float(planner.get(key)) <= 0.0):
                errors.append(f"aircraft.planner.{key} must be > 0")


def _validate_spacecraft(cfg: Dict[str, Any], errors: List[str]) -> None:
    required_keys = [
        "mission_days",
        "targets",
        "ground_stations",
        "battery_capacity_wh",
        "battery_min_wh",
        "battery_initial_wh",
        "data_buffer_capacity_mb",
        "slew_rate_deg_s",
        "downlink_rate_mb_s",
        "max_ops_per_orbit",
    ]
    for key in required_keys:
        _require(cfg, key, "spacecraft", errors)

    mission_days = cfg.get("mission_days")
    if not _is_number(mission_days):
        errors.append("spacecraft.mission_days must be numeric")
    elif int(mission_days) != 7:
        errors.append("spacecraft.mission_days must be 7 for official challenge compliance")

    targets = cfg.get("targets", [])
    if not isinstance(targets, list) or len(targets) == 0:
        errors.append("spacecraft.targets must be a non-empty list")
    else:
        for i, t in enumerate(targets):
            if not isinstance(t, dict):
                errors.append(f"spacecraft.targets[{i}] must be a mapping")
                continue
            for k in ["id", "lat_deg", "lon_deg", "value"]:
                if k not in t:
                    errors.append(f"spacecraft.targets[{i}] missing {k}")
            if _is_number(t.get("lat_deg")) and not (-90.0 <= float(t["lat_deg"]) <= 90.0):
                errors.append(f"spacecraft.targets[{i}].lat_deg must be in [-90,90]")
            if _is_number(t.get("lon_deg")) and not (-180.0 <= float(t["lon_deg"]) <= 180.0):
                errors.append(f"spacecraft.targets[{i}].lon_deg must be in [-180,180]")

    stations = cfg.get("ground_stations", [])
    if not isinstance(stations, list) or len(stations) == 0:
        errors.append("spacecraft.ground_stations must be a non-empty list")
    else:
        for i, s in enumerate(stations):
            if not isinstance(s, dict):
                errors.append(f"spacecraft.ground_stations[{i}] must be a mapping")
                continue
            for k in ["id", "lat_deg", "lon_deg"]:
                if k not in s:
                    errors.append(f"spacecraft.ground_stations[{i}] missing {k}")

    capacity = float(cfg.get("battery_capacity_wh", 0.0)) if _is_number(cfg.get("battery_capacity_wh")) else -1.0
    bmin = float(cfg.get("battery_min_wh", 0.0)) if _is_number(cfg.get("battery_min_wh")) else -1.0
    b0 = float(cfg.get("battery_initial_wh", 0.0)) if _is_number(cfg.get("battery_initial_wh")) else -1.0
    if capacity <= 0.0:
        errors.append("spacecraft.battery_capacity_wh must be > 0")
    if bmin < 0.0:
        errors.append("spacecraft.battery_min_wh must be >= 0")
    if b0 < 0.0:
        errors.append("spacecraft.battery_initial_wh must be >= 0")
    if capacity > 0.0 and bmin >= capacity:
        errors.append("spacecraft.battery_min_wh must be < battery_capacity_wh")
    if capacity > 0.0 and not (bmin <= b0 <= capacity):
        errors.append("spacecraft.battery_initial_wh must be within [battery_min_wh, battery_capacity_wh]")

    if not _is_number(cfg.get("max_ops_per_orbit")) or int(cfg.get("max_ops_per_orbit", 0)) <= 0:
        errors.append("spacecraft.max_ops_per_orbit must be > 0")

    if not _is_number(cfg.get("data_buffer_capacity_mb")) or float(cfg.get("data_buffer_capacity_mb", 0.0)) <= 0:
        errors.append("spacecraft.data_buffer_capacity_mb must be > 0")

    if not _is_number(cfg.get("slew_rate_deg_s")) or float(cfg.get("slew_rate_deg_s", 0.0)) <= 0:
        errors.append("spacecraft.slew_rate_deg_s must be > 0")

    if not _is_number(cfg.get("downlink_rate_mb_s")) or float(cfg.get("downlink_rate_mb_s", 0.0)) <= 0:
        errors.append("spacecraft.downlink_rate_mb_s must be > 0")

    if "solar_capture_efficiency" in cfg:
        if not _is_number(cfg.get("solar_capture_efficiency")) or not (0.0 < float(cfg.get("solar_capture_efficiency", 0.0)) <= 1.0):
            errors.append("spacecraft.solar_capture_efficiency must be in (0,1]")

    if "eccentricity" in cfg:
        if not _is_number(cfg.get("eccentricity")) or not (0.0 <= float(cfg.get("eccentricity", 0.0)) < 1.0):
            errors.append("spacecraft.eccentricity must be in [0,1)")
    if "ballistic_coefficient_kg_m2" in cfg:
        if not _is_number(cfg.get("ballistic_coefficient_kg_m2")) or float(cfg.get("ballistic_coefficient_kg_m2", 0.0)) <= 0.0:
            errors.append("spacecraft.ballistic_coefficient_kg_m2 must be > 0")
    if "atmosphere_ref_density_kg_m3" in cfg:
        if not _is_number(cfg.get("atmosphere_ref_density_kg_m3")) or float(cfg.get("atmosphere_ref_density_kg_m3", 0.0)) <= 0.0:
            errors.append("spacecraft.atmosphere_ref_density_kg_m3 must be > 0")
    if "atmosphere_scale_height_m" in cfg:
        if not _is_number(cfg.get("atmosphere_scale_height_m")) or float(cfg.get("atmosphere_scale_height_m", 0.0)) <= 0.0:
            errors.append("spacecraft.atmosphere_scale_height_m must be > 0")
    if "propagation_step_s" in cfg:
        if not _is_number(cfg.get("propagation_step_s")) or float(cfg.get("propagation_step_s", 0.0)) < 5.0:
            errors.append("spacecraft.propagation_step_s must be >= 5")
    if "propagation_backend" in cfg:
        backend = str(cfg.get("propagation_backend", "rk4")).lower()
        if backend not in {"rk4", "solve_ivp", "auto"}:
            errors.append("spacecraft.propagation_backend must be one of: rk4, solve_ivp, auto")

    min_elev = cfg.get("min_downlink_elevation_deg", 10.0)
    if not _is_number(min_elev) or not (0.0 <= float(min_elev) < 90.0):
        errors.append("spacecraft.min_downlink_elevation_deg must be in [0,90)")

    off_nadir = cfg.get("max_observation_off_nadir_deg", 30.0)
    if not _is_number(off_nadir) or not (0.0 < float(off_nadir) < 90.0):
        errors.append("spacecraft.max_observation_off_nadir_deg must be in (0,90)")

    planner = cfg.get("planner", {})
    if planner and not isinstance(planner, dict):
        errors.append("spacecraft.planner must be a mapping")
    elif isinstance(planner, dict):
        prune_keys = [
            "prune_time_bucket_s",
            "prune_energy_bucket_wh",
            "prune_battery_bucket_wh",
            "prune_data_buffer_bucket_mb",
            "prune_science_bucket",
        ]
        for key in prune_keys:
            if key in planner and (not _is_number(planner.get(key)) or float(planner.get(key)) <= 0.0):
                errors.append(f"spacecraft.planner.{key} must be > 0")


def validate_config(cfg: Dict[str, Any]) -> None:
    errors: List[str] = []

    if not isinstance(cfg, dict):
        raise ConfigValidationError("Configuration must be a mapping")

    for key in ["aircraft", "spacecraft"]:
        if key not in cfg:
            errors.append(f"Missing top-level key: {key}")

    aircraft = cfg.get("aircraft")
    spacecraft = cfg.get("spacecraft")

    if isinstance(aircraft, dict):
        _validate_aircraft(aircraft, errors)
    else:
        errors.append("aircraft must be a mapping")

    if isinstance(spacecraft, dict):
        _validate_spacecraft(spacecraft, errors)
    else:
        errors.append("spacecraft must be a mapping")

    validation = cfg.get("validation", {})
    if validation and isinstance(validation, dict):
        mode = str(validation.get("mode", "fast")).lower()
        if mode not in {"fast", "full"}:
            errors.append("validation.mode must be one of: fast, full")

    planner_strategy = str(cfg.get("planner_strategy", "beam")).lower()
    if planner_strategy not in {"beam", "greedy", "multistart", "hybrid_ocp", "auto_best"}:
        errors.append("planner_strategy must be one of: beam, greedy, multistart, hybrid_ocp, auto_best")

    if errors:
        msg = "Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors)
        raise ConfigValidationError(msg)
