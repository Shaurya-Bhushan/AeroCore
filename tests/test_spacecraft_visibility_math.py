from __future__ import annotations

from src.spacecraft.model import (
    OrbitConfig,
    R_EARTH_M,
    latlon_to_ecef,
    line_of_sight_clear_to_surface,
    observation_visibility,
    satellite_position_eci,
)


def test_line_of_sight_surface_clear_and_blocked() -> None:
    sat = [R_EARTH_M + 550000.0, 0.0, 0.0]
    target_near = [R_EARTH_M, 0.0, 0.0]
    target_far = [-R_EARTH_M, 0.0, 0.0]

    assert line_of_sight_clear_to_surface(sat_eci=sat, target_eci=target_near)
    assert not line_of_sight_clear_to_surface(sat_eci=sat, target_eci=target_far)


def test_observation_visibility_rejects_far_side_target() -> None:
    orbit = OrbitConfig(
        altitude_m=550000.0,
        inclination_deg=0.0,
        raan_deg=0.0,
        arg_lat0_deg=0.0,
        epoch_theta0_deg=0.0,
        horizon_s=3600.0,
    )
    sat = satellite_position_eci(0.0, orbit)

    near_target = latlon_to_ecef(0.0, 0.0)
    far_target = latlon_to_ecef(0.0, 180.0)

    vis_near = observation_visibility(
        sat_eci=sat,
        target_ecef=near_target,
        t_s=0.0,
        orbit=orbit,
        max_off_nadir_deg=80.0,
    )
    vis_far = observation_visibility(
        sat_eci=sat,
        target_ecef=far_target,
        t_s=0.0,
        orbit=orbit,
        max_off_nadir_deg=80.0,
    )

    assert vis_near.visible
    assert vis_near.elevation_deg > 0.0
    assert not vis_far.visible
