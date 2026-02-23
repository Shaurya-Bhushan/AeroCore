# Data Assets

This directory contains mission input datasets used by the default configuration:

- `aircraft_waypoints.json`
- `aircraft_no_fly_zones.json`
- `spacecraft_targets.json`
- `spacecraft_ground_stations.json`
- `sample_iss.tle` (example NORAD TLE for ingestion pipeline testing)

`configs/default.yaml` and `configs/full_validation.yaml` reference these files via `*.data_files`.
Inline copies are also kept in config for portability, but `run_all.py` resolves and injects these files at runtime.

Use `tools/tle_to_spacecraft_config.py` to convert `.tle` files into spacecraft config overrides.
