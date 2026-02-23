from __future__ import annotations

import csv
from pathlib import Path
import zipfile

from run_all import _spacecraft_horizon_covered, create_results_bundle
from src.validation.run_validation import _small_spacecraft_cfg


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action_type", "task_end_s"])
        writer.writeheader()
        writer.writerows(rows)


def test_horizon_guard_requires_end_and_horizon(tmp_path: Path) -> None:
    p = tmp_path / "sched.csv"
    _write_rows(
        p,
        [
            {"action_type": "observation", "task_end_s": 100.0},
            {"action_type": "end", "task_end_s": 604800.0},
        ],
    )
    assert _spacecraft_horizon_covered(p, 604800.0)


def test_horizon_guard_rejects_missing_end(tmp_path: Path) -> None:
    p = tmp_path / "sched.csv"
    _write_rows(
        p,
        [
            {"action_type": "observation", "task_end_s": 100.0},
            {"action_type": "downlink", "task_end_s": 604800.0},
        ],
    )
    assert not _spacecraft_horizon_covered(p, 604800.0)


def test_results_bundle_excludes_noncanonical_folders(tmp_path: Path) -> None:
    (tmp_path / "aircraft").mkdir(parents=True, exist_ok=True)
    (tmp_path / "spacecraft").mkdir(parents=True, exist_ok=True)
    (tmp_path / "validation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "validation_sanity_new").mkdir(parents=True, exist_ok=True)
    (tmp_path / "spacecraft_tmp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "aircraft" / "uav_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "spacecraft" / "spacecraft_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation" / "validation_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation_sanity_new" / "junk.txt").write_text("x", encoding="utf-8")
    (tmp_path / "spacecraft_tmp" / "junk.txt").write_text("x", encoding="utf-8")
    (tmp_path / "run_summary.json").write_text("{}", encoding="utf-8")

    bundle = create_results_bundle(tmp_path)
    with zipfile.ZipFile(bundle, "r") as zf:
        names = set(zf.namelist())

    assert "aircraft/uav_metrics.json" in names
    assert "spacecraft/spacecraft_metrics.json" in names
    assert "validation/validation_summary.json" in names
    assert "run_summary.json" in names
    assert "validation_sanity_new/junk.txt" not in names
    assert "spacecraft_tmp/junk.txt" not in names


def test_validation_profile_defaults_to_full_mission_days() -> None:
    cfg = {"mission_days": 7, "max_obs_windows": 180, "max_dl_windows": 160, "opportunity_step_s": 90.0}
    compact = _small_spacecraft_cfg(cfg)
    assert int(compact["mission_days"]) == 7
