from __future__ import annotations

import argparse
from pathlib import Path

from src.spacecraft.tle_ingest import tle_file_to_config_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NORAD TLE file into spacecraft config overrides JSON."
    )
    parser.add_argument("--tle", type=Path, required=True, help="Path to .tle file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/tle_overrides.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    payload = tle_file_to_config_json(args.tle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
