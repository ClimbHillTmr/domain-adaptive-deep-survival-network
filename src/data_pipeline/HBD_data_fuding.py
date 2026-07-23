"""Canonical V3 entry point for the Fuding/target cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RAW_SHA256 = "6fcfd523cd60aedc245d4b51e072aeca777f65e9a56554fdebaf45d3ee5f3813"
EXPECTED_SUMMARY = {
    "output_rows": 74947,
    "n_patients": 430,
    "n_idh_events": 28881,
    "n_ih_events": 8962,
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.data_pipeline.data_process import DATA_PROTOCOL_VERSION, write_cohort

    parser = argparse.ArgumentParser(
        description=f"Build the locked Fuding cohort under {DATA_PROTOCOL_VERSION}."
    )
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            write_cohort(
                args.raw_csv,
                "fuding",
                args.output_csv,
                expected_raw_sha256=EXPECTED_RAW_SHA256,
                expected_summary=EXPECTED_SUMMARY,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
