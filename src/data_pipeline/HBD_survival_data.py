"""Official V4 survival-cohort entry point for Shenyi/source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RAW_SHA256 = "607a7f115efa2db64c3f23190f82222978d10146570bdbf6e1f66118bd2320f9"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.data_pipeline.data_process import SURVIVAL_DATA_PROTOCOL_VERSION, write_survival_cohort

    parser = argparse.ArgumentParser(
        description=f"Build the Shenyi survival cohort under {SURVIVAL_DATA_PROTOCOL_VERSION}."
    )
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            write_survival_cohort(
                args.raw_csv,
                "shenyi",
                args.output_csv,
                expected_raw_sha256=EXPECTED_RAW_SHA256,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
