"""Build the Fuding cohort from the raw Hugging Face export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    from src.data_pipeline.data_process import write_cohort

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_cohort(args.raw_csv, "fuding", args.output_csv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
