"""Retired in-place history repair entrypoint."""


def main() -> None:
    raise RuntimeError(
        "In-place history repair is disabled because it corrupts dual-endpoint semantics. "
        "Rebuild cohorts with src/data_pipeline/HBD_data.py and HBD_data_fuding.py."
    )


if __name__ == "__main__":
    main()
