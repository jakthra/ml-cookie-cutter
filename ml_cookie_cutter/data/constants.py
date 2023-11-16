from pathlib import Path

import ml_cookie_cutter

root = Path(ml_cookie_cutter.__file__).parents[1]
DATASET_DIRECTORY = root / "data"
DAGSTER_HOME = root
RAW_DATASET_DIRECTORY: Path = DATASET_DIRECTORY / "raw"
DATALAKE_DIRECTORY: Path = DATASET_DIRECTORY / "datalake"
DUCKDB_PATH = DATALAKE_DIRECTORY / "duckdb.db"
