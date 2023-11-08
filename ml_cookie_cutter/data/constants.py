import ml_cookie_cutter


from pathlib import Path

DATASET_DIRECTORY = Path(ml_cookie_cutter.__file__).parents[1] / "data"
RAW_DATASET_DIRECTORY: Path = DATASET_DIRECTORY / "raw"
DATALAKE_DIRECTORY: Path = DATASET_DIRECTORY / "datalake"
DUCKDB_PATH = DATALAKE_DIRECTORY / "duckdb.db"
