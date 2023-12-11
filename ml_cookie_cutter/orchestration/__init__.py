import os
from pathlib import Path
from typing import Literal

from dagster import (
    Definitions,
    load_asset_checks_from_modules,
    load_assets_from_modules,
    materialize,
)
from dagster_duckdb import DuckDBResource
from dagster_duckdb_polars import DuckDBPolarsIOManager

from ml_cookie_cutter.data.constants import (
    DAGSTER_HOME,
    DATALAKE_DIRECTORY,
    DATASET_PREFIX,
    RAW_DATASET_DIRECTORY,
)
from ml_cookie_cutter.orchestration import timeseries_example
from ml_cookie_cutter.orchestration.io_managers import (
    LocalPolarsParquetIOManager,
    SourceAssetPolarsIOManager,
)
from ml_cookie_cutter.orchestration.timeseries_example import (
    timeseries_average_per_day,
    timeseries_example_asset,
    timeseries_example_cleaned,
    timeseries_example_df,
)

all_assets = load_assets_from_modules([timeseries_example])
all_asset_checks = load_asset_checks_from_modules([timeseries_example])

duckdb_resource = DuckDBResource(
    database=str(DATALAKE_DIRECTORY / "duckdb.db"),  # required
)
parquet_io_manager = LocalPolarsParquetIOManager(base_path=str(DATALAKE_DIRECTORY))
duckdb_polars_io_manager = DuckDBPolarsIOManager(database=str(DATALAKE_DIRECTORY / "duckdb.db"))


os.environ["DAGSTER_HOME"] = str(DAGSTER_HOME)
os.environ["RAW_DATA_VAULT"] = str(RAW_DATASET_DIRECTORY)

defs = Definitions(
    assets=all_assets,
    asset_checks=all_asset_checks,
    resources={
        "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
        "duckdb": duckdb_resource,
        "duckdb_polars_io_manager": duckdb_polars_io_manager,
        "local_polars_parquet_io_manager": parquet_io_manager,
    },
)


def materialize_timeseries_data_assets(
    io_manager: Literal["duckdb_polars_io_manager", "local_polars_parquet_io_manager"]
):
    # TODO: Add support for duckdb_polars_io_manager for dataset assets
    return materialize(
        [
            timeseries_example_asset,
            timeseries_example_df,
            timeseries_example_cleaned,
            timeseries_average_per_day,
        ],
        resources={
            "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
            "local_polars_parquet_io_manager": parquet_io_manager,
        },
    )


class Project:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_datasets(self) -> list[Path]:
        return list((DATALAKE_DIRECTORY / self.name / DATASET_PREFIX).rglob("*.parquet"))

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.name == __value
        return super().__eq__(__value)

    def __repr__(self) -> str:
        return f"Project(name={self.name})"

    def __str__(self) -> str:
        return self.name


projects = [Project(name=timeseries_example.DATASET_PREFIX)]
