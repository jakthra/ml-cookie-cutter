from dagster import Definitions, load_assets_from_modules, load_asset_checks_from_modules
from ml_cookie_cutter.data.constants import DATALAKE_DIRECTORY

from ml_cookie_cutter.data.dagster import timeseries_example
from ml_cookie_cutter.data.dagster.io_managers import LocalPolarsParquetIOManager, SourceAssetPolarsIOManager
from dagster_duckdb import DuckDBResource

from dagster_duckdb_polars import DuckDBPolarsIOManager

all_assets = load_assets_from_modules([timeseries_example])
all_asset_checks = load_asset_checks_from_modules([timeseries_example])

duckdb_resource = DuckDBResource(
    database=str(DATALAKE_DIRECTORY / "duckdb.db"),  # required
)

defs = Definitions(
    assets=all_assets,
    asset_checks=all_asset_checks,
    resources={
        "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
        "duckdb": duckdb_resource,
        "duckdb_polars_io_manager": DuckDBPolarsIOManager(database=str(DATALAKE_DIRECTORY / "duckdb.db")),
        "local_polars_parquet_io_manager": LocalPolarsParquetIOManager(base_path=str(DATALAKE_DIRECTORY)),
    },
)
