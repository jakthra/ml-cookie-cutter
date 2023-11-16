from dagster_duckdb import DuckDBResource
from dagster_duckdb_polars import DuckDBPolarsIOManager

from ml_cookie_cutter.data.constants import DATALAKE_DIRECTORY
from ml_cookie_cutter.orchestration.io_managers import LocalPolarsParquetIOManager

duckdb_polars_io_manager = DuckDBPolarsIOManager(database=str(DATALAKE_DIRECTORY / "duckdb.db"))
parquet_io_manager = LocalPolarsParquetIOManager(base_path=str(DATALAKE_DIRECTORY))
duckdb_resource = DuckDBResource(
    database=str(DATALAKE_DIRECTORY / "duckdb.db"),  # required
)
