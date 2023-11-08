from pathlib import Path
import shutil
from typing import Iterator, Tuple
import pytest
from dagster import materialize
from dagster_duckdb_polars import DuckDBPolarsIOManager
import duckdb

from ml_cookie_cutter.data.dagster.timeseries_example import (
    timeseries_example_asset,
    timeseries_example_cleaned,
    timeseries_example_df,
)
from ml_cookie_cutter.data.dagster.io_managers import LocalPolarsParquetIOManager, SourceAssetPolarsIOManager
import polars as pl


PERSIST_TEST_DATA = True


@pytest.fixture
def test_fixture_output(request: pytest.FixtureRequest) -> Iterator[Path]:
    folder: Path = (Path("tests/fixtures_output") / str(request.node.name)).with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    yield folder
    if not PERSIST_TEST_DATA:
        shutil.rmtree(folder)


@pytest.fixture
def duckdb_persisted_db(test_fixture_output: Path) -> Iterator[Tuple[str, duckdb.DuckDBPyConnection]]:
    db: str = str(test_fixture_output / "duckdb.db")
    conn = duckdb.connect(database=db)
    yield db, conn
    conn.close()
    Path(db).unlink()


@pytest.fixture
def local_parquet_persisted_io_manager(test_fixture_output: Path) -> Iterator[LocalPolarsParquetIOManager]:
    yield LocalPolarsParquetIOManager(base_path=str(test_fixture_output))


class TestTimeseriesMaterialization:
    def test_df_materialization(self):
        result = materialize(
            [timeseries_example_asset, timeseries_example_df],
            resources={"source_asset_polars_io_manager": SourceAssetPolarsIOManager()},
        )
        assert isinstance(result.asset_value(timeseries_example_df.key), pl.DataFrame)

    def test_duckdb_materialization(self, duckdb_persisted_db: Tuple[str, duckdb.DuckDBPyConnection]):
        db, conn = duckdb_persisted_db
        duckdb_resource = DuckDBPolarsIOManager(database=db)
        materialize(
            [timeseries_example_asset, timeseries_example_df, timeseries_example_cleaned],
            resources={
                "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
                "local_polars_parquet_io_manager": duckdb_resource,  # overwrite default io manager of timeseries_example_cleaned to be a duckdb resource
            },
        )

        df = conn.sql("SELECT * FROM timeseries.timeseries_example_cleaned").pl()
        assert isinstance(df, pl.DataFrame)

    def test_local_parquet_materialization(self, local_parquet_persisted_io_manager: LocalPolarsParquetIOManager):
        materialize(
            [timeseries_example_asset, timeseries_example_df, timeseries_example_cleaned],
            resources={
                "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
                "local_polars_parquet_io_manager": local_parquet_persisted_io_manager,
            },
        )
        assert Path("test_data/timeseries/timeseries_example_cleaned.parquet").exists()
