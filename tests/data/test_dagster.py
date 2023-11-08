from pathlib import Path
from typing import Iterator, Tuple
from dagster import materialize
import duckdb

from ml_cookie_cutter.data.dagster.raw_datasets import (
    timeseries_example_asset,
    timeseries_example_cleaned,
    timeseries_example_df,
)
from ml_cookie_cutter.data.dagster.io_managers import SourceAssetPolarsIOManager
import polars as pl
import pytest

from dagster_duckdb_polars import DuckDBPolarsIOManager


@pytest.fixture
def duckdb_persisted_db(request: pytest.FixtureRequest) -> Iterator[Tuple[str, duckdb.DuckDBPyConnection]]:
    db: str = str(Path(str(request.node.name)).with_suffix(".db"))
    conn = duckdb.connect(database=db)
    yield db, conn
    conn.close()
    Path(db).unlink()


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
                "duckdb_polars_io_manager": duckdb_resource,
            },
        )

        df = conn.sql("SELECT * FROM timeseries.timeseries_example_cleaned").pl()
        assert isinstance(df, pl.DataFrame)
