import warnings

import polars as pl

from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetIn,
    AssetKey,
    ExperimentalWarning,
    Output,
    SourceAsset,
    TableColumn,
    TableSchema,
    TableSchemaMetadataValue,
    asset,
)
from ml_cookie_cutter.data.raw import RawDataset

warnings.filterwarnings("ignore", category=ExperimentalWarning)

DATASET_PREFIX = "timeseries"

timeseries_example_dataset = RawDataset("timeseries-example")


timeseries_example_asset = SourceAsset(
    key=AssetKey([DATASET_PREFIX, timeseries_example_dataset.name]),
    metadata={**timeseries_example_dataset.to_dict(), "read_kwargs": {"separator": ";", "null_values": ["?"]}},
    description="A dataset of household power consumption.",
    io_manager_key="source_asset_polars_io_manager",
)


def polars_df_to_dagster_df(df: pl.DataFrame):
    return Output(
        df,
        metadata={
            "schema": TableSchemaMetadataValue(
                TableSchema([TableColumn(name=key, type=str(value)) for key, value in df.schema.items()])
            )
        },
    )


@asset(
    code_version="1",
    key_prefix=[DATASET_PREFIX],
    ins={"timeseries_example_asset": AssetIn(timeseries_example_asset.key)},
)
def timeseries_example_df(timeseries_example_asset: pl.DataFrame):
    return polars_df_to_dagster_df(timeseries_example_asset)


@asset(
    code_version="1",
    ins={"timeseries_example_df": AssetIn(timeseries_example_df.key)},
    io_manager_key="local_polars_parquet_io_manager",
    key_prefix=[DATASET_PREFIX],
    check_specs=[AssetCheckSpec(name="timeseries_has_no_nulls", asset=[DATASET_PREFIX, "timeseries_example_cleaned"])],
)
def timeseries_example_cleaned(timeseries_example_df: pl.DataFrame):
    rows_before_clean = timeseries_example_df.height
    df = timeseries_example_df.drop_nulls()
    rows_after_clean = df.height
    yield Output(value=df, metadata={"rows_before_clean": rows_before_clean, "rows_after_clean": rows_after_clean})

    yield AssetCheckResult(
        passed=(df.null_count().sum(axis=1).sum() == 0),
    )
