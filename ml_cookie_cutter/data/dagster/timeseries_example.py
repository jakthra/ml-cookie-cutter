import polars as pl

from dagster import (
    AssetIn,
    AssetKey,
    Output,
    SourceAsset,
    TableColumn,
    TableSchema,
    TableSchemaMetadataValue,
    asset,
    asset_check,
)
from ml_cookie_cutter.data.raw import RawDataset

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
)
def timeseries_example_cleaned(timeseries_example_df: pl.DataFrame):
    return timeseries_example_df


@asset_check(asset=timeseries_example_cleaned)
def no_nulls_in_timeseries_example_cleaned(context):
    df = context.upstream_output(timeseries_example_cleaned).load_input("timeseries_example_cleaned")
    return df.is_not_null().all().all()
