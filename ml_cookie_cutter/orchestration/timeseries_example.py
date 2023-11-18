import warnings
from typing import Tuple

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetIn,
    AssetKey,
    Config,
    ExperimentalWarning,
    Output,
    SourceAsset,
    TableColumn,
    TableSchema,
    TableSchemaMetadataValue,
    asset,
)
from pydantic import model_validator

from ml_cookie_cutter.data.raw import RawDataset

warnings.filterwarnings("ignore", category=ExperimentalWarning)

DATASET_PREFIX = "timeseries"

timeseries_example_dataset = RawDataset("timeseries-example")


timeseries_example_asset = SourceAsset(
    key=AssetKey([DATASET_PREFIX, timeseries_example_dataset.name]),
    metadata={**timeseries_example_dataset.to_dict(), "read_kwargs": timeseries_example_dataset.pl_read_kwargs},
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

    # Derive datetime column from date and time columns
    df = df.with_columns((df["Date"] + " " + df["Time"]).str.to_datetime(format="%d/%m/%Y %H:%M:%S").alias("Datetime"))
    rows_after_clean = df.height
    yield Output(value=df, metadata={"rows_before_clean": rows_before_clean, "rows_after_clean": rows_after_clean})

    yield AssetCheckResult(
        passed=(df.null_count().sum(axis=1).sum() == 0),
    )


def average_global_active_power_per_temporal_unit(df: pl.DataFrame, temporal_unit: str):
    return df.sort(["Datetime"]).rolling("Datetime", period=temporal_unit).agg(pl.col("Global_active_power").mean())


@asset(
    code_version="1",
    io_manager_key="local_polars_parquet_io_manager",
    ins={"timeseries_example_cleaned": AssetIn(timeseries_example_cleaned.key)},
    key_prefix=[DATASET_PREFIX, "dataset"],
)
def timeseries_average_per_day(timeseries_example_cleaned: pl.DataFrame):
    return average_global_active_power_per_temporal_unit(timeseries_example_cleaned, "1d")


# class TimeSeriesVersionedDatasetFactory:
#     def __init__(self, prefix: str) -> None:
#         self.prefix = prefix

#     def __call__(self, _asset: AssetsDefinition,
#           version: semver.Version = semver.Version("0.0.1")) -> AssetsDefinition:
#         @asset(name=f"versioned_{_asset.key.path[-1]}",
#           key_prefix=[self.prefix, "dataset", _asset.key.path[-1], f"v{str(version).replace('.', '_')}"])
#         def versioned_asset():
#             return Output(value=_asset, metadata={"version": str(version)})

#         return versioned_asset


# dataset_factory = TimeSeriesVersionedDatasetFactory(DATASET_PREFIX)

# timeseries_average_per_day_dataset = dataset_factory(timeseries_average_per_day, version=semver.Version("0.0.1"))


# @asset(
#     code_version="1",
#     io_manager_key="local_polars_parquet_io_manager",
#     key_prefix=[DATASET_PREFIX],
# )
# def timeseries_train_test_split(
#     config: TrainTestConfig,
#     timeseries_average_per_day: pl.DataFrame,
# ):
#     timeseries_average_per_day = timeseries_average_per_day
#     train, test = config.apply_split(timeseries_average_per_day)
#     return train, test


class TrainTestConfig(Config):
    train: float = 0.8
    test: float = 0.2

    @model_validator(mode="after")
    def validate_ratio(self) -> "TrainTestConfig":
        if self.train + self.test != 1:
            raise ValueError("Train test ratios must sum to 1")
        return self

    def apply_split(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        train_index = int(self.train * df.height)
        test_index = int(self.test * df.height)
        assert train_index + test_index == df.height
        return df[:train_index], df[train_index:]
