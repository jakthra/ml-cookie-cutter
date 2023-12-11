# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl
import seaborn as sns

from dagster import materialize
from ml_cookie_cutter.orchestration.definitions import parquet_io_manager
from ml_cookie_cutter.orchestration.io_managers import SourceAssetPolarsIOManager
from ml_cookie_cutter.orchestration.timeseries_example import (
    timeseries_average_per_day,
    timeseries_example_asset,
    timeseries_example_cleaned,
    timeseries_example_df,
)

# %%
result = materialize(
    [timeseries_example_asset, timeseries_example_df, timeseries_example_cleaned, timeseries_average_per_day],
    resources={
        "source_asset_polars_io_manager": SourceAssetPolarsIOManager(),
        "local_polars_parquet_io_manager": parquet_io_manager,
    },
)

# %%
df = result.asset_value(timeseries_example_cleaned.key)

# %%
df

# %% [markdown]
# https://plotly.com/python/ipython-notebook-tutorial/

# %%
active_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# %%

# fig = make_subplots(rows=2, cols=3)

# for idx, column in enumerate(active_columns):
#     col = idx % 3 + 1
#     row = int(idx // 3) + 1
#     print("row: ", row)
#     print("col: ", row)
#     fig.add_trace(go.Histogram(x=df[column]), row=row, col=col)

# fig.show(width=400)
df_pandas = df.to_pandas()
df_pandas.hist(column=active_columns, figsize=(20, 10), bins=100)

# %%
px.line(df[:2000], x="Datetime", y="Global_active_power", title="Global active power over time")


# %%
def average_global_active_power_per_temporal_unit(df: pl.DataFrame, temporal_unit: str):
    return df.sort(["Datetime"]).rolling("Datetime", period=temporal_unit).agg(pl.col("Global_active_power").mean())


average_global_active_power_per_day = average_global_active_power_per_temporal_unit(df, "1d")
average_global_active_power_per_hour = average_global_active_power_per_temporal_unit(df, "1h")

fig = plt.figure(figsize=(20, 10))
sns.set_theme(style="darkgrid")
sns.lineplot(
    average_global_active_power_per_hour,
    x="Datetime",
    y="Global_active_power",
    label="Average global active power per hour",
)
sns.lineplot(
    average_global_active_power_per_day,
    x="Datetime",
    y="Global_active_power",
    label="Average global active power per day",
)
# plt.plot(average_global_active_power_per_hour["Datetime"], average_global_active_power_per_hour["Global_active_power"], label="Average global active power per hour")
# plt.plot(average_global_active_power_per_day["Datetime"], average_global_active_power_per_day["Global_active_power"],  label="Average global active power per day")
# plt.legend()
plt.show()
# px.line(average_global_active_power_per_hour[:2000], x="Datetime", y="Global_active_power", title="Global active power over time")

# %%
# Visualize traces per day

days_to_sample = df.sort(["Datetime"]).group_by_dynamic("Datetime", every="1d").agg()

df = df.with_columns(pl.col("Datetime").dt.time().alias("Time"))

# Sample 10 random days
days_to_sample = days_to_sample.sample(10)

# Plot the traces
fig = plt.figure(figsize=(20, 10))
for day in days_to_sample["Datetime"]:
    day_df = df.filter(pl.col("Datetime").is_between(day, day + timedelta(hours=23, minutes=59, seconds=59)))
    sns.lineplot(day_df, x="Datetime", y="Global_active_power", label=day.strftime("%Y-%m-%d"))

# fig = plt.figure(figsize=(20, 10))
