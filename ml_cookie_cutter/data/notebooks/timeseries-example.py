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
import duckdb
import pandas as pd
import plotly.express as px

from ml_cookie_cutter.data.constants import DUCKDB_PATH

# %%
conn = duckdb.connect(str(DUCKDB_PATH))
df = conn.sql("SELECT * FROM timeseries_example_cleaned").to_df()
df["Datetime"] = df["Date"] + " " + df["Time"]
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H:%M:%S")
df.head()

# %% [markdown]
# https://plotly.com/python/ipython-notebook-tutorial/

# %%
active_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
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
df.hist(column=active_columns, figsize=(20, 10), bins=100)

# %%
px.line(df[:2000], x="Datetime", y="Global_active_power", title="Global active power over time")

# %%
df.info()

# %%
