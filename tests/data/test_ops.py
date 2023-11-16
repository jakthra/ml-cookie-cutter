import polars as pl
import pytest

from ml_cookie_cutter.orchestration.timeseries_example import TrainTestConfig


@pytest.fixture
def timeseries_example_df():
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10), eager=True),
        }
    )


def test_apply_split_timeseries(timeseries_example_df: pl.DataFrame):
    df_sorted = timeseries_example_df.sort("date")
    cfg = TrainTestConfig(train=0.8, test=0.2)
    train, test = cfg.apply_split(df_sorted)
    assert train.height == 8
    assert test.height == 2
