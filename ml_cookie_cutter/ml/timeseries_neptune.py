import os
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import neptune
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from ml_cookie_cutter.orchestration.timeseries_example import TrainTestConfig


def run():
    # Logging
    run = neptune.init_run(
        project="jakthra/ml-cookiecutter",
        api_token=os.environ["NEPTUNE_TOKEN"],
    )

    # Configuration
    target_column = "Global_active_power"
    features = ["weekday", "day_of_month", "day_of_year", "hour", "minute"]
    limit_dataset: Optional[int] = None
    run["parameters"] = {
        "target_column": target_column,
        "features": features,
        "limit_dataset": limit_dataset,
        "model": "RandomForestRegressor",
        "dataset": "timeseries_average_per_day.parquet",
    }

    # Load dataset
    dataset = Path("/workspaces/ml-cookie-cutter/data/datalake/timeseries/dataset/timeseries_average_per_day.parquet")
    run["dataset"].track_files(str(dataset))
    df = pl.read_parquet(dataset)

    # Split dataset
    train_df, test_df = TrainTestConfig().apply_split(df)

    X_train = train_df[features]
    y_train = train_df[target_column]
    X_test = test_df[features]
    y_test = test_df[target_column]

    if limit_dataset is not None:
        X_train = X_train.limit(limit_dataset)
        y_train = y_train.limit(limit_dataset)
        X_test = X_test.limit(limit_dataset)
        y_test = y_test.limit(limit_dataset)

    rfr = RandomForestRegressor(verbose=True, n_estimators=20, max_depth=10)
    rfr.fit(X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1))

    # Evaluate
    y_pred = rfr.predict(X_test.to_numpy())
    mae = mean_absolute_error(y_test.to_numpy(), y_pred)
    run["metrics/mae"] = mae
    print(f"MAE: {mae:.2f}")

    # Save predictions
    plt.figure()
    plt.plot(y_test.to_numpy().reshape(-1, 1), label="expected")
    plt.plot(y_pred, label="predictions")
    plt.savefig("predictions.png")
    run["predictions"].upload("predictions.png")

    # Save model version
    random_forest_dump = Path("random_forest.joblib")
    joblib.dump(rfr, str(random_forest_dump))
    model_version = neptune.init_model_version(
        model="MLCOOK-TSRANDF",
        project="jakthra/ml-cookiecutter",
        api_token=os.environ["NEPTUNE_TOKEN"],  # your credentials
    )
    model_version["model"].upload(str(random_forest_dump))
    model_version["dataset"].track_files(str(dataset))
    model_version["test/mae"] = mae

    run.stop()


if __name__ == "__main__":
    run()
