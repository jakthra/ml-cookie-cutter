from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import pandas as pd
import polars as pl
from pydantic import BaseModel
import yaml
import ml_cookie_cutter
from ml_cookie_cutter.data.factories import AbstractLoaderFactory

RAW_DATASET_DIRECTORY = Path(ml_cookie_cutter.__file__).parents[1] / "data" / "raw"


class DatasetDirectories:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self):
        return iter(self._get_datasets())

    def _get_datasets(self):
        for dataset_path in self.path.iterdir():
            yield dataset_path

    def get_by_name(self, name: str):
        for dataset_path in self:
            if dataset_path.name == name:
                return dataset_path
        raise ValueError(f"Dataset {name} not found")


raw_dataset_directories = DatasetDirectories(RAW_DATASET_DIRECTORY)

SINKS = Literal["parquet"]


class ParquetSink:
    containing_path: Path = Path("Parquet")

    def to_parquet(self, df: Union[pl.DataFrame, pd.DataFrame], target_path: Path):
        target_path = target_path.parent / self.containing_path / target_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path = target_path.with_suffix(".parquet")
        if isinstance(df, pl.DataFrame):
            df.write_parquet(parquet_path, compression="snappy")
        else:
            df.to_parquet(parquet_path, compression="snappy")  # type: ignore
        return parquet_path


class RawDatasetConfig(BaseModel):
    name: str
    sink: SINKS = "parquet"
    loader: Literal["DefaultCSVLoader"] = "DefaultCSVLoader"
    limit: Optional[int] = None
    asset: str
    loader_options: Optional[Dict[str, Any]] = None


class RawDataset:
    root_path: Path

    def __init__(self, name: str) -> None:
        self.name = name
        self.root_path = raw_dataset_directories.get_by_name(name)
        self.config = self._load_config()
        self._sink = ParquetSink()  # TODO: select sink from config

    @property
    def content(self):
        return list(self.root_path.iterdir())

    def _load_config(self) -> RawDatasetConfig:
        config_path = self.root_path / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return RawDatasetConfig.parse_obj(yaml.safe_load(f))

        raise ValueError(f"Config file {config_path} not found")

    def standardize(self) -> Path:
        # Get loader
        loader = AbstractLoaderFactory.get_loader(self)

        # Load data into memory
        # TODO: add limit
        # TODO: add stream to sink option
        df = loader.to_dataframe(self.root_path / self.config.asset)

        return self._sink.to_parquet(df, self.root_path / self.config.asset)
