from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel

from ml_cookie_cutter.data.constants import RAW_DATASET_DIRECTORY


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


class RawDatasetConfig(BaseModel):
    name: str
    asset: str


raw_dataset_directories = DatasetDirectories(RAW_DATASET_DIRECTORY)


class RawDataset:
    root_path: Path

    def __init__(self, name: str) -> None:
        self.name = name
        self.root_path = raw_dataset_directories.get_by_name(name)
        self.config = self._load_config()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.root_path,
            "content": [str(path) for path in self.content],
            "asset_path": self.asset_path,
            "format": self.asset_path.suffix,
        }

    @property
    def asset_path(self):
        return self.root_path / self.config.asset

    def _load_config(self) -> RawDatasetConfig:
        config_path = self.root_path / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return RawDatasetConfig.parse_obj(yaml.safe_load(f))

        raise ValueError(f"Config file {config_path} not found")

    @property
    def content(self):
        return list(self.root_path.iterdir())
