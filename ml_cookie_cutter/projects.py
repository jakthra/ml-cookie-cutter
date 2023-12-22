from pathlib import Path
from typing import Optional

from ml_cookie_cutter.data.constants import DATALAKE_DIRECTORY, DATASET_PREFIX, DATASET_PREFIX_TIMESERIES


class Project:
    def __init__(self, name: str, description: Optional[str] = None) -> None:
        self.name = name
        self.description = description

    def get_datasets(self) -> list[Path]:
        return list((DATALAKE_DIRECTORY / self.name / DATASET_PREFIX).rglob("*.parquet"))

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.name == __value
        return super().__eq__(__value)

    def __repr__(self) -> str:
        return f"Project(name={self.name})"

    def __str__(self) -> str:
        return self.name


projects = [Project(name=DATASET_PREFIX_TIMESERIES, description="Electricity consumption data from kaggle")]
