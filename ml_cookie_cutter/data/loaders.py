import abc
from pathlib import Path
from typing import Any, Dict, Generic, TypeVar, Union
import polars as pl
import pandas as pd

L = TypeVar("L")


class Loader(abc.ABC, Generic[L]):
    def to_dataframe(self, target_file: Path) -> Union[pl.DataFrame, pd.DataFrame]:
        raise NotImplementedError()


class DefaultCSVLoader(Loader["DefaultCSVLoader"]):
    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    def to_dataframe(self, target_file: Path) -> pl.DataFrame:
        return pl.read_csv(target_file, **self.options)
