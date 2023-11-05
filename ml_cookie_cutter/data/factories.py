from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_cookie_cutter.data.base import RawDataset
    from ml_cookie_cutter.data.loaders import Loader

from ml_cookie_cutter.data.loaders import DefaultCSVLoader


class AbstractLoaderFactory:
    @staticmethod
    def get_loader(raw_dataset: RawDataset) -> Loader[Any]:
        if raw_dataset.config.loader == "DefaultCSVLoader":
            return DefaultCSVLoader(raw_dataset.config.loader_options if raw_dataset.config.loader_options else {})
        raise ValueError(f"Loader {raw_dataset.config.loader} not found")
