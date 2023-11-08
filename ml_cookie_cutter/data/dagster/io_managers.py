from dagster import ConfigurableIOManager, ConfigurableIOManagerFactory, InputContext, OutputContext, UPathIOManager
import polars as pl
from upath import UPath


class SourceAssetPolarsIOManager(ConfigurableIOManager):
    """Translates between Pandas DataFrames and CSVs on the local filesystem."""

    def handle_output(self, context: OutputContext, obj: pl.DataFrame):
        """This saves the dataframe as a CSV."""
        pass

    def load_input(self, context: InputContext):
        """This reads a dataframe from a CSV."""
        if not context.upstream_output:
            raise ValueError("context.upstream_output is None")

        if not context.upstream_output.metadata:
            raise ValueError("Metadata is not configured")

        asset_path = context.upstream_output.metadata.get("asset_path", None)
        if asset_path is None:
            raise ValueError(f"Asset path not found in metadata for asset {context.asset_key.to_string()}")
        read_kwargs = context.upstream_output.metadata.get("read_kwargs", {})
        return pl.read_csv(str(asset_path), **read_kwargs)


class PolarsParquetIOManager(UPathIOManager):
    extension: str = ".parquet"

    def dump_to_path(self, context: OutputContext, obj: pl.DataFrame, path: UPath):
        with path.open("wb") as file:
            obj.write_parquet(file)

    def load_from_path(self, context: InputContext, path: UPath) -> pl.DataFrame:
        with path.open("rb") as file:
            return pl.read_parquet(file)


class LocalPolarsParquetIOManager(ConfigurableIOManagerFactory["PolarsParquetIOManager"]):
    base_path: str

    def create_io_manager(self, context: InputContext) -> PolarsParquetIOManager:
        base_path = UPath(self.base_path or context.instance.storage_directory())
        return PolarsParquetIOManager(base_path=base_path)
