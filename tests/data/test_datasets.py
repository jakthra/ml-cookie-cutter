import ml_cookie_cutter
from ml_cookie_cutter.data import RawDataset
from pathlib import Path
import pytest


class TestHouseHoldPowerConsumptionDataset:
    @pytest.fixture(scope="class")
    def raw_dataset(self):
        return RawDataset("timeseries-example")

    def test_base_api(self, raw_dataset: RawDataset):
        assert (
            raw_dataset.root_path == Path(ml_cookie_cutter.__file__).parents[1] / "data" / "raw" / "timeseries-example"
        )
        raw_dataset_content_name = [path.name for path in raw_dataset.content]
        assert "household_power_consumption.txt" in raw_dataset_content_name
        assert "README.md" in raw_dataset_content_name

    def test_standardization(self, raw_dataset: RawDataset):
        sink_path = raw_dataset.standardize()
        assert (
            sink_path
            == Path(ml_cookie_cutter.__file__).parents[1]
            / "data"
            / "raw"
            / "timeseries-example"
            / "Parquet"
            / "household_power_consumption.parquet"
        )
        assert sink_path.exists()
        assert sink_path.is_file()
