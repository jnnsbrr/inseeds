"""Tests for FAO data module (FaoProducerPrices, FaoCapitalStock, ensure, dummy fallback).

These tests cover the FAO API integration path including:
- ensure() with dummy fallback when API fails
- Path resolution (real vs dummy files)
- ConservationAgricultureFarmer loading FAO data
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from inseeds.components.data.fao import (
    FaoProducerPrices,
    FaoCapitalStock,
    ensure_dummy_fao_data,
)


class TestFaoPaths:
    """Test path resolution for FAO datasets."""

    def test_get_path_and_get_dummy_path(self):
        """Real and dummy paths use correct naming."""
        prices = FaoProducerPrices()
        capital = FaoCapitalStock()
        sim_path = Path("/tmp/sim")

        assert prices.get_path(sim_path) == sim_path / "input" / "fao_pft_prices.nc"
        assert prices.get_dummy_path(sim_path) == sim_path / "input" / "fao_pft_prices_DUMMY.nc"

        assert capital.get_path(sim_path) == sim_path / "input" / "fao_capital_stock.nc"
        assert capital.get_dummy_path(sim_path) == sim_path / "input" / "fao_capital_stock_DUMMY.nc"


class TestFaoEnsureDummyFallback:
    """Test ensure() creates dummy data when API fails."""

    def test_producer_prices_ensure_creates_dummy_when_api_fails(self):
        """FaoProducerPrices.ensure() creates _DUMMY.nc when prepare raises."""
        prices = FaoProducerPrices()
        years = (2010, 2012)

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            input_dir = sim_path / "input"
            input_dir.mkdir(parents=True)

            with patch.object(prices, "prepare", side_effect=RuntimeError("API unavailable")):
                path = prices.ensure(sim_path, years=years, use_dummy_on_failure=True)

            assert path == prices.get_dummy_path(sim_path)
            assert path.exists()
            assert "_DUMMY" in path.name

            ds = xr.open_dataset(path)
            assert "5532" in ds.data_vars  # Producer price element code
            assert "area_code" in ds.dims
            assert "time" in ds.dims or "npft" in ds.dims
            ds.close()

    def test_capital_stock_ensure_creates_dummy_when_api_fails(self):
        """FaoCapitalStock.ensure() creates _DUMMY.nc when prepare raises."""
        capital = FaoCapitalStock()
        years = (2010, 2012)

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            input_dir = sim_path / "input"
            input_dir.mkdir(parents=True)

            with patch.object(capital, "prepare", side_effect=RuntimeError("API unavailable")):
                path = capital.ensure(sim_path, years=years, use_dummy_on_failure=True)

            assert path == capital.get_dummy_path(sim_path)
            assert path.exists()
            assert "_DUMMY" in path.name

            ds = xr.open_dataset(path)
            assert "6186" in ds.data_vars  # Net Capital Stocks
            assert "depreciation_rate" in ds.data_vars
            assert "investment_rate" in ds.data_vars
            assert "area_code" in ds.dims
            ds.close()


class TestFaoEnsureUsesExistingFiles:
    """Test ensure() uses existing real or dummy files."""

    def test_ensure_uses_existing_real_file(self):
        """When real data exists, ensure() returns it without calling prepare."""
        prices = FaoProducerPrices()
        real_path = Path("input") / "fao_pft_prices.nc"

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            full_real = sim_path / real_path
            full_real.parent.mkdir(parents=True)
            full_real.touch()

            with patch.object(prices, "prepare") as mock_prepare:
                path = prices.ensure(sim_path)

            mock_prepare.assert_not_called()
            assert path == full_real

    def test_ensure_uses_existing_dummy_file(self):
        """When dummy data exists, ensure() returns it without calling prepare."""
        prices = FaoProducerPrices()
        dummy_path = Path("input") / "fao_pft_prices_DUMMY.nc"

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            full_dummy = sim_path / dummy_path
            full_dummy.parent.mkdir(parents=True)
            full_dummy.touch()

            with patch.object(prices, "prepare") as mock_prepare:
                path = prices.ensure(sim_path)

            mock_prepare.assert_not_called()
            assert path == full_dummy


class TestFaoIsAvailable:
    """Test is_available() and is_dummy() helpers."""

    def test_is_available_real(self):
        """is_available True when real file exists."""
        prices = FaoProducerPrices()
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            (sim_path / "input").mkdir(parents=True)
            (sim_path / "input" / "fao_pft_prices.nc").touch()
            assert prices.is_available(sim_path) is True
            assert prices.is_dummy(sim_path) is False

    def test_is_available_dummy(self):
        """is_available True when only dummy file exists."""
        prices = FaoProducerPrices()
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            (sim_path / "input").mkdir(parents=True)
            (sim_path / "input" / "fao_pft_prices_DUMMY.nc").touch()
            assert prices.is_available(sim_path) is True
            assert prices.is_dummy(sim_path) is True

    def test_is_available_false(self):
        """is_available False when no file exists."""
        prices = FaoProducerPrices()
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            assert prices.is_available(sim_path) is False
            assert prices.is_dummy(sim_path) is False


class TestEnsureDummyFaoData:
    """Test ensure_dummy_fao_data convenience function."""

    def test_ensure_dummy_creates_both_files(self):
        """ensure_dummy_fao_data creates both producer prices and capital stock."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            paths = ensure_dummy_fao_data(sim_path, data_type="both", years=(2010, 2012))

            assert "producer_prices" in paths
            assert "capital_stock" in paths
            assert paths["producer_prices"].exists()
            assert paths["capital_stock"].exists()
            assert "_DUMMY" in paths["producer_prices"].name
            assert "_DUMMY" in paths["capital_stock"].name


class TestConservationAgricultureFarmerFaoLoading:
    """Test that ConservationAgricultureFarmer loads FAO data correctly."""

    def test_ca_farmer_loads_fao_data_with_dummy(self):
        """ConservationAgricultureFarmer._load_fao_* loads dummy FAO data."""
        from inseeds.components.farming import ConservationAgricultureFarmer

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            ensure_dummy_fao_data(sim_path, data_type="both", years=(2010, 2015))

            # Create minimal mock model with config
            model = MagicMock()
            model.config.sim_path = str(sim_path)
            model.config.coupled_config.farm_economics = {"n_survival_years": 2}
            model.config.coupled_config.practice_costs = {
                "tillage": {"transition": 70.0, "direct": -50.0},
                "cover_crop": {"transition": 25.0, "direct": 75.0},
                "residue_on_field": {"transition": 10.0, "direct": 50.0},
            }

            # Create mock cell with country_code
            cell = MagicMock()
            cell.country_code = "NLD"

            # Create farmer instance (skip full super().__init__ to avoid complex deps)
            farmer = object.__new__(ConservationAgricultureFarmer)
            farmer.model = model
            farmer._cell = cell

            # Call the FAO loading methods directly
            farmer._load_fao_pft_prices()
            farmer._load_fao_capital_stock()

            assert farmer.fao_pft_prices is not None
            assert farmer.fao_capital_stock is not None
            assert "5532" in farmer.fao_pft_prices.data_vars
            assert "6186" in farmer.fao_capital_stock.data_vars
            assert "depreciation_rate" in farmer.fao_capital_stock.data_vars
