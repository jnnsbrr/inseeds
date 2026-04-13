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
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            input_dir = sim_path / "input"
            input_dir.mkdir(parents=True)

            with patch.object(prices, "prepare", side_effect=RuntimeError("API unavailable")):
                path = prices.ensure(
                    sim_path,
                    reference_year=2012,
                    years_before=2,
                    use_dummy_on_failure=True,
                )

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

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            input_dir = sim_path / "input"
            input_dir.mkdir(parents=True)

            with patch.object(capital, "prepare", side_effect=RuntimeError("API unavailable")):
                path = capital.ensure(
                    sim_path,
                    reference_year=2012,
                    years_before=2,
                    use_dummy_on_failure=True,
                )

            assert path == capital.get_dummy_path(sim_path)
            assert path.exists()
            assert "_DUMMY" in path.name

            ds = xr.open_dataset(path)
            assert "ncs" in ds.data_vars  # Net Capital Stocks
            assert "gfcf" in ds.data_vars  # Gross Fixed Capital Formation
            assert "cfc" in ds.data_vars  # Consumption of Fixed Capital
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

    def test_ensure_tries_download_when_only_dummy_exists(self):
        """When only dummy data exists, ensure() tries to download real data first."""
        prices = FaoProducerPrices()
        dummy_path = Path("input") / "fao_pft_prices_DUMMY.nc"

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            full_dummy = sim_path / dummy_path
            full_dummy.parent.mkdir(parents=True)
            full_dummy.touch()

            # Mock prepare to fail (simulating API unavailable)
            with patch.object(prices, "prepare", side_effect=RuntimeError("API unavailable")):
                path = prices.ensure(sim_path)

            # Should fall back to existing dummy
            assert path == full_dummy
            assert path.exists()


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


class TestCACountryFaoLoading:
    """Test that CACountry loads FAO data correctly at country level."""

    def test_ca_country_loads_fao_data_with_dummy(self):
        """CACountry loads dummy FAO data once per country."""
        from inseeds.components.farming.ca_country import CACountry
        from inseeds.components.data.fao.dummy import generate_dummy_gpv

        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            # Create dummy FAO data files
            dummy_paths = ensure_dummy_fao_data(sim_path, data_type="both", years=(2010, 2015))

            # Create dummy crop share (GPV) data
            crop_share_path = Path(sim_path) / "input" / "fao_crop_share_DUMMY.nc"
            crop_share_path.parent.mkdir(parents=True, exist_ok=True)
            generate_dummy_gpv(years=(2010, 2015), output_path=crop_share_path)

            # Create minimal mock model with config
            model = MagicMock()
            model.config.sim_path = str(sim_path)

            # Create country instance (skip full super().__init__)
            country = object.__new__(CACountry)
            country.model = model
            country.country_code = "NLD"
            country._cropland_area = 5000.0  # Pre-set cropland area in ha

            # Clear class-level cache to ensure fresh load
            CACountry._fao_prices_ds = None
            CACountry._fao_capital_ds = None
            CACountry._fao_crop_share_ds = None

            # Pre-load the datasets into class cache to avoid API calls
            CACountry._fao_prices_ds = xr.open_dataset(dummy_paths["producer_prices"])
            CACountry._fao_capital_ds = xr.open_dataset(dummy_paths["capital_stock"])
            CACountry._fao_crop_share_ds = xr.open_dataset(crop_share_path)

            # Call the extraction methods directly (bypassing ensure())
            country._extract_capital_parameters("NLD")
            country._extract_prices("NLD")

            # Verify country has FAO-derived attributes (internal attributes)
            assert hasattr(country, "_depreciation_rate")
            assert hasattr(country, "_initial_capital_per_ha")
            assert hasattr(country, "_pft_prices")
            assert country._depreciation_rate > 0
            assert country._initial_capital_per_ha > 0
            assert country._pft_prices is not None

            # Clean up
            CACountry._fao_prices_ds.close()
            CACountry._fao_capital_ds.close()
            CACountry._fao_crop_share_ds.close()
            CACountry._fao_prices_ds = None
            CACountry._fao_capital_ds = None
            CACountry._fao_crop_share_ds = None
