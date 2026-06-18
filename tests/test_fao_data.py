"""Tests for FAO data module (FaoProducerPrices, FaoCapitalStock).

These tests cover the FAO API integration path including:
- Path resolution
- ensure() using existing files
- is_available() helper
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from inseeds.components.exogenous.faostat import (
    FaoProducerPrices,
    FaoCapitalStock,
)


class TestFaoPaths:
    """Test path resolution for FAO datasets."""

    def test_get_path(self):
        """Real paths use correct naming."""
        prices = FaoProducerPrices()
        capital = FaoCapitalStock()
        sim_path = Path("/tmp/sim")

        assert prices.get_path(sim_path) == sim_path / "input" / "fao_pft_prices.nc"
        assert capital.get_path(sim_path) == sim_path / "input" / "fao_capital_stock.nc"


class TestFaoEnsureUsesExistingFiles:
    """Test ensure() uses existing real files."""

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


class TestFaoIsAvailable:
    """Test is_available() helper."""

    def test_is_available_real(self):
        """is_available True when real file exists."""
        prices = FaoProducerPrices()
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            (sim_path / "input").mkdir(parents=True)
            (sim_path / "input" / "fao_pft_prices.nc").touch()
            assert prices.is_available(sim_path) is True

    def test_is_available_false(self):
        """is_available False when no file exists."""
        prices = FaoProducerPrices()
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            assert prices.is_available(sim_path) is False


class TestFaoEnsureRaisesOnApiFailure:
    """Test ensure() raises RuntimeError when API fails and no data exists."""

    def test_ensure_raises_when_api_fails(self):
        """ensure() should raise RuntimeError when API fails."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            (sim_path / "input").mkdir(parents=True)
            
            with patch.object(prices, "prepare", side_effect=RuntimeError("API unavailable")):
                with pytest.raises(RuntimeError, match="FAO API failed"):
                    prices.ensure(sim_path, reference_year=2020)


# =============================================================================
# CACountry Country Stats Cache Tests
# =============================================================================

class TestCACountryStatsCache:
    """Tests for CACountry country-level statistics caching."""

    def test_compute_country_stats_method_exists(self):
        """CACountry should compute management performance stats."""
        from inseeds.components.farming.ca_country import CACountry
        assert hasattr(CACountry, "compute_management_performance")

    def test_country_stats_cache_structure(self):
        """Country stats cache should store a performance store object."""
        from inseeds.components.farming.ca_management import (
            ManagementBundle,
            RegionManagementPerformance,
            RegionManagementPerformanceStore,
        )

        store = RegionManagementPerformanceStore(
            year=2020,
            total_farmers=8,
            bundle_counts={ManagementBundle.notill: 5.0, ManagementBundle.conservation: 3.0},
        )
        store._bundles[ManagementBundle.notill] = RegionManagementPerformance(
            yield_sum=25.0,
            soilc_sum=50.0,
            moisture_sum=2.5,
            yield_trend_sum=0.05,
            soilc_trend_sum=0.10,
            moisture_trend_sum=0.0,
            count=5,
        )

        assert store.year == 2020
        assert store.total_farmers == 8
        assert store.bundle_counts[ManagementBundle.notill] == 5.0

    def test_bundle_performance_has_required_fields(self):
        """Per-bundle performance should expose soilc-based averages."""
        from inseeds.components.farming.ca_management import (
            RegionManagementPerformance,
        )

        perf = RegionManagementPerformance(
            yield_sum=25.0,
            soilc_sum=50.0,
            moisture_sum=2.5,
            yield_trend_sum=0.05,
            soilc_trend_sum=0.10,
            moisture_trend_sum=0.0,
            count=5,
        )

        assert perf.mean_yield == 5.0
        assert perf.mean_soilc == 10.0
        assert perf.mean_moisture == 0.5
        assert perf.mean_yield_trend == 0.01
        assert perf.mean_soilc_trend == 0.02
        assert perf.count == 5

    def test_update_method_exists(self):
        """CACountry should have update method that computes management performance."""
        from inseeds.components.farming.ca_country import CACountry
        assert hasattr(CACountry, "update")
        assert hasattr(CACountry, "compute_management_performance")
