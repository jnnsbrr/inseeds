"""Tests for FAO tiered fallback mechanism.

Tests cover the generic get_value_with_fallback() function that implements:
- Tier A: Same country with expanding time window
- Tier B: Mean from neighbouring countries
- Tier C: Global mean across all countries
"""

import numpy as np
import pytest
import xarray as xr

from inseeds.components.data.fao.base import (
    FallbackResult,
    get_value_with_fallback,
)


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""

    def test_fallback_result_has_required_fields(self):
        """FallbackResult should have value, tier, and detail fields."""
        result = FallbackResult(value=0.5, tier="country", detail="window=2015-2020")
        
        assert result.value == 0.5
        assert result.tier == "country"
        assert result.detail == "window=2015-2020"

    def test_fallback_result_tier_values(self):
        """FallbackResult tier should be one of country, neighbours, global."""
        for tier in ["country", "neighbours", "global"]:
            result = FallbackResult(value=1.0, tier=tier, detail="test")
            assert result.tier == tier


class TestTierACountryFallback:
    """Tests for Tier A: same country with expanding time window."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataArray with time and area_code dimensions."""
        return xr.DataArray(
            data=[[0.05, 0.06], [0.04, 0.05], [0.03, 0.04]],
            dims=["time", "area_code"],
            coords={
                "time": [2018, 2019, 2020],
                "area_code": ["NLD", "DEU"],
            },
        )

    def test_tier_a_uses_recent_window(self, sample_data):
        """Should use data from recent years within avg_years window."""
        result = get_value_with_fallback(
            sample_data,
            "NLD",
            year=2020,
            avg_years=5,
        )
        
        assert result.tier == "country"
        assert "window=" in result.detail
        # Mean of 0.05, 0.04, 0.03 = 0.04
        assert result.value == pytest.approx(0.04, rel=0.01)

    def test_tier_a_expands_window_when_needed(self):
        """Should expand window backward when recent years have no data."""
        # Data only in older years
        data = xr.DataArray(
            data=[[0.05, 0.06], [np.nan, np.nan], [np.nan, np.nan]],
            dims=["time", "area_code"],
            coords={
                "time": [2010, 2019, 2020],
                "area_code": ["NLD", "DEU"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            avg_years=2,
            max_lookback=15,
        )
        
        assert result.tier == "country"
        assert result.value == pytest.approx(0.05)

    def test_tier_a_uses_all_years_as_last_resort(self):
        """Should use all available years if window expansion fails."""
        data = xr.DataArray(
            data=[[0.05], [np.nan], [np.nan]],
            dims=["time", "area_code"],
            coords={
                "time": [2000, 2019, 2020],
                "area_code": ["NLD"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            avg_years=2,
            max_lookback=10,  # Won't reach 2000
        )
        
        assert result.tier == "country"
        assert result.value == pytest.approx(0.05)

    def test_tier_a_handles_no_time_dimension(self):
        """Should work with data that has no time dimension."""
        data = xr.DataArray(
            data=[0.05, 0.06],
            dims=["area_code"],
            coords={"area_code": ["NLD", "DEU"]},
        )
        
        result = get_value_with_fallback(data, "NLD")
        
        assert result.tier == "country"
        assert result.value == pytest.approx(0.05)

    def test_tier_a_aggregator_mean(self, sample_data):
        """Should compute mean when aggregator='mean'."""
        result = get_value_with_fallback(
            sample_data,
            "NLD",
            year=2020,
            aggregator="mean",
        )
        
        # Mean of 0.05, 0.04, 0.03
        assert result.value == pytest.approx(0.04, rel=0.01)

    def test_tier_a_aggregator_last(self, sample_data):
        """Should use last value when aggregator='last'."""
        result = get_value_with_fallback(
            sample_data,
            "NLD",
            year=2020,
            aggregator="last",
        )
        
        # Last value is 0.03 (year 2020)
        assert result.value == pytest.approx(0.03)

    def test_tier_a_aggregator_sum(self, sample_data):
        """Should compute sum when aggregator='sum'."""
        result = get_value_with_fallback(
            sample_data,
            "NLD",
            year=2020,
            aggregator="sum",
        )
        
        # Sum of 0.05 + 0.04 + 0.03 = 0.12
        assert result.value == pytest.approx(0.12, rel=0.01)


class TestTierBNeighbourFallback:
    """Tests for Tier B: mean from neighbouring countries."""

    def test_tier_b_uses_neighbours_when_country_missing(self):
        """Should use neighbour mean when target country has no data."""
        data = xr.DataArray(
            data=[[np.nan, 0.05, 0.06]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "BEL"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["DEU", "BEL"],
        )
        
        assert result.tier == "neighbours"
        assert "neighbours=" in result.detail
        # Mean of 0.05 and 0.06 = 0.055
        assert result.value == pytest.approx(0.055)

    def test_tier_b_uses_neighbours_when_country_not_in_dataset(self):
        """Should use neighbours when country not in area_code at all."""
        data = xr.DataArray(
            data=[[0.05, 0.06]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["DEU", "BEL"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",  # Not in dataset
            year=2020,
            neighbour_codes=["DEU", "BEL"],
        )
        
        assert result.tier == "neighbours"
        assert result.value == pytest.approx(0.055)

    def test_tier_b_skips_missing_neighbours(self):
        """Should skip neighbours that are not in the dataset."""
        data = xr.DataArray(
            data=[[np.nan, 0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["DEU", "FRA", "BEL"],  # FRA, BEL not in data
        )
        
        assert result.tier == "neighbours"
        assert result.value == pytest.approx(0.05)

    def test_tier_b_expands_window_for_neighbours(self):
        """Should expand time window for neighbours too."""
        data = xr.DataArray(
            data=[[np.nan, np.nan], [np.nan, 0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2010, 2020],
                "area_code": ["NLD", "DEU"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            avg_years=2,
            max_lookback=15,
            neighbour_codes=["DEU"],
        )
        
        assert result.tier == "neighbours"
        assert result.value == pytest.approx(0.05)

    def test_tier_b_skipped_when_no_neighbour_codes(self):
        """Should skip to Tier C when neighbour_codes is None or empty."""
        data = xr.DataArray(
            data=[[np.nan, 0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU"],
            },
        )
        
        # No neighbour_codes provided
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=None,
        )
        
        # Should fall through to global
        assert result.tier == "global"


class TestTierCGlobalFallback:
    """Tests for Tier C: global mean across all countries."""

    def test_tier_c_uses_global_mean(self):
        """Should use global mean when country and neighbours unavailable."""
        data = xr.DataArray(
            data=[[np.nan, 0.05, 0.06, 0.07]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "FRA", "ESP"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["BEL"],  # BEL not in data
        )
        
        assert result.tier == "global"
        assert "n_countries=" in result.detail
        # Mean of 0.05, 0.06, 0.07 = 0.06
        assert result.value == pytest.approx(0.06)

    def test_tier_c_ignores_nan_in_global_mean(self):
        """Should ignore NaN values when computing global mean."""
        data = xr.DataArray(
            data=[[np.nan, 0.05, np.nan, 0.07]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "FRA", "ESP"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=[],
        )
        
        assert result.tier == "global"
        # Mean of 0.05, 0.07 = 0.06
        assert result.value == pytest.approx(0.06)


class TestFallbackRaisesOnNoData:
    """Tests for error handling when no data available."""

    def test_raises_when_all_nan(self):
        """Should raise ValueError when all data is NaN."""
        data = xr.DataArray(
            data=[[np.nan, np.nan]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU"],
            },
        )
        
        with pytest.raises(ValueError, match="No valid FAO data found"):
            get_value_with_fallback(
                data,
                "NLD",
                year=2020,
                neighbour_codes=["DEU"],
            )

    def test_raises_when_country_not_found_and_no_global(self):
        """Should raise when country not found and global fails."""
        data = xr.DataArray(
            data=[[np.nan]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["DEU"],
            },
        )
        
        with pytest.raises(ValueError, match="No valid FAO data found"):
            get_value_with_fallback(
                data,
                "NLD",  # Not in dataset
                year=2020,
                neighbour_codes=[],
            )


class TestFallbackYearSelection:
    """Tests for year selection behavior."""

    def test_uses_most_recent_year_when_none(self):
        """Should use most recent year when year=None."""
        data = xr.DataArray(
            data=[[0.03], [0.04], [0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2018, 2019, 2020],
                "area_code": ["NLD"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=None,
            avg_years=1,
        )
        
        # Should use 2020 data
        assert result.value == pytest.approx(0.05)

    def test_respects_target_year(self):
        """Should respect the specified target year."""
        data = xr.DataArray(
            data=[[0.03], [0.04], [0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2018, 2019, 2020],
                "area_code": ["NLD"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2019,
            avg_years=1,
        )
        
        # Should use 2019 data
        assert result.value == pytest.approx(0.04)


class TestFallbackIntegration:
    """Integration tests for the full fallback chain."""

    def test_full_fallback_chain(self):
        """Should try all tiers in order: country -> neighbours -> global."""
        # Country NLD has no data, neighbour DEU has no data, only FRA has data
        data = xr.DataArray(
            data=[[np.nan, np.nan, 0.05]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "FRA"],
            },
        )
        
        # DEU is a neighbour but has no data
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["DEU"],
        )
        
        # Should fall through to global (FRA has data)
        assert result.tier == "global"
        assert result.value == pytest.approx(0.05)

    def test_prefers_country_over_neighbours(self):
        """Should prefer country data over neighbours even if neighbours have more data."""
        data = xr.DataArray(
            data=[[0.03, 0.05, 0.06]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "BEL"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["DEU", "BEL"],
        )
        
        assert result.tier == "country"
        assert result.value == pytest.approx(0.03)

    def test_prefers_neighbours_over_global(self):
        """Should prefer neighbour data over global mean."""
        data = xr.DataArray(
            data=[[np.nan, 0.05, 0.10]],
            dims=["time", "area_code"],
            coords={
                "time": [2020],
                "area_code": ["NLD", "DEU", "FRA"],
            },
        )
        
        result = get_value_with_fallback(
            data,
            "NLD",
            year=2020,
            neighbour_codes=["DEU"],  # Only DEU is neighbour
        )
        
        assert result.tier == "neighbours"
        assert result.value == pytest.approx(0.05)
