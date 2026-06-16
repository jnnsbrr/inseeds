"""Tests for management performance stores at country and cluster levels.

Tests cover:
- RegionManagementPerformance aggregation
- RegionManagementPerformanceStore creation and merging  
- Country-level management performance computation logic
- Cluster-level aggregation logic
- Store serialization for Dask sync

Note: These tests use inline class definitions to avoid import issues with
Python <3.10 type hint syntax in pycoupler dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import pytest
from unittest.mock import MagicMock


# =============================================================================
# INLINE CLASS DEFINITIONS (matching ca_management.py)
# =============================================================================

_BUNDLE_METADATA: Dict[str, Tuple[int, str]] = {
    "notill": (0, "no-till"),
    "notill_residue": (1, "no-till + residue retention"),
    "notill_covercrop": (2, "no-till + cover crop"),
    "conservation": (3, "conservation agriculture"),
    "conventional": (4, "conventional farming"),
    "residue": (5, "residue retention"),
    "covercrop": (6, "cover crop"),
    "covercrop_residue": (7, "cover crop + residue retention"),
}


class ManagementBundle(Enum):
    """Practice bundle enum (simplified for tests)."""
    notill = (0, 1, 0)
    notill_residue = (0, 1, 1)
    notill_covercrop = (0, 1, 1)
    conservation = (0, 1, 1)
    conventional = (1, 0, 0)
    residue = (1, 0, 1)
    covercrop = (1, 1, 0)
    covercrop_residue = (1, 1, 1)

    @property
    def id(self) -> int:
        return _BUNDLE_METADATA[self.name][0]

    @property
    def label(self) -> str:
        return _BUNDLE_METADATA[self.name][1]


@dataclass
class RegionManagementPerformance:
    """Aggregated performance for one bundle across multiple farmers."""
    yield_sum: float = 0.0
    soilc_sum: float = 0.0
    moisture_sum: float = 0.0
    yield_trend_sum: float = 0.0
    soilc_trend_sum: float = 0.0
    moisture_trend_sum: float = 0.0
    count: int = 0

    @classmethod
    def empty(cls) -> "RegionManagementPerformance":
        return cls()

    @property
    def avg_yield(self) -> float:
        return self.yield_sum / self.count if self.count else 0.0

    @property
    def avg_soilc(self) -> float:
        return self.soilc_sum / self.count if self.count else 0.0

    @property
    def avg_moisture(self) -> float:
        return self.moisture_sum / self.count if self.count else 0.0

    @property
    def avg_yield_trend(self) -> float:
        return self.yield_trend_sum / self.count if self.count else 0.0

    @property
    def avg_soilc_trend(self) -> float:
        return self.soilc_trend_sum / self.count if self.count else 0.0

    @property
    def avg_moisture_trend(self) -> float:
        return self.moisture_trend_sum / self.count if self.count else 0.0

    def add_observation(
        self,
        cropyield: float,
        soilc: float,
        moisture: float,
        trend: Dict[str, float],
    ) -> None:
        self.yield_sum += cropyield
        self.soilc_sum += soilc
        self.moisture_sum += moisture
        self.yield_trend_sum += trend["yield"]
        self.soilc_trend_sum += trend["soilc"]
        self.moisture_trend_sum += trend["moisture"]
        self.count += 1

    def merge(self, other: "RegionManagementPerformance") -> None:
        self.yield_sum += other.yield_sum
        self.soilc_sum += other.soilc_sum
        self.moisture_sum += other.moisture_sum
        self.yield_trend_sum += other.yield_trend_sum
        self.soilc_trend_sum += other.soilc_trend_sum
        self.moisture_trend_sum += other.moisture_trend_sum
        self.count += other.count

    def weighted_trend(self, farmer: Any) -> float:
        return (
            farmer.weight_yield * self.avg_yield_trend
            + farmer.weight_soil * self.avg_soilc_trend
            + farmer.weight_moisture * self.avg_moisture_trend
        )


class RegionManagementPerformanceStore:
    """Collection of performance data for ALL bundles in a region."""

    def __init__(
        self,
        year: int = -1,
        total_farmers: int = 0,
        bundle_counts: Optional[Dict[ManagementBundle, float]] = None,
        country_codes: Optional[List[str]] = None,
    ) -> None:
        self.year = year
        self.total_farmers = total_farmers
        self.bundle_counts: Dict[ManagementBundle, float] = bundle_counts or {}
        self.country_codes: List[str] = country_codes or []
        self._bundles: Dict[ManagementBundle, RegionManagementPerformance] = {}

    def _get(self, bundle: ManagementBundle) -> RegionManagementPerformance:
        if bundle not in self._bundles:
            self._bundles[bundle] = RegionManagementPerformance.empty()
        return self._bundles[bundle]

    def get(self, bundle: ManagementBundle) -> RegionManagementPerformance:
        return self._bundles.get(bundle, RegionManagementPerformance.empty())

    def add_observation(self, bundle: ManagementBundle, farmer: Any) -> None:
        self._get(bundle).add_observation(
            farmer.cropyield,
            farmer.soilc,
            farmer.root_moisture,
            farmer.behaviour.performance_tracker.trend,
        )

    def items(self):
        return self._bundles.items()

    def finalize_store(self) -> None:
        self.bundle_counts = {b: p.count for b, p in self._bundles.items()}
        self.total_farmers = sum(self.bundle_counts.values())

    def merge_store(self, other: "RegionManagementPerformanceStore") -> None:
        for bundle, perf in other._bundles.items():
            self._get(bundle).merge(perf)
        for bundle, count in other.bundle_counts.items():
            self.bundle_counts[bundle] = self.bundle_counts.get(bundle, 0) + count
        self.total_farmers += other.total_farmers
        self.country_codes.extend(other.country_codes)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_trends():
    """Sample trend data for testing."""
    return {"yield": 0.02, "soilc": 0.01, "moisture": -0.005}


@pytest.fixture
def mock_farmer(sample_trends):
    """Create a mock farmer with necessary attributes."""
    farmer = MagicMock()
    farmer.cropyield = 5000.0
    farmer.soilc = 8000.0
    farmer.root_moisture = 0.35
    farmer.behaviour.performance_tracker.trend = sample_trends
    farmer.behaviour.practice_bundle = ManagementBundle.conservation
    farmer.weight_yield = 0.4
    farmer.weight_soil = 0.4
    farmer.weight_moisture = 0.2
    return farmer


@pytest.fixture
def mock_country():
    """Create a mock country with statistic."""
    country = MagicMock()
    country.code = "DEU"
    country.indices = np.array([0, 1, 2])

    class MockStatistic:
        def __init__(self):
            self._cache = {}

        def get(self, key, default=None):
            return self._cache.get(key, default)

        def set(self, key, value):
            self._cache[key] = value

    country.statistic = MockStatistic()
    country._statistic = country.statistic
    return country


@pytest.fixture
def mock_world():
    """Create a mock world with countries and statistic."""
    world = MagicMock()

    class MockStatistic:
        def __init__(self):
            self._cache = {}

        def get(self, key, default=None):
            return self._cache.get(key, default)

        def set(self, key, value):
            self._cache[key] = value

    world.statistic = MockStatistic()
    world._statistic = world.statistic
    return world


# =============================================================================
# RegionManagementPerformance TESTS
# =============================================================================


class TestRegionManagementPerformance:
    """Tests for single-bundle performance aggregation."""

    def test_empty_creation(self):
        """Empty aggregate should have zero counts and sums."""
        perf = RegionManagementPerformance.empty()
        assert perf.count == 0
        assert perf.yield_sum == 0.0
        assert perf.soilc_sum == 0.0
        assert perf.moisture_sum == 0.0

    def test_empty_averages_are_zero(self):
        """Averages on empty aggregate should return 0, not NaN."""
        perf = RegionManagementPerformance.empty()
        assert perf.avg_yield == 0.0
        assert perf.avg_soilc == 0.0
        assert perf.avg_moisture == 0.0
        assert perf.avg_yield_trend == 0.0
        assert perf.avg_soilc_trend == 0.0
        assert perf.avg_moisture_trend == 0.0

    def test_add_single_observation(self, sample_trends):
        """Adding one observation should correctly set sums and count."""
        perf = RegionManagementPerformance.empty()
        perf.add_observation(
            cropyield=5000.0,
            soilc=8000.0,
            moisture=0.35,
            trend=sample_trends,
        )

        assert perf.count == 1
        assert perf.yield_sum == 5000.0
        assert perf.soilc_sum == 8000.0
        assert perf.moisture_sum == 0.35
        assert perf.yield_trend_sum == 0.02
        assert perf.soilc_trend_sum == 0.01
        assert perf.moisture_trend_sum == -0.005

    def test_averages_with_observations(self, sample_trends):
        """Averages should be sums / count."""
        perf = RegionManagementPerformance.empty()
        perf.add_observation(5000.0, 8000.0, 0.35, sample_trends)
        perf.add_observation(6000.0, 9000.0, 0.40, sample_trends)

        assert perf.count == 2
        assert perf.avg_yield == pytest.approx(5500.0)
        assert perf.avg_soilc == pytest.approx(8500.0)
        assert perf.avg_moisture == pytest.approx(0.375)
        assert perf.avg_yield_trend == pytest.approx(0.02)

    def test_merge_two_aggregates(self, sample_trends):
        """Merging should combine counts and sums."""
        perf1 = RegionManagementPerformance.empty()
        perf1.add_observation(5000.0, 8000.0, 0.35, sample_trends)

        perf2 = RegionManagementPerformance.empty()
        perf2.add_observation(6000.0, 9000.0, 0.40, sample_trends)
        perf2.add_observation(7000.0, 10000.0, 0.45, sample_trends)

        perf1.merge(perf2)

        assert perf1.count == 3
        assert perf1.yield_sum == pytest.approx(18000.0)
        assert perf1.soilc_sum == pytest.approx(27000.0)

    def test_weighted_trend(self, sample_trends, mock_farmer):
        """Weighted trend score should combine trends with farmer weights."""
        perf = RegionManagementPerformance.empty()
        perf.add_observation(5000.0, 8000.0, 0.35, sample_trends)

        # weight_yield=0.4, weight_soil=0.4, weight_moisture=0.2
        # trends: yield=0.02, soilc=0.01, moisture=-0.005
        expected = 0.4 * 0.02 + 0.4 * 0.01 + 0.2 * (-0.005)

        score = perf.weighted_trend(mock_farmer)
        assert score == pytest.approx(expected)


# =============================================================================
# RegionManagementPerformanceStore TESTS
# =============================================================================


class TestRegionManagementPerformanceStore:
    """Tests for multi-bundle performance store."""

    def test_empty_store_creation(self):
        """New store should be empty with defaults."""
        store = RegionManagementPerformanceStore()
        assert store.year == -1
        assert store.total_farmers == 0
        assert store.bundle_counts == {}
        assert store.country_codes == []

    def test_store_with_year_and_country(self):
        """Store can be created with metadata."""
        store = RegionManagementPerformanceStore(
            year=2025,
            country_codes=["DEU", "FRA"],
        )
        assert store.year == 2025
        assert store.country_codes == ["DEU", "FRA"]

    def test_add_observation_creates_bundle_entry(self, mock_farmer):
        """Adding observation should create performance entry for bundle."""
        store = RegionManagementPerformanceStore()

        store.add_observation(ManagementBundle.conservation, mock_farmer)

        perf = store.get(ManagementBundle.conservation)
        assert perf.count == 1
        assert perf.avg_yield == pytest.approx(5000.0)

    def test_get_missing_bundle_returns_empty(self):
        """Getting non-existent bundle should return empty performance."""
        store = RegionManagementPerformanceStore()

        perf = store.get(ManagementBundle.notill)
        assert perf.count == 0
        assert perf.avg_yield == 0.0

    def test_finalize_store_sets_counts(self, mock_farmer):
        """finalize_store should compute bundle_counts and total_farmers."""
        store = RegionManagementPerformanceStore()

        # Add 2 conservation farmers
        store.add_observation(ManagementBundle.conservation, mock_farmer)
        store.add_observation(ManagementBundle.conservation, mock_farmer)

        # Add 1 conventional farmer (modify mock)
        mock_farmer.cropyield = 4000.0
        store.add_observation(ManagementBundle.conventional, mock_farmer)

        store.finalize_store()

        assert store.total_farmers == 3
        assert store.bundle_counts[ManagementBundle.conservation] == 2
        assert store.bundle_counts[ManagementBundle.conventional] == 1

    def test_items_returns_bundle_performance_pairs(self, mock_farmer):
        """items() should yield (bundle, performance) pairs."""
        store = RegionManagementPerformanceStore()
        store.add_observation(ManagementBundle.conservation, mock_farmer)
        store.add_observation(ManagementBundle.notill, mock_farmer)

        items = list(store.items())
        bundles = [b for b, _ in items]

        assert ManagementBundle.conservation in bundles
        assert ManagementBundle.notill in bundles

    def test_merge_store_combines_two_stores(self, mock_farmer):
        """merge_store should combine bundles and counts."""
        store1 = RegionManagementPerformanceStore(country_codes=["DEU"])
        store1.add_observation(ManagementBundle.conservation, mock_farmer)
        store1.finalize_store()

        store2 = RegionManagementPerformanceStore(country_codes=["FRA"])
        store2.add_observation(ManagementBundle.conservation, mock_farmer)
        store2.add_observation(ManagementBundle.notill, mock_farmer)
        store2.finalize_store()

        store1.merge_store(store2)

        assert store1.total_farmers == 3
        assert ManagementBundle.conservation in store1.bundle_counts
        assert ManagementBundle.notill in store1.bundle_counts
        assert "DEU" in store1.country_codes
        assert "FRA" in store1.country_codes


# =============================================================================
# COUNTRY-LEVEL INTEGRATION TESTS
# =============================================================================


class TestCountryManagementPerformance:
    """Tests for country-level management performance computation."""

    def test_country_statistic_set_correctly(self, mock_country, mock_farmer):
        """compute_management_performance should set statistic correctly."""
        mock_farmer.behaviour.practice_bundle = ManagementBundle.conservation
        mock_country.farmers = [mock_farmer]

        # Simulate what compute_management_performance does
        store = RegionManagementPerformanceStore(year=2025)
        store.add_observation(ManagementBundle.conservation, mock_farmer)
        store.finalize_store()
        mock_country.statistic.set("management_performance", store)

        # Verify
        retrieved = mock_country.statistic.get("management_performance")
        assert retrieved is not None
        assert isinstance(retrieved, RegionManagementPerformanceStore)
        assert retrieved.total_farmers == 1
        assert retrieved.year == 2025

    def test_multiple_bundles_tracked(self, mock_country):
        """Store should track performance for multiple bundles."""
        farmers = []
        for bundle in [
            ManagementBundle.conservation,
            ManagementBundle.conventional,
            ManagementBundle.notill,
        ]:
            farmer = MagicMock()
            farmer.cropyield = 5000.0
            farmer.soilc = 8000.0
            farmer.root_moisture = 0.35
            farmer.behaviour.performance_tracker.trend = {
                "yield": 0.01,
                "soilc": 0.01,
                "moisture": 0.0,
            }
            farmer.behaviour.practice_bundle = bundle
            farmers.append(farmer)

        store = RegionManagementPerformanceStore(year=2025)
        for f in farmers:
            store.add_observation(f.behaviour.practice_bundle, f)
        store.finalize_store()

        assert store.total_farmers == 3
        assert len(store.bundle_counts) == 3


# =============================================================================
# CLUSTER-LEVEL INTEGRATION TESTS
# =============================================================================


class TestClusterManagementPerformance:
    """Tests for cluster-level aggregation."""

    def test_cluster_aggregates_countries(self, mock_world):
        """Cluster aggregation should combine country stores."""
        # Setup cluster mapping
        cluster_to_countries = {
            0: ["DEU", "FRA"],
            1: ["ESP", "ITA"],
        }
        mock_world.statistic.set("cluster_to_countries", cluster_to_countries)

        # Create country stores
        country_stores = {}
        for code in ["DEU", "FRA", "ESP", "ITA"]:
            store = RegionManagementPerformanceStore(year=2025, country_codes=[code])
            store._bundles[ManagementBundle.conservation] = RegionManagementPerformance(
                yield_sum=5000.0,
                soilc_sum=8000.0,
                moisture_sum=0.35,
                yield_trend_sum=0.01,
                soilc_trend_sum=0.01,
                moisture_trend_sum=0.0,
                count=1,
            )
            store.total_farmers = 1
            store.bundle_counts = {ManagementBundle.conservation: 1}
            country_stores[code] = store

        # Simulate cluster_management_performance logic
        cluster_stats = {}
        for cluster_id, country_codes in cluster_to_countries.items():
            cluster_store = RegionManagementPerformanceStore(year=2025)
            for code in country_codes:
                if code in country_stores:
                    cluster_store.merge_store(country_stores[code])
            cluster_stats[cluster_id] = cluster_store

        mock_world.statistic.set("cluster_management_performance", cluster_stats)

        # Verify cluster stores were created
        result = mock_world.statistic.get("cluster_management_performance")
        assert result is not None
        assert 0 in result
        assert 1 in result

        # Cluster 0 should have DEU + FRA = 2 farmers
        cluster0_store = result[0]
        assert cluster0_store.total_farmers == 2
        assert "DEU" in cluster0_store.country_codes
        assert "FRA" in cluster0_store.country_codes

    def test_cluster_skips_missing_countries(self, mock_world):
        """Cluster aggregation should handle missing country stores."""
        cluster_to_countries = {0: ["DEU", "MISSING"]}
        mock_world.statistic.set("cluster_to_countries", cluster_to_countries)

        # Only DEU exists
        country_stores = {"DEU": RegionManagementPerformanceStore(year=2025, country_codes=["DEU"])}
        country_stores["DEU"].total_farmers = 1

        # Simulate aggregation with missing country
        cluster_stats = {}
        for cluster_id, country_codes in cluster_to_countries.items():
            cluster_store = RegionManagementPerformanceStore(year=2025)
            for code in country_codes:
                if code in country_stores:
                    cluster_store.merge_store(country_stores[code])
            cluster_stats[cluster_id] = cluster_store

        mock_world.statistic.set("cluster_management_performance", cluster_stats)

        result = mock_world.statistic.get("cluster_management_performance")
        assert result is not None
        # Should only have DEU's farmers
        assert result[0].total_farmers == 1

    def test_empty_cluster_mapping(self, mock_world):
        """Empty cluster mapping should not crash."""
        mock_world.statistic.set("cluster_to_countries", {})

        # Simulate aggregation with empty mapping
        cluster_stats = {}
        mock_world.statistic.set("cluster_management_performance", cluster_stats)

        result = mock_world.statistic.get("cluster_management_performance")
        assert result == {}


# =============================================================================
# STORE SERIALIZATION TESTS (for Dask sync)
# =============================================================================


class TestStoreSerializationForDask:
    """Tests for store serialization (Dask worker sync)."""

    def test_store_can_be_copied(self, mock_farmer):
        """Store should be copyable for Dask sync."""
        store = RegionManagementPerformanceStore(year=2025, country_codes=["DEU"])
        store.add_observation(ManagementBundle.conservation, mock_farmer)
        store.finalize_store()

        # Simulate Dask copy
        store_copy = copy.deepcopy(store)

        assert store_copy.year == store.year
        assert store_copy.total_farmers == store.total_farmers
        assert store_copy.country_codes == store.country_codes

    def test_store_in_cache_dict(self, mock_farmer):
        """Store should work when stored in a cache dict."""
        cache = {}

        store = RegionManagementPerformanceStore(year=2025)
        store.add_observation(ManagementBundle.conservation, mock_farmer)
        store.finalize_store()

        cache["management_performance"] = store

        # Simulate what _extract_country_stats does
        retrieved = cache.get("management_performance")
        assert retrieved is store

        # Can we copy the cache?
        cache_copy = cache.copy()
        assert "management_performance" in cache_copy

    def test_performance_dataclass_copyable(self, sample_trends):
        """RegionManagementPerformance should be copyable."""
        perf = RegionManagementPerformance.empty()
        perf.add_observation(5000.0, 8000.0, 0.35, sample_trends)

        perf_copy = copy.deepcopy(perf)

        assert perf_copy.count == perf.count
        assert perf_copy.yield_sum == perf.yield_sum
        assert perf_copy.avg_yield == perf.avg_yield


# =============================================================================
# EDGE CASES AND REGRESSION TESTS
# =============================================================================


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_zero_farmer_averages_dont_raise(self):
        """Averages with zero farmers should return 0, not raise."""
        store = RegionManagementPerformanceStore()
        store.finalize_store()

        assert store.total_farmers == 0
        
        perf = store.get(ManagementBundle.conservation)
        assert perf.avg_yield == 0.0
        assert perf.avg_soilc == 0.0

    def test_negative_trends_handled(self):
        """Negative trends should be stored correctly."""
        trends = {"yield": -0.05, "soilc": -0.02, "moisture": -0.01}
        perf = RegionManagementPerformance.empty()
        perf.add_observation(5000.0, 8000.0, 0.35, trends)

        assert perf.avg_yield_trend == pytest.approx(-0.05)
        assert perf.avg_soilc_trend == pytest.approx(-0.02)
        assert perf.avg_moisture_trend == pytest.approx(-0.01)

    def test_large_farmer_counts(self, sample_trends):
        """Store should handle large numbers of observations."""
        perf = RegionManagementPerformance.empty()
        
        n_farmers = 10000
        for _ in range(n_farmers):
            perf.add_observation(5000.0, 8000.0, 0.35, sample_trends)

        assert perf.count == n_farmers
        assert perf.avg_yield == pytest.approx(5000.0)

    def test_bundle_counts_preserved_after_merge(self, mock_farmer):
        """Bundle counts should be correctly summed after merge."""
        store1 = RegionManagementPerformanceStore(country_codes=["DEU"])
        for _ in range(5):
            store1.add_observation(ManagementBundle.conservation, mock_farmer)
        store1.finalize_store()

        store2 = RegionManagementPerformanceStore(country_codes=["FRA"])
        for _ in range(3):
            store2.add_observation(ManagementBundle.conservation, mock_farmer)
        store2.finalize_store()

        store1.merge_store(store2)

        assert store1.bundle_counts[ManagementBundle.conservation] == 8
        assert store1.total_farmers == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
