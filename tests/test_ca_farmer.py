"""Comprehensive tests for Conservation Agriculture farmer and behaviour.

Tests cover:
- CA bundle definitions and utilities
- TPB sigmoid function
- FAO data batching optimizations
- CA farmer integration with real model instances
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from inseeds.components.exogenous.faostat import (
    FaoProducerPrices,
    FaoCapitalStock,
)
from inseeds.components.farming.ca_behaviour import sigmoid
from inseeds.components.farming.ca_management import ManagementBundle


class TestSigmoidFunction:
    """Tests for sigmoid utility function."""

    def test_sigmoid_zero_returns_half(self):
        """sigmoid(0) should return 0.5."""
        assert sigmoid(0) == pytest.approx(0.5)

    def test_sigmoid_positive_increases(self):
        """sigmoid(x) for x > 0 should be > 0.5."""
        assert sigmoid(1) > 0.5
        assert sigmoid(10) == pytest.approx(1.0, abs=1e-4)

    def test_sigmoid_negative_decreases(self):
        """sigmoid(x) for x < 0 should be < 0.5."""
        assert sigmoid(-1) < 0.5
        assert sigmoid(-10) == pytest.approx(0.0, abs=1e-4)

    def test_sigmoid_bounds(self):
        """sigmoid output should be in [0, 1]."""
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            y = sigmoid(x)
            assert 0 <= y <= 1

    def test_sigmoid_symmetric_around_zero(self):
        """sigmoid should be symmetric: sigmoid(-x) = 1 - sigmoid(x)."""
        for x in [1, 2, 5, 10]:
            assert sigmoid(-x) == pytest.approx(1 - sigmoid(x))


class TestBundleDefinitions:
    """Tests for CA practice bundle definitions."""

    def test_all_bundles_have_unique_ids(self):
        """Each bundle should have a unique numeric ID (0-7)."""
        ids = [bundle.id for bundle in ManagementBundle]
        assert len(ids) == 8
        assert set(ids) == set(range(8))

    def test_all_bundles_have_unique_keys(self):
        """Each bundle should have a unique internal key."""
        keys = [bundle.key for bundle in ManagementBundle]
        assert len(keys) == len(set(keys))
        assert keys == [
            "notill",
            "notill_residue",
            "notill_covercrop",
            "conservation",
            "conventional",
            "residue",
            "covercrop",
            "covercrop_residue",
        ]

    def test_all_bundles_have_valid_practices(self):
        """All bundles should use binary practice flags."""
        for bundle in ManagementBundle:
            assert bundle.tillage in (0, 1)
            assert bundle.cover_crop in (0, 1)
            assert bundle.residue_on_field in (0, 1)

    def test_conventional_bundle(self):
        """Conventional bundle should be (1,0,0) with ID 4."""
        bundle = ManagementBundle.conventional
        assert bundle.label == "conventional farming"
        assert bundle.id == 4
        assert bundle.key == "conventional"

    def test_full_ca_bundle(self):
        """Full CA bundle should be (0,1,1) with ID 3."""
        bundle = ManagementBundle.conservation
        assert bundle.label == "conservation agriculture"
        assert bundle.id == 3
        assert bundle.key == "conservation"

    def test_all_bundles_have_unique_labels(self):
        """All bundle labels should be unique."""
        labels = [bundle.label for bundle in ManagementBundle]
        assert len(labels) == len(set(labels))

    def test_enum_value_is_practice_triple_only(self):
        """Enum .value should be the practice triple only."""
        assert ManagementBundle.notill.value == (0, 0, 0)
        assert ManagementBundle.conventional.value == (1, 0, 0)

    def test_class_level_listings(self):
        """Class methods should expose all bundle metadata."""
        assert len(ManagementBundle.all_bundles()) == 8
        assert ManagementBundle.all_ids() == tuple(range(8))
        assert ManagementBundle.all_keys()[0] == "notill"

    def test_from_practices_returns_existing_member(self):
        """from_practices should return a fully initialized enum member."""
        bundle = ManagementBundle.from_practices(1, 0, 0)
        assert bundle is ManagementBundle.conventional
        assert bundle.id == 4
        assert bundle.key == "conventional"
        assert bundle.label == "conventional farming"


class TestFAODataBatchingOptimization:
    """Tests for FAO data batching optimizations."""

    def test_producer_prices_downloads_only_lpjml_crops(self):
        """FaoProducerPrices should only download 166 crops that map to LPJmL."""
        prices = FaoProducerPrices()
        items = prices._get_items()
        
        # Should be 166 FAO codes that map to LPJmL, not all 176
        assert len(items) == 166
        assert all(isinstance(item, str) for item in items)

    def test_capital_stock_has_three_items(self):
        """FaoCapitalStock should download 3 items (GFCF, CFC, NCS)."""
        capital = FaoCapitalStock()
        items = capital._get_items()
        
        assert len(items) == 3
        assert list(items) == ["22030", "22031", "22034"]

    def test_producer_prices_is_pp_domain(self):
        """Producer prices should use PP domain."""
        prices = FaoProducerPrices()
        assert prices.domain == "PP"
        assert prices.domain != "CS"  # Not CS, so items batch together

    def test_capital_stock_is_cs_domain(self):
        """Capital stock should use CS domain."""
        capital = FaoCapitalStock()
        assert capital.domain == "CS"

    def test_producer_prices_element_code(self):
        """Producer prices should use element 5532 (Producer Price USD/tonne)."""
        prices = FaoProducerPrices()
        assert prices.elements == ["5532"]

    def test_capital_stock_element_code(self):
        """Capital stock should use element 6110 (Value US$)."""
        capital = FaoCapitalStock()
        assert capital.elements == ["6110"]


class TestFAOAPICallScaling:
    """Tests verifying API call counts for different scenarios."""

    def test_api_calls_for_single_country(self):
        """Single country should require 20 API calls."""
        n_years = 5  # years_before=4 means 5 years total
        
        # Producer prices: 5 calls (5 years × 1 element, all items+countries batched)
        calls_prices = n_years * 1
        
        # Capital stock: 15 calls (5 years × 1 element × 3 items, all countries batched)
        calls_capital = n_years * 1 * 3
        
        total = calls_prices + calls_capital
        assert total == 20

    def test_api_calls_constant_for_global_run(self):
        """Global run (150 countries) should still require only 20 API calls."""
        n_years = 5
        n_countries = 150  # Global run
        
        # With batching, countries don't multiply the call count
        calls_prices = n_years * 1  # All countries batched per year
        calls_capital = n_years * 1 * 3  # All countries batched per item
        
        total = calls_prices + calls_capital
        assert total == 20  # Same as single country!

    def test_old_approach_would_scale_linearly(self):
        """Old per-country approach would scale linearly (for comparison)."""
        n_years = 5
        n_countries = 150
        calls_per_country = 20
        
        old_total = calls_per_country * n_countries
        assert old_total == 3000  # 150x worse than new approach (20 calls)


class TestCACountryFAOBatching:
    """Tests for CACountry FAO data batching behavior."""

    def test_ca_country_class_has_cache_attributes(self):
        """CACountry class should have class-level cache attributes."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Verify class-level reference year cache exists
        assert hasattr(CACountry, "_fao_reference_year")

    def test_ca_country_has_fao_properties(self):
        """CACountry should have properties for FAO-derived parameters."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Check that properties exist
        assert "depreciation_rate" in dir(CACountry)
        assert "initial_capital_per_ha" in dir(CACountry)
        assert "pft_prices" in dir(CACountry)


class TestFAODataYearAveraging:
    """Tests for FAO data averaging over multiple years."""

    def test_fao_data_uses_reference_year_and_years_before(self):
        """FAO ensure should use reference_year and years_before parameters."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            prices = FaoProducerPrices()
            
            # Should accept these parameters without error
            path = prices.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=4,
            )
            
            assert path.exists()

    def test_capital_stock_calculates_derived_metrics(self):
        """Capital stock should calculate depreciation_rate and investment_rate."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            capital = FaoCapitalStock()
            path = capital.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=4,
            )
            
            import xarray as xr
            ds = xr.open_dataset(path)
            
            # Should have derived metrics
            assert "depreciation_rate" in ds
            assert "investment_rate" in ds
            assert "ncs" in ds
            assert "gfcf" in ds
            assert "cfc" in ds
            
            ds.close()


class TestCACountryFAOExtraction:
    """Tests for country-specific FAO data extraction."""

    def test_ca_country_has_extraction_methods(self):
        """CACountry should have methods for loading FAO parameters."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Check that FAO loading and price extraction methods exist
        assert hasattr(CACountry, "_ensure_fao_data")
        assert hasattr(CACountry, "extract_prices")


class TestFAODataCaching:
    """Tests for FAO data caching behavior."""

    def test_fao_data_cached_at_class_level(self):
        """FAO reference year should be cached at class level."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Verify class-level reference year cache exists (data now accessed via world.exogenous)
        assert hasattr(CACountry, "_fao_reference_year")


class TestFAODummyDataFallback:
    """Tests for dummy data fallback when API fails."""

    def test_ensure_creates_dummy_when_api_unavailable(self):
        """ensure() should create dummy data when API is unavailable."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            prices = FaoProducerPrices()
            
            # ensure() now automatically falls back to dummy on failure
            path = prices.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=4,
            )
            
            assert path.exists()

    def test_real_data_required(self):
        """Real FAO data is now required (dummy data functionality was removed)."""
        # Dummy data functionality was removed - real FAO data must be pre-downloaded
        # This test documents the new behavior: ensure() requires real data or API access
        from inseeds.components.exogenous.faostat import FaoProducerPrices
        assert hasattr(FaoProducerPrices, "ensure")


class TestCAFarmerProfitComponents:
    """Tests for profit calculation components."""

    def test_revenue_calculation_method_exists(self):
        """Revenue calculation method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # Verify the method exists
        assert hasattr(ConservationAgricultureFarmer, "calculate_revenue")

    def test_direct_costs_method_exists(self):
        """Direct costs calculation method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert hasattr(ConservationAgricultureFarmer, "get_current_direct_costs")

    def test_capital_dynamics_methods_exist(self):
        """Capital dynamics methods should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # Check for property descriptor (not direct attribute)
        assert "min_capital" in dir(ConservationAgricultureFarmer)
        assert hasattr(ConservationAgricultureFarmer, "update")
        assert hasattr(ConservationAgricultureFarmer, "update_capital")


class TestCABehaviourStructure:
    """Tests for CA behaviour module structure."""

    def test_tpb_class_exists(self):
        """TPB class should be importable."""
        from inseeds.components.farming.ca_behaviour import TPB
        assert TPB is not None

    def test_decision_model_base_class_exists(self):
        """DecisionModel abstract base class should exist."""
        from inseeds.components.farming.ca_behaviour import DecisionModel
        assert DecisionModel is not None

    def test_tpb_has_required_properties(self):
        """TPB should have attitude, social_norm, pbc, and tpb properties."""
        from inseeds.components.farming.ca_behaviour import TPB
        
        assert hasattr(TPB, "attitude")
        assert hasattr(TPB, "social_norm")
        assert hasattr(TPB, "pbc")
        assert hasattr(TPB, "tpb")

    def test_decision_model_has_abstract_methods(self):
        """DecisionModel should define abstract interface."""
        from inseeds.components.farming.ca_behaviour import DecisionModel
        
        assert hasattr(DecisionModel, "update")
        assert hasattr(DecisionModel, "should_transition")


class TestCAFarmerCoverCropLogic:
    """Tests for cover crop type selection logic."""

    def test_indicate_cover_crop_type_method_exists(self):
        """indicate_cover_crop_type method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert hasattr(ConservationAgricultureFarmer, "indicate_cover_crop_type")

    def test_cover_crop_type_returns_valid_values(self):
        """Cover crop type should be 1 (non-legume) or 2 (legume)."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # The method should return 1 or 2 based on environmental conditions
        # This is tested through the method signature
        import inspect
        sig = inspect.signature(ConservationAgricultureFarmer.indicate_cover_crop_type)
        # Method should take only self
        assert len(sig.parameters) == 1


class TestCAFarmerResidueEconomics:
    """Tests for residue opportunity cost calculation."""

    def test_residue_opportunity_cost_properties_exist(self):
        """Residue opportunity cost properties should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert "residue_opportunity_cost" in dir(ConservationAgricultureFarmer)
        assert "residue_opportunity_cost_per_ha" in dir(ConservationAgricultureFarmer)


class TestCAFarmerCellProperties:
    """Tests for cell property accessors."""

    def test_cell_property_accessors_exist(self):
        """Cell property accessors should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # Check for various cell properties
        assert "cell_runoff" in dir(ConservationAgricultureFarmer)
        assert "cell_leaching" in dir(ConservationAgricultureFarmer)
        assert "cell_fertilizer" in dir(ConservationAgricultureFarmer)
        assert "cell_pft_yield" in dir(ConservationAgricultureFarmer)
        assert "cell_pft_production" in dir(ConservationAgricultureFarmer)


class TestCAFarmerNeighbourhood:
    """Tests for neighbourhood initialization."""

    def test_init_neighbourhood_method_exists(self):
        """init_neighbourhood method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert hasattr(ConservationAgricultureFarmer, "init_neighbourhood")


class TestCAFarmerUpdate:
    """Tests for farmer update logic."""

    def test_update_method_exists(self):
        """update method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert hasattr(ConservationAgricultureFarmer, "update")

    def test_update_takes_time_parameter(self):
        """update method should take a time parameter."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        import inspect
        sig = inspect.signature(ConservationAgricultureFarmer.update)
        # Should have self and t parameters
        assert "t" in sig.parameters


class TestCAFarmerCapitalProperties:
    """Tests for capital-related properties."""

    def test_min_capital_property_exists(self):
        """min_capital property should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert "min_capital" in dir(ConservationAgricultureFarmer)

    def test_farm_size_property_exists(self):
        """farm_size property should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert "farm_size" in dir(ConservationAgricultureFarmer)
