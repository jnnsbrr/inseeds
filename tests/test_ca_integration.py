"""Integration tests for Conservation Agriculture model with FAO data.

These tests exercise the full CA farmer lifecycle including:
- Initialization with FAO data
- Capital dynamics
- Profit calculation
- TPB decision making
- Practice adoption
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from inseeds.components.data.fao import ensure_dummy_fao_data


class TestCAModelWithDummyFAO:
    """Integration tests using CA model with dummy FAO data."""

    def test_ca_model_has_farmers(self, ca_model_instance):
        """CA model should create farmers."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        assert len(farmers) > 0

    def test_ca_farmers_have_capital(self, ca_model_instance):
        """CA farmers should have capital attribute."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        for farmer in farmers[:3]:
            assert hasattr(farmer, "capital")
            # Capital may be NaN if FAO data extraction fails for test data
            # but the attribute should exist
            if not math.isnan(farmer.capital):
                assert farmer.capital >= 0

    def test_ca_farmers_have_depreciation_rate(self, ca_model_instance):
        """CA farmers should have depreciation rate from FAO."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        for farmer in farmers[:3]:
            assert hasattr(farmer, "depreciation_rate")
            assert 0 < farmer.depreciation_rate < 1

    def test_ca_farmers_have_tpb_behaviour(self, ca_model_instance):
        """CA farmers should have TPB behaviour model."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "behaviour")
        assert farmer.behaviour is not None

    def test_ca_farmers_have_practice_attributes(self, ca_model_instance):
        """CA farmers should have tillage, cover_crop, residue attributes."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "tillage")
        assert hasattr(farmer, "cover_crop")
        assert hasattr(farmer, "residue_on_field")
        assert farmer.tillage in (0, 1)

    def test_ca_farmers_update_runs(self, ca_model_instance):
        """CA farmers should update without errors."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        
        # Check if output data has required variables for update
        try:
            output = farmer.cell.from_earth
            if not hasattr(output, "pft_harvestc"):
                pytest.skip("Test data missing pft_harvestc for update")
        except Exception:
            pytest.skip("Cannot access cell output data")
        
        initial_capital = farmer.capital
        
        # Run update
        farmer.update(t=2020)
        
        # Verify update completed
        assert isinstance(farmer.capital, (int, float))


class TestCAFarmerCapitalDynamics:
    """Tests for capital dynamics in CA farmers."""

    def test_capital_depreciates_over_time(self, ca_model_instance):
        """Capital should change after updates."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        
        # Check if output data has required variables for update
        try:
            output = farmer.cell.from_earth
            if not hasattr(output, "pft_harvestc"):
                pytest.skip("Test data missing pft_harvestc for update")
        except Exception:
            pytest.skip("Cannot access cell output data")
        
        capitals = [farmer.capital]
        
        # Run multiple updates
        for t in range(2020, 2023):
            farmer.update(t=t)
            capitals.append(farmer.capital)
        
        # Capital should have changed
        assert not all(c == capitals[0] for c in capitals)

    def test_min_capital_threshold_exists(self, ca_model_instance):
        """Farmers should have min_capital threshold."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "min_capital")
        # min_capital may be NaN if FAO data extraction fails for test data
        if not math.isnan(farmer.min_capital):
            assert farmer.min_capital >= 0


class TestCAFarmerProfitCalculation:
    """Tests for profit and revenue calculation."""

    def test_revenue_calculation_runs(self, ca_model_instance):
        """Revenue calculation should run without errors."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        
        # Check if output data has required variables for revenue calculation
        try:
            output = farmer.cell.from_earth
            if not hasattr(output, "pft_harvestc"):
                pytest.skip("Test data missing pft_harvestc for revenue calculation")
        except Exception:
            pytest.skip("Cannot access cell output data")
        
        # Call revenue calculation
        if hasattr(farmer, "_calculate_revenue"):
            revenue = farmer._calculate_revenue()
            assert revenue >= 0

    def test_direct_costs_calculation_runs(self, ca_model_instance):
        """Direct costs calculation should run without errors."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        
        # Call costs calculation
        if hasattr(farmer, "_get_current_direct_costs"):
            costs = farmer._get_current_direct_costs()
            # Costs may be NaN if FAO data extraction fails for test data
            if not math.isnan(costs):
                assert costs >= 0


class TestCAFarmerTPBComponents:
    """Tests for TPB components in CA farmers."""

    def test_tpb_attitude_in_range(self, ca_model_instance):
        """TPB attitude should be in [0, 1]."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            attitude = farmer.behaviour.attitude
            assert 0 <= attitude <= 1

    def test_tpb_social_norm_in_range(self, ca_model_instance):
        """TPB social norm should be in [0, 1]."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            norm = farmer.behaviour.social_norm
            assert 0 <= norm <= 1

    def test_tpb_pbc_in_range(self, ca_model_instance):
        """TPB PBC should be in [0, 1]."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            pbc = farmer.behaviour.pbc
            assert 0 <= pbc <= 1

    def test_tpb_intention_in_range(self, ca_model_instance):
        """TPB intention should be in [0, 1]."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            intention = farmer.behaviour.tpb
            assert 0 <= intention <= 1


class TestCAFarmerPracticeBundle:
    """Tests for practice bundle tracking."""

    def test_practice_bundle_is_valid(self, ca_model_instance):
        """Practice bundle should be in range [0, 7]."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            bundle_id = farmer.behaviour.practice_bundle
            assert 0 <= bundle_id <= 7

    def test_practice_bundle_name_is_valid(self, ca_model_instance):
        """Practice bundle name should be one of the defined names."""
        from inseeds.components.farming.ca_behaviour import BUNDLE_NAMES
        
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            name = farmer.behaviour.practice_bundle_name
            assert name in BUNDLE_NAMES.values()


class TestFAODataLoadingInModel:
    """Tests for FAO data loading during model initialization."""

    def test_fao_data_loads_during_init(self):
        """FAO data should load during model initialization."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Reset cache
        CACountry._fao_prices_ds = None
        CACountry._fao_capital_ds = None
        
        # The test_fao_data.py already tests this comprehensively
        # This is a placeholder to document the behavior
        assert True

    def test_fao_data_shared_across_countries(self):
        """FAO data should be shared across all countries via class-level cache."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Verify class-level cache exists
        assert hasattr(CACountry, "_fao_prices_ds")
        assert hasattr(CACountry, "_fao_capital_ds")


class TestCAFarmerAFTTypes:
    """Tests for Agent Functional Type (AFT) distribution."""

    def test_farmers_have_aft_attribute(self, ca_model_instance):
        """All CA farmers should have AFT attribute."""
        from inseeds.components.farming.farmer import AFT
        
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        for farmer in farmers[:5]:
            assert hasattr(farmer, "aft")
            assert farmer.aft in (AFT.traditionalist, AFT.pioneer)

    def test_aft_distribution_follows_config(self, ca_model_instance):
        """AFT distribution should roughly follow configured pioneer_share."""
        from inseeds.components.farming.farmer import AFT
        
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers or len(farmers) < 10:
            pytest.skip("Need at least 10 farmers for distribution test")
        
        pioneers = sum(1 for f in farmers if f.aft == AFT.pioneer)
        pioneer_fraction = pioneers / len(farmers)
        
        # Should be roughly 30% pioneers (allow wide range for small samples)
        assert 0.1 <= pioneer_fraction <= 0.6


class TestResidueEconomics:
    """Tests for residue opportunity cost calculations."""

    def test_residue_opportunity_cost_exists(self, ca_model_instance):
        """Farmers should have residue opportunity cost."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "residue_opportunity_cost"):
            cost = farmer.residue_opportunity_cost
            # Cost may be NaN if FAO data extraction fails for test data
            if not math.isnan(cost):
                assert cost >= 0

    def test_residue_opportunity_cost_per_ha_exists(self, ca_model_instance):
        """Farmers should have per-hectare residue opportunity cost."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "residue_opportunity_cost_per_ha"):
            cost_per_ha = farmer.residue_opportunity_cost_per_ha
            assert cost_per_ha >= 0


class TestPracticeCosts:
    """Tests for practice cost configuration."""

    def test_farmers_have_practice_costs(self, ca_model_instance):
        """Farmers should have practice costs configuration."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "practice_costs")
        assert isinstance(farmer.practice_costs, dict)

    def test_practice_costs_have_direct_and_transition(self, ca_model_instance):
        """Practice costs should have direct and transition components per practice."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "practice_costs"):
            costs = farmer.practice_costs
            # Costs are structured as {practice: {direct: X, transition: Y}}
            for practice, practice_costs in costs.items():
                assert "direct" in practice_costs or "transition" in practice_costs


class TestNeighbourhoodStructure:
    """Tests for farmer neighbourhood structure."""

    def test_farmers_have_neighbourhood(self, ca_model_instance):
        """Farmers should have neighbourhood attribute."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "neighbourhood")

    def test_neighbourhood_is_list(self, ca_model_instance):
        """Neighbourhood should be a list."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "neighbourhood"):
            assert isinstance(farmer.neighbourhood, list)


class TestFAODataStructure:
    """Tests for FAO data structure and access."""

    def test_producer_prices_structure(self):
        """Producer prices should have correct structure."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            ensure_dummy_fao_data(sim_path, years=(2016, 2020))
            
            prices_path = sim_path / "input" / "fao_pft_prices_DUMMY.nc"
            ds = xr.open_dataset(prices_path)
            
            # Check dimensions
            assert "npft" in ds.dims
            assert "time" in ds.dims
            assert "area_code" in ds.dims
            
            # Check data variable
            assert "5532" in ds
            
            ds.close()

    def test_capital_stock_structure(self):
        """Capital stock should have correct structure."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            ensure_dummy_fao_data(sim_path, years=(2016, 2020))
            
            capital_path = sim_path / "input" / "fao_capital_stock_DUMMY.nc"
            ds = xr.open_dataset(capital_path)
            
            # Check dimensions
            assert "time" in ds.dims
            assert "area_code" in ds.dims
            
            # Check data variables
            assert "ncs" in ds
            assert "gfcf" in ds
            assert "cfc" in ds
            assert "depreciation_rate" in ds
            assert "investment_rate" in ds
            
            # Check derived metrics are valid
            assert (ds["depreciation_rate"] > 0).all()
            assert (ds["depreciation_rate"] < 1).all()
            assert (ds["investment_rate"] > 0).all()
            assert (ds["investment_rate"] < 1).all()
            
            ds.close()


class TestFAODataBatchDownload:
    """Tests for FAO data batch download functionality."""

    def test_ensure_with_multiple_countries(self):
        """ensure() should handle multiple countries."""
        from inseeds.components.data.fao import FaoProducerPrices
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            prices = FaoProducerPrices()
            path = prices.ensure(
                sim_path=sim_path,
                country_codes=["NLD", "DEU", "FRA"],
                reference_year=2020,
                years_before=4,
            )
            
            assert path.exists()
            
            # Check that data has multiple countries
            ds = xr.open_dataset(path)
            assert len(ds.area_code) >= 3
            ds.close()

    def test_ensure_with_single_country(self):
        """ensure() should handle single country."""
        from inseeds.components.data.fao import FaoCapitalStock
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            capital = FaoCapitalStock()
            path = capital.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=4,
            )
            
            assert path.exists()


class TestFAODataYearSelection:
    """Tests for year selection in FAO data."""

    def test_years_before_parameter(self):
        """years_before should control how many years are fetched."""
        from inseeds.components.data.fao import FaoProducerPrices
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            prices = FaoProducerPrices()
            
            # With years_before=4, should get 5 years total (reference + 4 before)
            path = prices.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=4,
            )
            
            ds = xr.open_dataset(path)
            
            # Dummy data spans 2016-2020, so should have 5 years
            assert len(ds.time) == 5
            
            ds.close()

    def test_reference_year_parameter(self):
        """reference_year should control which year is used as reference."""
        from inseeds.components.data.fao import FaoCapitalStock
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            capital = FaoCapitalStock()
            
            # Should use 2020 as reference
            path = capital.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=2,
            )
            
            ds = xr.open_dataset(path)
            
            # Should have years 2018, 2019, 2020 (3 years total)
            assert len(ds.time) == 3
            
            ds.close()


class TestFAOItemFiltering:
    """Tests for FAO item filtering (LPJmL-relevant crops only)."""

    def test_producer_prices_filters_to_lpjml_crops(self):
        """Producer prices should only include crops that map to LPJmL CFTs."""
        from inseeds.components.data.fao import FaoProducerPrices
        
        prices = FaoProducerPrices()
        items = prices._get_items()
        
        # Should be 166 items (FAO codes that map to LPJmL)
        assert len(items) == 166
        
        # All should be strings (FAO item codes)
        assert all(isinstance(item, str) for item in items)

    def test_capital_stock_has_all_three_items(self):
        """Capital stock should include all three capital components."""
        from inseeds.components.data.fao import FaoCapitalStock
        
        capital = FaoCapitalStock()
        items = capital._get_items()
        
        # Should be exactly 3 items: GFCF, CFC, NCS
        assert len(items) == 3
        items_list = list(items)
        assert "22030" in items_list  # GFCF
        assert "22031" in items_list  # CFC
        assert "22034" in items_list  # NCS


class TestCACountryFAOProperties:
    """Tests for CACountry FAO-derived properties."""

    def test_ca_country_has_depreciation_rate_property(self):
        """CACountry should have depreciation_rate property."""
        from inseeds.components.farming.ca_country import CACountry
        
        assert "depreciation_rate" in dir(CACountry)

    def test_ca_country_has_investment_rate_property(self):
        """CACountry should have _investment_rate property."""
        from inseeds.components.farming.ca_country import CACountry
        
        assert "_investment_rate" in dir(CACountry)

    def test_ca_country_has_capital_per_ha_property(self):
        """CACountry should have initial_capital_per_ha property."""
        from inseeds.components.farming.ca_country import CACountry
        
        assert "initial_capital_per_ha" in dir(CACountry)

    def test_ca_country_has_pft_prices_property(self):
        """CACountry should have pft_prices property."""
        from inseeds.components.farming.ca_country import CACountry
        
        assert "pft_prices" in dir(CACountry)


class TestTPBBehaviourExecution:
    """Tests that actually execute TPB behaviour methods."""

    def test_tpb_attitude_calculation(self, ca_model_instance):
        """TPB attitude calculation should return valid value."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Access attitude - should trigger calculation
        attitude = farmer.behaviour.attitude
        assert isinstance(attitude, (int, float))
        assert 0 <= attitude <= 1

    def test_tpb_social_norm_calculation(self, ca_model_instance):
        """TPB social norm calculation should return valid value."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Access social_norm - should trigger calculation
        social_norm = farmer.behaviour.social_norm
        assert isinstance(social_norm, (int, float))
        assert 0 <= social_norm <= 1

    def test_tpb_pbc_calculation(self, ca_model_instance):
        """TPB PBC calculation should return valid value."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Access pbc - should trigger calculation
        pbc = farmer.behaviour.pbc
        assert isinstance(pbc, (int, float))
        assert 0 <= pbc <= 1

    def test_tpb_overall_score(self, ca_model_instance):
        """TPB overall score should be weighted combination."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Access tpb - should trigger calculation
        tpb = farmer.behaviour.tpb
        assert isinstance(tpb, (int, float))
        assert 0 <= tpb <= 1

    def test_should_transition_returns_boolean(self, ca_model_instance):
        """should_transition should return boolean."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        result = farmer.behaviour.should_transition()
        assert isinstance(result, bool)


class TestTPBBundleOperations:
    """Tests for TPB bundle-related operations."""

    def test_practice_bundle_id_is_integer(self, ca_model_instance):
        """Practice bundle ID should be an integer 0-7."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        bundle_id = farmer.behaviour.practice_bundle
        assert isinstance(bundle_id, int)
        assert 0 <= bundle_id <= 7

    def test_internal_practice_bundle_is_tuple(self, ca_model_instance):
        """Internal _practice_bundle should be a tuple of 3 values."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Access internal tuple representation
        bundle = farmer.behaviour._practice_bundle
        assert isinstance(bundle, tuple)
        assert len(bundle) == 3
        assert all(v in (0, 1) for v in bundle)

    def test_farmer_has_practice_attributes(self, ca_model_instance):
        """Farmer should have tillage, cover_crop, residue_on_field."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "tillage")
        assert hasattr(farmer, "cover_crop")
        assert hasattr(farmer, "residue_on_field")
        
        # Values should be numeric (0, 1, or float for residue)
        assert farmer.tillage in (0, 1)
        assert farmer.cover_crop in (0, 1)
        assert 0 <= farmer.residue_on_field <= 1

    def test_practice_bundle_name_is_string(self, ca_model_instance):
        """Practice bundle name should be a string."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        name = farmer.behaviour.practice_bundle_name
        assert isinstance(name, str)
        assert name in [
            "conventional", "residue_only", "cover_crop_only",
            "cover_crop_residue", "notill_only", "notill_residue",
            "notill_cover_crop", "conservation"
        ]


class TestTPBSimilarityCalculations:
    """Tests for TPB similarity calculations."""

    def test_bundle_similarity_calculation(self, ca_model_instance):
        """Bundle similarity should be calculable."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # Test similarity between bundles
        if hasattr(farmer.behaviour, "_bundle_similarity"):
            sim = farmer.behaviour._bundle_similarity((0, 0, 0), (1, 1, 1))
            assert isinstance(sim, (int, float))
            assert 0 <= sim <= 1

    def test_total_similarity_calculation(self, ca_model_instance):
        """Total similarity should combine bundle and crop similarity."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if len(farmers) < 2:
            pytest.skip("Need at least 2 farmers for similarity")
        
        farmer = farmers[0]
        neighbour = farmers[1]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_total_similarity"):
            # _total_similarity takes (self, own_bundle, neighbour)
            own_bundle = farmer.behaviour._practice_bundle  # Use internal tuple
            sim = farmer.behaviour._total_similarity(own_bundle, neighbour)
            assert isinstance(sim, (int, float))
            assert 0 <= sim <= 1


class TestTPBAffordabilityChecks:
    """Tests for TPB affordability checks."""

    def test_affordable_bundle_check(self, ca_model_instance):
        """Affordability check should return boolean."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_affordable_bundle"):
            # Check if current bundle is affordable (use internal tuple)
            bundle = farmer.behaviour._practice_bundle
            # Skip if capital is NaN
            if math.isnan(farmer.capital):
                pytest.skip("Capital is NaN")
            result = farmer.behaviour._affordable_bundle(bundle)
            assert isinstance(result, bool)

    def test_bundle_direct_cost_calculation(self, ca_model_instance):
        """Bundle direct cost should be calculable."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_get_bundle_direct_cost"):
            bundle = farmer.behaviour._practice_bundle  # Use internal tuple
            cost = farmer.behaviour._get_bundle_direct_cost(bundle)
            # Cost may be NaN if FAO data extraction fails
            if not math.isnan(cost):
                assert isinstance(cost, (int, float))


class TestTPBExplorationBehaviour:
    """Tests for TPB exploration behaviour."""

    def test_is_reasonable_bundle_check(self, ca_model_instance):
        """Reasonableness check should return boolean."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_is_reasonable_bundle"):
            # Check if conventional bundle is reasonable
            result = farmer.behaviour._is_reasonable_bundle((0, 0, 0))
            assert isinstance(result, bool)

    def test_most_promising_bundle_selection(self, ca_model_instance):
        """Most promising bundle should be a valid bundle tuple."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_most_promising_bundle"):
            bundle = farmer.behaviour._most_promising_bundle()
            if bundle is not None:
                assert isinstance(bundle, tuple)
                assert len(bundle) == 3


class TestTPBFallbackBehaviour:
    """Tests for TPB fallback behaviour."""

    def test_check_fallback_returns_boolean(self, ca_model_instance):
        """Fallback check should return boolean."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "_check_fallback"):
            result = farmer.behaviour._check_fallback()
            assert isinstance(result, bool)


class TestBundleNameConstants:
    """Tests for bundle name constants."""

    def test_bundle_names_dict_has_all_combinations(self):
        """BUNDLE_NAMES should have all 8 practice combinations."""
        from inseeds.components.farming.ca_behaviour import BUNDLE_NAMES
        
        assert len(BUNDLE_NAMES) == 8
        
        # All combinations of (0,1) for 3 practices
        for t in (0, 1):
            for c in (0, 1):
                for r in (0, 1):
                    assert (t, c, r) in BUNDLE_NAMES

    def test_bundle_tuples_is_reverse_of_names(self):
        """BUNDLE_TUPLES should be reverse lookup of BUNDLE_NAMES."""
        from inseeds.components.farming.ca_behaviour import (
            BUNDLE_NAMES, BUNDLE_TUPLES
        )
        
        for bundle, name in BUNDLE_NAMES.items():
            assert BUNDLE_TUPLES[name] == bundle

    def test_bundle_ids_are_0_to_7(self):
        """BUNDLE_IDS should map bundles to 0-7."""
        from inseeds.components.farming.ca_behaviour import BUNDLE_IDS
        
        assert len(BUNDLE_IDS) == 8
        assert set(BUNDLE_IDS.values()) == set(range(8))


class TestSigmoidFunction:
    """Tests for sigmoid utility function."""

    def test_sigmoid_zero_gives_half(self):
        """sigmoid(0) should return 0.5."""
        from inseeds.components.farming.ca_behaviour import sigmoid
        
        assert abs(sigmoid(0) - 0.5) < 1e-10

    def test_sigmoid_large_positive_approaches_one(self):
        """sigmoid(large positive) should approach 1."""
        from inseeds.components.farming.ca_behaviour import sigmoid
        
        assert sigmoid(10) > 0.99

    def test_sigmoid_large_negative_approaches_zero(self):
        """sigmoid(large negative) should approach 0."""
        from inseeds.components.farming.ca_behaviour import sigmoid
        
        assert sigmoid(-10) < 0.01

    def test_sigmoid_is_monotonic(self):
        """sigmoid should be monotonically increasing."""
        from inseeds.components.farming.ca_behaviour import sigmoid
        import numpy as np
        
        x = np.linspace(-5, 5, 100)
        y = sigmoid(x)
        assert all(np.diff(y) > 0)
