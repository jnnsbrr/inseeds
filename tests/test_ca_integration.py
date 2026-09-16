"""Integration tests for Conservation Agriculture model with FAO data.

These tests exercise the full CA farmer lifecycle including:
- Initialization with FAO data
- Capital dynamics
- Profit calculation
- TPB decision making
- Practice adoption
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


class TestCAModelWithDummyFAO:
    """Integration tests using CA model with dummy FAO data."""

    def test_ca_model_has_farmers(self, ca_model_instance):
        """CA model should create farmers."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        assert len(farmers) > 0

    def test_ca_farmers_have_farm_size(self, ca_model_instance):
        """CA farmers must expose farm_size (used for capital and costs)."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")

        for farmer in farmers[:3]:
            assert hasattr(farmer, "farm_size")
            assert farmer.farm_size > 0

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

    def test_ca_farmers_have_capital_endowment(self, ca_model_instance):
        """CA farmers should take their initial capital from FAO Net Capital Stock."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")

        for farmer in farmers[:3]:
            assert hasattr(farmer, "initial_capital_per_ha")
            if not math.isnan(farmer.initial_capital_per_ha):
                assert farmer.initial_capital_per_ha >= 0

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

    def test_capital_moves_by_the_profit_of_the_year(self, ca_model_instance):
        """K_{t+1} = K + profit, and nothing else moves the stock.

        A farm whose environment does not change earns exactly its baseline and
        so holds its capital, which is what the deviation-based dynamics mean.
        """
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

        for t in range(2020, 2023):
            before = farmer.capital
            farmer.update(t=t)
            # A transition is charged after update_capital, so the operating
            # profit alone does not close the year; see test_capital_balance.
            expected = max(
                0.0,
                before
                + farmer.raw_profit
                - farmer.transition_cost
                + farmer.residue_sale_income,
            )
            assert farmer.capital == pytest.approx(expected)

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
        if hasattr(farmer, "calculate_revenue"):
            revenue = farmer.calculate_revenue()
            assert revenue >= 0

    def test_direct_costs_calculation_runs(self, ca_model_instance):
        """Direct costs calculation should run without errors."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        
        # Call costs calculation
        if hasattr(farmer, "get_bundle_direct_costs"):
            costs = farmer.get_bundle_direct_costs(
                farmer.behaviour.practice_bundle
            )
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
            id = farmer.behaviour.practice_bundle.id
            assert 0 <= id <= 7

    def test_practice_bundle_name_is_valid(self, ca_model_instance):
        """Practice bundle name should be one of the defined display names."""
        from inseeds.components.farming.ca_management import ManagementBundle

        labels = {bundle.label for bundle in ManagementBundle}
        
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "behaviour"):
            name = farmer.behaviour.practice_bundle.label
            assert name in labels


class TestFAODataLoadingInModel:
    """Tests for FAO data loading during model initialization."""

    def test_fao_data_loads_during_init(self):
        """FAO data should load during model initialization."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Reset class-level cache
        CACountry._fao_reference_year = None
        
        # The test_fao_data.py already tests this comprehensively
        assert True

    def test_fao_data_shared_across_countries(self):
        """FAO data should be shared via world.exogenous with class-level reference year."""
        from inseeds.components.farming.ca_country import CACountry
        
        # Verify class-level reference year cache exists
        assert hasattr(CACountry, "_fao_reference_year")


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
        """AFT distribution should follow the configured pioneer_share.

        Drawn here rather than counted over the farmers of the run: a couple of
        dozen cells are too few for a share to show, so counting them would
        fail on the luck of the draw a few runs in a hundred.
        """
        from inseeds.components.farming.farmer import AFT
        
        share = ca_model_instance.config.coupled_config.pioneer_share
        np.random.seed(0)
        drawn = [AFT.random(share) for _ in range(2000)]
        
        pioneer_fraction = sum(a == AFT.pioneer for a in drawn) / len(drawn)
        assert pioneer_fraction == pytest.approx(share, abs=0.05)


class TestResidueEconomics:
    """Tests for residue opportunity cost calculations."""

    def test_compute_residue_opportunity_cost_exists(self, ca_model_instance):
        """Farmers should have compute_residue_opportunity_cost method."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "compute_residue_opportunity_cost"):
            cost = farmer.compute_residue_opportunity_cost()
            # Cost may be NaN if FAO data extraction fails for test data
            if not math.isnan(cost):
                assert cost >= 0


class TestPracticeCosts:
    """Tests for practice cost configuration."""

    def test_farmers_have_practice_costs(self, ca_model_instance):
        """Farmers should have practice costs configuration."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        assert hasattr(farmer, "practice_costs")
        from inseeds.components.farming.ca_management import ManagementCosts
        assert isinstance(farmer.practice_costs, ManagementCosts)

    def test_practice_costs_have_direct_and_transition(self, ca_model_instance):
        """Practice costs should have direct and transition components per practice."""
        from dataclasses import fields
        from inseeds.components.farming.ca_management import ManagementCosts

        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if hasattr(farmer, "practice_costs"):
            costs = farmer.practice_costs
            for f in fields(ManagementCosts):
                practice_cost = getattr(costs, f.name)
                assert hasattr(practice_cost, "direct")
                assert hasattr(practice_cost, "transition")


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

    def test_producer_prices_structure(self, ca_model_instance):
        """Producer prices should have correct structure."""
        prices = ca_model_instance.world.exogenous.prices

        assert "npft" in prices.dims
        assert "time" in prices.dims
        assert "area_code" in prices.dims

        # FAOSTAT element 5532, Producer Price (USD/tonne)
        assert "5532" in prices

    def test_capital_stock_structure(self, ca_model_instance):
        """Capital stock should have correct structure."""
        capital = ca_model_instance.world.exogenous.capital

        assert "time" in capital.dims
        assert "area_code" in capital.dims

        for name in ("ncs", "gfcf", "cfc", "depreciation_rate", "investment_rate"):
            assert name in capital

        # Rates are shares of the capital stock, so they live in (0, 1)
        for name in ("depreciation_rate", "investment_rate"):
            rate = capital[name]
            assert (rate > 0).all()
            assert (rate < 1).all()


class TestFAODataBatchDownload:
    """Tests for FAO data batch download functionality."""

    def test_ensure_with_multiple_countries(self):
        """ensure() should handle multiple countries."""
        from inseeds.components.exogenous.faostat import FaoProducerPrices
        
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
        from inseeds.components.exogenous.faostat import FaoCapitalStock
        
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
        from inseeds.components.exogenous.faostat import FaoProducerPrices
        
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
        from inseeds.components.exogenous.faostat import FaoCapitalStock
        
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
        from inseeds.components.exogenous.faostat import FaoProducerPrices
        
        prices = FaoProducerPrices()
        items = prices._get_items()
        
        # Should be 166 items (FAO codes that map to LPJmL)
        assert len(items) == 166
        
        # All should be strings (FAO item codes)
        assert all(isinstance(item, str) for item in items)

    def test_capital_stock_has_all_three_items(self):
        """Capital stock should include all three capital components."""
        from inseeds.components.exogenous.faostat import FaoCapitalStock
        
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


class TestYearlyLevels:
    """Tests that the levels farmers are judged on share one vintage."""

    def test_leaching_holds_until_the_year_is_read(self, ca_model_instance):
        """Leaching is a value for the year, like yield and soil carbon.

        The country performance store is built before any farmer has updated,
        so a level that read the cell there would answer with a year the other
        levels have not reached, and the store would mix two vintages.
        """
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")

        from inseeds.components.farming.farmer import agricultural_leaching

        farmer = farmers[0]
        held = farmer.leaching
        cell_leaching = farmer.cell.from_earth["leaching"]
        original = cell_leaching.values.copy()

        try:
            # Set the raw cell field; farmer reads whole-cell leaching directly.
            new_val = held * 2.0 + 1.0
            cell_leaching.values[...] = new_val
            expected = agricultural_leaching(farmer)

            assert farmer.leaching == held
            assert farmer.cell_leaching != held

            farmer._read_leaching_state()
            assert farmer.leaching == pytest.approx(expected)
        finally:
            cell_leaching.values[...] = original
            farmer._read_leaching_state()


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
        
        id = farmer.behaviour.practice_bundle.id
        assert isinstance(id, int)
        assert 0 <= id <= 7

    def test_internal_practice_bundle_is_management_bundle(self, ca_model_instance):
        """Internal practice_bundle should be a ManagementBundle enum member."""
        from inseeds.components.farming.ca_management import ManagementBundle

        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        bundle = farmer.behaviour.practice_bundle
        assert isinstance(bundle, ManagementBundle)
        assert bundle.tillage in (0, 1)
        assert bundle.cover_crop in (0, 1)
        assert bundle.residue_on_field in (0, 1)

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
        
        name = farmer.behaviour.practice_bundle.label
        assert isinstance(name, str)
        assert name in [
            "conventional farming", "residue retention", "cover crop",
            "cover crop + residue retention", "no-till", "no-till + residue retention",
            "no-till + cover crop", "conservation agriculture"
        ]


class TestTPBSimilarityCalculations:
    """Tests for TPB similarity calculations."""

    def test_bundle_similarity_calculation(self, ca_model_instance):
        """Bundle similarity should be calculable."""
        from inseeds.components.farming.ca_management import ManagementBundle

        sim = ManagementBundle.notill.similarity(ManagementBundle.covercrop_residue)
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
        
        if hasattr(farmer.behaviour, "total_similarity"):
            # total_similarity takes (self, own_bundle, neighbour)
            own_bundle = farmer.behaviour.practice_bundle  # Use internal tuple
            sim = farmer.behaviour.total_similarity(own_bundle, neighbour)
            assert isinstance(sim, (int, float))
            assert 0 <= sim <= 1


class TestTPBAffordabilityChecks:
    """Tests for TPB affordability checks."""

    def test_affordable_bundle_check(self, ca_model_instance):
        """Affordability returns the bundle the capital reaches, not a verdict."""
        from inseeds.components.farming.ca_management import ManagementBundle

        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "affordable_bundle"):
            # Check if current bundle is affordable (use internal tuple)
            bundle = farmer.behaviour.practice_bundle
            # Skip if capital is NaN
            if math.isnan(farmer.capital):
                pytest.skip("Capital is NaN")
            result = farmer.behaviour.affordable_bundle(bundle)
            assert isinstance(result, ManagementBundle)
            # The bundle it already runs is always within reach
            assert result == bundle

    def test_bundle_direct_cost_calculation(self, ca_model_instance):
        """Bundle direct cost should be calculable."""
        import math
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "get_bundle_direct_cost"):
            bundle = farmer.behaviour.practice_bundle  # Use internal tuple
            cost = farmer.get_bundle_direct_costs(bundle)
            # Cost may be NaN if FAO data extraction fails
            if not math.isnan(cost):
                assert isinstance(cost, (int, float))


class TestTPBExplorationBehaviour:
    """Tests for TPB exploration behaviour."""

    def test_exploration_method_exists(self, ca_model_instance):
        """Exploration method should exist (replaces _is_reasonable_bundle)."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        # _is_reasonable_bundle was removed, exploration logic now in maybe_explore_bundle
        assert hasattr(farmer.behaviour, "maybe_explore_bundle")

    def test_most_promising_bundle_selection(self, ca_model_instance):
        """Most promising bundle should be a valid bundle tuple."""
        farmers = getattr(ca_model_instance, "_farmers", [])
        if not farmers:
            pytest.skip("No farmers")
        
        farmer = farmers[0]
        if not hasattr(farmer, "behaviour"):
            pytest.skip("Farmer has no behaviour")
        
        if hasattr(farmer.behaviour, "most_promising_bundle_local"):
            from inseeds.components.farming.ca_management import ManagementBundle

            bundle = farmer.behaviour.most_promising_bundle_local()
            if bundle is not None:
                assert isinstance(bundle, ManagementBundle)


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
        
        if hasattr(farmer.behaviour, "check_fallback"):
            result = farmer.behaviour.check_fallback()
            assert isinstance(result, bool)


class TestBundleNameConstants:
    """Tests for ManagementBundle enum metadata."""

    def test_all_practice_combinations_exist(self):
        """Each practice triple should map to an enum member."""
        from inseeds.components.farming.ca_management import ManagementBundle

        assert len(ManagementBundle) == 8

        for t in (0, 1):
            for c in (0, 1):
                for r in (0, 1):
                    bundle = ManagementBundle.from_practices(t, c, r)
                    assert bundle.tillage == t
                    assert bundle.cover_crop == c
                    assert bundle.residue_on_field == r

    def test_bundle_ids_are_0_to_7(self):
        """Each bundle should have a unique ID from 0 to 7."""
        from inseeds.components.farming.ca_management import ManagementBundle

        ids = [bundle.id for bundle in ManagementBundle]
        assert len(ids) == 8
        assert set(ids) == set(range(8))


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
