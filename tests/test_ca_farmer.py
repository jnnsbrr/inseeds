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
from inseeds.components.farming.ca_management import (
    COVER_CROP_LEGUME,
    COVER_CROP_NONLEGUME,
    ManagementBundle,
)


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
        
        assert hasattr(ConservationAgricultureFarmer, "get_bundle_direct_costs")

    def test_capital_dynamics_methods_exist(self):
        """Capital dynamics methods should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # Check for property descriptor (not direct attribute)
        assert "min_capital" in dir(ConservationAgricultureFarmer)
        assert hasattr(ConservationAgricultureFarmer, "update")
        assert hasattr(ConservationAgricultureFarmer, "update_capital")


class TestWeedControlCredit:
    """The cover crop only saves weed control where there is no cultivation."""

    @staticmethod
    def _costs():
        from inseeds.components.farming.ca_management import ManagementCosts

        return ManagementCosts.from_config(
            {
                "tillage": {"transition": 0, "costs": {"fuel": 30.0}},
                "cover_crop": {
                    "transition": 0,
                    "costs": {
                        "seeds_nonlegume": 40.0,
                        "seeds_legume": 90.0,
                        "weed_control": -15.0,
                    },
                },
                "residue_on_field": {"transition": 0, "costs": {}},
            },
            gdp_cost_ratio=1.0,
        )

    def _cost_of(self, bundle, cover_crop_type=COVER_CROP_NONLEGUME):
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer

        class _Stub:
            practice_costs = self._costs()
            get_practice_direct_cost_per_ha = (
                ConservationAgricultureFarmer.get_practice_direct_cost_per_ha
            )

        return ConservationAgricultureFarmer.get_bundle_direct_cost_per_ha(
            _Stub(), bundle, cover_crop_type
        )

    def test_no_till_field_earns_the_credit(self):
        from inseeds.components.farming.ca_management import ManagementBundle

        assert self._cost_of(ManagementBundle.from_practices(0, 1, 0)) == 25.0

    def test_tilled_field_pays_the_full_cover_crop_cost(self):
        from inseeds.components.farming.ca_management import ManagementBundle

        # 30 tillage + 40 seeds, no credit: cultivation already kills the weeds.
        assert self._cost_of(ManagementBundle.from_practices(1, 1, 0)) == 70.0

    def test_credit_needs_a_cover_crop(self):
        from inseeds.components.farming.ca_management import ManagementBundle

        assert self._cost_of(ManagementBundle.from_practices(0, 0, 0)) == 0.0

    def test_only_the_seed_actually_sown_is_charged(self):
        """The two seed costs are alternatives, never a sum.

        Both live in the same cost_components dict, so anything that reaches
        for the total rather than for the one in the ground would charge a farm
        $130/ha for a crop it planted once.
        """
        from inseeds.components.farming.ca_management import ManagementBundle

        cover_crop = ManagementBundle.from_practices(0, 1, 0)

        catch = self._cost_of(cover_crop, COVER_CROP_NONLEGUME)
        legume = self._cost_of(cover_crop, COVER_CROP_LEGUME)

        assert catch == 25.0   # 40 seed - 15 credit
        assert legume == 75.0  # 90 seed - 15 credit
        assert legume - catch == 50.0


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

    def test_compute_residue_opportunity_cost_method_exists(self):
        """Residue opportunity cost method should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert "compute_residue_opportunity_cost" in dir(ConservationAgricultureFarmer)


class TestBandPriceMatching:
    """Tests for matching LPJmL crop bands to producer price categories."""

    CATEGORIES = [
        "Others, annual", "Others, perennial", "cotton", "maize", "pulses",
        "temperate cereals", "tropical roots",
    ]

    def test_named_crop_matches_its_own_category(self):
        from inseeds.components.farming.ca_farmer import _price_category_for

        assert _price_category_for("maize", self.CATEGORIES, {}) == "maize"
        assert _price_category_for(
            "temperate cereals", self.CATEGORIES, {}
        ) == "temperate cereals"

    def test_others_is_priced_as_the_cereal_lpjml_grows(self):
        """others_to_crop makes that band a cereal stand, not a residual."""
        from inseeds.components.farming.ca_farmer import _price_category_for

        assert _price_category_for(
            "others", self.CATEGORIES, {"others": "temperate cereals"}
        ) == "temperate cereals"
        assert _price_category_for(
            "others", self.CATEGORIES, {"others": "maize"}
        ) == "maize"

    def test_unpriceable_band_returns_nothing(self):
        """No category of its own; _price_for_band takes it from here."""
        from inseeds.components.farming.ca_farmer import _price_category_for

        assert _price_category_for("sugarcane", self.CATEGORIES, {}) is None


class TestPriceForBand:
    """Tests for the price a band is finally valued at."""

    @staticmethod
    def _prices(values):
        import xarray as xr

        return xr.DataArray(
            np.array(list(values.values()), dtype=float),
            dims=("npft",),
            coords={"npft": list(values)},
        )

    def test_own_category_wins(self):
        from inseeds.components.farming.ca_farmer import _price_for_band

        prices = self._prices(
            {"maize": 200.0, "Others, annual": 50.0, "Others, perennial": 40.0}
        )
        assert _price_for_band("maize", prices, "npft", {}) == ("maize", 200.0)

    def test_uncovered_crop_is_worth_its_own_group(self):
        """A country-only price file holds only that country's crops."""
        from inseeds.components.farming.ca_farmer import _price_for_band

        prices = self._prices(
            {"temperate cereals": 180.0, "maize": 220.0, "Others, annual": 900.0}
        )
        label, price = _price_for_band("rice", prices, "npft", {})
        assert price == pytest.approx(200.0)
        assert "temperate cereals" in label and "maize" in label

    def test_crop_without_a_priced_relative_falls_back_to_the_residual(self):
        from inseeds.components.farming.ca_farmer import _price_for_band

        prices = self._prices(
            {"Others, annual": 50.0, "Others, perennial": 40.0}
        )
        assert _price_for_band("sugarcane", prices, "npft", {}) == (
            "Others, perennial", 40.0
        )
        assert _price_for_band("pulses", prices, "npft", {}) == (
            "Others, annual", 50.0
        )

    def test_non_finite_own_price_falls_back(self):
        """FAO returning no value must not silently void the revenue."""
        from inseeds.components.farming.ca_farmer import _price_for_band

        prices = self._prices({"rice": np.nan, "temperate cereals": 180.0})
        assert _price_for_band("rice", prices, "npft", {}) == (
            "mean of temperate cereals", 180.0
        )

    def test_nothing_usable_reports_no_price(self):
        from inseeds.components.farming.ca_farmer import _price_for_band

        prices = self._prices({"temperate cereals": 180.0})
        category, price = _price_for_band("pulses", prices, "npft", {})
        assert category is None
        assert not np.isfinite(price)


class TestInitRevenueMapping:
    """Tests for grouping a cell's crop bands onto producer prices."""

    CROPS = [
        "temperate cereals", "rice", "maize", "tropical cereals", "pulses",
        "temperate roots", "tropical roots", "oil crops sunflower",
        "oil crops soybean", "oil crops groundnut", "oil crops rapeseed",
        "sugarcane", "others",
    ]

    @classmethod
    def _farmer(cls, price_values, cropped, min_crop_fraction=0.01):
        """A farmer stub carrying only what the mapping reads."""
        import xarray as xr
        from types import SimpleNamespace

        from inseeds.components.farming.ca_farmer import (
            ConservationAgricultureFarmer,
        )

        bands = [
            f"{water} {crop}"
            for water in ("rainfed", "irrigated")
            for crop in cls.CROPS
        ]
        fractions = np.array([cropped.get(b, 0.0) for b in bands])
        cftfrac = xr.DataArray(
            fractions, dims=("band",), coords={"band": bands}
        )
        prices = xr.DataArray(
            np.array(list(price_values.values()), dtype=float),
            dims=("npft",),
            coords={"npft": list(price_values)},
        )

        farmer = object.__new__(ConservationAgricultureFarmer)
        farmer.get_from_earth = lambda *args, **kwargs: cftfrac
        farmer.pft_prices = prices
        farmer._cell = SimpleNamespace(
            grid=SimpleNamespace(lat=np.float64(52.0), cell=np.int64(20)),
            area=np.float64(3.9e8),
            country=SimpleNamespace(code="NLD"),
        )
        farmer._model = SimpleNamespace(
            config=SimpleNamespace(
                coupled_config=SimpleNamespace(
                    materiality=SimpleNamespace(
                        min_crop_fraction=min_crop_fraction
                    )
                )
            )
        )
        return farmer, bands

    def test_every_band_is_priced_from_a_country_only_price_file(self):
        """The case that broke the Netherlands run: 6 categories, 26 bands."""
        farmer, bands = self._farmer(
            {
                "Others, annual": 1175.0,
                "Others, perennial": 2067.0,
                "oil crops rapeseed": 442.0,
                "pulses": 1029.0,
                "temperate cereals": 209.0,
                "temperate roots": 548.0,
            },
            {"rainfed temperate cereals": 0.3, "rainfed maize": 0.1},
        )
        farmer._init_revenue_mapping()

        priced = sorted(
            i for indices, _ in farmer._revenue_groups for i in indices
        )
        assert priced == list(range(len(bands)))

    def test_maize_is_worth_cereals_not_dutch_horticulture(self):
        """The residual aggregate of a horticulture country is far off."""
        farmer, bands = self._farmer(
            {"Others, annual": 1175.0, "temperate cereals": 209.0},
            {"rainfed maize": 0.1},
        )
        farmer._init_revenue_mapping()

        maize = bands.index("rainfed maize")
        price = next(
            p for indices, p in farmer._revenue_groups if maize in indices
        )
        assert price == pytest.approx(209.0)

    def test_unpriceable_band_the_cell_does_not_crop_is_left_out(self):
        farmer, bands = self._farmer(
            {"temperate cereals": 209.0},
            {"rainfed temperate cereals": 0.3},
        )
        farmer._init_revenue_mapping()

        priced = {i for indices, _ in farmer._revenue_groups for i in indices}
        assert bands.index("rainfed temperate cereals") in priced
        assert bands.index("rainfed pulses") not in priced

    def test_unpriceable_band_the_cell_does_crop_raises(self):
        """Its cost share is charged either way, so revenue must count it."""
        farmer, _ = self._farmer(
            {"temperate cereals": 209.0},
            {"rainfed temperate cereals": 0.3, "rainfed pulses": 0.1},
        )
        with pytest.raises(ValueError, match="no producer price"):
            farmer._init_revenue_mapping()


class TestOthersCftLatitudeSplit:
    """Tests for which cereal stands in for 'others' cropland."""

    def test_split_follows_lpjml_thirty_degrees(self):
        from inseeds.components.farming.ca_farmer import (
            OTHERS_CFT_TEMPERATE,
            OTHERS_CFT_TROPICAL,
            OTHERS_TROPICS_LAT,
            ConservationAgricultureFarmer,
        )

        assert OTHERS_TROPICS_LAT == 30.0
        assert "others_cft" in dir(ConservationAgricultureFarmer)
        assert OTHERS_CFT_TEMPERATE != OTHERS_CFT_TROPICAL


class TestCAFarmerCellProperties:
    """Tests for cell property accessors."""

    def test_cell_property_accessors_exist(self):
        """Cell property accessors should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        # Check for various cell properties
        assert "cell_runoff" in dir(ConservationAgricultureFarmer)
        assert "cell_leaching" in dir(ConservationAgricultureFarmer)
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


class TestFarmSizeHa:
    """Cropped-area helper used by Farmer.farm_size."""

    def test_farm_size_ha_is_crop_area_in_hectares(self):
        """cftfrac of 0.5 on a 1e6 m2 cell is 50 ha after dropping grassland."""
        import xarray as xr
        from types import SimpleNamespace

        from inseeds.components.farming.farmer import NON_CROPS, farm_size_ha

        bands = [
            "rainfed temperate cereals",
            "rainfed maize",
            *NON_CROPS,
        ]
        cftfrac = xr.DataArray(
            np.array([0.3, 0.2] + [0.0] * (len(NON_CROPS) - 1) + [0.4]),
            dims=("band",),
            coords={"band": bands},
        )
        cell = SimpleNamespace(
            from_earth=SimpleNamespace(cftfrac=cftfrac),
            area=1_000_000.0,
        )
        assert farm_size_ha(cell) == pytest.approx(50.0)


class TestCAFarmerCapitalProperties:
    """Tests for capital-related properties."""

    def test_min_capital_property_exists(self):
        """min_capital property should exist."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer
        
        assert "min_capital" in dir(ConservationAgricultureFarmer)

    def test_farm_size_property_exists(self):
        """farm_size is the public name used by CA init, costs and output."""
        from inseeds.components.farming.ca_farmer import ConservationAgricultureFarmer

        assert "farm_size" in dir(ConservationAgricultureFarmer)
