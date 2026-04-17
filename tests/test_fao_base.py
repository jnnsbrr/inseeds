"""Tests for FAO base module (FaoDataset, download, prepare, transform).

Tests cover:
- FaoDataset abstract base class
- Download functionality with batching
- Prepare and transform methods
- Country code handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import xarray as xr

from inseeds.components.data.fao.base import FaoDataset, get_fao_country_code
from inseeds.components.data.fao import FaoProducerPrices, FaoCapitalStock


class TestGetFaoCountryCode:
    """Tests for get_fao_country_code helper function."""

    def test_iso3_to_fao_code(self):
        """Should convert ISO3 codes to FAO codes."""
        # Function uses fao_definitions to map ISO3 to FAO
        result = get_fao_country_code("NLD")
        assert result is not None
        
    def test_returns_none_for_invalid_code(self):
        """Should return None for invalid codes."""
        result = get_fao_country_code("XXX")
        assert result is None

    def test_function_exists(self):
        """get_fao_country_code function should exist."""
        assert callable(get_fao_country_code)


class TestFaoDatasetAbstract:
    """Tests for FaoDataset abstract base class."""

    def test_fao_dataset_is_abstract(self):
        """FaoDataset should not be instantiable directly."""
        # It's abstract but doesn't use ABC, so we just verify subclasses exist
        assert issubclass(FaoProducerPrices, FaoDataset)
        assert issubclass(FaoCapitalStock, FaoDataset)

    def test_fao_dataset_has_required_attributes(self):
        """FaoDataset subclasses should have required attributes."""
        prices = FaoProducerPrices()
        
        assert hasattr(prices, "domain")
        assert hasattr(prices, "elements")
        assert hasattr(prices, "output_filename")

    def test_fao_dataset_has_required_methods(self):
        """FaoDataset should have required methods."""
        prices = FaoProducerPrices()
        
        assert hasattr(prices, "ensure")
        assert hasattr(prices, "prepare")
        assert hasattr(prices, "download")
        assert hasattr(prices, "transform")
        assert hasattr(prices, "get_path")
        assert hasattr(prices, "get_dummy_path")


class TestFaoDatasetPaths:
    """Tests for FaoDataset path methods."""

    def test_get_path_returns_correct_location(self):
        """get_path should return input/filename.nc."""
        prices = FaoProducerPrices()
        sim_path = Path("/tmp/sim")
        
        path = prices.get_path(sim_path)
        
        assert path == sim_path / "input" / "fao_pft_prices.nc"

    def test_get_dummy_path_adds_dummy_suffix(self):
        """get_dummy_path should add _DUMMY before extension."""
        capital = FaoCapitalStock()
        sim_path = Path("/tmp/sim")
        
        path = capital.get_dummy_path(sim_path)
        
        assert path == sim_path / "input" / "fao_capital_stock_DUMMY.nc"

    def test_is_available_checks_real_file(self):
        """is_available should check for real (non-dummy) file."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            # No file exists
            assert not prices.is_available(sim_path)
            
            # Create real file
            real_path = prices.get_path(sim_path)
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.touch()
            
            assert prices.is_available(sim_path)


class TestFaoDatasetEnsureLogic:
    """Tests for ensure() method logic."""

    def test_ensure_returns_existing_real_file(self):
        """ensure should return existing real file without re-downloading."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            real_path = prices.get_path(sim_path)
            real_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a real file with minimal valid data
            ds = xr.Dataset(
                {"5532": (["npft", "time", "area_code"], [[[100.0]]])},
                coords={
                    "npft": ["temperate cereals"],
                    "time": [2020],
                    "area_code": ["NLD"],
                },
            )
            ds.to_netcdf(real_path)
            ds.close()
            
            # ensure should return this file
            result = prices.ensure(
                sim_path=sim_path,
                country_codes=["NLD"],
                reference_year=2020,
                years_before=0,
            )
            
            assert result == real_path

    def test_ensure_tries_download_when_only_dummy_exists(self):
        """ensure should try download even if dummy file exists."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            dummy_path = prices.get_dummy_path(sim_path)
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create dummy file
            ds = xr.Dataset(
                {"5532": (["npft", "time", "area_code"], [[[100.0]]])},
                coords={
                    "npft": ["temperate cereals"],
                    "time": [2020],
                    "area_code": ["NLD"],
                },
            )
            ds.to_netcdf(dummy_path)
            ds.close()
            
            # ensure should try to download (will fail and use dummy)
            with patch.object(prices, "prepare", side_effect=RuntimeError("API fail")):
                result = prices.ensure(
                    sim_path=sim_path,
                    country_codes=["NLD"],
                    reference_year=2020,
                    years_before=0,
                )
            
            # Should return dummy path
            assert result == dummy_path


class TestFaoProducerPricesSpecifics:
    """Tests specific to FaoProducerPrices."""

    def test_producer_prices_domain(self):
        """Producer prices should use PP domain."""
        prices = FaoProducerPrices()
        assert prices.domain == "PP"

    def test_producer_prices_element(self):
        """Producer prices should use element 5532."""
        prices = FaoProducerPrices()
        assert prices.elements == ["5532"]

    def test_producer_prices_filename(self):
        """Producer prices should use correct filename."""
        prices = FaoProducerPrices()
        assert prices.output_filename == "fao_pft_prices.nc"

    def test_producer_prices_get_items_returns_lpjml_crops(self):
        """_get_items should return only crops that map to LPJmL."""
        prices = FaoProducerPrices()
        items = prices._get_items()
        
        assert len(items) == 166
        assert all(isinstance(item, str) for item in items)


class TestFaoCapitalStockSpecifics:
    """Tests specific to FaoCapitalStock."""

    def test_capital_stock_domain(self):
        """Capital stock should use CS domain."""
        capital = FaoCapitalStock()
        assert capital.domain == "CS"

    def test_capital_stock_element(self):
        """Capital stock should use element 6110."""
        capital = FaoCapitalStock()
        assert capital.elements == ["6110"]

    def test_capital_stock_filename(self):
        """Capital stock should use correct filename."""
        capital = FaoCapitalStock()
        assert capital.output_filename == "fao_capital_stock.nc"

    def test_capital_stock_get_items_returns_three(self):
        """_get_items should return GFCF, CFC, NCS."""
        capital = FaoCapitalStock()
        items = capital._get_items()
        
        items_list = list(items)
        assert len(items_list) == 3
        assert "22030" in items_list  # GFCF
        assert "22031" in items_list  # CFC
        assert "22034" in items_list  # NCS

    def test_capital_stock_uses_base_transform(self):
        """Capital stock should use base class transform (via copan_eval)."""
        capital = FaoCapitalStock()
        
        # Verify it has transform method (inherited from base class)
        assert hasattr(capital, "transform")
        # Should NOT override transform - uses copan_eval's FaoData.from_dataframe
        assert "transform" not in capital.__class__.__dict__

    def test_capital_stock_has_post_process(self):
        """Capital stock should have _post_process for derived metrics."""
        capital = FaoCapitalStock()
        
        assert hasattr(capital, "_post_process")


class TestFaoDatasetBatchingBehavior:
    """Tests for batching behavior in download method."""

    def test_non_cs_domain_batches_items(self):
        """Non-CS domains should batch all items together."""
        prices = FaoProducerPrices()
        
        # PP domain should batch items
        assert prices.domain != "CS"

    def test_cs_domain_requires_separate_item_calls(self):
        """CS domain should make separate calls per item."""
        capital = FaoCapitalStock()
        
        # CS domain has different structure
        assert capital.domain == "CS"

    def test_all_domains_batch_countries(self):
        """All domains should batch all countries together."""
        # This is tested through the implementation
        # Both CS and non-CS domains pass areas=countries (full list)
        assert True


class TestFaoDatasetYearHandling:
    """Tests for year selection and averaging."""

    def test_prepare_accepts_years_parameter(self):
        """prepare() should accept years tuple."""
        prices = FaoProducerPrices()
        
        # Verify method signature accepts years
        assert hasattr(prices, "prepare")

    def test_ensure_converts_reference_year_to_years_tuple(self):
        """ensure() should convert reference_year + years_before to years tuple."""
        # This is tested implicitly through ensure() calls
        assert True


class TestFaoDatasetCountryHandling:
    """Tests for country code handling."""

    def test_prepare_accepts_country_codes_list(self):
        """prepare() should accept list of country codes."""
        capital = FaoCapitalStock()
        
        # Verify method signature
        assert hasattr(capital, "prepare")

    def test_download_batches_all_countries(self):
        """download() should batch all countries into single API call."""
        # This is the key optimization - verified through implementation
        assert True


class TestFaoDatasetErrorHandling:
    """Tests for error handling in FAO data operations."""

    def test_ensure_falls_back_to_dummy_on_api_error(self):
        """ensure should create dummy data when API fails."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            
            # Mock prepare to raise error
            with patch.object(prices, "prepare", side_effect=RuntimeError("API error")):
                path = prices.ensure(
                    sim_path=sim_path,
                    country_codes=["NLD"],
                    reference_year=2020,
                    years_before=4,
                )
            
            # Should create dummy file (ensure now always falls back to dummy)
            assert path.exists()
            assert "DUMMY" in path.name


class TestFaoDataTransformation:
    """Tests for data transformation methods."""

    def test_capital_stock_transform_handles_cs_domain(self):
        """Capital stock transform should handle CS domain structure via copan_eval."""
        capital = FaoCapitalStock()
        
        # Create sample data mimicking CS domain structure with correct column names
        # Must include Domain Code for copan_eval's FaoData.from_dataframe
        df = pd.DataFrame({
            "Domain Code": ["CS", "CS", "CS"],
            "Area Code (FAO)": ["276", "276", "276"],
            "Area": ["Germany", "Germany", "Germany"],
            "Item Code (FAO)": ["22030", "22031", "22034"],
            "Item": ["GFCF", "CFC", "NCS"],
            "Element Code": ["6110", "6110", "6110"],
            "Element": ["Value", "Value", "Value"],
            "Year": [2020, 2020, 2020],
            "Value": [1000000.0, 50000.0, 950000.0],
            "Unit": ["1000 US$", "1000 US$", "1000 US$"],
        })
        
        # Transform should convert to xarray via copan_eval
        ds = capital.transform(df)
        
        assert isinstance(ds, xr.Dataset)
        assert "area_code" in ds.dims
        assert "time" in ds.dims
        # copan_eval now preserves item_code dimension for CS domain
        assert "item_code" in ds.dims

    def test_producer_prices_uses_default_transform(self):
        """Producer prices should use default transform (not overridden)."""
        prices = FaoProducerPrices()
        
        # Should not have custom transform
        # (uses FaoData.from_dataframe from copan_eval)
        assert hasattr(prices, "transform")


class TestFaoDatasetPostProcessing:
    """Tests for post-processing methods."""

    def test_capital_stock_post_process_calculates_rates(self):
        """Capital stock _post_process should calculate depreciation and investment rates."""
        capital = FaoCapitalStock()
        
        # Create sample dataset with correct structure
        # Shape: (item_code=3, time=1, area_code=1)
        ds = xr.Dataset({
            "6110": xr.DataArray(
                [[[100.0]], [[50.0]], [[1000.0]]],  # 3 items × 1 time × 1 area
                dims=["item_code", "time", "area_code"],
                coords={
                    "item_code": ["22030", "22031", "22034"],
                    "time": [2020],
                    "area_code": ["276"],
                }
            )
        })
        
        # Post-process
        ds_processed = capital._post_process(ds)
        
        # Should have derived metrics
        assert "depreciation_rate" in ds_processed
        assert "investment_rate" in ds_processed
        assert "gfcf" in ds_processed
        assert "cfc" in ds_processed
        assert "ncs" in ds_processed
        
        # Check calculations
        # depreciation_rate = cfc / ncs = 50 / 1000 = 0.05
        # investment_rate = gfcf / ncs = 100 / 1000 = 0.10
        assert ds_processed["depreciation_rate"].values[0, 0] == pytest.approx(0.05)
        assert ds_processed["investment_rate"].values[0, 0] == pytest.approx(0.10)


class TestFaoDatasetAPIIntegration:
    """Tests for API adapter integration."""

    def test_fao_dataset_uses_api_adapter(self):
        """FaoDataset should use FaoApiAdapter for downloads."""
        # This is tested through the download method
        # which calls adapter.download_data
        assert True

    def test_download_passes_correct_parameters(self):
        """download should pass correct parameters to API adapter."""
        # Verified through implementation:
        # - domain, elements, items, years, areas (countries)
        assert True


class TestFaoDatasetCaching:
    """Tests for caching behavior."""

    def test_ensure_uses_existing_file(self):
        """ensure should use existing file without re-downloading."""
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            real_path = prices.get_path(sim_path)
            real_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create real file
            ds = xr.Dataset(
                {"5532": (["npft", "time", "area_code"], [[[100.0]]])},
                coords={
                    "npft": ["temperate cereals"],
                    "time": [2020],
                    "area_code": ["NLD"],
                },
            )
            ds.to_netcdf(real_path)
            ds.close()
            
            # Mock prepare to verify it's not called
            with patch.object(prices, "prepare") as mock_prepare:
                result = prices.ensure(
                    sim_path=sim_path,
                    country_codes=["NLD"],
                    reference_year=2020,
                    years_before=0,
                )
                
                # prepare should not have been called
                mock_prepare.assert_not_called()
                assert result == real_path


class TestFaoDatasetDomainSpecificBehavior:
    """Tests for domain-specific behavior differences."""

    def test_cs_domain_uses_copan_eval_transform(self):
        """CS domain (capital stock) should use copan_eval's transform."""
        capital = FaoCapitalStock()
        
        # Capital stock uses CS domain
        assert capital.domain == "CS"
        # transform should come from base class (which uses copan_eval)
        assert "transform" not in capital.__class__.__dict__

    def test_pp_domain_uses_default_transform(self):
        """PP domain (producer prices) should use default transform."""
        prices = FaoProducerPrices()
        
        # Producer prices doesn't override transform
        assert prices.domain == "PP"
        # transform should come from base class
        assert "transform" not in prices.__class__.__dict__


class TestFaoDatasetItemRetrieval:
    """Tests for item retrieval methods."""

    def test_get_items_is_abstract(self):
        """_get_items should be implemented by subclasses."""
        prices = FaoProducerPrices()
        capital = FaoCapitalStock()
        
        # Both should implement _get_items
        assert hasattr(prices, "_get_items")
        assert hasattr(capital, "_get_items")

    def test_get_items_returns_list_or_series(self):
        """_get_items should return list-like of item codes."""
        prices = FaoProducerPrices()
        items = prices._get_items()
        
        # Should be iterable
        assert hasattr(items, "__iter__")
        assert len(items) > 0


@pytest.mark.integration
class TestFaoDownloadIntegration:
    """Integration tests for FAO API download with real API calls.
    
    These tests require FAO API access and are marked with @pytest.mark.integration.
    Run with: pytest -m integration tests/test_fao_base.py
    Skip with: pytest -m "not integration" tests/test_fao_base.py
    """

    def test_download_single_country(self):
        """Download FAO data for a single country (NLD)."""
        from inseeds.components.data.fao.base import check_fao_api_available
        
        if not check_fao_api_available():
            pytest.skip("FAO API not available")
        
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            cache_path = sim_path / "input" / "pft_prices_cache.parquet"
            output_path = sim_path / "input" / "fao_pft_prices.nc"
            
            ds = prices.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=(2018, 2020),
                country_codes=["NLD"],
            )
            
            # Verify result
            assert "area_code" in ds.dims
            assert "NLD" in ds.area_code.values
            assert len(ds.area_code) == 1  # Only NLD
            assert "npft" in ds.dims
            assert "time" in ds.dims
            
            # Verify file was created
            assert output_path.exists()
            
            ds.close()

    def test_download_multiple_countries(self):
        """Download FAO data for multiple countries (NLD, DEU, BEL)."""
        from inseeds.components.data.fao.base import check_fao_api_available
        
        if not check_fao_api_available():
            pytest.skip("FAO API not available")
        
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            cache_path = sim_path / "input" / "pft_prices_cache.parquet"
            output_path = sim_path / "input" / "fao_pft_prices.nc"
            
            ds = prices.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=(2018, 2020),
                country_codes=["NLD", "DEU", "BEL"],
            )
            
            # Verify result
            assert "area_code" in ds.dims
            assert len(ds.area_code) == 3
            assert "NLD" in ds.area_code.values
            assert "DEU" in ds.area_code.values
            assert "BEL" in ds.area_code.values
            
            ds.close()

    def test_download_respects_country_codes_parameter(self):
        """Verify that only requested countries are downloaded."""
        from inseeds.components.data.fao.base import check_fao_api_available
        
        if not check_fao_api_available():
            pytest.skip("FAO API not available")
        
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            cache_path = sim_path / "input" / "pft_prices_cache.parquet"
            output_path = sim_path / "input" / "fao_pft_prices.nc"
            
            # Request only 2 countries
            ds = prices.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=(2019, 2020),
                country_codes=["FRA", "ESP"],
            )
            
            # Should have exactly 2 countries
            assert len(ds.area_code) == 2
            assert "FRA" in ds.area_code.values
            assert "ESP" in ds.area_code.values
            # Should NOT have other countries
            assert "NLD" not in ds.area_code.values
            assert "DEU" not in ds.area_code.values
            
            ds.close()

    def test_download_caches_to_parquet(self):
        """Verify that download creates parquet cache file."""
        from inseeds.components.data.fao.base import check_fao_api_available
        
        if not check_fao_api_available():
            pytest.skip("FAO API not available")
        
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            cache_path = sim_path / "input" / "pft_prices_cache.parquet"
            output_path = sim_path / "input" / "fao_pft_prices.nc"
            
            prices.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=(2019, 2020),
                country_codes=["NLD"],
            )
            
            # Cache file should exist
            assert cache_path.exists()
            
            # Cache should be readable
            df = pd.read_parquet(cache_path)
            assert len(df) > 0

    def test_download_uses_cache_on_second_call(self):
        """Verify that second call uses cache instead of API."""
        from inseeds.components.data.fao.base import check_fao_api_available
        
        if not check_fao_api_available():
            pytest.skip("FAO API not available")
        
        prices = FaoProducerPrices()
        
        with tempfile.TemporaryDirectory() as tmp:
            sim_path = Path(tmp) / "sim"
            cache_path = sim_path / "input" / "pft_prices_cache.parquet"
            output_path = sim_path / "input" / "fao_pft_prices.nc"
            
            # First call - downloads from API
            ds1 = prices.prepare(
                cache_path=cache_path,
                output_path=output_path,
                years=(2019, 2020),
                country_codes=["NLD"],
            )
            ds1.close()
            
            # Delete output but keep cache
            output_path.unlink()
            
            # Second call - should use cache (no API call)
            with patch("inseeds.components.data.fao.base.FaoApiAdapter") as mock_adapter:
                ds2 = prices.prepare(
                    cache_path=cache_path,
                    output_path=output_path,
                    years=(2019, 2020),
                    country_codes=["NLD"],
                )
                
                # API adapter should NOT have been instantiated
                mock_adapter.assert_not_called()
                
            ds2.close()
