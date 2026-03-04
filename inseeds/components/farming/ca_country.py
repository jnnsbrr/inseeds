"""Conservation Agriculture Country with FAO economic data.

This module provides the CACountry class that loads FAO data once per country
(not per farmer) to minimize API calls and provide country-level economic
parameters for Conservation Agriculture farmers.
"""

import xarray as xr

from inseeds.components.farming.region import Country
from inseeds.components.data.fao import FaoProducerPrices, FaoCapitalStock


class CACountry(Country):
    """Country with FAO economic data for Conservation Agriculture farmers.

    Loads FAO data once for all countries to minimize API calls.
    Provides country-level economic parameters that farmers can access.

    FAO data is loaded lazily on first access to any economic parameter,
    since self.model is not available during __init__ (model is still being
    constructed at that point).

    For global runs, data for ALL countries is downloaded in a single batch
    to minimize API calls and avoid race conditions during parallelization.

    Attributes
    ----------
    depreciation_rate : float
        Annual depreciation rate δ = CFC / NCS
    investment_rate : float
        Structural investment rate i = GFCF / NCS
    initial_capital_per_ha : float
        Initial capital per hectare from NCS / cropland_area
    pft_prices : xr.DataArray
        Producer prices by crop type (USD/tonne dry matter)
    """

    _fao_prices_ds = None  # Class-level cache for FAO prices dataset
    _fao_capital_ds = None  # Class-level cache for FAO capital dataset

    def _ensure_fao_data(self):
        """Load FAO data if not already loaded (lazy initialization).

        Called automatically when any FAO-derived property is accessed.
        This is necessary because self.model is not available during __init__.

        For global runs, downloads data for ALL countries in the simulation
        to minimize API calls (20 calls total instead of 20 × N_countries).
        """
        # Check if already loaded using instance attribute
        if getattr(self, "_fao_loaded", False):
            return

        sim_path = self.model.config.sim_path
        country_code = self.country_code

        # Get reference year from config (simulation start year)
        try:
            reference_year = self.model.config.coupled_config.start_year
        except AttributeError:
            reference_year = 2020

        # Load FAO datasets - for ALL countries in simulation
        # Cached at class level to avoid repeated API calls
        if CACountry._fao_prices_ds is None:
            # Get all country codes from the model
            all_country_codes = [
                country.country_code for country in self.model.world.countries
            ]
            
            prices = FaoProducerPrices()
            CACountry._fao_prices_ds = xr.open_dataset(
                prices.ensure(
                    sim_path,
                    country_codes=all_country_codes,
                    reference_year=reference_year,
                    years_before=4,
                )
            )

        if CACountry._fao_capital_ds is None:
            # Get all country codes from the model
            all_country_codes = [
                country.country_code for country in self.model.world.countries
            ]
            
            capital = FaoCapitalStock()
            CACountry._fao_capital_ds = xr.open_dataset(
                capital.ensure(
                    sim_path,
                    country_codes=all_country_codes,
                    reference_year=reference_year,
                    years_before=4,
                )
            )

        # Extract country-specific values
        self._extract_capital_parameters(country_code)
        self._extract_prices(country_code)

        # Mark as loaded (instance attribute)
        self._fao_loaded = True

    def _extract_capital_parameters(self, country_code):
        """Extract capital stock parameters for this country.

        Uses mean over available years for robustness against year-to-year
        fluctuations.
        """
        ds = CACountry._fao_capital_ds

        # Depreciation rate: δ = CFC / NCS (mean over years for robustness)
        dep_rate = ds["depreciation_rate"]
        if country_code in dep_rate.area_code.values:
            self._depreciation_rate = float(
                dep_rate.sel(area_code=country_code).mean(dim="time").values
            )
        else:
            self._depreciation_rate = float(dep_rate.mean().values)

        # Investment rate: i = GFCF / NCS (mean over years)
        inv_rate = ds["investment_rate"]
        if country_code in inv_rate.area_code.values:
            self.__investment_rate = float(
                inv_rate.sel(area_code=country_code).mean(dim="time").values
            )
        else:
            self.__investment_rate = float(inv_rate.mean().values)

        # Initial capital per ha from NCS (use most recent year)
        ncs = ds["ncs"]  # Net Capital Stocks (million USD)
        cropland_ha = self.cropland_area

        if country_code in ncs.area_code.values:
            ncs_million_usd = float(
                ncs.sel(area_code=country_code).isel(time=-1).values
            )
        else:
            ncs_million_usd = float(ncs.isel(time=-1).mean().values)

        # Convert: million USD → USD, then divide by hectares
        self._initial_capital_per_ha = (ncs_million_usd * 1e6) / cropland_ha

    def _extract_prices(self, country_code):
        """Extract producer prices for this country.

        Uses mean over available years for robustness.
        """
        ds = CACountry._fao_prices_ds
        var_name = list(ds.data_vars)[0]
        prices = ds[var_name]

        # Average over time dimension for robustness
        if "time" in prices.dims:
            prices = prices.mean(dim="time")

        # Check if dataset has country dimension
        if "area_code" not in prices.dims:
            self._pft_prices = prices
            return

        # Use country-specific prices if available
        if country_code in prices.area_code.values:
            self._pft_prices = prices.sel(area_code=country_code)
        else:
            # Fall back to global mean
            self._pft_prices = prices.mean(dim="area_code")

    # Properties that trigger lazy loading
    @property
    def depreciation_rate(self):
        """Annual depreciation rate δ = CFC / NCS."""
        try:
            self._ensure_fao_data()
            return self._depreciation_rate
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def _investment_rate(self):
        """Structural investment rate i = GFCF / NCS."""
        try:
            self._ensure_fao_data()
            return self.__investment_rate
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def initial_capital_per_ha(self):
        """Initial capital per hectare from NCS / cropland_area."""
        try:
            self._ensure_fao_data()
            return self._initial_capital_per_ha
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def pft_prices(self):
        """Producer prices by crop type (USD/tonne dry matter)."""
        try:
            self._ensure_fao_data()
            return self._pft_prices
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e
