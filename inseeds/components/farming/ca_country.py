"""Conservation Agriculture Country with FAO economic data.

This module provides the CACountry class that loads FAO data once per country
(not per farmer) to minimize API calls and provide country-level economic
parameters for Conservation Agriculture farmers.
"""

import xarray as xr

from inseeds.components.farming.region import Country
from inseeds.components.data.fao import (
    FaoProducerPrices,
    FaoCapitalStock,
    FaoGrossProductionValue,
    get_ag_share_of_aff,
    get_crop_share_from_qv,
)


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
    _fao_crop_share_ds = None  # Class-level cache for FAO crop share dataset

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

        # Load FAO datasets - download ALL countries for global averages
        # This ensures we have price data for all crops, using global averages
        # when country-specific data is missing (e.g., NL doesn't report maize)
        # Cached at class level to avoid repeated API calls
        if CACountry._fao_prices_ds is None:
            prices = FaoProducerPrices()
            # Download ALL countries (country_codes=None) to get global coverage
            # This is essential for filling gaps in country-specific data
            CACountry._fao_prices_ds = xr.open_dataset(
                prices.ensure(
                    sim_path,
                    country_codes=None,  # Download all countries for global averages
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

        # Load FAO QV data for crop shares (REQUIRED - no fallback)
        if CACountry._fao_crop_share_ds is None:
            crop_share = FaoGrossProductionValue()
            crop_share_path = crop_share.ensure(
                sim_path,
                country_codes=None,  # All countries
                reference_year=reference_year,
                years_before=4,
            )
            CACountry._fao_crop_share_ds = xr.open_dataset(crop_share_path)

        # Extract country-specific values
        self._extract_capital_parameters(country_code, reference_year)
        self._extract_prices(country_code)

        # Mark as loaded (instance attribute)
        self._fao_loaded = True

    def _extract_capital_parameters(self, country_code, reference_year=None):
        """Extract capital stock parameters for this country.

        Uses mean over available years for robustness against year-to-year
        fluctuations.

        Capital is scaled by crop_capital_share to account for the fact that
        FAO NCS includes all agricultural capital (crops + livestock + forestry
        + fishing), but LPJmL only simulates field crops.

        The scaling factor has two components:
        1. ag_share: Agriculture share of Ag+Forestry+Fishing (static table)
        2. crop_share: Crop share of agriculture (from FAO QV data)

        Final: crop_capital_share = ag_share × crop_share

        See inseeds/components/data/fao/crop_capital_share.py for methodology.
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

        # Get crop capital share for this country
        # Two components: ag_share (static) × crop_share (from FAO QV)
        qv_ds = CACountry._fao_crop_share_ds

        # Agriculture share of Ag+Forestry+Fishing (static table)
        ag_share = get_ag_share_of_aff(country_code)
        self._ag_share_of_aff = ag_share

        # Crop share of agriculture (from real FAO QV data - no fallback)
        crop_share = get_crop_share_from_qv(qv_ds, country_code, year=reference_year)
        self._crop_share_of_ag = crop_share

        # Total crop capital share
        crop_capital_share = ag_share * crop_share
        self._crop_capital_share = crop_capital_share

        # Convert: million USD → USD, apply crop share, then divide by hectares
        # crop_capital = total_NCS × ag_share × crop_share
        crop_ncs_usd = ncs_million_usd * 1e6 * crop_capital_share
        self._initial_capital_per_ha = crop_ncs_usd / cropland_ha

    def _extract_prices(self, country_code):
        """Extract producer prices for this country.

        Uses mean over available years for robustness.
        For crops without country-specific prices, uses global FAO average.
        Renames 'npft' dimension to 'band' for consistency with LPJmL.

        Note: FAO Producer Prices only contains data for crops that a country
        actually reports. For example, Netherlands doesn't report maize grain
        prices (they grow silage maize for feed). To ensure complete price
        coverage, we download global data and fill gaps with global averages.
        """
        ds = CACountry._fao_prices_ds
        var_name = list(ds.data_vars)[0]
        prices = ds[var_name]

        # Average over time dimension for robustness
        if "time" in prices.dims:
            prices = prices.mean(dim="time")

        # Get all available crop types from the dataset
        crop_dim = "npft" if "npft" in prices.dims else "band"
        all_crops = prices[crop_dim].values

        # Compute global average prices (across all countries)
        if "area_code" in prices.dims:
            global_avg = prices.mean(dim="area_code")
        else:
            global_avg = prices

        # Check if dataset has country dimension and country exists
        if "area_code" not in prices.dims:
            result = prices
        elif country_code in prices.area_code.values:
            # Get country-specific prices
            country_prices = prices.sel(area_code=country_code)

            # Fill NaN values (missing crops) with global average
            # This handles cases where a country doesn't report prices for
            # certain crops (e.g., Netherlands doesn't report maize prices)
            result = country_prices.fillna(global_avg)
        else:
            # Country not in dataset - use global average
            result = global_avg

        # Rename 'npft' to 'band' for consistency with LPJmL
        if "npft" in result.dims:
            result = result.rename({"npft": "band"})

        self._pft_prices = result

    # Properties that trigger lazy loading
    @property
    def depreciation_rate(self):
        """Annual depreciation rate δ = CFC / NCS.

        Note: This is the depreciation rate for the combined "Agriculture,
        Forestry and Fishing" sector from FAO CS domain. Since agriculture
        dominates this sector (85-95% in most countries), we apply this rate
        to crop capital as well.

        The rate is computed as: CFC (Consumption of Fixed Capital) / NCS
        (Net Capital Stocks), averaged over available years for robustness.

        Typical values: 5-10% per year depending on capital composition
        (machinery depreciates faster than land improvements).
        """
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
        """Initial capital per hectare from (NCS × crop_share) / cropland_area.

        Note: Capital is scaled by crop_capital_share to match LPJmL's field
        crop simulation. See crop_capital_share property for the scaling factor.
        """
        try:
            self._ensure_fao_data()
            return self._initial_capital_per_ha
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def crop_capital_share(self):
        """Fraction of agricultural capital attributed to field crops.

        This scaling factor converts total FAO NCS (which includes crops,
        livestock, forestry, fishing) to crop-specific capital that matches
        LPJmL's field crop simulation.

        Computed as: ag_share_of_aff × crop_share_of_ag

        Where:
        - ag_share_of_aff: Agriculture share of Ag+Forestry+Fishing (static table)
        - crop_share_of_ag: Crop share of agriculture (from FAO QV data)

        Returns
        -------
        float
            Crop capital share in [0, 1].
        """
        try:
            self._ensure_fao_data()
            return self._crop_capital_share
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def ag_share_of_aff(self):
        """Agriculture share of Ag+Forestry+Fishing.

        From static table based on World Bank and FAO data.
        Varies by country: ~60% for Norway (major fishing/forestry)
        to ~98% for landlocked countries.

        Returns
        -------
        float
            Agriculture share in [0, 1].
        """
        try:
            self._ensure_fao_data()
            return self._ag_share_of_aff
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to load FAO data for {self.code}: {e}. "
                "Ensure model is fully initialized before accessing FAO data."
            ) from e

    @property
    def crop_share_of_ag(self):
        """Crop share of agriculture (crops vs livestock).

        Computed from FAO Gross Production Value (QV domain):
        crop_share = GPV_crops / GPV_agriculture

        Returns
        -------
        float
            Crop share in [0, 1].
        """
        try:
            self._ensure_fao_data()
            return self._crop_share_of_ag
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
