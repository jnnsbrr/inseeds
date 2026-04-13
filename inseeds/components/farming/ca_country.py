"""Country entity with FAO economic data for Conservation Agriculture.

Provides country-level economic parameters (capital, prices, depreciation)
loaded from FAOSTAT. Data is cached at class level to minimize API calls.

Missing Data Handling
---------------------
Uses tiered fallback when country data is unavailable:
1. Same country with expanded time window
2. Mean from neighbouring countries
3. Global mean
"""

import numpy as np
import xarray as xr

from inseeds.components.farming.region import Country
from inseeds.components.data.fao import (
    FaoProducerPrices,
    FaoCapitalStock,
    FaoGrossProductionValue,
    get_ag_share_of_aff,
    get_value_with_fallback,
)


class CACountry(Country):
    """Country with FAO economic data for Conservation Agriculture.
    
    FAO data is loaded lazily on first property access and cached at
    class level for efficiency across multiple country instances.
    
    Attributes
    ----------
    depreciation_rate : float
        Annual depreciation rate δ = CFC / NCS
    investment_rate : float
        Structural investment rate i = GFCF / NCS
    initial_capital_per_ha : float
        Initial capital per hectare (USD/ha)
    pft_prices : xr.DataArray
        Producer prices by crop type (USD/tonne dry matter)
    """

    # Class-level cache for FAO datasets
    _fao_prices_ds = None
    _fao_capital_ds = None
    _fao_crop_share_ds = None

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    
    @classmethod
    def preload_fao_data(cls, sim_path, country_codes, reference_year=2020):
        """Pre-download FAO data for multiple countries.
        
        Call before init_countries() for multi-country simulations.
        This ensures all data is available before parallelization.
        
        Parameters
        ----------
        sim_path : Path
            Simulation directory for caching.
        country_codes : list[str]
            ISO3 codes to download (e.g., ["NLD", "DEU", "BEL"]).
        reference_year : int
            Reference year for data (default: 2020).
        """
        if not country_codes:
            raise ValueError("country_codes cannot be empty")

        print(f"Pre-loading FAO data for {len(country_codes)} countries...")

        if cls._fao_prices_ds is None:
            prices = FaoProducerPrices()
            cls._fao_prices_ds = xr.open_dataset(
                prices.ensure(sim_path, country_codes, reference_year, years_before=4)
            )

        if cls._fao_capital_ds is None:
            capital = FaoCapitalStock()
            cls._fao_capital_ds = xr.open_dataset(
                capital.ensure(sim_path, country_codes, reference_year, years_before=4)
            )

        if cls._fao_crop_share_ds is None:
            gpv = FaoGrossProductionValue()
            cls._fao_crop_share_ds = xr.open_dataset(
                gpv.ensure(sim_path, country_codes, reference_year, years_before=10)
            )

        print(f"FAO data loaded for: {country_codes}")

    def _ensure_fao_data(self):
        """Load FAO data lazily on first access."""
        if getattr(self, "_fao_loaded", False):
            return

        sim_path = self.model.config.sim_path
        country_code = self.country_code
        reference_year = getattr(
            self.model.config.coupled_config, "start_year", 2020
        )

        # Get neighbour codes for fallback
        self._neighbour_codes = [
            n.country_code for n in getattr(self, "neighbourhood", set())
            if hasattr(n, "country_code")
        ]

        # Load datasets if not preloaded
        codes = [country_code]
        
        if CACountry._fao_prices_ds is None:
            prices = FaoProducerPrices()
            CACountry._fao_prices_ds = xr.open_dataset(
                prices.ensure(sim_path, codes, reference_year, years_before=4)
            )

        if CACountry._fao_capital_ds is None:
            capital = FaoCapitalStock()
            CACountry._fao_capital_ds = xr.open_dataset(
                capital.ensure(sim_path, codes, reference_year, years_before=4)
            )

        if CACountry._fao_crop_share_ds is None:
            gpv = FaoGrossProductionValue()
            CACountry._fao_crop_share_ds = xr.open_dataset(
                gpv.ensure(sim_path, codes, reference_year, years_before=10)
            )

        # Extract values for this country
        self._extract_capital_parameters(country_code, reference_year)
        self._extract_prices(country_code, reference_year)
        self._fao_loaded = True

    # -------------------------------------------------------------------------
    # Parameter Extraction
    # -------------------------------------------------------------------------
    
    def _extract_capital_parameters(self, country_code, reference_year):
        """Extract capital stock parameters with tiered fallback."""
        ds = CACountry._fao_capital_ds
        neighbours = self._neighbour_codes

        # Depreciation rate: δ = CFC / NCS
        result = get_value_with_fallback(
            ds["depreciation_rate"], country_code, reference_year,
            neighbour_codes=neighbours, aggregator="mean"
        )
        self._depreciation_rate = result.value

        # Investment rate: i = GFCF / NCS
        result = get_value_with_fallback(
            ds["investment_rate"], country_code, reference_year,
            neighbour_codes=neighbours, aggregator="mean"
        )
        self._investment_rate_value = result.value

        # Net Capital Stocks (most recent value)
        result = get_value_with_fallback(
            ds["ncs"], country_code, reference_year,
            neighbour_codes=neighbours, aggregator="last"
        )
        ncs_million_usd = result.value

        # Compute crop capital share
        qv_ds = CACountry._fao_crop_share_ds
        
        # Agriculture share of Ag+Forestry+Fishing (static table)
        ag_share = get_ag_share_of_aff(country_code)
        self._ag_share_of_aff = ag_share

        # Crop share of agriculture (from GPV data)
        crops = get_value_with_fallback(
            qv_ds["gpv_crops"], country_code, reference_year,
            neighbour_codes=neighbours, max_lookback=20, aggregator="mean"
        )
        ag = get_value_with_fallback(
            qv_ds["gpv_agriculture"], country_code, reference_year,
            neighbour_codes=neighbours, max_lookback=20, aggregator="mean"
        )
        crop_share = crops.value / ag.value
        self._crop_share_of_ag = crop_share

        # Total crop capital share
        self._crop_capital_share = ag_share * crop_share

        # Capital per hectare: (NCS × crop_share) / cropland_area
        crop_ncs_usd = ncs_million_usd * 1e6 * self._crop_capital_share
        self._initial_capital_per_ha = crop_ncs_usd / self.cropland_area

    def _extract_prices(self, country_code, reference_year):
        """Extract producer prices with tiered fallback per crop."""
        ds = CACountry._fao_prices_ds
        var_name = list(ds.data_vars)[0]
        prices = ds[var_name]
        neighbours = self._neighbour_codes

        crop_dim = "npft" if "npft" in prices.dims else "band"
        crops = prices[crop_dim].values

        # Get price for each crop with fallback
        values = []
        for crop in crops:
            crop_data = prices.sel({crop_dim: crop})
            try:
                result = get_value_with_fallback(
                    crop_data, country_code, reference_year,
                    neighbour_codes=neighbours, aggregator="mean"
                )
                values.append(result.value)
            except ValueError:
                values.append(np.nan)

        # Create result array
        result = xr.DataArray(data=values, dims=[crop_dim], coords={crop_dim: crops})
        
        # Rename to 'band' for LPJmL consistency
        if "npft" in result.dims:
            result = result.rename({"npft": "band"})
        
        self._pft_prices = result

    # -------------------------------------------------------------------------
    # Properties (lazy-loaded)
    # -------------------------------------------------------------------------
    
    def _get_fao_attr(self, attr_name):
        """Helper to get FAO attribute with lazy loading."""
        self._ensure_fao_data()
        return getattr(self, attr_name)

    @property
    def depreciation_rate(self):
        """Annual depreciation rate δ = CFC / NCS (typically 5-10%)."""
        return self._get_fao_attr("_depreciation_rate")

    @property
    def _investment_rate(self):
        """Structural investment rate i = GFCF / NCS."""
        return self._get_fao_attr("_investment_rate_value")

    @property
    def initial_capital_per_ha(self):
        """Initial capital per hectare (USD/ha), scaled for crops only."""
        return self._get_fao_attr("_initial_capital_per_ha")

    @property
    def crop_capital_share(self):
        """Fraction of agricultural capital for field crops (0-1)."""
        return self._get_fao_attr("_crop_capital_share")

    @property
    def ag_share_of_aff(self):
        """Agriculture share of Ag+Forestry+Fishing sector (0-1)."""
        return self._get_fao_attr("_ag_share_of_aff")

    @property
    def crop_share_of_ag(self):
        """Crop share of agriculture (crops vs livestock, 0-1)."""
        return self._get_fao_attr("_crop_share_of_ag")

    @property
    def pft_prices(self):
        """Producer prices by crop type (USD/tonne dry matter)."""
        return self._get_fao_attr("_pft_prices")
