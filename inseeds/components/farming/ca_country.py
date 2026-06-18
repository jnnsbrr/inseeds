"""Country-level entity with FAO economic data for Conservation Agriculture.

This module defines the CACountry class, which extends the base Country class
with FAO economic data (capital, prices, depreciation) and country-level
social learning statistics.

Overview
--------
Each country:
1. Loads FAO economic data lazily (capital stock, producer prices, etc.)
2. Aggregates farmer performance by practice bundle each year
3. Optionally blends its statistics with neighboring countries

Data Sources
------------
- **FAO Capital Stock**: Depreciation rates, investment rates, net capital
- **FAO Producer Prices**: Farm-gate prices per crop type (PFT)
- **FAO Gross Production Value**: For calculating crop vs livestock shares

Fallback Logic
--------------
Missing FAO data is handled via tiered fallback:
    country → neighboring countries → global mean

This ensures all countries have usable parameters even with sparse FAO coverage.

Key Methods
-----------
compute_management_performance(t)
    Aggregates farmer data into RegionManagementPerformanceStore and
    optionally merges with neighboring country statistics.

Attributes (Lazy-Loaded)
------------------------
depreciation_rate : float
    Annual capital depreciation (typically 3-8%/year)
investment_rate : float
    Annual capital formation rate (typically 5-15%/year)
initial_capital_per_ha : float
    Starting capital endowment (USD/ha)
pft_prices : xarray.DataArray
    Producer prices per plant functional type (USD/tonne)

See Also
--------
ca_management : RegionManagementPerformanceStore definition
ca_agroecology : Cluster-level aggregation across countries
ca_behaviour : How farmers use country-level statistics
"""

import logging
import numpy as np
import xarray as xr

from inseeds.components.farming.region import Country
from inseeds.components.exogenous.faostat import get_ag_share_of_aff, get_value_with_fallback
from inseeds.components.exogenous import Exogenous
from inseeds.components.farming.ca_management import (
    RegionManagementPerformanceStore,
)

logger = logging.getLogger(__name__)


def _find_nearest_year(ds: xr.Dataset, target: int) -> int:
    """Find nearest available year <= target, or earliest if target is before all data."""
    if "time" not in ds.dims:
        return target
    years = sorted(int(y) for y in ds.time.values)
    if not years:
        return target
    valid = [y for y in years if y <= target]
    return max(valid) if valid else min(years)


def _extract_with_fallback(ds, var, country, year, neighbours, **kwargs):
    """Extract single value with fallback. Returns float."""
    return get_value_with_fallback(ds[var], country, year, neighbour_codes=neighbours, **kwargs).value


class CACountry(Country):
    """Country with FAO economic data for Conservation Agriculture."""

    _fao_reference_year = None  # Class-level cache
    _fao_loaded = False

    @property
    def exogenous(self):
        """Exogenous data sliced for this country."""
        if not hasattr(self, "_exogenous"):
            self._exogenous = Exogenous.for_entity(self, self.model.world.exogenous)
        return self._exogenous

    @property
    def neighbourhood_codes(self):
        """Neighbourhood country codes for fallback."""
        if not hasattr(self, "_neighbour_codes"):
            self._neighbour_codes = [n.code for n in self.neighbourhood]
        return self._neighbour_codes

    def _ensure_fao_data(self):
        """Load FAO economic parameters lazily on first access.

        All parameters are loaded together since they share the same reference
        year and fallback logic. Loading is idempotent - subsequent calls return
        immediately.
        """
        if self._fao_loaded:
            return

        # ---------------------------------------------------------------------
        # Step 1: Determine reference year (shared across all countries)
        # ---------------------------------------------------------------------
        # FAO data has annual time series, but we need a single snapshot.
        # Use the year closest to (but not after) the coupling start year.
        # This is cached at class level so all countries use the same year.
        if CACountry._fao_reference_year is None:
            start = self.model.config.start_coupling
            CACountry._fao_reference_year = _find_nearest_year(
                self.model.world.exogenous.capital, start
            )
            logger.info(
                f"FAO reference year: {CACountry._fao_reference_year} (nearest to {start})"
            )

        year = CACountry._fao_reference_year
        code = self.code

        # Fallback countries if data missing
        neighbour_codes = self.neighbourhood_codes

        world = self.model.world.exogenous

        # ---------------------------------------------------------------------
        # Step 2: Extract capital dynamics parameters from FAO Capital Stock
        # ---------------------------------------------------------------------
        # Source: FAO Capital Stock database for Agriculture+Forestry+Fishing
        cap = world.capital

        # Depreciation rate: annual fraction of capital value lost (wear/obsolescence)
        # Aggregator="mean" averages over available years if multiple exist
        self._depreciation_rate = _extract_with_fallback(
            cap, "depreciation_rate", code, year, neighbour_codes, aggregator="mean"
        )

        # Investment rate: annual gross fixed capital formation / existing stock
        self._investment_rate_value = _extract_with_fallback(
            cap, "investment_rate", code, year, neighbour_codes, aggregator="mean"
        )

        # Net Capital Stock (NCS): total value of agricultural capital in million USD
        # Aggregator="last" uses most recent available year
        ncs = _extract_with_fallback(cap, "ncs", code, year, neighbour_codes, aggregator="last")

        # ---------------------------------------------------------------------
        # Step 3: Compute crop-specific capital share
        # ---------------------------------------------------------------------
        # FAO reports capital for combined AFF sector. We need crop-only portion.
        # Approach: NCS_crops = NCS_aff * ag_share_of_aff * crop_share_of_ag
        gpv = world.gpv

        # Gross Production Value of crops (USD) - with 20-year lookback for sparse data
        crops_gpv = _extract_with_fallback(
            gpv, "gpv_crops", code, year, neighbour_codes, max_lookback=20, aggregator="mean"
        )

        # Gross Production Value of all agriculture (crops + livestock + ...)
        ag_gpv = _extract_with_fallback(
            gpv, "gpv_agriculture", code, year, neighbour_codes, max_lookback=20, aggregator="mean"
        )

        # Agriculture's share of AFF sector (country-specific, accounts for
        # fishing/forestry-heavy economies like Norway or Finland)
        self._ag_share_of_aff = get_ag_share_of_aff(code)

        # Crops' share within agriculture (from production value ratio)
        self._crop_share_of_ag = crops_gpv / ag_gpv

        # Combined share: what fraction of total AFF capital belongs to crops
        self._crop_capital_share = self._ag_share_of_aff * self._crop_share_of_ag

        # Initial capital per hectare: total crop capital / total cropland area
        # NCS is in million USD, so multiply by 1e6 to get USD
        self._initial_capital_per_ha = (
            (ncs * 1e6 * self._crop_capital_share) / self.cropland_area
        )

        # ---------------------------------------------------------------------
        # Step 4: Extract producer prices per crop type
        # ---------------------------------------------------------------------
        # Farm-gate prices (USD/tonne) for each PFT, used to compute farm revenue
        self._pft_prices = self.extract_prices(world.prices, code, year, neighbour_codes)

        self._fao_loaded = True

    # Properties (lazy-loaded via _ensure_fao_data)
    def _fao(self, attr):
        self._ensure_fao_data()
        return getattr(self, attr)

    def extract_prices(self, prices_ds, country, year, neighbours):
        """Extract producer prices for all crops."""
        var = list(prices_ds.data_vars)[0]
        prices = prices_ds[var]
        dim = "npft" if "npft" in prices.dims else "band"
        
        values = []
        for crop in prices[dim].values:
            try:
                val = get_value_with_fallback(prices.sel({dim: crop}), country, year, neighbour_codes=neighbours, aggregator="mean").value
            except ValueError:
                val = np.nan
            values.append(val)
        
        result = xr.DataArray(values, dims=[dim], coords={dim: prices[dim].values})
        return result.rename({dim: "band"}) if dim == "npft" else result

    @property
    def depreciation_rate(self):
        """Annual capital depreciation rate (fraction per year).

        The rate at which agricultural capital loses value due to wear,
        obsolescence, and physical deterioration. Used in the capital
        dynamics model to compute capital loss each year.

        Source: FAO Capital Stock database (consumption of fixed capital
        divided by net capital stock).

        Returns
        -------
        float
            Depreciation rate, typically 0.03-0.08 (3-8% per year).
            Higher values indicate faster capital turnover.

        Example
        -------
        If depreciation_rate = 0.05 and capital = 10000:
        annual_depreciation = 0.05 * 10000 = 500 currency units lost
        """
        return self._fao("_depreciation_rate")

    @property
    def investment_rate(self):
        """Annual gross fixed capital formation rate (fraction per year).

        The rate at which new capital is added to the existing stock through
        investment. Represents farmer reinvestment behavior at the country
        level, derived from national accounts data.

        Source: FAO Capital Stock database (gross fixed capital formation
        divided by net capital stock).

        Returns
        -------
        float
            Investment rate, typically 0.05-0.15 (5-15% per year).
            Higher values indicate more active capital accumulation.

        Notes
        -----
        Net capital change = investment_rate - depreciation_rate.
        If investment_rate > depreciation_rate, capital grows over time.
        """
        return self._fao("_investment_rate_value")

    @property
    def initial_capital_per_ha(self):
        """Initial agricultural capital stock per hectare (USD/ha).

        Starting capital endowment for farmers, computed from national
        capital stock data scaled to cropland area. Used to initialize
        farmer capital at simulation start.

        Computed as:
            (NCS * 1e6 * crop_capital_share) / cropland_area

        Where NCS is Net Capital Stock in million USD.

        Returns
        -------
        float
            Capital per hectare in USD. Varies widely by country:
            - Low-income countries: ~100-500 USD/ha
            - Middle-income: ~500-2000 USD/ha
            - High-income: ~2000-10000+ USD/ha

        See Also
        --------
        crop_capital_share : Fraction of total capital attributable to crops.
        """
        return self._fao("_initial_capital_per_ha")

    @property
    def crop_capital_share(self):
        """Fraction of agricultural capital attributable to crop production.

        The share of total Agriculture+Forestry+Fishing (AFF) sector capital
        that belongs specifically to crop production. Used to disaggregate
        national capital stock data to the crop sector.

        Computed as:
            ag_share_of_aff * crop_share_of_ag

        Returns
        -------
        float
            Share in [0, 1]. Typically 0.3-0.8 depending on country's
            economic structure (fishing, forestry, livestock importance).

        See Also
        --------
        ag_share_of_aff : Agriculture's share of AFF sector.
        crop_share_of_ag : Crops' share within agriculture.
        """
        return self._fao("_crop_capital_share")

    @property
    def ag_share_of_aff(self):
        """Agriculture's share of the Agriculture+Forestry+Fishing sector.

        FAO capital stock data is reported for the combined AFF sector.
        This coefficient extracts the agriculture-only portion, accounting
        for countries where fishing (e.g., Norway, Iceland) or forestry
        (e.g., Finland, Sweden) dominate the sector.

        Source: Country-specific estimates based on value-added data.
        Falls back to global default (0.85) if country not in database.

        Returns
        -------
        float
            Share in [0, 1]. Examples:
            - Norway (major fishing): ~0.60
            - Bolivia (landlocked): ~0.95
            - USA (diversified): ~0.85

        Notes
        -----
        This is a structural parameter that changes slowly over time.
        Currently uses static country-level estimates.
        """
        return self._fao("_ag_share_of_aff")

    @property
    def crop_share_of_ag(self):
        """Crops' share of total agricultural gross production value.

        The fraction of agricultural output value (in USD) that comes from
        crop production vs. livestock, aquaculture, etc. Derived from FAO
        Gross Production Value (GPV) data.

        Computed as:
            GPV_crops / GPV_agriculture

        Returns
        -------
        float
            Share in [0, 1]. Typically 0.4-0.7 for most countries.
            Higher values indicate crop-dominated agriculture.

        Source
        ------
        FAO Gross Production Value database, averaged over recent years
        with fallback to neighbours if country data unavailable.
        """
        return self._fao("_crop_share_of_ag")

    @property
    def pft_prices(self):
        """Producer prices per plant functional type (USD per tonne).

        Farm-gate prices received by farmers for each crop type, used to
        calculate farm revenue from harvest yields. Prices are matched to
        LPJmL's plant functional types (PFTs).

        Source: FAO Producer Prices database, with fallback to neighbour
        countries or regional averages if country data unavailable.

        Returns
        -------
        xarray.DataArray
            Prices indexed by 'band' dimension (PFT names).
            Units: USD per metric tonne of dry matter.

        Notes
        -----
        Prices are fixed at simulation start year and do not change
        during the simulation (exogenous price assumption).

        See Also
        --------
        ca_farmer.revenue : Uses these prices to compute farm income.
        """
        return self._fao("_pft_prices")

    # -------------------------------------------------------------------------
    # Country-Level Reference Values (for performance score normalization)
    # -------------------------------------------------------------------------

    # Default reference values for LEVELS (overwritten each year in update())
    reference_yield_level: float = 1.0
    reference_soilc_level: float = 1.0
    reference_moisture_level: float = 1.0

    # Default reference values for TRENDS (overwritten each year in update())
    # These represent "meaningful" trends for normalization, computed as
    # the standard deviation of trends across farmers (not mean, since mean
    # trend is often near zero and would cause division issues)
    reference_yield_trend: float = 1.0
    reference_soilc_trend: float = 1.0
    reference_moisture_trend: float = 1.0

    def compute_reference_values(self):
        """Compute country-mean reference values for performance normalization.

        Called once per year in update() BEFORE farmers are updated.
        This ensures all farmers see consistent reference values and avoids
        repeated computation during performance comparisons.

        Reference values for levels use country means.
        Reference values for trends use standard deviation (spread of trends),
        which provides a natural scale for "what counts as meaningful change".
        """
        yields, soilcs, moistures = [], [], []
        yield_trends, soilc_trends, moisture_trends = [], [], []

        for farmer in self.farmers:
            if hasattr(farmer, "cropyield"):
                yields.append(farmer.cropyield)
            if hasattr(farmer, "soilc"):
                soilcs.append(farmer.soilc)
            if hasattr(farmer, "root_moisture"):
                moistures.append(farmer.root_moisture)

            # Collect trends from trackers
            tracker = farmer.behaviour.performance_tracker
            if tracker is not None:
                yield_trends.append(tracker.yield_trend)
                soilc_trends.append(tracker.soilc_trend)
                moisture_trends.append(tracker.moisture_trend)

        # Level references: country mean (with fallback to 1.0)
        self.reference_yield_level = float(np.mean(yields)) if yields else 1.0
        self.reference_soilc_level = float(np.mean(soilcs)) if soilcs else 1.0
        self.reference_moisture_level = float(np.mean(moistures)) if moistures else 1.0

        # Trend references: standard deviation of trends (what counts as meaningful)
        # Use std rather than mean because mean trend can be ~0 which breaks normalization
        # The std represents "typical variation in trends" - a natural scale
        # Minimum of 0.1 to avoid extreme values when all trends are identical
        self.reference_yield_trend = max(0.1, float(np.std(yield_trends))) if yield_trends else 1.0
        self.reference_soilc_trend = max(0.1, float(np.std(soilc_trends))) if soilc_trends else 1.0
        self.reference_moisture_trend = max(0.1, float(np.std(moisture_trends))) if moisture_trends else 1.0

    # -------------------------------------------------------------------------
    # Country-Level Statistics (for non-local spreading)
    # -------------------------------------------------------------------------

    def compute_management_performance(self, t):
        """Compute management performance for the country.

        Computes bundle distribution and average performance per bundle
        across all farmers in this country, then merges with neighbouring
        countries' stats (from previous year) for cross-border social learning.

        Enables O(1) lookup for country-level social influence
        instead of O(n²) pairwise comparisons.

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # Skip if already computed for this year
        existing = self.statistic.get("management_performance")
        if isinstance(existing, RegionManagementPerformanceStore) and existing.year == t:
            return

        # Initialize new store to collect observations from farmers and
        #   aggregate with add_observation method.
        store = RegionManagementPerformanceStore(year=t)
        for farmer in self.farmers:
            if not hasattr(farmer, "behaviour"):
                continue
            store.add_observation(farmer.behaviour.practice_bundle, farmer)

        nb_weight = self.model.config.coupled_config.tpb.neighbour_country_weight

        if nb_weight > 0:
            store = store.merge_with_neighbours(
                1.0 - nb_weight,  # own-country share (weights must sum to 1.0)
                self._neighbour_country_stats(),
                nb_weight,  # neighbour share from config (e.g. 0.25 → 75/25 split)
            )
        else:
            # compute final counts for bundles and farmers
            store.finalize_store()

        # add management performance store to statistic
        self.statistic.set("management_performance", store)

    def _neighbour_country_stats(self) -> list:
        """Stats from neighbouring countries (previous year, world broadcast)."""
        all_country_stats = self.world.statistic.get(
            "countries_management_performance", {}
        )
        neighbour_codes = getattr(self, "neighbourhood_codes", [])
        return [
            all_country_stats[code]
            for code in neighbour_codes
            if code in all_country_stats
            and isinstance(all_country_stats[code], RegionManagementPerformanceStore)
        ]

    def update(self, t):
        """Update country statistics and its farmers.

        Computes country-level statistics BEFORE updating farmers to ensure
        all farmers see the same country-level data for this timestep.
        """
        # Compute reference values for performance normalization
        # Must run BEFORE farmers update so they see consistent references
        self.compute_reference_values()

        # Compute country-level statistics for non-local spreading
        # Must run BEFORE farmers update so they see current country stats
        self.compute_management_performance(t)

        # Call parent's update which handles farmer updates via Region.update()
        super().update(t)
