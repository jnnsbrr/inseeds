"""Country entity with FAO economic data for Conservation Agriculture.

Overview
--------
Each country:
1. Loads FAO economic data lazily (capital stock, producer prices)
2. Aggregates farmer performance by practice bundle each year
3. Optionally blends statistics with neighbouring countries

Data sources: FAO Capital Stock, Producer Prices, Gross Production Value.
Missing values fall back: country → neighbours → global mean.

See Also
--------
ca_management : RegionManagementPerformanceStore
ca_behaviour : farmer use of country-level statistics
"""

import logging
import numpy as np
import xarray as xr

from inseeds.components.farming.region import Country
from inseeds.components.exogenous.faostat import get_ag_share_of_aff, get_value_with_fallback, get_gdp_cost_ratio
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
    return get_value_with_fallback(ds[var], country, year, neighbour_codes=neighbours, field_name=var, **kwargs).value


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
        """Load FAO economic parameters on first access (idempotent)."""
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

        # Net Capital Stock (NCS): total value of agricultural capital in million USD
        # Aggregator="last" uses most recent available year
        ncs = _extract_with_fallback(cap, "ncs", code, year, neighbour_codes, aggregator="last")

        # Step 3: Compute crop-specific capital share
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

        # ---------------------------------------------------------------------
        # Step 5: Extract GDP per capita for cost scaling
        # ---------------------------------------------------------------------
        # GDP per capita determines where a country falls on the cost spectrum.
        # Countries below 10th percentile use minimum costs, above 90th use maximum.
        if "gdp" in world.keys():
            gdp = world.gdp
            self._gdp_per_capita = _extract_with_fallback(
                gdp, "gdp_per_capita", code, year, neighbour_codes, aggregator="mean"
            )
            # Get percentiles from dataset attributes (computed during preprocessing)
            self._gdp_p10 = gdp.attrs.get("gdp_percentile_10", 1000.0)
            self._gdp_p90 = gdp.attrs.get("gdp_percentile_90", 40000.0)
            # Compute cost ratio for practice cost scaling
            self._gdp_cost_ratio = get_gdp_cost_ratio(
                self._gdp_per_capita, self._gdp_p10, self._gdp_p90
            )
        else:
            # Fallback if GDP data unavailable: use middle of cost range
            logger.warning(f"{code}: GDP data unavailable, using default cost ratio 0.5")
            self._gdp_per_capita = None
            self._gdp_cost_ratio = 0.5

        self._fao_loaded = True

    # Properties (lazy-loaded via _ensure_fao_data)
    def _fao(self, attr):
        self._ensure_fao_data()
        return getattr(self, attr)

    def extract_prices(self, prices_ds, country, year, neighbours):
        """Producer prices per PFT with country→neighbour→global fallback.

        Crops that fail all tiers become NaN here; ``_init_revenue_mapping``
        raises if a cropped LPJmL band would map to a non-finite price.
        """
        var = list(prices_ds.data_vars)[0]
        prices = prices_ds[var]
        dim = "npft" if "npft" in prices.dims else "band"
        
        values = []
        for crop in prices[dim].values:
            try:
                val = get_value_with_fallback(prices.sel({dim: crop}), country, year, neighbour_codes=neighbours, aggregator="mean", field_name=f"price_{crop}").value
            except ValueError:
                # No country / neighbour / global value (e.g. cotton, tea in FAO)
                val = np.nan
            values.append(val)
        
        result = xr.DataArray(values, dims=[dim], coords={dim: prices[dim].values})
        return result.rename({dim: "band"}) if dim == "npft" else result

    @property
    def initial_capital_per_ha(self):
        """Initial agricultural capital per hectare (USD/ha).

        Computed as ``(NCS × 1e6 × crop_capital_share) / cropland_area`` from
        FAO Net Capital Stock (million USD).

        Returns
        -------
        float
            Capital endowment used to initialise farmer capital.
        """
        return self._fao("_initial_capital_per_ha")

    @property
    def pft_prices(self):
        """Farm-gate producer prices by PFT (USD/t dry matter).

        Source: FAO Producer Prices; neighbour/global fallback if missing.
        Fixed at coupling start (exogenous prices).

        Returns
        -------
        xarray.DataArray
            Prices indexed by ``band`` (PFT names).
        """
        return self._fao("_pft_prices")

    @property
    def gdp_cost_ratio(self):
        """GDP-based practice cost scaling ratio in [0, 1].

        Linear interpolation between global 10th and 90th GDP percentiles;
        values below p10 → 0, above p90 → 1.

        Returns
        -------
        float
            Weight for interpolating min/max practice costs.
        """
        return self._fao("_gdp_cost_ratio")

    # -------------------------------------------------------------------------
    # Country-Level Statistics (for non-local spreading)
    # -------------------------------------------------------------------------

    def compute_management_performance(self, t):
        """Aggregate bundle-level performance for country social learning.

        Merges with neighbouring countries' prior-year stats when
        ``neighbour_country_weight > 0``.

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
        """Update country statistics, then farmers.

        Country-level performance is computed before farmer updates so all
        farmers share the same country stats for timestep ``t``. Those stats
        therefore describe the farms as they stood at the end of the previous
        year - levels and trends alike, which is why the farmer holds its
        levels as values for the year rather than reading them off the cell.
        Farmers learn from what the country did last year, not from a mixture
        of last year's trends and this year's harvests.
        """
        # Compute country-level statistics for non-local spreading
        # Must run BEFORE farmers update so they see current country stats
        self.compute_management_performance(t)

        # Call parent's update which handles farmer updates via Region.update()
        super().update(t)
