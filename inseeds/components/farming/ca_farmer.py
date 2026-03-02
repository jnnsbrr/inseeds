"""Conservation Agriculture farmer with TPB-based multi-practice decisions.

This module implements a farmer agent that manages Conservation Agriculture (CA)
practices using Theory of Planned Behaviour (TPB) for adoption decisions.

Key features:
- Capital dynamics grounded in FAO data (OECD 2009 methodology)
- Two-component investment model (structural + discretionary)
- Practice costs and affordability constraints
- Residue economics (opportunity cost of retention)
- TPB-based decision making for practice adoption

The farmer manages three CA practices as a bundle:
1. Tillage: conventional (0) vs no-till (1)
2. Cover crop: none (0) vs planted (1)
3. Residue: baseline (0) vs retained (1)

References:
- Jorgenson, D.W. (1963). Capital Theory and Investment Behavior. AER.
- OECD (2009). Measuring Capital - OECD Manual, 2nd ed.
- FAO (2023). FAOSTAT Capital Stock methodology.
- Ajzen, I. (1991). The theory of planned behavior. OBHDP.
"""

import numpy as np
import xarray as xr

from inseeds.components.farming.farmer import Farmer
from inseeds.components.farming.ca_behaviour import TPB
from inseeds.components.data.fao import FaoProducerPrices, FaoCapitalStock


class ConservationAgricultureFarmer(Farmer):
    """Farmer with TPB-based Conservation Agriculture adoption decisions.

    Manages three CA practices as a bundle: (tillage, cover_crop, residue).
    Each practice is binary (0=off, 1=on). Decisions are delegated to TPB
    (Theory of Planned Behaviour) which weighs attitudes, social norms, and
    perceived behavioural control.

    Capital Dynamics (hybrid FAO + behavioral model)
    ------------------------------------------------
    Initial capital:
        K₀ = NCS / agricultural_land_area
        where NCS = Net Capital Stocks from FAO (million USD)

    Depreciation (Jorgenson 1963):
        Dep = δ × K
        where δ = CFC / NCS (Consumption of Fixed Capital / Net Capital Stocks)

    Two-component investment:
        1. Replacement: i × K
           where i = GFCF / NCS (structural, FAO-derived)
           Farmers replace worn equipment regardless of profit.

        2. Discretionary: s × max(profit, 0)
           where s = savings_rate (behavioral parameter)
           Profitable farmers invest more to expand.

    Annual update:
        K_next = K - δK + iK + s × max(profit, 0)

    Profit Calculation
    ------------------
    Revenue = LPJmL harvest (gC) × crop fraction × area × FAO prices
    Variable costs = direct costs per practice × farm size
    Profit = Revenue - Variable costs - Depreciation

    Affordability Constraints
    -------------------------
    - If capital < min_capital: deselect costly practices until affordable
    - Transition costs deducted when switching practices (if capital suffices)

    Cover Crop Type Selection
    -------------------------
    - Legume (N-fixing) vs non-legume based on leaching/fertilizer thresholds
    - High leaching + high fertilizer → non-legume (catch crop)
    - Otherwise → legume (for nitrogen fixation)
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, **kwargs):
        """Initialize farmer with economics, FAO data, costs, and TPB behaviour.

        Loads:
        - Farm economics parameters from config
        - FAO producer prices for profit calculation
        - FAO capital stock for depreciation/investment rates
        - Practice costs (direct and transition)
        - Residue economics (opportunity cost)

        Initializes:
        - Capital based on FAO Net Capital Stocks
        - TPB behaviour model
        """
        super().__init__(**kwargs)

        # -----------------------------------------------------------------
        # Load farm economics configuration
        # -----------------------------------------------------------------
        econ = self.model.config.coupled_config.farm_economics
        econ = econ.to_dict() if hasattr(econ, "to_dict") else dict(econ)

        # -----------------------------------------------------------------
        # Policy/behavioral parameters (not derivable from FAO)
        # -----------------------------------------------------------------

        # Survival buffer: years of depreciation the farmer can sustain
        # min_capital = n_survival_years × δ × initial_capital
        # This ties the threshold to actual capital dynamics (Jorgenson 1963)
        # and adapts to country-specific depreciation rates from FAO.
        # Default 2 years is conservative: US farms maintain liquid assets of
        # ~75% of annual cash expenses (Zulauf & Schnitkey 2024, farmdoc daily),
        # roughly 0.75-1.4 years of operating buffer. 2 years of depreciation
        # (a subset of total expenses) provides a reasonable survival threshold.
        self._n_survival_years = econ.get("n_survival_years", 2)

        # -----------------------------------------------------------------
        # Load FAO data
        # -----------------------------------------------------------------

        # FAO producer prices: crop prices by country for profit calculation
        self._load_fao_pft_prices()

        # FAO capital stock: depreciation rate, investment rate, initial capital
        self._load_fao_capital_stock()

        # -----------------------------------------------------------------
        # Initialize capital
        # -----------------------------------------------------------------
        # Capital = initial capital per ha × farm size
        # Initial capital per ha derived from FAO NCS / agricultural land area
        self.capital = self._initial_capital_per_ha * self.farm_size

        # Track capital history for risk aversion calculation (Chavas & Holt 1996)
        # Stores last N years of capital values to compute volatility
        self._capital_history = [self.capital]
        self._capital_history_max_years = 10  # Keep last 10 years

        # -----------------------------------------------------------------
        # Load practice costs
        # -----------------------------------------------------------------
        # Costs include:
        # - direct: annual operating cost per ha
        # - transition: one-time cost when adopting practice
        pc = self.model.config.coupled_config.practice_costs
        self.practice_costs = pc.to_dict() if hasattr(pc, "to_dict") else dict(pc)

        # -----------------------------------------------------------------
        # Load residue economics
        # -----------------------------------------------------------------
        # Opportunity cost: value of residue if sold/used elsewhere (e.g., feed)
        # Retaining residue means forgoing this income
        res = self.model.config.coupled_config.residue_economics
        use_costs = res.use_costs.to_dict() if hasattr(res.use_costs, "to_dict") else dict(res.use_costs)
        self._residue_opportunity_cost_per_ha = use_costs[res.default_removal_use]

        # -----------------------------------------------------------------
        # Initialize behaviour
        # -----------------------------------------------------------------
        # Store baseline residue level (before any CA adoption)
        self.residue_baseline = self.residue_on_field

        # Create TPB decision model
        self.behaviour = TPB(self)

    # =========================================================================
    # CAPITAL BOUNDS PROPERTY
    # =========================================================================

    @property
    def min_capital(self):
        """Minimum sustainable capital threshold.

        Defined as the capital needed to cover n_survival_years of depreciation:
            min_capital = n_survival_years × δ × initial_capital

        This ties the threshold to actual capital dynamics (Jorgenson 1963)
        and adapts to country-specific depreciation rates from FAO.
        Below this threshold, farmer must deselect costly practices to survive.

        References
        ----------
        - Jorgenson, D.W. (1963). Capital Theory and Investment Behavior. AER.
        - Bandiera et al. (2017). Why Do People Stay Poor? (poverty trap thresholds)

        Returns
        -------
        float
            Minimum capital in currency units.
        """
        initial_capital = self._initial_capital_per_ha * self.farm_size
        return self._n_survival_years * self.depreciation_rate * initial_capital

    # =========================================================================
    # RESIDUE ECONOMICS PROPERTIES
    # =========================================================================

    @property
    def residue_opportunity_cost(self):
        """Opportunity cost of retaining residue (total for farm).

        Returns
        -------
        float
            Opportunity cost scaled by farm size.
        """
        return self._residue_opportunity_cost_per_ha * self.farm_size

    @property
    def residue_opportunity_cost_per_ha(self):
        """Opportunity cost per hectare (for behaviour reasonableness checks).

        Returns
        -------
        float
            Opportunity cost per ha.
        """
        return self._residue_opportunity_cost_per_ha

    # =========================================================================
    # FAO DATA LOADING
    # =========================================================================

    def _load_fao_pft_prices(self):
        """Load FAO producer prices for profit calculation.

        Prices are used to convert LPJmL harvest (in carbon) to monetary value.
        Country-specific prices are used when available, otherwise global mean.
        """
        sim_path = self.model.config.sim_path
        prices = FaoProducerPrices()
        self.fao_pft_prices = xr.open_dataset(prices.ensure(sim_path))

    def _load_fao_capital_stock(self):
        """Load FAO capital stock and derive capital parameters.

        Derives country-specific values following standard capital accounting
        (OECD 2009, Jorgenson 1963):

        - Depreciation rate: δ = CFC / NCS
          Annual rate of capital wear (typically 3-8% for agriculture)

        - Investment rate: i = GFCF / NCS
          Gross investment as fraction of capital stock (structural)

        - Initial capital per ha: NCS / agricultural_land_area
          Starting capital endowment based on country's capital stock
        """
        sim_path = self.model.config.sim_path
        capital = FaoCapitalStock()
        self.fao_capital_stock = xr.open_dataset(capital.ensure(sim_path))

        country_code = self.cell.country_code

        # -----------------------------------------------------------------
        # Depreciation rate: δ = CFC / NCS
        # -----------------------------------------------------------------
        # CFC = Consumption of Fixed Capital (annual capital wear)
        # NCS = Net Capital Stocks (total capital value)
        # δ represents the fraction of capital that depreciates each year

        dep_rate = self.fao_capital_stock["depreciation_rate"]

        if country_code in dep_rate.area_code.values:
            # Use country-specific depreciation rate
            self.depreciation_rate = float(
                dep_rate.sel(area_code=country_code).isel(time=-1).values
            )
        else:
            # Fallback to global mean if country not in dataset
            self.depreciation_rate = float(dep_rate.isel(time=-1).mean().values)

        # -----------------------------------------------------------------
        # Investment rate: i = GFCF / NCS
        # -----------------------------------------------------------------
        # GFCF = Gross Fixed Capital Formation (annual investment)
        # This represents structural investment behavior: how much of the
        # capital stock is renewed each year through new investment

        inv_rate = self.fao_capital_stock["investment_rate"]

        if country_code in inv_rate.area_code.values:
            self._investment_rate = float(
                inv_rate.sel(area_code=country_code).isel(time=-1).values
            )
        else:
            self._investment_rate = float(inv_rate.isel(time=-1).mean().values)

        # -----------------------------------------------------------------
        # Initial capital per ha from NCS
        # -----------------------------------------------------------------
        # NCS is in million USD; convert to USD/ha using agricultural land area
        # This gives a per-hectare capital endowment for the farmer

        ncs = self.fao_capital_stock["6186"]  # Net Capital Stocks (million USD)
        agri_land_ha = self._get_fao_agricultural_land_ha(country_code)

        if country_code in ncs.area_code.values:
            ncs_million_usd = float(
                ncs.sel(area_code=country_code).isel(time=-1).values
            )
        else:
            ncs_million_usd = float(ncs.isel(time=-1).mean().values)

        # Convert: million USD → USD, then divide by hectares
        self._initial_capital_per_ha = (ncs_million_usd * 1e6) / agri_land_ha

    def _get_fao_agricultural_land_ha(self, country_code):
        """Get agricultural land area in hectares for a country.

        Used to convert FAO NCS (total capital) to per-hectare values.

        Note: Currently uses config fallback. Could be extended to use
        FAO Land Use dataset (RL domain) for country-specific values.

        Parameters
        ----------
        country_code : str
            ISO3 country code.

        Returns
        -------
        float
            Agricultural land area in hectares.
        """
        econ = self.model.config.coupled_config.farm_economics
        econ = econ.to_dict() if hasattr(econ, "to_dict") else dict(econ)

        # Fallback: use config value if FAO land use not integrated
        # Default ~500M ha is roughly global agricultural land / number of countries
        return econ.get("agricultural_land_ha", 500_000_000)

    # =========================================================================
    # COVER CROP TYPE SELECTION
    # =========================================================================

    def _indicate_cover_crop_type(self):
        """Determine cover crop type based on environmental conditions.

        Cover crop types:
        - 1 = Non-legume (catch crop): captures excess nutrients
        - 2 = Legume: fixes atmospheric nitrogen

        Selection logic:
        - High leaching + high fertilizer → non-legume (catch excess N)
        - Otherwise → legume (add N through fixation)

        Returns
        -------
        int
            Cover crop type: 1 (non-legume) or 2 (legume).
        """
        # Get environmental conditions from cell
        runoff = self.cell_runoff
        leaching_val = self.cell_leaching
        fertilizer_val = self.cell_fertilizer

        # Get thresholds from config
        cc = self.model.config.coupled_config.practice_dimensions.cover_crop
        leaching_limit = cc.leaching_high
        fertilizer_limit = getattr(cc, "fertilizer_high", 5)

        # Calculate leaching rate (normalized by runoff)
        leaching = leaching_val / runoff if runoff > 0 else 0

        # -----------------------------------------------------------------
        # Decision logic
        # -----------------------------------------------------------------
        # High leaching + high fertilizer: soil has excess N that's being lost
        # → Use non-legume catch crop to capture nutrients
        if leaching >= leaching_limit and fertilizer_val >= fertilizer_limit:
            return 1  # Non-legume

        # Otherwise: soil may be N-limited
        # → Use legume for biological N fixation
        return 2  # Legume

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def _get_current_direct_costs(self):
        """Calculate annual direct costs of current practice bundle.

        Direct costs are ongoing annual costs for each active practice
        (e.g., seeds for cover crops, foregone income from residue retention).

        Returns
        -------
        float
            Total annual direct cost (scaled by farm size).
        """
        bundle = self.behaviour._practice_bundle
        costs = self.practice_costs
        total = 0.0

        # Tillage cost (index 0)
        # Note: no-till may have lower fuel costs but higher herbicide costs
        if bundle[0] == 1:
            total += costs.get("tillage", {}).get("direct", 0)

        # Cover crop cost (index 1)
        # Seeds, planting, termination
        if bundle[1] == 1:
            total += costs.get("cover_crop", {}).get("direct", 0)

        # Residue retention cost (index 2)
        # Foregone income from not selling/using residue
        if bundle[2] == 1:
            total += costs.get("residue_on_field", {}).get("direct", 0)

        # Scale by farm size (costs are per-hectare in config)
        return total * self.farm_size

    # =========================================================================
    # CAPITAL UPDATE
    # =========================================================================

    def _update_capital(self):
        """Update farmer's capital based on profit and FAO-derived rates.

        Capital changes through two mechanisms:

        1. Depreciation (capital wear)
           - Machinery, equipment, and infrastructure lose value over time
           - Rate from FAO: depreciation_rate = CFC / NCS (Jorgenson 1963)
           - This is a real cost that reduces capital regardless of profit

        2. Reinvestment (from profit)
           - Farmers reinvest a fraction of their profit back into the farm
           - Rate from FAO: investment_rate = GFCF / NCS (OECD 2009)
           - Only positive profit contributes; losses don't add capital

        The FAO investment rate represents observed reinvestment behavior
        at the national level, providing a data-driven (not arbitrary)
        estimate of how much farmers typically reinvest.

        References
        ----------
        - Jorgenson, D.W. (1963). Capital Theory and Investment Behavior. AER.
        - OECD (2009). Measuring Capital - OECD Manual, 2nd ed.
        - FAO (2023). FAOSTAT Capital Stock methodology.
        """

        # -----------------------------------------------------------------
        # Step 1: Calculate profit from farming
        # -----------------------------------------------------------------
        # Revenue from crop sales (LPJmL yields × FAO prices)
        revenue = self._calculate_revenue()

        # Variable costs for current practices (per-ha costs × farm size)
        variable_costs = self._get_current_direct_costs()

        # Gross profit before depreciation
        gross_profit = revenue - variable_costs

        # -----------------------------------------------------------------
        # Step 2: Depreciation (capital wear)
        # -----------------------------------------------------------------
        # Depreciation rate from FAO: CFC / NCS
        # CFC = Consumption of Fixed Capital (annual capital used up)
        # NCS = Net Capital Stock (total capital value)
        # This represents the fraction of capital that wears out each year.
        depreciation = self.depreciation_rate * self.capital

        # Net profit after accounting for capital wear
        net_profit = gross_profit - depreciation

        # -----------------------------------------------------------------
        # Step 3: Reinvestment (from profit)
        # -----------------------------------------------------------------
        # Investment rate from FAO: GFCF / NCS
        # GFCF = Gross Fixed Capital Formation (annual investment)
        # This represents the fraction of income farmers typically reinvest.
        #
        # Key insight: investment comes FROM profit, not in addition to it.
        # Farmers can only reinvest what they earn.
        # If profit is negative, no reinvestment occurs.
        reinvestment = self._investment_rate * max(net_profit, 0.0)

        # -----------------------------------------------------------------
        # Step 4: External financing (TODO)
        # -----------------------------------------------------------------
        # TODO: Add loans/credits from banks as external capital source.
        #
        # This would allow farmers to:
        # - Access capital beyond their own profit (especially for transitions)
        # - Take on debt to invest in CA practices with high upfront costs
        # - Model credit constraints as barrier to CA adoption
        #
        # Implementation considerations:
        # - Interest rates (country-specific, possibly from World Bank data)
        # - Loan repayment schedules (reduce future net_profit)
        # - Credit access based on farm size, collateral, or credit history
        # - Debt-to-asset ratio limits
        # - Microfinance vs. commercial bank access
        #
        # Potential references:
        # - Feder et al. (1990). The relationship between credit and productivity
        # - Guirkinger & Boucher (2008). Credit constraints and productivity
        external_financing = 0.0  # Placeholder for future implementation

        # -----------------------------------------------------------------
        # Step 5: Update capital
        # -----------------------------------------------------------------
        # new_capital = current_capital - depreciation + reinvestment + loans
        #
        # - Profitable farmers: capital grows (reinvestment > depreciation)
        # - Unprofitable farmers: capital shrinks (only depreciation, no
        #   reinvestment)
        # - Break-even: capital stable if reinvestment covers depreciation
        # - With loans: farmers can grow capital even without profit
        self.capital = (
            self.capital - depreciation + reinvestment + external_financing
        )

        # Capital cannot go negative
        self.capital = max(0.0, self.capital)

        # -----------------------------------------------------------------
        # Step 6: Track capital history for risk calculation
        # -----------------------------------------------------------------
        # Used by _compute_risk_factor() to assess capital volatility
        self._capital_history.append(self.capital)
        if len(self._capital_history) > self._capital_history_max_years:
            self._capital_history.pop(0)  # Remove oldest entry

    @property
    def capital_history(self):
        """Recent capital values for volatility calculation.

        Returns
        -------
        list
            Last N years of capital values.
        """
        return self._capital_history

    # =========================================================================
    # REVENUE CALCULATION
    # =========================================================================

    def _calculate_revenue(self):
        """Calculate revenue from crop production.

        Converts LPJmL harvest (in grams of carbon) to monetary value
        using FAO producer prices.

        Note: This is gross revenue, not profit. Costs are subtracted
        separately in _update_capital().

        Conversion steps
        ----------------
        1. Get harvest in gC/m² from LPJmL
        2. Multiply by crop fraction and area to get total production
        3. Convert gC to tonnes dry matter (using 0.45 C fraction)
        4. Multiply by FAO prices to get revenue

        Returns
        -------
        float
            Total revenue in currency units (USD).
        """
        # Get LPJmL outputs
        pft_harvestc = self.cell.from_earth.pft_harvestc  # Harvest in gC/m²
        cftfrac = self.cell.from_earth.cftfrac  # Crop fractions

        # Get country-specific prices
        prices = self._get_country_prices(self.fao_pft_prices, self.cell.country_code)

        # -----------------------------------------------------------------
        # Align bands (crop types) between harvest and prices
        # -----------------------------------------------------------------
        # Prices may have 'band' or 'npft' dimension depending on source
        price_dim = None
        if "band" in prices.dims:
            price_dim = "band"
        elif "npft" in prices.dims:
            price_dim = "npft"

        if price_dim and "band" in pft_harvestc.dims:
            # Get common crop types between LPJmL bands and price dimension
            lpjml_bands = set(pft_harvestc.band.values)
            price_bands = set(prices[price_dim].values)
            common = list(lpjml_bands & price_bands)

            if common:
                pft_harvestc = pft_harvestc.sel(band=common)
                cftfrac = cftfrac.sel(band=common)
                prices = prices.sel({price_dim: common})
                # Rename price dimension to match LPJmL if needed
                if price_dim != "band":
                    prices = prices.rename({price_dim: "band"})

        # -----------------------------------------------------------------
        # Calculate production
        # -----------------------------------------------------------------
        # Farm size is in hectares; convert to m²
        area_m2 = self.farm_size * 10000

        # Production in gC = yield (gC/m²) × crop fraction × area (m²)
        production_gC = pft_harvestc * cftfrac * area_m2

        # -----------------------------------------------------------------
        # Convert gC to tonnes dry matter
        # -----------------------------------------------------------------
        # Carbon fraction in dry matter is approximately 0.45
        # 1 tonne = 1e6 grams
        # tonnes_DM = gC / (0.45 × 1e6)
        production_tonnes_dm = production_gC / (0.45 * 1e6)

        # -----------------------------------------------------------------
        # Calculate revenue
        # -----------------------------------------------------------------
        # Revenue = production × price, summed across all crops
        return float(np.nansum((production_tonnes_dm * prices).values))

    def _get_country_prices(self, prices_ds, country_code):
        """Select prices for a country, with fallback to global mean.

        Parameters
        ----------
        prices_ds : xr.Dataset
            FAO producer prices dataset.
        country_code : str
            ISO3 country code.

        Returns
        -------
        xr.DataArray
            Prices for the country (or global mean if country missing).
            Time dimension is collapsed to most recent year to avoid
            dtype conflicts with LPJmL datetime coordinates.
        """
        var_name = list(prices_ds.data_vars)[0]
        prices = prices_ds[var_name]

        # Collapse time dimension - use most recent year's prices
        # This avoids dtype conflicts between FAO int years and LPJmL datetime
        if "time" in prices.dims:
            prices = prices.isel(time=-1)

        # Check if dataset has country dimension
        if "area_code" not in prices.dims:
            return prices

        # Use country-specific prices if available
        if country_code in prices.area_code.values:
            return prices.sel(area_code=country_code)

        # Fallback to global mean
        return prices.mean(dim="area_code")

    # =========================================================================
    # AFFORDABILITY CHECK
    # =========================================================================

    def _check_practice_affordability(self):
        """Deselect practices if capital falls below minimum threshold.

        When capital is too low, farmer must abandon costly practices
        to reduce expenses. Practices are deselected in order of cost
        (most expensive first).

        Deselection order:
        1. Cover crop (typically highest cost)
        2. Residue retention
        3. Tillage change (no-till)
        """
        # No action needed if capital is sufficient
        if self.capital >= self.min_capital:
            return

        # -----------------------------------------------------------------
        # Deselect practices until affordable
        # -----------------------------------------------------------------
        bundle = list(self.behaviour._practice_bundle)

        # Order: cover_crop (idx 1), residue (idx 2), tillage (idx 0)
        # Most to least costly (typical ordering)
        practices = [("cover_crop", 1), ("residue_on_field", 2), ("tillage", 0)]

        for practice_name, idx in practices:
            # Skip if practice already off
            if bundle[idx] == 0:
                continue

            # Only deselect practices with positive direct cost
            direct_cost = self.practice_costs.get(practice_name, {}).get("direct", 0)
            if direct_cost > 0:
                bundle[idx] = 0

                # Check if now affordable
                if self.capital >= self.min_capital:
                    break

        # -----------------------------------------------------------------
        # Apply changes if bundle changed
        # -----------------------------------------------------------------
        new_bundle = tuple(bundle)

        if new_bundle != self.behaviour._practice_bundle:
            self.behaviour.apply_bundle(new_bundle)
            self.behaviour.record_switch(new_bundle)

    # =========================================================================
    # MAIN UPDATE METHOD
    # =========================================================================

    def update(self, t):
        """Update farmer state for current timestep.

        Update sequence:
        1. Call parent update (base farmer logic)
        2. Skip if control run (no CA dynamics)
        3. Update capital (depreciation, investment, profit)
        4. Check affordability (deselect practices if needed)
        5. Run TPB decision logic
        6. Apply practice switch if TPB threshold exceeded

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # -----------------------------------------------------------------
        # Step 1: Parent update
        # -----------------------------------------------------------------
        super().update(t)

        # -----------------------------------------------------------------
        # Step 2: Skip CA dynamics in control run
        # -----------------------------------------------------------------
        if self.control_run:
            return

        # -----------------------------------------------------------------
        # Step 3: Update capital
        # -----------------------------------------------------------------
        self._update_capital()

        # -----------------------------------------------------------------
        # Step 4: Check affordability
        # -----------------------------------------------------------------
        # May deselect practices if capital too low
        self._check_practice_affordability()

        # -----------------------------------------------------------------
        # Step 5: Skip TPB if capital-constrained
        # -----------------------------------------------------------------
        # Farmer is in survival mode; no voluntary practice changes
        if self.capital < self.min_capital:
            return

        # -----------------------------------------------------------------
        # Step 6: Run TPB decision logic
        # -----------------------------------------------------------------
        self.behaviour.update()

        # -----------------------------------------------------------------
        # Step 7: Apply switch if TPB threshold exceeded
        # -----------------------------------------------------------------
        if self.behaviour.should_switch():
            new_bundle = self.behaviour._proposed_bundle

            if new_bundle is not None:
                old_bundle = self.behaviour._practice_bundle

                # Identify which practices are changing
                practice_names = ["tillage", "cover_crop", "residue_on_field"]
                flipped = [
                    p for i, p in enumerate(practice_names)
                    if old_bundle[i] != new_bundle[i]
                ]

                if flipped:
                    # Calculate transition cost
                    cost = sum(
                        self.practice_costs.get(p, {}).get("transition", 0)
                        for p in flipped
                    ) * self.farm_size

                    # Only apply if farmer can afford transition
                    if self.capital >= cost:
                        # Deduct transition cost
                        self.capital -= cost

                        # Apply new practices
                        self.behaviour.apply_bundle(new_bundle)
                        self.behaviour.record_switch(new_bundle)
