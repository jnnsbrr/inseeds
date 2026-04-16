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

from inseeds.components.farming.farmer import Farmer
from inseeds.components.farming.ca_behaviour import TPB


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
        # Default 1 year based on USDA farm financial indicators: farms typically
        # maintain working capital ratio of 0.3-0.5 (Katchova & Dinterman 2018),
        # and current ratio ~1.5-2.0 (USDA ERS). One year of depreciation buffer
        # represents a conservative minimum for operational continuity.
        self._n_survival_years = econ.get("n_survival_years")

        # Savings rate: fraction of profit reinvested into farm capital
        # This is a behavioral parameter representing farmer investment decisions.
        # Literature suggests farm savings rates of 10-30% depending on region
        # and farm type (Lowder et al. 2016; FAO 2017).
        # Default 15% is a moderate estimate for smallholder farmers.
        self._savings_rate = econ.get("savings_rate")

        # -----------------------------------------------------------------
        # Get FAO data from country (loaded once per country, not per farmer)
        # -----------------------------------------------------------------
        country = self.cell.country

        # Economic parameters from FAO capital stock
        self.depreciation_rate = country.depreciation_rate
        self._investment_rate = country._investment_rate
        self._initial_capital_per_ha = country.initial_capital_per_ha

        # Producer prices for profit calculation
        self._pft_prices = country.pft_prices

        # -----------------------------------------------------------------
        # Capital initialization
        # -----------------------------------------------------------------
        self.capital = self._initial_capital_per_ha * self.farm_size

        # -----------------------------------------------------------------
        # Load practice costs
        # -----------------------------------------------------------------
        # Costs include:
        # - direct: annual operating cost per ha
        # - transition: one-time cost when adopting practice
        pc = self.model.config.coupled_config.practice_costs
        self.practice_costs = pc.to_dict() if hasattr(pc, "to_dict") else dict(pc)

        # -----------------------------------------------------------------
        # Load residue economics (spatially-explicit from MADRaT data)
        # -----------------------------------------------------------------
        # Opportunity cost: value of residue if sold/used elsewhere
        # Weighted by spatial fractions of burnt, removed, and left on field
        self._residue_opportunity_cost_per_ha = self._compute_residue_opportunity_cost()

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
    # RESIDUE ECONOMICS
    # =========================================================================

    def _compute_residue_opportunity_cost(self):
        """Compute opportunity cost of retaining residue using spatial data.

        Uses MADRaT data for spatially-explicit fractions of residue burnt,
        removed, and recycled. The opportunity cost is the weighted sum
        of per-use costs from config, weighted by these fractions.
        
        MADRaT data structure (Smerald et al. 2023):
            production = recycled + removed + burnt
            - burnt: burned (no economic value)
            - removed: animal feed + other purposes (has opportunity cost)
            - recycled: bedding that returns to field with manure

        If spatial data is unavailable, falls back to config default.

        Returns
        -------
        float
            Opportunity cost per hectare ($/ha/yr).
        """
        res_cfg = self.model.config.coupled_config.residue_economics
        use_costs = res_cfg.use_costs.to_dict() if hasattr(res_cfg.use_costs, "to_dict") else dict(res_cfg.use_costs)

        # Try to get spatial fractions
        fracs = self._get_residue_fractions()
        if fracs is not None:
            # Weighted opportunity cost from spatial data
            # Only removed residues have economic value (animal feed + other)
            # Burnt has no value, recycled returns to field via manure
            return (
                fracs['burnt'] * use_costs.get('burnt', 0.0) +
                fracs['removed'] * use_costs.get('removed', use_costs.get('other', 40.0)) +
                fracs['recycled'] * use_costs.get('recycled', 0.0)
            )
        else:
            # Fallback to config default
            return use_costs.get(res_cfg.default_removal_use, 40.0)

    def _get_residue_fractions(self):
        """Get residue use fractions for this cell, weighted by crop composition.

        Uses MADRaT CFT-specific data weighted by actual cftfrac from LPJmL.
        This ensures residue fractions reflect the actual crop mix in each cell.

        Returns
        -------
        dict or None
            Dict with 'burnt', 'removed', 'recycled' fractions (0-1),
            or None if data not available.
        """
        if not hasattr(self.model.world, 'residue_fractions') or self.model.world.residue_fractions is None:
            return None

        rf = self.model.world.residue_fractions

        # Get cell index from grid
        cell_idx = self.cell.grid.cell.item()

        # Get crop fractions from LPJmL for weighting
        cftfrac = self._get_from_earth("cftfrac")
        if cftfrac is None:
            return None

        try:
            from inseeds.components.data.residue import ResidueData
            return ResidueData.weighted_fractions(rf, cell_idx, cftfrac)
        except Exception:
            return None

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
        fertilizer_limit = cc.fertilizer_high

        # Calculate leaching rate (normalized by runoff)
        leaching = leaching_val *1e3 / runoff if runoff > 0 else 0

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
        # Savings rate: behavioral parameter representing farmer investment decisions.
        # This is the fraction of net profit that farmers choose to reinvest.
        #
        # Note: FAO investment rate (GFCF/NCS) is used for initialization but NOT
        # for annual reinvestment, because:
        # - GFCF/NCS is a national aggregate ratio, not individual behavior
        # - It conflates new investment with replacement investment
        # - Individual farmers vary widely in savings behavior
        #
        # The savings_rate parameter (default 15%) is configurable and represents
        # the behavioral choice of how much profit to reinvest vs. consume.
        # Literature: Lowder et al. (2016); FAO (2017) suggest 10-30% range.
        #
        # Key insight: investment comes FROM profit, not in addition to it.
        # Farmers can only reinvest what they earn.
        # If profit is negative, no reinvestment occurs.
        reinvestment = self._savings_rate * max(net_profit, 0.0)

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
        2. Multiply by crop fraction and cell area to get total production
        3. Convert gC to tonnes dry matter (using 0.45 C fraction)
        4. Aggregate rainfed/irrigated variants to match price categories
        5. Multiply by FAO prices to get revenue

        Returns
        -------
        float
            Total revenue in currency units (USD).
        """
        # Get LPJmL outputs (uses _get_from_earth which handles multi-year data)
        pft_harvestc = self._get_from_earth("pft_harvestc")  # Harvest in gC/m²
        cftfrac = self._get_from_earth("cftfrac")  # Crop fractions

        # -----------------------------------------------------------------
        # Calculate production per band
        # -----------------------------------------------------------------
        # Production in gC = yield (gC/m²) × crop fraction × cell area (m²)
        # Cell area from pycopanlpjml is already in m²
        production_gC = pft_harvestc * cftfrac * self.cell.area.item()

        # Convert gC to tonnes dry matter (C fraction ~0.45, 1 tonne = 1e6 g)
        production_tonnes_dm = production_gC / (0.45 * 1e6)

        # -----------------------------------------------------------------
        # Aggregate production by crop type (strip rainfed/irrigated prefix)
        # -----------------------------------------------------------------
        # LPJmL bands: 'rainfed temperate cereals', 'irrigated temperate cereals'
        # Price categories: 'temperate cereals'
        # Sum production across irrigation variants
        if "band" in production_tonnes_dm.dims:
            # Create mapping from LPJmL band to price category
            band_to_category = {}
            for band in production_tonnes_dm.band.values:
                # Strip 'rainfed ' or 'irrigated ' prefix
                category = band
                if band.startswith("rainfed "):
                    category = band[8:]  # len("rainfed ") = 8
                elif band.startswith("irrigated "):
                    category = band[10:]  # len("irrigated ") = 10
                band_to_category[band] = category

            # Group production by category and sum
            category_production = {}
            for band, category in band_to_category.items():
                prod = float(production_tonnes_dm.sel(band=band).sum().values)
                if category not in category_production:
                    category_production[category] = 0.0
                category_production[category] += prod

        # -----------------------------------------------------------------
        # Match with prices and calculate revenue
        # -----------------------------------------------------------------
        prices = self._pft_prices

        # Get price dimension name
        price_dim = "npft" if "npft" in prices.dims else "band"
        price_categories = set(prices[price_dim].values)

        total_revenue = 0.0
        for category, production in category_production.items():
            if category in price_categories:
                price = float(prices.sel({price_dim: category}).values)
                total_revenue += production * price

        return total_revenue

    # =========================================================================
    # AFFORDABILITY CHECK
    # =========================================================================

    def _check_practice_affordability(self):
        """Deselect practices if annual direct costs exceed capital buffer.

        When capital is too low to sustain current practices, farmer must
        abandon costly practices to reduce expenses. Practices are deselected
        in order of cost (most expensive first) until direct costs are
        within the affordable range.

        The affordability criterion is: direct_costs <= capital - min_capital
        This ensures the farmer retains a survival buffer (min_capital).

        Deselection order:
        1. Cover crop (typically highest cost)
        2. Residue retention
        3. Tillage change (no-till, often has negative cost = savings)
        """
        # Calculate current annual direct costs
        current_direct_costs = self._get_current_direct_costs()

        # Available capital for costs (above survival threshold)
        available_capital = self.capital - self.min_capital

        # No action needed if costs are within budget
        if current_direct_costs <= available_capital:
            return

        # -----------------------------------------------------------------
        # Deselect practices until costs are affordable
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

                # Recalculate costs with this practice removed
                # (simplified: subtract the practice's direct cost)
                current_direct_costs -= direct_cost * self.farm_size

                # Check if costs are now within budget
                if current_direct_costs <= available_capital:
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

        Sets behaviour._switch_blocker to indicate why switch didn't happen.

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # Import blocker constants here to avoid circular imports
        from inseeds.components.farming.ca_behaviour import (
            BLOCKER_NONE, BLOCKER_CONTROL_RUN, BLOCKER_CAPITAL_SURVIVAL,
            BLOCKER_TRANSITION_UNAFFORDABLE
        )

        # -----------------------------------------------------------------
        # Step 1: Parent update
        # -----------------------------------------------------------------
        super().update(t)

        # -----------------------------------------------------------------
        # Step 2: Skip CA dynamics in control run
        # -----------------------------------------------------------------
        if self.control_run:
            self.behaviour._switch_blocker = BLOCKER_CONTROL_RUN
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
            self.behaviour._switch_blocker = BLOCKER_CAPITAL_SURVIVAL
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
                        # Clear blocker since switch succeeded
                        self.behaviour._switch_blocker = BLOCKER_NONE
                    else:
                        # Can't afford transition cost
                        self.behaviour._switch_blocker = BLOCKER_TRANSITION_UNAFFORDABLE
        else:
            # TPB score below threshold - identify which component is limiting
            self.behaviour.set_tpb_switch_blocker()
