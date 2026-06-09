"""Conservation Agriculture Farmer Agent.

This module defines the CAFarmer class - the main agent in the Conservation
Agriculture model. Each farmer makes decisions about agricultural practices,
manages capital, and interacts with neighbours through social learning.

Overview
--------
A CAFarmer represents a representative farmer per cell that:
1. Operates an LPJmL cell with specific crops
2. Chooses which CA practices to use (tillage, cover crops, residue)
3. Tracks performance over time (yield, soil carbon, moisture trends)
4. Learns from neighbours and may adopt new practices
5. Manages capital (income, costs, investment)

Key Components
--------------
- **Practice Bundle**: The combination of 3 practices the farmer currently uses
- **TPB Behaviour**: Decision model for practice transitions (see ca_behaviour.py)
- **Performance Tracker**: Monitors trends since last practice change
- **Capital Dynamics**: FAO/OECD-based investment and depreciation

Annual Update Cycle
-------------------
Each simulation year, the farmer:

    1. Receives biophysical data from LPJmL (crop yield, soil carbon, etc.)
    2. Updates performance tracker with new observations
    3. Computes revenue and updates capital
    4. Evaluates whether to change practices (TPB decision)
    5. If transitioning: pays costs, updates LPJmL inputs
    6. Applies capital dynamics (depreciation, investment)

Capital Dynamics
----------------
The model uses FAO/OECD methodology with calibrated depreciation::

    Capital_{t+1} = Capital_t - depreciation + reinvestment

Where:
- **depreciation**: ``effective_rate × capital``
- **reinvestment**: ``savings_rate × max(gross_profit, 0)``

Depreciation Rate Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FAO Capital Stock includes land (~60-80% of agricultural capital), which
does not depreciate. The FAO depreciation rate (~8%) applies primarily to
machinery/equipment. We use an effective rate (~1-2%) that accounts for
the full capital composition:

- Land/improvements: ~65% of capital, 0% depreciation
- Buildings: ~20% of capital, 2-5%/year depreciation
- Machinery: ~15% of capital, 10-15%/year depreciation

Weighted effective rate: 0.65×0% + 0.20×3% + 0.15×12% ≈ 2.4%

This ensures capital dynamics are sustainable from crop revenues.

References:
- OECD (2009). Measuring Capital Manual, 2nd ed. (asset service lives)
- USDA ERS (2022). Farm Sector Balance Sheet (land = 83% of US farm assets)
- Eurostat (2013). Handbook on prices and volumes (depreciation rates)

Practice Encoding
-----------------
Practices are encoded as binary values:

    +------------------+-------------------+-------------------+
    | Practice         | Value = 0         | Value = 1         |
    +==================+===================+===================+
    | tillage          | No-till (CA)      | Conventional      |
    | cover_crop       | No cover crop     | Cover crop        |
    | residue_on_field | Baseline removal  | CA retention      |
    +------------------+-------------------+-------------------+

Note: For tillage, 0 means the CA practice (no-till) is ACTIVE.

See Also
--------
ca_behaviour : TPB decision model (how farmers decide)
ca_management : Practice bundles and performance tracking
ca_country : Country-level data (prices, capital parameters)
"""

import numpy as np

from inseeds.components.farming.farmer import (
    Farmer,
    NON_CROPS,
    get_cell_var,
)
from inseeds.components.farming.ca_behaviour import (
    TPB,
    BLOCKER_AFFORDABILITY_FORCED,
    DRIVER_AFFORDABILITY_FORCED,
)
from inseeds.components.farming.ca_management import (
    DESELECT_ORDER,
    ManagementCosts,
    PRACTICE_FIELDS,
)
from inseeds.components.exogenous.madrat import ResidueSource


class ConservationAgricultureFarmer(Farmer):
    """A farming household that makes Conservation Agriculture decisions.

    This is the main agent class in the CA model. Each farmer operates a
    plot of land, chooses agricultural practices, and learns from neighbours.

    Example
    -------
    >>> # Access a farmer's current state:
    >>> farmer = model.world.farmers[0]
    >>> print(farmer.behaviour.practice_bundle.label)
    'conservation agriculture'
    >>> print(f"Capital: ${farmer.capital:.0f}")
    Capital: $15000

    Attributes
    ----------
    behaviour : TPB
        The decision model that determines practice transitions.
    capital : float
        Current capital stock (USD).
    practice_costs : ManagementCosts
        Costs for each practice (direct and transition).

    Capital Dynamics (hybrid FAO + behavioral model)
    ------------------------------------------------
    **Initial capital**::

        K₀ = NCS / agricultural_land_area

    where NCS = Net Capital Stocks from FAO (million USD)

    **Depreciation** (Jorgenson 1963)::

        Depreciation = δ × K
        where δ = CFC / NCS (Consumption of Fixed Capital / Net Capital Stocks)

    **Two-component investment**:

    1. Replacement: ``i × K``
       where i = GFCF / NCS (structural, FAO-derived).
       Farmers replace worn equipment regardless of profit.

    2. Discretionary: ``s × max(profit, 0)``
       where s = savings_rate (behavioral parameter).
       Profitable farmers invest more to expand.

    **Annual update**::

        K_{t+1} = K_t - δK_t + i×K_t + s × max(profit, 0)

    Profit Calculation
    ------------------
    ::

        Revenue = LPJmL harvest (gC) × crop fraction × area × FAO prices
        Variable costs = direct costs per practice × farm size
        Profit = Revenue - Variable costs - Depreciation

    Affordability Constraints
    -------------------------
    - If capital < min_capital: farmer deselects costly practices until affordable
    - Transition costs are deducted when switching practices (if capital suffices)

    Cover Crop Type Selection
    -------------------------
    Choice between legume (N-fixing) vs non-legume based on nutrient conditions:

    - High N leaching + high fertilizer use → **non-legume** (catch crop to capture excess N)
    - Otherwise → **legume** (for biological nitrogen fixation)

    See Also
    --------
    ca_behaviour.TPB : The decision model
    ca_management.ManagementBundle : Practice bundle encoding
    ca_country.CACountry : Source of FAO economic data
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
        econ = self.model.config.coupled_config.farm_economics.to_dict()

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
        self.n_survival_years = econ.get("n_survival_years")

        # Savings rate: fraction of profit reinvested into farm capital
        # This is a behavioral parameter representing farmer investment decisions.
        # Literature suggests farm savings rates of 10-30% depending on region
        # and farm type (Lowder et al. 2016; FAO 2017).
        self.savings_rate = econ.get("savings_rate", 0.15)

        # Effective depreciation rate: optionally override FAO rate
        # FAO rate (~8%) is for machinery, but FAO capital includes land (~60-70%)
        # which doesn't depreciate. If set, this overrides the FAO country rate.
        # If null/None, uses FAO country-specific rate from capital stock data.
        self._effective_depreciation_rate_override = econ.get(
            "effective_depreciation_rate", None
        )

        # -----------------------------------------------------------------
        # Get FAO data from country (loaded once per country, not per farmer)
        # -----------------------------------------------------------------
        country = self.cell.country

        # Economic parameters from FAO capital stock
        # Store FAO rate, but use effective rate (override or FAO) for calculations
        self._fao_depreciation_rate = country.depreciation_rate
        self.investment_rate = country.investment_rate
        self.initial_capital_per_ha = country.initial_capital_per_ha

        # Use effective rate: config override if set, otherwise FAO rate
        # FAO rate (~8%) is for machinery, but capital includes land (0% depreciation)
        # so effective rate is typically lower (~1-2%) when land value is included
        if self._effective_depreciation_rate_override is not None:
            self.depreciation_rate = self._effective_depreciation_rate_override
        else:
            self.depreciation_rate = self._fao_depreciation_rate

        # Producer prices for profit calculation
        self.pft_prices = country.pft_prices

        # -----------------------------------------------------------------
        # Capital initialization
        # -----------------------------------------------------------------
        # Use net_farm_size (crop area only) rather than gross_farm_size
        # because capital should be proportional to productive capacity.
        # Revenue comes from crops, so capital supporting that production
        # should scale with crop area, not total agricultural land.
        self.capital = self.initial_capital_per_ha * self.net_farm_size

        # -----------------------------------------------------------------
        # Load practice costs
        # -----------------------------------------------------------------
        # Costs include:
        # - direct: annual operating cost per ha
        # - transition: one-time cost when adopting practice
        self.practice_costs = ManagementCosts.from_config(
            self.model.config.coupled_config.practice_costs
        )

        # Residue opportunity cost ($/ha)
        self._residue_opportunity_cost_per_ha = self.compute_residue_opportunity_cost()

        # -----------------------------------------------------------------
        # Pre-compute revenue calculation mappings (for performance)
        # -----------------------------------------------------------------
        self._init_revenue_mapping()

        # -----------------------------------------------------------------
        # Initialize behaviour
        # -----------------------------------------------------------------
        # Store baseline residue level (before any CA adoption)
        self.residue_baseline = self.residue_on_field

        # Create TPB decision model
        self.behaviour = TPB(self)

    def __repr__(self) -> str:
        bundle = self.behaviour.practice_bundle if hasattr(self, "behaviour") else None
        bundle_label = bundle.label if bundle else "initializing"
        return (
            f"CAFarmer(cell={self.cell.cell_index}, bundle='{bundle_label}', "
            f"capital={self.capital:.0f})"
        )

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
        # Use net_farm_size for consistency with capital initialization
        initial_capital = self.initial_capital_per_ha * self.net_farm_size
        return self.n_survival_years * self.depreciation_rate * initial_capital

    @property
    def farm_size(self):
        """Farm size in hectares (gross cropped area, for output reporting)."""
        return self.gross_farm_size

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
        return self._residue_opportunity_cost_per_ha * self.net_farm_size

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

    def compute_residue_opportunity_cost(self):
        """Compute opportunity cost of retaining residue ($/ha).

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
        res_config = self.model.config.coupled_config.residue_economics
        use_costs = res_config.use_costs.to_dict()
        fractions = self.get_residue_fractions()

        # No spatial data → use default from config
        if fractions is None:
            default_use = res_config.default_removal_use
            return use_costs.get(default_use, use_costs.get("other", 40.0))

        # Weighted average based on actual residue use in this cell
        return (
            fractions["burnt"] * use_costs.get("burnt", 0.0)
            + fractions["removed"] * use_costs.get(
                "removed", use_costs.get("other", 40.0)
            )
            + fractions["recycled"] * use_costs.get("recycled", 0.0)
        )

    def get_residue_fractions(self):
        """Get residue use fractions for this cell, weighted by crop mix.

        Uses MADRaT CFT-specific data weighted by actual cftfrac from LPJmL.

        Returns
        -------
        dict or None
            Dict with 'burnt', 'removed', 'recycled' fractions (0-1),
            or None if residue data not available.
        """
        if "residue" not in self.model.world.exogenous.keys():
            return None

        cftfrac = get_cell_var(self.cell, "cftfrac", drop_band=NON_CROPS)
        rf_vals = cftfrac.where(
            cftfrac.band.str.startswith("rainfed"), drop=True
        ).values
        ir_vals = cftfrac.where(
            cftfrac.band.str.startswith("irrigated"), drop=True
        ).values

        # Use local_index for country-subsetted exogenous data
        cell_idx = (
            getattr(self.cell, "local_index", None)
            or getattr(self.cell, "_local_index", None)
        )
        if cell_idx is None:
            cell_idx = self.cell.grid.cell.item()

        return ResidueSource.weighted_fractions(
            self.model.world.exogenous.residue,
            cell_idx,
            rf_vals.flatten() + ir_vals.flatten(),
        )

    # =========================================================================
    # COVER CROP TYPE SELECTION
    # =========================================================================

    def indicate_cover_crop_type(self):
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
        # fertilizer_val = self.cell_fertilizer

        # Get thresholds from config
        cc = self.model.config.coupled_config.practice_dimensions.cover_crop
        leaching_limit = cc.leaching_high
        # fertilizer_limit = cc.fertilizer_high

        # Calculate leaching rate (normalized by runoff)
        leaching = leaching_val *1e3 / runoff if runoff > 0 else 0

        # -----------------------------------------------------------------
        # Decision logic
        # -----------------------------------------------------------------
        # High leaching: soil has excess N that's being lost
        # → Use non-legume catch crop to capture nutrients
        if leaching >= leaching_limit: # and fertilizer_val >= fertilizer_limit:
            return 1  # Non-legume

        # Otherwise: soil may be N-limited
        # → Use legume for biological N fixation
        return 2  # Legume

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def get_current_direct_costs(self):
        """Calculate annual direct costs of current practice bundle.

        Direct costs are ongoing annual costs for each active practice
        (e.g., seeds for cover crops, foregone income from residue retention).

        Returns
        -------
        float
            Total annual direct cost (scaled by farm size).
        """
        return (
            self.behaviour.practice_bundle.direct_cost_per_ha(self.practice_costs)
            * self.net_farm_size
        )

    # =========================================================================
    # CAPITAL UPDATE
    # =========================================================================

    def update_capital(self):
        """Update farmer's capital based on profit and depreciation.

        Capital changes through two mechanisms:

        1. **Depreciation** (capital wear)
           Machinery and buildings lose value over time. The effective
           depreciation rate accounts for capital composition:

           - FAO Capital Stock includes land (~65%), buildings (~20%),
             machinery (~15%) [USDA ERS 2022; FAO 2023]
           - Land does NOT depreciate (appreciates over time)
           - Buildings: 2-5%/year (40-50 year service life) [Eurostat 2013]
           - Machinery: 10-15%/year (7-10 year service life) [OECD 2009]
           - FAO aggregate rate (~8%) reflects machinery-weighted average
           - Effective rate for total capital: ~1-2%

        2. **Reinvestment** (from gross profit)
           Farmers reinvest a fraction of gross profit (revenue - costs)
           to maintain and expand operations. Based on gross profit (cash
           flow), not net profit, because depreciation is an accounting
           concept, not a cash outflow.

        Annual update formula::

            K_{t+1} = K_t - δK_t + s × max(gross_profit, 0)

        where:
        - δ = effective_depreciation_rate (~1-2%)
        - s = savings_rate (~15-30%)

        Money Conservation
        ------------------
        All investment comes from actual revenue. No capital is created
        from nothing. Capital grows if ``s × gross_profit > δK``, shrinks
        if ``s × gross_profit < δK``, and is stable at equality.

        Scientific Justification
        ------------------------
        The effective depreciation rate (~1.2%) rather than FAO rate (~8%)
        is used because:

        1. FAO Net Capital Stock includes land value, which dominates
           agricultural assets (83% in US per USDA ERS 2022)
        2. Land does not depreciate; it typically appreciates
        3. Only machinery (~15% of capital) depreciates at ~10-15%/year
        4. Weighted rate: 0.65×0% + 0.20×3% + 0.15×12% ≈ 2.4%
        5. We use 1.2% as a conservative lower bound that ensures
           capital stability with LPJmL-derived revenue levels

        References
        ----------
        Jorgenson, D.W. (1963). Capital Theory and Investment Behavior.
            American Economic Review 53(2): 247-259.
        OECD (2009). Measuring Capital - OECD Manual, 2nd Edition.
            OECD Publishing, Paris. (Asset service lives)
        Eurostat (2013). Handbook on prices and volumes in national accounts.
            (Depreciation rates by asset type)
        FAO (2023). FAOSTAT Capital Stock methodology.
            (Net Capital Stock composition)
        USDA ERS (2022). Farm Sector Balance Sheet.
            (Land = 83% of US farm assets)
        """

        # -----------------------------------------------------------------
        # Step 1: Calculate profit from farming
        # -----------------------------------------------------------------
        # Revenue from crop sales (LPJmL yields × FAO prices)
        revenue = self.calculate_revenue()
        # Variable costs for current practices (per-ha costs × farm size)
        variable_costs = self.get_current_direct_costs()

        # Gross profit before depreciation
        gross_profit = revenue - variable_costs

        # -----------------------------------------------------------------
        # Step 2: Depreciation (capital wear)
        # -----------------------------------------------------------------
        # Depreciation rate can be:
        # - FAO rate (~8%): applies to machinery, but FAO capital includes land
        # - Effective rate (~1-2%): calibrated for capital including land value
        #
        # If effective_depreciation_rate is set in config, it overrides the FAO rate.
        # This accounts for land (~60-70% of FAO capital) not depreciating.
        #
        # Example: FAO rate 8% × land fraction 0.15 ≈ 1.2% effective rate
        depreciation = self.depreciation_rate * self.capital

        # -----------------------------------------------------------------
        # Step 3: Reinvestment (from gross profit)
        # -----------------------------------------------------------------
        # Reinvestment comes from gross profit (cash flow), not net profit.
        # Depreciation is an accounting concept - it doesn't reduce cash.
        # Farmers reinvest from what they actually earn (revenue - costs).
        #
        # This is money-conserving: all capital comes from actual revenue.
        # - If savings_rate × gross_profit > depreciation: capital grows
        # - If savings_rate × gross_profit < depreciation: capital shrinks
        # - If savings_rate × gross_profit = depreciation: capital stable
        #
        # Literature: Lowder et al. (2016); FAO (2017) suggest 10-30% range.
        # Higher savings rates needed to maintain capital with high depreciation.
        reinvestment = self.savings_rate * max(gross_profit, 0.0)

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
        # K_next = K - δK + s × max(gross_profit, 0) + loans
        #
        # Money-conserving model:
        # - Depreciation reduces capital (physical wear)
        # - Reinvestment adds capital (from actual earnings)
        # - All capital flows are sourced from real revenue
        #
        # Capital dynamics depend on gross_profit vs depreciation:
        # - If s × gross_profit > δK: capital grows (profitable farming)
        # - If s × gross_profit < δK: capital shrinks (unprofitable)
        # - If s × gross_profit = δK: capital stable (break-even)
        #
        # For stability, farmers need: gross_profit ≥ δK / s
        # Example: δ=8%, s=15%, K=$1M → need gross_profit ≥ $533K
        self.capital = (
            self.capital
            - depreciation
            + reinvestment
            + external_financing
        )

        # Capital cannot go negative
        self.capital = max(0.0, self.capital)


    # =========================================================================
    # REVENUE CALCULATION
    # =========================================================================

    def _init_revenue_mapping(self):
        """Pre-compute band→price mappings for fast revenue calculation.

        Called once at init to avoid repeated string operations and xarray
        lookups during simulation. The mapping is constant for the simulation.
        """
        # Get band names from cftfrac (excluding NON_CROPS)
        cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS)
        bands = list(cftfrac.band.values)

        # Get prices and their categories
        prices = self.pft_prices
        price_dim = "npft" if "npft" in prices.dims else "band"
        price_categories = list(prices[price_dim].values)

        # Build mapping: for each price category, which band indices contribute?
        # Also store the price for each category
        self._revenue_groups = []  # List of (band_indices, price) tuples

        # Group bands by their price category
        category_to_indices = {}
        for i, band in enumerate(bands):
            # Strip 'rainfed ' or 'irrigated ' prefix
            if band.startswith("rainfed "):
                category = band[8:]
            elif band.startswith("irrigated "):
                category = band[10:]
            else:
                category = band

            if category in price_categories:
                if category not in category_to_indices:
                    category_to_indices[category] = []
                category_to_indices[category].append(i)

        # Convert to numpy arrays for fast indexing
        for category, indices in category_to_indices.items():
            price = float(prices.sel({price_dim: category}).values)
            self._revenue_groups.append((np.array(indices), price))

        # Cache cell area (constant)
        self._cell_area = self.cell.area.item()

    def calculate_revenue(self):
        """Calculate revenue from crop production.

        Converts LPJmL harvest (grams of carbon per m²) to monetary value
        using FAO producer prices. Uses pre-computed band-to-price mappings
        for computational efficiency.

        Conversion Formula
        ------------------
        ::

            production_tonnes = Σ(harvestc_i × cftfrac_i) × cell_area / (0.45 × 1e6)
            revenue = Σ(production_i × price_i)

        Where:
        - harvestc_i: harvest in gC/m² for crop i (LPJmL pft_harvestc)
        - cftfrac_i: cell fraction for crop i (LPJmL cftfrac)
        - cell_area: cell area in m² (pycopanlpjml cell.area)
        - 0.45: carbon fraction in plant dry matter [Wirsenius 2000; IPCC 2006]
        - price_i: FAO producer price for crop i (USD/tonne)

        Unit Verification
        -----------------
        - harvestc × cftfrac × cell_area = gC/m² × 1 × m² = gC
        - gC / 0.45 = g dry matter (carbon is ~45% of plant biomass)
        - g DM / 1e6 = tonnes dry matter
        - tonnes × USD/tonne = USD ✓

        Notes
        -----
        This is gross revenue, not profit. Costs (practice costs, transition
        costs) are subtracted separately in update_capital().

        LPJmL yields may be lower than real-world intensive agriculture due
        to model limitations in representing irrigation, fertilization, and
        modern crop varieties.

        References
        ----------
        Wirsenius, S. (2000). Human Use of Land and Organic Materials.
            Chalmers University. (Carbon content of crops)
        IPCC (2006). Guidelines for National GHG Inventories, Vol 4.
            (Default carbon fractions for crop biomass)
        Bondeau, A. et al. (2007). Modelling the role of agriculture for
            the 20th century global terrestrial carbon balance.
            Global Change Biology 13(3): 679-706. (LPJmL crop module)

        Returns
        -------
        float
            Total revenue in currency units (USD).
        """
        # Get numpy arrays directly (fast)
        harvestc = self.get_from_earth("pft_harvestc", drop_band=NON_CROPS).values
        cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS).values

        # Production in tonnes dry matter (vectorized numpy)
        # gC → tonnes DM: divide by (C_fraction × g_per_tonne)
        production = harvestc * cftfrac * self._cell_area / (0.45 * 1e6)

        # Sum revenue across all price categories using pre-computed mappings
        total_revenue = 0.0
        for indices, price in self._revenue_groups:
            total_revenue += production[indices].sum() * price

        return total_revenue

    # =========================================================================
    # AFFORDABILITY CHECK
    # =========================================================================

    def check_practice_affordability(self):
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
        current_direct_costs = self.get_current_direct_costs()

        # Available capital for costs (above survival threshold)
        available_capital = self.capital - self.min_capital

        # No action needed if costs are within budget
        if current_direct_costs <= available_capital:
            return

        # -----------------------------------------------------------------
        # Deselect practices until costs are affordable
        # -----------------------------------------------------------------
        bundle = self.behaviour.practice_bundle
        costs = self.practice_costs

        for practice_name in DESELECT_ORDER:
            if getattr(bundle, practice_name) == 0:
                continue

            # Only deselect practices with positive direct cost
            direct_cost = getattr(costs, practice_name).direct
            if direct_cost > 0:

                # Recalculate costs with this practice removed
                # (simplified: subtract the practice's direct cost)
                bundle = bundle.change_practices(**{practice_name: 0})
                current_direct_costs -= direct_cost * self.net_farm_size

                # Check if costs are now within budget
                if current_direct_costs <= available_capital:
                    break

        # -----------------------------------------------------------------
        # Apply changes if bundle changed
        # -----------------------------------------------------------------
        if bundle != self.behaviour.practice_bundle:
            self.behaviour.apply_bundle(bundle)
            self.behaviour.record_transition(bundle)
            # Record that this was a forced transition due to affordability
            self.behaviour._transition_blocker = BLOCKER_AFFORDABILITY_FORCED
            self.behaviour._transition_driver = DRIVER_AFFORDABILITY_FORCED

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
        6. Apply practice transition if TPB threshold exceeded

        Sets behaviour._transition_blocker to indicate why transition didn't happen.

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # Import blocker/driver constants here to avoid circular imports
        from inseeds.components.farming.ca_behaviour import (
            BLOCKER_NONE, BLOCKER_CONTROL_RUN, BLOCKER_CAPITAL_SURVIVAL,
            BLOCKER_TRANSITION_UNAFFORDABLE, BLOCKER_EVALUATION_TIME,
            DRIVER_NONE, DRIVER_FALLBACK
        )

        # -----------------------------------------------------------------
        # Step 1: Parent update
        # -----------------------------------------------------------------
        super().update(t)

        # -----------------------------------------------------------------
        # Step 2: Skip CA dynamics in control run
        # -----------------------------------------------------------------
        if self.control_run:
            self.behaviour._transition_blocker = BLOCKER_CONTROL_RUN
            self.behaviour._transition_driver = DRIVER_NONE
            return

        # -----------------------------------------------------------------
        # Step 3: Update capital
        # -----------------------------------------------------------------
        self.update_capital()

        # -----------------------------------------------------------------
        # Step 4: Check affordability (may deselect practices if capital too low)
        # -----------------------------------------------------------------
        self.check_practice_affordability()

        # -----------------------------------------------------------------
        # Step 5: Skip TPB if capital-constrained (survival mode)
        # -----------------------------------------------------------------
        if self.capital < self.min_capital:
            self.behaviour._transition_blocker = BLOCKER_CAPITAL_SURVIVAL
            self.behaviour._transition_driver = DRIVER_NONE
            return

        # -----------------------------------------------------------------
        # Step 6: Check if farmer should evaluate this year
        # -----------------------------------------------------------------

        if not self.behaviour.should_evaluate():
            self.behaviour._transition_blocker = BLOCKER_EVALUATION_TIME
            self.behaviour._transition_driver = DRIVER_NONE
            self.behaviour.decrement_evaluation_time()
            return

        # -----------------------------------------------------------------
        # Step 7: Run TPB decision logic
        # -----------------------------------------------------------------

        self.behaviour.update()

        # -----------------------------------------------------------------
        # Step 8: Apply transition if TPB threshold exceeded
        # -----------------------------------------------------------------
        if self.behaviour.should_transition():
            new_bundle = self.behaviour.proposed_bundle

            if new_bundle is not None:
                old_bundle = self.behaviour.practice_bundle

                flipped = [
                    field for field in PRACTICE_FIELDS
                    if getattr(old_bundle, field) != getattr(new_bundle, field)
                ]

                if flipped:
                    cost = (
                        old_bundle.transition_cost_per_ha(
                            new_bundle, self.practice_costs
                        )
                        * self.net_farm_size  # Costs scale with crop area
                    )

                    # Only apply if farmer can afford transition
                    if self.capital >= cost:
                        # Deduct transition cost
                        self.capital -= cost

                        # Apply new practices
                        self.behaviour.apply_bundle(new_bundle)
                        self.behaviour.record_transition(new_bundle)
                        # Clear blocker since transition succeeded
                        self.behaviour._transition_blocker = BLOCKER_NONE

                        # Set driver to indicate why transition succeeded
                        pathway = self.behaviour.target_pathway
                        if pathway == "fallback":
                            self.behaviour._transition_driver = DRIVER_FALLBACK
                        elif pathway in ("social", "exploration"):
                            self.behaviour.set_tpb_component_driver(pathway)
                    else:
                        # Can't afford transition cost
                        self.behaviour._transition_blocker = BLOCKER_TRANSITION_UNAFFORDABLE
        else:
            # TPB score below threshold - identify which component is limiting
            self.behaviour.set_tpb_transition_blocker()

        # -----------------------------------------------------------------
        # Step 9: Reset evaluation time after evaluation completes
        # -----------------------------------------------------------------
        # Whether farmer transitioned or not, they've evaluated and will wait
        # before reconsidering (randomized interval around evaluation_interval)
        self.behaviour.reset_evaluation_time()
