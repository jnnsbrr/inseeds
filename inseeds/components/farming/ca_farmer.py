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
FAO baseline + deviations model::

    K_{t+1} = K × (1 + i - δ) + Δrevenue - Δcosts

Where:
- **(i - δ)** = FAO net rate (investment - depreciation, typically +2-7%/year)
- **Δrevenue** = current_revenue - baseline_revenue (from historic 2015-2025)
- **Δcosts** = cost difference between current and baseline bundle

Why This Works
~~~~~~~~~~~~~~
FAO (i - δ) captures everything at sector average: depreciation, reinvestment,
subsidies, loans, typical yields and costs. We only track DEVIATIONS:

- Better yields → Δrevenue > 0 → capital grows faster
- Worse yields → Δrevenue < 0 → capital grows slower
- Higher costs (CA adoption) → Δcosts > 0 → capital grows slower
- Lower costs (no-till savings) → Δcosts < 0 → capital grows faster

References:
- Solow, R.M. (1956). "A Contribution to the Theory of Economic Growth."
- OECD (2009). Measuring Capital Manual, 2nd ed. (methodology)
- FAO (2023). FAOSTAT Capital Stock (source data)

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
from inseeds.components.farming.ca_management import ManagementBundle

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

    Capital Dynamics (FAO baseline + deviations)
    --------------------------------------------
    **Initial capital**::

        K₀ = NCS / agricultural_land_area

    where NCS = Net Capital Stocks from FAO (million USD)

    **Annual update**::

        K_{t+1} = K × (1 + i - δ) + Δrevenue - Δcosts

    Where:

    - ``i - δ`` = FAO net rate (investment - depreciation, typically +2-7%/year)
    - ``Δrevenue`` = current_revenue - baseline_revenue
    - ``Δcosts`` = cost difference between current and baseline bundle

    **Why this works**: FAO (i - δ) captures sector average dynamics including
    subsidies, typical costs, typical yields. We only track DEVIATIONS from
    this baseline - no need to model all components explicitly.

    Baselines
    ---------
    - ``baseline_revenue``: Average over historic period (2015-2025)
    - ``baseline_bundle``: Initial practice bundle at simulation start

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
        # Get FAO data from country (loaded once per country, not per farmer)
        # -----------------------------------------------------------------
        # The FAO-based capital dynamics model uses observed sector-level rates
        # that implicitly capture all income sources and cost structures.
        country = self.cell.country

        # FAO depreciation rate: δ = CFC / NCS (consumption of fixed capital / net capital stock)
        # This is the observed rate at which agricultural capital loses value.
        # Typically 3-8% per year (higher for machinery-intensive countries).
        self.fao_depreciation_rate = country.depreciation_rate

        # FAO investment rate: i = GFCF / NCS (gross fixed capital formation / net capital stock)
        # This is the observed rate at which new capital is added to the stock.
        # Crucially, GFCF includes ALL sources of capital formation:
        #   - Farmer reinvestment from profits
        #   - Bank loans and credit
        #   - Government subsidies and grants (EU CAP, US farm bill, etc.)
        #   - External/foreign investment
        # Typically 5-15% per year.
        self.fao_investment_rate = country.investment_rate

        # Initial capital per hectare from FAO Net Capital Stock
        self.initial_capital_per_ha = country.initial_capital_per_ha

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
        # Costs are scaled by farmer's capital relative to reference country.
        # This accounts for global variation in equipment and input costs.
        # Costs include:
        # - direct: annual operating cost per ha (scaled)
        # - transition: one-time cost when adopting practice (scaled)
        self.practice_costs = ManagementCosts.from_config(
            self.model.config.coupled_config.practice_costs,
            capital_per_ha=self.initial_capital_per_ha,
            model=self.model,
        )

        # -----------------------------------------------------------------
        # Pre-compute revenue calculation mappings (for performance)
        # -----------------------------------------------------------------
        self._init_revenue_mapping()

        # Cache for current revenue (updated each year in update_capital)
        # Used for both capital dynamics and residue opportunity cost
        self._current_revenue = None

        # -----------------------------------------------------------------
        # Compute baseline revenue from historic data
        # -----------------------------------------------------------------
        # The baseline is the average revenue over the historic period
        # (pre-coupling years, e.g., 2015-2025). This represents what
        # the FAO capital dynamics already account for.
        self.baseline_revenue = self._compute_baseline_revenue()

        # Store baseline residue opportunity cost (frozen at simulation start)
        # Used for cost difference calculations to avoid dynamic cost changes
        self.baseline_residue_cost = self.compute_residue_opportunity_cost(
            revenue=self.baseline_revenue
        )

        # -----------------------------------------------------------------
        # Initialize behaviour
        # -----------------------------------------------------------------
        # Store baseline residue level (before any CA adoption)
        self.residue_baseline = self.residue_on_field

        # Create TPB decision model
        self.behaviour = TPB(self)

        # Store initial practice bundle as baseline
        # Cost differences are computed relative to this bundle, since FAO
        # baseline already includes whatever costs the farmer had at start.
        self.baseline_bundle = self.behaviour.practice_bundle


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

        Defined as a fraction of initial capital:
            min_capital = min_capital_fraction × initial_capital

        Based on Farm Financial Scorecard (CFFM, University of Minnesota):
        - Working capital / gross revenue > 0.35 = strong liquidity
        - Working capital / gross revenue < 0.20 = vulnerable
        Default fraction of 0.25 is the midpoint.

        Below this threshold, farmer must deselect costly practices to survive.

        Note: FAO NCS excludes land (SNA 2008 definition), so this fraction
        applies to productive capital (machinery, equipment, buildings, crops).

        References
        ----------
        - Farm Financial Scorecard (CFFM, University of Minnesota)
        - Davis, J. (2022). "Farm's Target for Working Capital." SDSU Extension.
        - Wisconsin Extension (2021). Farm Finance Scorecard guidelines.

        Returns
        -------
        float
            Minimum capital in currency units.
        """
        initial_capital = self.initial_capital_per_ha * self.net_farm_size
        fraction = self.model.config.coupled_config.farm_economics.min_capital_fraction
        return fraction * initial_capital

    @property
    def farm_size(self):
        """Farm size in hectares (gross cropped area, for output reporting)."""
        return self.gross_farm_size

    # =========================================================================
    # RESIDUE ECONOMICS
    # =========================================================================

    def compute_residue_opportunity_cost(self, revenue: float | None = None):
        """Compute opportunity cost of retaining residue (total for farm).

        Implements the Singh & Schiere (1995) finding that straw value represents
        10-15% of total crop value. The cost is computed dynamically as:

            residue_cost = crop_revenue × use_cost_fraction

        This makes residue costs spatially and temporally variable:
        - Higher yields → higher residue opportunity cost
        - Lower yields → lower residue opportunity cost

        Uses MADRaT data (Smerald et al. 2023) for spatially-explicit fractions
        of residue burnt, removed, and recycled. The final cost is the weighted
        sum across use types.

        MADRaT data structure:
            production = recycled + removed + burnt
            - burnt: burned (no economic value → 0% of crop value)
            - removed: animal feed (has value → 12.5% of crop value)
            - recycled: returns to field (no cost → 0% of crop value)

        Parameters
        ----------
        revenue : float, optional
            Revenue to use for computation. If None, uses cached current revenue
            or computes it. Pass baseline_revenue for baseline cost calculation.

        Returns
        -------
        float
            Total opportunity cost for the farm ($/yr).

        References
        ----------
        Singh, K. & Schiere, J.B. (eds.) (1995). Handbook for Straw Feeding
            Systems. ICAR, New Delhi. Ch. 1.1, Box 1: "straw value = 10-15%
            of crop value". https://edepot.wur.nl/333326
        """
        res_config = self.model.config.coupled_config.residue_economics
        cost_fractions = res_config.use_cost_fractions.to_dict()
        use_fractions = self.get_residue_fractions()

        # Use provided revenue, cached revenue, or compute fresh
        if revenue is None:
            if self._current_revenue is not None:
                revenue = self._current_revenue
            else:
                revenue = self.calculate_revenue()

        # Weighted average based on actual residue use in this cell
        # residue_cost = revenue × Σ(use_fraction × cost_fraction)
        weighted_cost_fraction = (
            use_fractions["burnt"] * cost_fractions["burnt"]
            + use_fractions["removed"] * cost_fractions["removed"]
            + use_fractions["recycled"] * cost_fractions["recycled"]
        )

        return revenue * weighted_cost_fraction

    def get_residue_fractions(self):
        """Get residue use fractions for this cell, weighted by crop mix.

        Uses MADRaT CFT-specific data weighted by actual cftfrac from LPJmL.
        Result is cached since residue use fractions don't change during simulation.

        Returns
        -------
        dict or None
            Dict with 'burnt', 'removed', 'recycled' fractions (0-1),
            or None if residue data not available.
        """
        # Return cached value if available (computed once, used throughout simulation)
        if hasattr(self, "_cached_residue_fractions"):
            return self._cached_residue_fractions

        if "residue" not in self.model.world.exogenous.keys():
            self._cached_residue_fractions = None
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

        result = ResidueSource.weighted_fractions(
            self.model.world.exogenous.residue,
            cell_idx,
            rf_vals.flatten() + ir_vals.flatten(),
        )

        # Cache for subsequent calls
        self._cached_residue_fractions = result
        return result

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
        # High leaching: soil has excess N that's being lost
        # → Use non-legume catch crop to capture nutrients
        if leaching >= leaching_limit or fertilizer_val >= fertilizer_limit:
            return 1  # Non-legume

        # Otherwise: soil may be N-limited
        # → Use legume for biological N fixation
        return 2  # Legume

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def get_practice_direct_cost_per_ha(self, field: str, value: int) -> float:
        """Get per-hectare direct cost for a single practice state.

        Costs are defined relative to the CA reference point (no-till, no cover crop):
        - tillage=0 (no-till): 0 (reference point)
        - tillage=1 (conventional): +cost (pays more than no-till)
        - cover_crop=0: 0 (no cover crop)
        - cover_crop=1: +cost (pays for cover crop)

        This encoding ensures correct cost differences when comparing bundles:
        conventional→no-till: 0 - (+cost) = -cost (saves money) ✓

        Parameters
        ----------
        field : str
            Practice name ('tillage', 'cover_crop')
        value : int
            Practice state (0 or 1)

        Returns
        -------
        float
            Per-hectare direct cost relative to CA reference.
        """
        costs = self.practice_costs

        if field == "tillage":
            # Conventional (tillage=1) costs more; no-till (tillage=0) is reference
            return costs.tillage.direct if value == 1 else 0.0
        elif field == "cover_crop":
            # Cover crop (value=1) has cost; no cover crop (value=0) is reference
            return costs.cover_crop.direct if value == 1 else 0.0
        else:
            # Residue handled separately (not per-ha)
            return 0.0

    def get_residue_cost_for_bundle(self, residue_on_field: int) -> float:
        """Get residue cost/income based on retention state.

        The residue model captures opportunity costs and income dynamics:

        **Retention (residue=1):**
        - Farmer committed to leaving residue on field
        - Cost = baseline (frozen) — the income they committed to give up
        - Higher yields: extra residue is "free" to leave (no extra cost)
        - Lower yields: still pays baseline cost, just leaves less physically

        **Selling (residue=0):**
        - Farmer sells residue at market price
        - Cost = baseline - current (can be negative = income)
        - Higher yields: sells more → negative cost (extra income)
        - Lower yields: sells less → positive cost (lost income vs baseline)

        Parameters
        ----------
        residue_on_field : int
            1 = retaining residue, 0 = selling residue

        Returns
        -------
        float
            Residue cost (positive) or income (negative).
        """
        if residue_on_field == 1:
            # Retention: fixed opportunity cost (committed to giving up baseline income)
            # No need to compute current cost - we use frozen baseline
            return self.baseline_residue_cost
        else:
            # Selling: income/loss relative to baseline
            # Good year (current > baseline): negative cost = extra income
            # Bad year (current < baseline): positive cost = lost income
            current = self.compute_residue_opportunity_cost()
            return self.baseline_residue_cost - current

    def get_bundle_direct_costs(self, bundle: ManagementBundle, per_ha: bool = False):
        """Calculate annual direct costs of current practice bundle.

        Direct costs are ongoing annual costs for each practice state, defined
        relative to the CA reference point (no-till, no cover crop):
        - Tillage: conventional (tillage=1) costs +$X/ha; no-till (tillage=0) = 0
        - Cover crop: active (value=1) costs +$Y/ha; none (value=0) = 0
        - Residue: see get_residue_cost_for_bundle() for retention vs selling logic

        Note: Tillage and cover_crop costs are per-ha (scaled by farm size).
        Residue cost is already total (computed from total revenue).

        This encoding ensures correct cost differences between bundles:
        - Full CA bundle (no-till + cover crop): 0 + cover_cost + residue_cost
        - Conventional bundle: tillage_cost + 0 + 0
        - Difference correctly reflects actual cost change

        Returns
        -------
        float
            Total annual direct cost (or per-ha if per_ha is True).
        """
        # Per-ha costs (tillage, cover_crop)
        total_per_ha = (
            self.get_practice_direct_cost_per_ha("tillage", bundle.tillage)
            + self.get_practice_direct_cost_per_ha("cover_crop", bundle.cover_crop)
        )

        # Scale per-ha costs to total farm cost
        total = total_per_ha * self.net_farm_size

        # Residue: cost depends on retention state (see get_residue_cost_for_bundle)
        total += self.get_residue_cost_for_bundle(bundle.residue_on_field)

        return total / self.net_farm_size if per_ha else total

    def get_bundle_costs_difference(
        self,
        from_bundle: ManagementBundle,
        to_bundle: ManagementBundle,
        per_ha: bool = False,
    ) -> float:
        """Compute direct cost DIFFERENCE when switching between bundles.

        Computes: to_bundle_cost - from_bundle_cost

        Cost differences:
        - Tillage: conventional→no-till = -cost (saves), no-till→conv = +cost
        - Cover crop: adopting = +cost, abandoning = -cost
        - Residue: uses baseline_residue_cost (FROZEN at sim start) for practice
          changes to separate practice-driven costs from yield-driven income

        Note: Tillage and cover_crop costs are per-ha (scaled by farm size).
        Residue cost is already total (computed from total revenue).

        Parameters
        ----------
        from_bundle : ManagementBundle
            Starting bundle (e.g., baseline at simulation start).
        to_bundle : ManagementBundle
            Current bundle.
        per_ha : bool
            If True, return per-hectare cost; if False, scale by farm size.

        Returns
        -------
        float
            Cost difference: positive = to_bundle costs more than from_bundle.
        """
        # Per-hectare cost differences (tillage, cover_crop)
        # diff = to_cost - from_cost
        diff_per_ha = (
            self.get_practice_direct_cost_per_ha("tillage", to_bundle.tillage)
            - self.get_practice_direct_cost_per_ha("tillage", from_bundle.tillage)
            + self.get_practice_direct_cost_per_ha("cover_crop", to_bundle.cover_crop)
            - self.get_practice_direct_cost_per_ha("cover_crop", from_bundle.cover_crop)
        )

        # Scale per-ha costs to total farm cost
        diff_total = diff_per_ha * self.net_farm_size

        # Residue: use frozen baseline cost for practice changes
        # (yield-driven income changes are in delta_revenue)
        # Cost is ALREADY total (from revenue × fraction), don't scale again
        if from_bundle.residue_on_field != to_bundle.residue_on_field:
            if to_bundle.residue_on_field == 1:  # Started retaining
                diff_total += self.baseline_residue_cost
            else:  # Stopped retaining (selling residue)
                diff_total -= self.baseline_residue_cost

        if per_ha:
            return diff_total / self.net_farm_size
        return diff_total

    def get_practice_transition_cost(
        self,
        field: str,
        from_val: int,
        to_val: int,
        per_ha: bool = False,
    ) -> float:
        """Get transition cost for a single practice change.

        Cost asymmetry by practice:
        - Tillage: SYMMETRIC - both directions require equipment
        - Cover crop: ASYMMETRIC - only adoption (0→1) has cost
        - Residue: No transition cost either direction

        Parameters
        ----------
        field : str
            Practice name ('tillage', 'cover_crop', 'residue_on_field')
        from_val : int
            Current practice value (0 or 1)
        to_val : int
            Target practice value (0 or 1)
        per_ha : bool
            If True, return per-hectare cost; if False, scale by farm size.

        Returns
        -------
        float
            Transition cost for this single practice change.
        """
        if from_val == to_val:
            return 0.0

        costs = self.practice_costs
        cost_per_ha = 0.0

        if field == "cover_crop":
            # ASYMMETRIC: only adoption (0→1) has cost
            if from_val == 0 and to_val == 1:
                cost_per_ha = costs.cover_crop.transition
            # Abandoning (1→0): no cost, just stop planting
        elif field == "tillage":
            # SYMMETRIC: both directions require equipment
            cost_per_ha = costs.tillage.transition
        # Residue: no transition cost (choppers are standard equipment)

        return cost_per_ha if per_ha else cost_per_ha * self.net_farm_size

    def get_bundle_transition_costs(
        self,
        current_bundle: ManagementBundle,
        target_bundle: ManagementBundle,
        per_ha: bool = False
    ) -> float:
        """One-time transition cost for all practice changes in a bundle switch.

        See get_practice_transition_cost() for per-practice cost logic.

        Returns
        -------
        float
            Total transition cost (per-ha if per_ha is True).
        """
        total = 0.0
        for field, old, new in zip(PRACTICE_FIELDS, current_bundle.value, target_bundle.value):
            total += self.get_practice_transition_cost(field, old, new, per_ha=True)

        return total if per_ha else total * self.net_farm_size


    # =========================================================================
    # CAPITAL UPDATE
    # =========================================================================

    def update_capital(self):
        """Update farmer's capital using FAO baseline + deviations model.

        Baseline + Deviations Approach
        ------------------------------
        FAO provides the baseline capital dynamics at sector level. We track
        DEVIATIONS from this baseline for individual farmers:

        - **FAO baseline**: (i - δ) × K captures average sector behavior including
          depreciation, reinvestment, subsidies, loans, and typical costs/yields.

        - **Δrevenue**: How this farmer's revenue differs from historic baseline.
          Computed from LPJmL yields vs. average over 2015-2025 (pre-coupling).

        - **Δcosts**: How this farmer's costs differ from initial practice costs.
          Initial costs are assumed part of FAO baseline.

        Annual Update Formula
        ---------------------
        ::

            K_{t+1} = K × (1 + i - δ) + Δrevenue - Δcosts

        where:
        - i = FAO investment_rate (GFCF/NCS, typically 5-15%/year)
        - δ = FAO depreciation_rate (CFC/NCS, typically 3-8%/year)
        - Δrevenue = current_revenue - baseline_revenue
        - Δcosts = cost difference from baseline_bundle to current bundle

        Key Behaviors
        -------------
        - If Δrevenue = 0 and Δcosts = 0: farmer follows FAO baseline exactly
        - Better yields (Δrevenue > 0): capital grows faster than baseline
        - Worse yields (Δrevenue < 0): capital grows slower than baseline
        - CA adoption with higher costs: Δcosts > 0 → reduces capital
        - CA adoption with savings (no-till): Δcosts < 0 → increases capital

        Why This Works
        --------------
        FAO (i - δ) already captures everything at sector average: typical yields,
        typical costs, subsidies, loans, reinvestment behavior. We don't need to
        model all these components explicitly - just the deviations from baseline.

        References
        ----------
        Solow, R.M. (1956). "A Contribution to the Theory of Economic Growth."
        OECD (2009). Measuring Capital Manual (depreciation methodology).
        FAO (2023). FAOSTAT Capital Stock (source data: GFCF, CFC, NCS).
        """

        # -----------------------------------------------------------------
        # Step 1: FAO baseline capital change
        # -----------------------------------------------------------------
        # The FAO rates (i - δ) capture the NET capital change at sector level,
        # including: depreciation, reinvestment, subsidies, loans, average costs.
        # This is our baseline - what happens to capital at sector average.
        if (self.model.config.coupled_config.farm_economics.fao_capital_baseline):
            fao_net_rate = self.fao_investment_rate - self.fao_depreciation_rate
            baseline_capital_change = fao_net_rate * self.capital
        else:
            baseline_capital_change = 0.0

        # -----------------------------------------------------------------
        # Step 2: Deviation in revenue from baseline
        # -----------------------------------------------------------------
        # baseline_revenue = average revenue from historic period (2015-2025)
        # This is what FAO already accounts for. We only track the CHANGE.
        #
        # Δrevenue > 0: yields improved → farmer does better than baseline
        # Δrevenue < 0: yields declined → farmer does worse than baseline
        #
        # Cache revenue for reuse (e.g., residue_opportunity_cost calculation)
        self._current_revenue = self.calculate_revenue()
        delta_revenue = self._current_revenue - self.baseline_revenue

        # -----------------------------------------------------------------
        # Step 3: Deviation in costs from baseline bundle
        # -----------------------------------------------------------------
        # Computes cost change from baseline bundle (at simulation start) to
        # current bundle. FAO baseline already includes starting practice costs.
        #
        # Δcosts > 0: current bundle costs more than baseline → reduces capital
        # Δcosts < 0: current bundle costs less (e.g., no-till savings) → adds capital
        delta_costs = self.get_bundle_costs_difference(
            self.baseline_bundle,
            self.behaviour.practice_bundle,
        )
        # -----------------------------------------------------------------
        # Step 4: Update capital
        # -----------------------------------------------------------------
        # K_{t+1} = K × (1 + i - δ) + Δrevenue - Δcosts
        #
        # Interpretation:
        # - FAO baseline: what happens at sector average (includes subsidies,
        #   typical costs, typical reinvestment behavior)
        # - Δrevenue: how this farmer's yields differ from baseline
        # - Δcosts: how this farmer's costs differ from baseline
        #
        # If Δrevenue = 0 and Δcosts = 0: farmer follows FAO baseline exactly
        # If yields improve or costs decrease: farmer does better than baseline
        # If yields decline or costs increase: farmer does worse than baseline
        # if delta_costs != 0:
        #     breakpoint()
        self.capital = self.capital + baseline_capital_change + delta_revenue - delta_costs

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
        cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS, time_idx=-1)
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

    def _compute_baseline_revenue(self):
        """Compute baseline revenue from historic (pre-coupling) yield data.

        The baseline revenue is the average revenue over the historic period
        (typically 2015-2025, before coupled simulation starts). This baseline
        represents what FAO capital dynamics already account for - the typical
        yields and revenues at the sector level.

        During simulation, we track DEVIATIONS from this baseline:
        - Higher yields → positive Δrevenue → capital grows faster
        - Lower yields → negative Δrevenue → capital grows slower

        Returns
        -------
        float
            Average revenue over historic period (USD).
        """
        # Get historic yield data (all available time steps)
        harvestc_da = self.get_from_earth("pft_harvestc", drop_band=NON_CROPS)
        cftfrac_da = self.get_from_earth("cftfrac", drop_band=NON_CROPS)

        # Get underlying numpy arrays (faster than xarray operations)
        harvestc = harvestc_da.values
        cftfrac = cftfrac_da.values

        # Compute production for all time steps at once (vectorized)
        # Shape: (time, band) or (band,) if single time step
        production = harvestc * cftfrac * self._cell_area / (0.45 * 1e6)

        # Average over time if multiple time steps (faster than xarray operations)
        if production.ndim > 1 and "time" in harvestc_da.dims:
            # Mean over time axis (axis=0 for time-first arrays)
            time_axis = harvestc_da.dims.index("time")
            mean_production = np.mean(production, axis=time_axis)
        else:
            mean_production = production

        # Compute revenue from mean production using pre-computed price mappings
        total_revenue = 0.0
        for indices, price in self._revenue_groups:
            total_revenue += mean_production[indices].sum() * price

        return total_revenue

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
        harvestc = self.get_from_earth("pft_harvestc", drop_band=NON_CROPS, time_idx=-1).values
        cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS, time_idx=-1).values

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
        3. Tillage change (conventional→no-till gives savings)

        Special cases:
        - Residue: When deselected, farmer sells residue and gets immediate income
          equal to the opportunity cost they were paying to retain.
        - Tillage: When switching from conventional (1) to no-till (0), farmer
          gains SAVINGS (negative cost), not just removal of cost. This is handled
          correctly by recalculating costs after each change.
        """
        # Available capital for costs (above survival threshold)
        available_capital = self.capital - self.min_capital

        # Calculate current annual direct costs
        bundle = self.behaviour.practice_bundle
        current_direct_costs = self.get_bundle_direct_costs(bundle=bundle)

        # No action needed if costs are within budget
        if current_direct_costs <= available_capital:
            return

        # -----------------------------------------------------------------
        # Deselect practices until costs are affordable
        # Recalculate costs after each change to handle bidirectional tillage
        # -----------------------------------------------------------------
        for practice_name in DESELECT_ORDER:
            if getattr(bundle, practice_name) == 0:
                continue

            # Try deselecting this practice
            new_bundle = bundle.change_practices(**{practice_name: 0})
            new_costs = self.get_bundle_direct_costs(bundle=new_bundle)

            # Only deselect if it actually reduces costs (or provides savings)
            if new_costs < current_direct_costs:
                # Special case: deselecting residue retention means SELLING residue
                # Farmer gets immediate cash from the sale (income = opportunity cost)
                if practice_name == "residue_on_field":
                    residue_value = self.compute_residue_opportunity_cost()
                    self.capital += residue_value  # Get cash from selling
                    # Update available capital since we just got income
                    available_capital = self.capital - self.min_capital

                bundle = new_bundle
                current_direct_costs = new_costs

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
            self.behaviour.transition_blocker = BLOCKER_AFFORDABILITY_FORCED
            self.behaviour.transition_driver = DRIVER_AFFORDABILITY_FORCED

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

        Sets behaviour.transition_blocker to indicate why transition didn't happen.

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # Import blocker/driver constants here to avoid circular imports
        from inseeds.components.farming.ca_behaviour import (
            BLOCKER_NONE, BLOCKER_CONTROL_RUN, BLOCKER_CAPITAL_SURVIVAL,
            BLOCKER_TRANSITION_UNAFFORDABLE, BLOCKER_OBSERVATION_YEARS,
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
            self.behaviour.transition_blocker = BLOCKER_CONTROL_RUN
            self.behaviour.transition_driver = DRIVER_NONE
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
            self.behaviour.transition_blocker = BLOCKER_CAPITAL_SURVIVAL
            self.behaviour.transition_driver = DRIVER_NONE
            # Still decrement observation counter so farmer can evaluate
            # promptly when they recover from survival mode (avoids limbo)
            self.behaviour.decrement_observation_years()
            return

        # -----------------------------------------------------------------
        # Step 6: Run TPB decision logic
        # -----------------------------------------------------------------
        # This records observations every year and checks if observation
        # period is complete before running full TPB evaluation.

        self.behaviour.update()

        # -----------------------------------------------------------------
        # Step 7: Apply transition if TPB threshold exceeded
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
                    cost = self.get_bundle_transition_costs(
                        old_bundle, new_bundle
                    )

                    # Only apply if farmer can afford transition
                    if self.capital >= cost:
                        # Deduct transition cost
                        self.capital -= cost

                        # Apply new practices
                        self.behaviour.apply_bundle(new_bundle)
                        self.behaviour.record_transition(new_bundle)
                        # Clear blocker since transition succeeded
                        self.behaviour.transition_blocker = BLOCKER_NONE

                        # Set driver to indicate why transition succeeded
                        pathway = self.behaviour.target_pathway
                        if pathway == "fallback":
                            self.behaviour.transition_driver = DRIVER_FALLBACK
                        elif pathway in ("local", "country", "cluster", "exploration"):
                            self.behaviour.set_tpb_component_driver(pathway)
                    else:
                        # Can't afford transition cost
                        self.behaviour.transition_blocker = BLOCKER_TRANSITION_UNAFFORDABLE
        else:
            # TPB score below threshold - identify which component is limiting
            self.behaviour.set_tpb_transition_blocker()

        # -----------------------------------------------------------------
        # Step 8: Reset observation years after FULL evaluation completes
        # -----------------------------------------------------------------
        # Only reset if we actually ran the full TPB evaluation (not blocked
        # by observation period). After evaluation, farmer waits again.
        if self.behaviour.transition_blocker != BLOCKER_OBSERVATION_YEARS:
            self.behaviour.reset_observation_years()
