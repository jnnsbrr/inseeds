"""Conservation Agriculture farmer agent.

One representative farmer per LPJmL cell. Each year the farmer reads the cell's
biophysical outcome from LPJmL, records it in a performance tracker, earns
revenue and pays costs, and decides through its TPB behaviour whether to change
the practices it writes back to LPJmL as land-use input.

Overview
--------
A CAFarmer represents a representative farmer per cell that:

1. Operates an LPJmL cell with specific crops
2. Chooses which CA practices to use (tillage, cover crops, residue)
3. Tracks performance over time (yield, soil carbon, moisture, leaching)
4. Learns from neighbours and may adopt new practices
5. Manages capital (income, costs, optional policy transfers)

Key Components
--------------
- **Practice Bundle**: The combination of 3 practices the farmer currently uses
- **TPB Behaviour**: Decision model for practice transitions (see ca_behaviour)
- **Performance Tracker**: Monitors trends since last practice change
- **Capital**: FAO endowment, then deviations from the farm's own baseline

Annual Update Cycle
-------------------
Each simulation year, ``update`` does:

1. Read this year's levels from LPJmL (parent update, leaching)
2. Return immediately on a control run (no CA dynamics)
3. Update capital from this year's deviation from baseline
4. If capital constraints bind and the stock is below min_capital: observe
   the year, skip the decision (survival mode)
5. Run the TPB decision (includes affordability deselection)
6. If transitioning and the one-time cost is payable: charge it and write
   the new practices back to LPJmL
7. Decrement the observation-year counter

Capital
-------
The endowment is exogenous, taken from the FAO Net Capital Stock for the
country's crop sector. Everything after that is what this farm earns and
spends relative to its own starting point::

    K_{t+1} = K + Δrevenue - Δcosts + payments - charges

where

- **Δrevenue** = current_revenue - baseline_revenue (2015-2025 pre-coupling)
- **Δcosts** = cost of the current bundle minus the start-of-run bundle
- **payments / charges** = carbon payment, practice subsidy, nitrogen charge;
  zero unless a scenario turns an instrument on

A farm that neither gains nor spends more than it used to holds its stock
steady. Better yields or no-till savings grow it; worse yields or the added
cost of a practice shrink it. Whether the stock may *refuse* a practice
change is set separately by ``farm_economics.capital_constraints``; see
``min_capital``.

Practice Encoding
-----------------
A practice bundle is three binary practices, where 0 is the conservation
agriculture state for tillage but the conventional state for the other two:

    +------------------+-------------------+-------------------+
    | Practice         | Value = 0         | Value = 1         |
    +==================+===================+===================+
    | tillage          | No-till (CA)      | Conventional      |
    | cover_crop       | No cover crop     | Cover crop        |
    | residue_on_field | Baseline removal  | Residue retained  |
    +------------------+-------------------+-------------------+

Full conservation agriculture is therefore (0, 1, 1). For tillage, 0 means
the CA practice (no-till) is active.

See Also
--------
ca_behaviour : how farmers decide
ca_management : practice bundles and performance tracking
ca_country : country-level prices and capital data

References
----------
- FAO (2023). FAOSTAT Capital Stock (endowment) and producer prices.
"""

import logging

import numpy as np

from inseeds.components.farming.farmer import (
    Farmer,
    GC_M2_TO_TCO2_HA,
    GN_M2_TO_KGN_HA,
    NON_CROPS,
    agricultural_leaching,
    get_cell_var,
    log_rate,
    moisture_var,
    nitrate_concentration,
)
from inseeds.components.farming.ca_behaviour import (
    TPB,
    DRIVER_AFFORDABILITY_FORCED,
)
from inseeds.components.farming.ca_management import (
    COVER_CROP_LEGUME,
    COVER_CROP_NONLEGUME,
    ManagementCosts,
    PolicySettings,
    default_policy_span,
    PRACTICE_FIELDS,
)
from inseeds.components.exogenous.madrat import ResidueSource
from inseeds.components.farming.ca_management import ManagementBundle


logger = logging.getLogger(__name__)


# LPJmL's aggregate "others" band is not a residual crop but a real cereal
# stand: under others_to_crop it is cultivated as cft_temp outside the tropics
# and cft_tropic inside them, on the abs(lat) > 30 deg split. The harvest read
# back for that band is therefore cereal harvest and is priced as cereal, not
# against the FAO "Others, annual"/"Others, perennial" residuals. Mirrors
# lpjml_config.cjson; change these together with it.
OTHERS_CFT_TEMPERATE = "temperate cereals"
OTHERS_CFT_TROPICAL = "maize"
OTHERS_TROPICS_LAT = 30.0

# FAO returns a producer price only for the crops a country reports, so a price
# file prepared for a single country holds only that country's crops while LPJmL
# keeps reporting all crop bands. A band without a price of its own is valued at
# the crops closest to it: first the mean of the crops in its own group that are
# priced, then the FAO residual aggregate for its life form. The group comes
# first because the residual is whatever the country does not report separately,
# which can be far off - Dutch horticulture prices it four times above cereals.
PRICE_PROXY_GROUPS = (
    frozenset({"temperate cereals", "rice", "maize", "tropical cereals"}),
    frozenset({"temperate roots", "tropical roots"}),
    frozenset(
        {
            "oil crops sunflower",
            "oil crops soybean",
            "oil crops groundnut",
            "oil crops rapeseed",
        }
    ),
)
OTHERS_ANNUAL_PRICE = "Others, annual"
OTHERS_PERENNIAL_PRICE = "Others, perennial"
PERENNIAL_CFTS = frozenset({"sugarcane", "oil palm", "tea"})

# Cover crop cost component that is only earned under no-till, where weed
# control runs on herbicide rather than on cultivation.
WEED_CONTROL_COST = "weed_control"

# Substitutions already reported, so a global run logs each country and crop
# once instead of once per cell.
_LOGGED_PRICE_SUBSTITUTES = set()


def _price_category_for(band_category, price_categories, aliases):
    """Producer price category that prices one LPJmL crop band.

    Parameters
    ----------
    band_category : str
        Band name with any 'rainfed '/'irrigated ' prefix already stripped.
    price_categories : list of str
        Category names available in the producer price data.
    aliases : dict
        Case-folded band name to the category to price it as, for bands whose
        LPJmL name is not what LPJmL actually grows there.

    Returns
    -------
    str or None
        Matching category, None if the band cannot be priced.
    """
    target = aliases.get(band_category.casefold(), band_category).casefold()

    return next(
        (c for c in price_categories if c.casefold() == target), None
    )


def _price_for_band(band_category, prices, price_dim, aliases):
    """Usable producer price for one LPJmL crop band.

    Tried in order: the band's own category, the mean of the priced crops in
    its group (``PRICE_PROXY_GROUPS``), the FAO residual aggregate for its life
    form. A band the price file does not cover therefore still earns its
    harvest rather than earning nothing while its cost share is charged.

    Parameters
    ----------
    band_category : str
        Band name with any 'rainfed '/'irrigated ' prefix already stripped.
    prices : xarray.DataArray
        Producer prices of one country, indexed by price category.
    price_dim : str
        Name of the category dimension of ``prices``.
    aliases : dict
        Case-folded band name to the category to price it as.

    Returns
    -------
    tuple of (str or None, float)
        What the price was taken from and the price (USD/t dry matter),
        ``(None, nan)`` if no tier holds a usable value.
    """
    categories = [str(c) for c in prices[price_dim].values]
    target = aliases.get(band_category.casefold(), band_category)

    def finite_price(category):
        price = float(prices.sel({price_dim: category}).values)
        return price if np.isfinite(price) else None

    own = _price_category_for(target, categories, {})
    if own is not None:
        price = finite_price(own)
        if price is not None:
            return own, price

    group = next(
        (g for g in PRICE_PROXY_GROUPS if target.casefold() in g), frozenset()
    )
    related = sorted(
        c for c in categories if c.casefold() in group and finite_price(c) is not None
    )
    if related:
        mean = float(np.mean([finite_price(c) for c in related]))
        return f"mean of {', '.join(related)}", mean

    residual = _price_category_for(
        OTHERS_PERENNIAL_PRICE
        if target.casefold() in PERENNIAL_CFTS
        else OTHERS_ANNUAL_PRICE,
        categories,
        {},
    )
    if residual is not None:
        price = finite_price(residual)
        if price is not None:
            return residual, price

    return None, float("nan")


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

    Baselines
    ---------
    Every rate the farmer judges itself by is a deviation from its own historic
    state, averaged over the pre-coupling period 2015-2025: baseline_revenue and
    the baseline yield, soil carbon, moisture and leaching. baseline_bundle is
    the practice bundle in place at the start of the simulation.

    Affordability
    -------------
    While farm_economics.capital_constraints is set, a farmer below min_capital
    deselects costly practices until it can pay for them, and a transition only
    proceeds if the capital covers its one-time cost. Switched off, the stock is
    still tracked and still drives profit but never refuses a change, which takes
    credit to be available whenever a practice pays off.

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
        - FAO capital stock for the initial capital endowment
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

        country = self.cell.country

        # Initial capital per hectare from FAO Net Capital Stock
        self.initial_capital_per_ha = country.initial_capital_per_ha

        # Producer prices for profit calculation
        self.pft_prices = country.pft_prices

        # -----------------------------------------------------------------
        # Capital initialization
        # -----------------------------------------------------------------

        # Capital scales with cropped area, the area that earns the revenue it
        # has to be compared against.
        self.capital = self.initial_capital_per_ha * self.farm_size

        # -----------------------------------------------------------------
        # Load practice costs
        # -----------------------------------------------------------------

        # Costs are scaled by country GDP per capita (FAOSTAT MK domain).
        # Countries below 10th percentile get minimum costs, above 90th get
        # maximum costs. This accounts for global variation in equipment and
        # input costs based on national income levels.
        # Costs include:
        # - direct: annual operating cost per ha (scaled)
        # - transition: one-time cost when adopting practice (scaled)
        self.practice_costs = ManagementCosts.from_config(
            self.model.config.coupled_config.practice_costs,
            gdp_cost_ratio=self.cell.country.gdp_cost_ratio,
        )

        # -----------------------------------------------------------------
        # Pre-compute revenue calculation mappings (for performance)
        # -----------------------------------------------------------------
        self._init_revenue_mapping()

        # Cache for current revenue (updated each year in update_capital)
        # Used for both capital dynamics and residue opportunity cost
        self._current_revenue = None

        # Cache for capital accounting components, updated each year in
        # update_capital. All in dollars for the whole farm.
        self._delta_revenue = 0.0  # revenue - baseline_revenue
        self._delta_costs = 0.0    # current_bundle_costs - baseline_bundle_costs
        self._carbon_payment = 0.0
        self._practice_subsidy = 0.0
        self._nitrogen_charge = 0.0
        self._transfers = 0.0      # carbon_payment + practice_subsidy - nitrogen_charge
        self._current_profit = 0.0 # delta_revenue - delta_costs + transfers
        self._residue_opportunity_cost = 0.0  # this year's straw value

        # One-off movements from a transition, charged after update_capital and
        # so outside _current_profit. Reported so that the year's change in
        # capital can be closed exactly; see the ``transition_cost`` property.
        self._transition_cost = 0.0
        self._residue_sale_income = 0.0

        # This year's water and nitrogen fluxes, read once like the other levels
        self._read_leaching_state()

        # -----------------------------------------------------------------
        # Compute baseline revenue from historic data
        # -----------------------------------------------------------------
        # The baseline is the average revenue over the historic period
        # (pre-coupling years, e.g., 2015-2025). This represents what
        # the FAO capital dynamics already account for.
        self.baseline_revenue = self._compute_baseline_revenue()

        # The offsets the log deviations are taken about, needed before any rate
        # property is read
        self._read_negligible_levels()

        # -----------------------------------------------------------------
        # Compute environmental baselines from historic data
        # -----------------------------------------------------------------
        # Historic mean of each environmental metric (yield, soil carbon,
        # moisture, leaching). Performance scoring expresses every metric as a
        # log deviation from these self-referencing baselines, so all five
        # metrics are dimensionless and centred at zero - the same
        # self-referenced logic as profit, which uses the signed analogue asinh.
        self._compute_baseline_metrics()

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
        # Committed rather than certified, so a farm that starts out above the
        # residue threshold is not later credited with income for giving up a
        # retention it never chose.
        self.baseline_bundle = self.behaviour.committed_bundle

        # Frozen alongside the bundle, for the reason baseline_residue_cost is:
        # the crop the baseline was grown with does not change when the farm
        # re-sows. Read dynamically instead, a switch to the legume would
        # re-price both legs of the cost difference and cancel its own seed
        # bill, so the farm would sow the dearer crop for free.
        self.baseline_cover_crop_type = (
            int(self.cover_crop)
            if self.baseline_bundle.cover_crop and self.cover_crop
            else None
        )


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
    def capital_constrained(self):
        """Whether the capital stock may refuse a change the farmer wants.

        Governs the three places where the stock can override the decision: the
        survival gate that suspends evaluation below min_capital, the forced
        deselection of practices whose running costs have outgrown the stock, and
        the trimming of a target bundle to what the stock can buy. Profit,
        revenue and the stock itself are computed either way, as is PBC, which
        asks whether a switch pays for itself out of income rather than whether
        it can be paid for out of the stock.

        A stock of productive capital is only a proxy for access to finance, and
        a practice that pays for itself is normally financed somehow - a loan, a
        subsidy, a deferred purchase. Switching the constraint off treats finance
        as available whenever the economics justify it; leaving it on represents
        the credit markets of the poorest countries, where it does bind.

        Returns
        -------
        bool
            True if capital constraints bind (the default).
        """
        return self.model.config.coupled_config.farm_economics.capital_constraints

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
        Only consulted while capital_constrained is set.

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
        initial_capital = self.initial_capital_per_ha * self.farm_size
        fraction = self.model.config.coupled_config.farm_economics.min_capital_fraction
        return fraction * initial_capital

    @property
    def profit(self) -> float:
        """Current profit as a compressed rate relative to baseline revenue.

        This is the economic metric for performance scoring:
            profit = asinh(delta_profit / baseline_revenue)

        Where delta_profit = (revenue - baseline_revenue) - delta_costs

        Being a rate relative to the farmer's OWN baseline revenue makes it
        self-normalizing: independent of farm size, crop mix, price level and
        climate by construction. This isolates the effect of management changes
        from confounding ambient conditions, which is why the economic metric
        stays self-referenced rather than being country-centered like the
        environmental metrics (soil carbon, moisture, yield, leaching).

        Why asinh
        ---------
        The four environmental metrics enter the score as log deviations, so a
        profit left as a plain fraction would be the only metric on a linear
        scale, and, carrying the largest decision weight, a single windfall or
        crop failure would move the score further than an equally extreme
        environmental change. asinh is the signed counterpart of the log
        transform: antisymmetric, indistinguishable from the plain fraction for
        small deviations (asinh(0.1) = 0.0998), and logarithmic thereafter.
        Unlike ln(1 + rate) it stays defined when costs exceed baseline revenue
        and the rate falls below -1, which a whole-farm margin can do.

        Direct costs enter through delta_costs (recurring annual costs only,
        NOT one-time transition costs, which are handled by PBC/affordability).

        The historic series behind the reference scales computes the same thing
        in ManagementPerformanceTracker._load_history, spelled out as
        revenue/baseline - 1 because costs are flat before anyone transitions.
        A change here needs the same change there.

        Returns
        -------
        float
            Compressed profit rate, ~0.05 for a 5% improvement over baseline.
        """
        if self.baseline_revenue <= 0:
            return 0.0
        return float(np.arcsinh(self._current_profit / self.baseline_revenue))

    @property
    def current_revenue(self) -> float:
        """This year's harvest valued at producer prices, USD for the farm.

        Computed in ``update_capital`` from LPJmL yields and FAO prices.
        Before the first step it is None; the property returns 0.0 in that
        case so an output collected early does not crash.
        """
        return self._current_revenue if self._current_revenue is not None else 0.0

    @property
    def delta_revenue(self) -> float:
        """Revenue change from baseline, in dollars for the whole farm.

        Positive when this year's harvest is worth more than the historic
        average, negative when it is worth less.
        ``delta_revenue = current_revenue - baseline_revenue``.
        """
        return self._delta_revenue

    @property
    def delta_costs(self) -> float:
        """Cost change from baseline bundle, in dollars for the whole farm.

        Positive when the current bundle costs more than the baseline bundle
        (e.g., adopted cover crop), negative when it costs less (e.g., switched
        to no-till).
        """
        return self._delta_costs

    @property
    def transfers(self) -> float:
        """Net policy transfers, in dollars for the whole farm.

        Sum of carbon payment + practice subsidy - nitrogen charge. Positive
        when the farm receives more than it is charged, negative otherwise.
        """
        return self._transfers

    @property
    def raw_profit(self) -> float:
        """This year's operating profit in dollars for the whole farm.

        ``raw_profit = delta_revenue - delta_costs + transfers``, the
        un-transformed cash flow the performance score reads.

        It is the whole of the change in capital only in a year the farm did
        not move: a transition is charged after ``update_capital`` has run, so
        the year's full balance is

        ``capital_t - capital_{t-1} = raw_profit - transition_cost
        + residue_sale_income``

        exact except where the floor at zero clamped it.
        """
        return self._current_profit

    @property
    def residue_opportunity_cost(self) -> float:
        """This year's straw, priced at what it would fetch, USD for the farm.

        The dynamic figure: this year's revenue times the cell's weighted use
        fraction, so it moves with the harvest. It is what a selling farm is
        paid and what a retaining farm gives up *this* year, as against
        ``baseline_residue_cost``, which is the same calculation frozen on
        baseline revenue and is the figure retention is actually charged at.
        Reported because the difference between the two is the whole of the
        residue term in ``delta_costs`` and cannot otherwise be recovered.
        """
        return self._residue_opportunity_cost

    @property
    def transition_cost(self) -> float:
        """One-off cost of this year's practice change, USD for the farm.

        Zero in a year the farm held its bundle. Charged directly against
        capital after ``update_capital``, so it is outside ``raw_profit`` and
        has to be reported separately for the year to balance.
        """
        return self._transition_cost

    @property
    def residue_sale_income(self) -> float:
        """Cash from selling standing straw on a forced deselection, USD.

        Only a farm driven off retention by affordability raises it, and only
        once: it is the stock of straw it still had, priced at
        ``residue_opportunity_cost``. Zero in every other year.
        """
        return self._residue_sale_income

    @property
    def yield_rate(self) -> float:
        """Crop yield as log deviation from the farmer's own baseline.

        Self-referencing, so it is dimensionless and centred at zero, comparable
        across farmers regardless of absolute productivity, and symmetric, so a
        doubling and a halving are equidistant from zero. See profit for the
        same logic applied to economics, and farmer.log_rate for the form.
        """
        return log_rate(self.cropyield, self.baseline_yield, self.negligible_yield)

    @property
    def soilc_rate(self) -> float:
        """Top-layer soil carbon as log deviation from own baseline."""
        return log_rate(self.soilc, self.baseline_soilc, self.negligible_soilc)

    @property
    def moisture_rate(self) -> float:
        """Root-zone moisture as log deviation from own baseline."""
        return log_rate(
            self.root_moisture, self.baseline_moisture, self.negligible_moisture
        )

    @property
    def soilc_accumulation(self) -> float:
        """Annual soil carbon gain in gC/m2/yr, positive when the stock builds.

        Soil carbon is a stock and the score reads it as one, but an instrument
        that pays for sequestration is owed the flow: how much carbon went in
        this year, not how much is there. This is that flow, in the unit the
        payment would be written in.

        Taken from the tracker's fitted slope rather than the difference between
        two years, because the year-to-year figure carries the weather and a
        payment following it would swing on rainfall. The slope is over the
        current episode only, so it reads zero for a year or two after a switch,
        which understates rather than overstates a new practice.

        The conversion out of log space is first order: the rate is a log
        deviation about ``baseline_soilc``, so ``baseline x expm1(slope)`` is the
        gain per year in gC/m2. Exact for soil carbon because the scored
        quantity and the physical one are the same thing (see
        ``metrics.METRICS``).
        """
        tracker = getattr(self.behaviour, "performance_tracker", None)
        if tracker is None:
            return 0.0
        return float(self.baseline_soilc * np.expm1(tracker.trend["soilc"]))

    @property
    def creditable_soilc_accumulation(self) -> float:
        """Soil carbon gain a payment could be made on, in gC/m2/yr.

        The gain, floored at zero: a scheme pays for carbon put in and does not
        invoice a farm for carbon coming back out. That is how soil carbon
        schemes are written, and the alternative would make enrolling a wager on
        the weather - a run of dry years releases carbon nobody chose to release.

        The asymmetry is deliberate and it is not free: a farm can build the
        stock, take the payments, shed the practice and keep both. Pricing the
        decline would close that, at the cost of making the instrument something
        a cautious farm declines to touch.
        """
        return max(0.0, self.soilc_accumulation)

    # =========================================================================
    # POLICY INSTRUMENTS
    # =========================================================================

    @property
    def policy(self):
        """Instrument rates in force for this farm's country, or None.

        None rather than a zeroed record, so that a run without instruments
        does no arithmetic at all and the counterfactual is bit-identical to
        the model as it stood before they existed.

        Resolved on each read rather than frozen at initialisation, because
        that is what lets a scenario introduce or withdraw an instrument
        part-way through a run - which ``policy.windows`` can do for each one
        separately, so a carbon price can arrive late or a subsidy be pulled
        early to see whether the practice survives it.
        """
        raw = getattr(self.model.config.coupled_config, "policy", None)
        if raw is None:
            return None

        settings = PolicySettings.from_config(
            raw,
            self.cell.country.gdp_cost_ratio,
            default_policy_span(self.model.config),
        ).in_force(self.model.lpjml.sim_year)

        if not settings.any_in_force:
            return None
        return settings

    def carbon_payment_for(self, accumulation: float) -> float:
        """What this farm would be paid for sequestering ``accumulation`` gC/m2.

        The rates are read from ``policy``, so this is the payment as of the
        moment it is called; ``update_capital`` calls it once a year and the
        ``carbon_payment`` property reports what it settled on.
        """
        settings = self.policy
        if settings is None or not settings.carbon_price:
            return 0.0
        return (
            accumulation
            * GC_M2_TO_TCO2_HA
            * settings.carbon_price
            * self.farm_size
        )

    @property
    def carbon_payment(self) -> float:
        """The sequestration payment that reached capital this year, USD.

        The figure ``update_capital`` settled on, not a fresh resolution of the
        instrument. Reporting the payment that was actually made is the only
        way the output can be reconciled against ``raw_profit``: ``policy``
        resolves against ``lpjml.sim_year``, which the coupler advances to the
        next year once the year's exchange is done, so a property recomputed
        when the output is collected answers for the wrong year - zero in the
        run's final year, and nonzero in the year before a late ``start_year``.

        Never negative: see ``creditable_soilc_accumulation``.
        """
        return self._carbon_payment

    @property
    def levy_free_leaching(self) -> float:
        """Nitrogen this farm may lose untaxed this year, in gN/m2.

        The drinking-water standard turned back into a mass. A nitrate limit is
        written as a concentration - 50 mg NO3/l, or 11.3 mgN/l - because that
        is what determines whether the water is fit to drink, and the mass of
        nitrogen that corresponds to is however much the year's drainage can
        carry at that strength::

            allowance [gN/m2] = concentration [mgN/l] x drainage [mm] / 1000

        one millimetre over a square metre being one litre. So a wet cell may
        lose more before it is charged, because the same nitrogen arrives in
        the aquifer more dilute, and a dry one may lose less. This is how the
        EU-27 critical nitrogen assessment derives a critical leaching rate
        from the directive's concentration standard [21], and it is the only
        place the model consults a concentration: as a threshold read once,
        not as a quantity carried through baselines, rulers and peer memory,
        which is what made it a poor metric to score on.

        Zero unless a scenario sets the standard, in which case every kilogram
        is charged.
        """
        settings = self.policy
        if settings is None:
            return 0.0
        return settings.levy_free_concentration * self.runoff / 1e3

    def nitrogen_charge_for(self, leaching: float) -> float:
        """What this farm would owe if it leached ``leaching`` gN/m2.

        One function for the bill, the anticipated saving from a switch and the
        cover crop comparison, because a charge with an allowance is not linear
        in the loss - below the threshold the marginal rate is zero - and three
        copies of that would not stay in step.
        """
        settings = self.policy
        if settings is None or not settings.nitrogen_charge:
            return 0.0
        taxable = max(0.0, leaching - self.levy_free_leaching)
        return (
            taxable
            * GN_M2_TO_KGN_HA
            * settings.nitrogen_charge
            * self.farm_size
        )

    @property
    def taxable_leaching(self) -> float:
        """Leached nitrogen above the allowance, gN/m2, the charge's base."""
        return max(0.0, self.leaching - self.levy_free_leaching)

    def practice_subsidy_for(self, bundle, cover_crop_type=None) -> float:
        """Annual payment for holding ``bundle``, USD for the whole farm.

        Paid every year the practice is held rather than once on adoption,
        which is how EQIP and the CAP eco-schemes are written and what makes
        this an instrument rather than a discount. It is also what separates
        it from the cost component this used to be: as a negative cost it
        entered ``update_capital`` through a difference against the baseline
        bundle, so it cancelled entirely for a farm that already held the
        practice, and only converts were ever paid.

        Two shapes of retention payment, and a config may set only one. The
        flat rate is per hectare; the cost share is a fraction of
        ``baseline_residue_cost``, which is the figure
        ``get_residue_cost_for_bundle`` charges a retaining farm, so the share
        is exactly the fraction of the cost that the farm does not bear. Using
        the frozen baseline rather than this year's straw value keeps it that
        way: a payment worked out on current revenue would drift away from the
        cost it is meant to cover, and would add a weather signal to an
        instrument whose rate is posted.

        The cover crop payment carries an enhancement for a legume, so the
        crop has to be resolved and not just the practice. Left to itself that
        is ``cover_crop_type_for``, which returns the crop in the ground for a
        bundle in force and the one the farm would sow for a candidate -
        exactly the reading each caller wants, since the payment on the
        committed bundle is for what is growing and the payment in a
        comparison is for what would be.
        """
        settings = self.policy
        if settings is None:
            return 0.0
        if cover_crop_type is None and bundle.cover_crop == 1:
            cover_crop_type = self.behaviour.cover_crop_type_for(bundle)
        total = settings.subsidy_for(bundle, cover_crop_type) * self.farm_size
        if bundle.residue_on_field == 1 and settings.residue_subsidy_share:
            total += settings.residue_subsidy_share * self.baseline_residue_cost
        return total

    @property
    def practice_subsidy(self) -> float:
        """The practice payments that reached capital this year, USD.

        The figure ``update_capital`` settled on, for the reason given under
        ``carbon_payment``. It is paid on the committed bundle, not the
        certified one: a farm whose stubble happens to clear the 30%
        litter-cover threshold never enrolled in anything and is not paid for
        it; see ``TPB.committed_bundle``.
        """
        return self._practice_subsidy

    @property
    def nitrogen_charge(self) -> float:
        """The charge on leached nitrogen that reached capital this year, USD.

        The figure ``update_capital`` settled on, for the reason given under
        ``carbon_payment``.

        Levied on the nitrogen actually lost, as a real tax would be: a farm
        pays for what leaves its field, not for the difference between that and
        what used to leave it. It is therefore a bill the farm owes every year
        whatever it does, and only the part it can change by farming
        differently enters any comparison it makes - which is why
        ``anticipated_policy_transfer`` carries the difference and not this.

        The bill is real money out, so it lands on capital. Note what that
        means in a ledger where every other term is a deviation from baseline:
        revenue and costs average near zero over a run, and this does not, so
        an instrument at a high enough rate walks every farm down to the
        capital floor whether or not it changes practice. That is the tax
        incidence the model can represent, and it is harsher than reality,
        where the bill is paid out of gross income rather than out of the
        machinery shed. An allowance is the main thing standing between a run
        and that outcome, since it exempts the farms that were never the
        problem.
        """
        return self._nitrogen_charge

    @property
    def leaching_rate(self) -> float:
        """Leached nitrogen as log deviation from own baseline.

        Taken on the flux, the mass of nitrogen that left the field, rather than
        on its concentration in the drainage. The flux is what an instrument can
        be written on and what adds up over a catchment, and it is the same
        quantity the peer estimator differences, so a change remembered by a
        neighbour converts to kilograms without assuming their drainage matched.

        A year with no drainage lost no nitrogen, and reads here as the large
        improvement it physically was. That leaves the metric weather-driven in
        the same way yield is, which the fitted trend averages out over an
        episode; ``negligible_leaching`` sets how hard a dry year can swing it.

        Higher leaching is worse, but the rate is still a plain log deviation;
        the inversion is applied in the scoring step.
        """
        return log_rate(
            self.leaching, self.baseline_leaching, self.negligible_leaching
        )

    @property
    def leaching_concentration(self) -> float:
        """Nitrate concentration (mgN/l) in the leachate, for reporting only.

        Nothing decides on this. It is here because a nitrate standard is
        written as a concentration - the EU limit is 50 mg NO3/l, or 11.3 mgN/l
        - so it is what makes the output comparable to one. Reported as 0.0 in
        years too dry for a concentration to be defined, which keeps the output
        a plain number per year and must not be read as clean water.

        Returns
        -------
        float
            Nitrate concentration in mgN/l, or 0.0 when undefined.
        """
        conc = nitrate_concentration(self.leaching, self.runoff)
        return 0.0 if conc is None else conc

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
        if use_fractions is None:
            raise ValueError(
                f"cell {self.cell.grid.cell.item()}: no MADRaT residue use "
                "data, so the opportunity cost of retaining straw cannot be "
                "priced. Check that the residue source loaded; "
                "exogenous.load_all downgrades a failed source to a warning."
            )

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
        """Which cover crop this farm would sow, chosen on what it is worth.

        The two do different jobs and cost different money. A legume fixes
        nitrogen, which shows up as yield, and its seed costs about twice as
        much. A catch crop only holds on to nitrogen already there, which shows
        up as less leaching, and is worth what that nitrogen is charged at.
        The farm sows whichever leaves it better off::

            value(type) = Δrevenue(type) - seed(type) - Δcharge(type)

        all against the other type, over the same discounted horizon the rest
        of the decision uses, and all read off peers running each crop under
        this farm's own bundle - which is what the memory being keyed by
        ``(bundle, cover_crop_type)`` was for.

        With no peer evidence the comparison is decided by seed cost alone and
        the catch crop wins, which on its own would lock the model into it: no
        farm sows a legume, so no record of one accumulates, so no farm ever
        learns it pays. ``TPB.explores_cover_crop_type`` is what breaks that,
        overriding this preference now and again as the crop is committed.

        The preference itself, with no draw in it and nothing cached: a
        question about prices and peer records that can be asked as often as
        costing a candidate bundle needs to, and answers the same way each
        time. What the farm actually sows is settled once, by ``apply_bundle``.

        Returns
        -------
        int
            ``COVER_CROP_NONLEGUME`` or ``COVER_CROP_LEGUME``.
        """
        # Priced under the bundle the farm actually runs, or the plainest one
        # that has a cover crop if it runs none, since that is the bundle any
        # adoption would start from.
        bundle = self.behaviour.practice_bundle
        if not bundle.cover_crop:
            bundle = bundle.change_practices(cover_crop=1)

        legume_key = (bundle, COVER_CROP_LEGUME)
        catch_key = (bundle, COVER_CROP_NONLEGUME)

        seed = self.practice_costs.cover_crop
        # Seed is per hectare and per year; the horizon puts it on the same
        # footing as the two streams below.
        horizon = self.behaviour.projection_weight()
        extra_seed = (
            seed.seed_cost(COVER_CROP_LEGUME) - seed.seed_cost(COVER_CROP_NONLEGUME)
        ) * self.farm_size * horizon

        # What peers gained in yield by sowing the legume instead of the catch
        # crop, as a fraction of their own yield, applied to this farm's
        # revenue - the same conversion pbc_for_bundle makes.
        extra_revenue = (
            self.behaviour.anticipated_change_between("yield", catch_key, legume_key)
            * self.baseline_revenue
            * horizon
        )

        settings = self.policy

        # An enhancement paid only on the legume, so unlike the base cover crop
        # payment it does not cancel between the two sides of this comparison.
        # This is the term a scenario steering the crop acts through, and the
        # only one large enough to settle a choice whose other two sides differ
        # by about a per cent of revenue.
        extra_subsidy = 0.0
        if settings is not None and settings.cover_crop_legume_subsidy:
            extra_subsidy = (
                settings.cover_crop_legume_subsidy * self.farm_size * horizon
            )

        extra_charge = 0.0
        if settings is not None and settings.nitrogen_charge:
            # A catch crop is bought for this: nitrogen it holds in the soil is
            # nitrogen the farm is not charged for. Worth nothing at all until
            # the charge exists, which is the point of the instrument.
            delta_flux = (
                self.behaviour.anticipated_change_between(
                    "leaching", catch_key, legume_key
                )
                * self.baseline_leaching
            )
            # Two bills differenced rather than a rate applied, so the kink at
            # the allowance is respected: a farm already clean enough to be
            # exempt sees no reason here to prefer the catch crop, which is
            # correct, because holding nitrogen it is not charged for earns it
            # nothing.
            extra_charge = self.nitrogen_charge_for(
                self.leaching + delta_flux
            ) - self.nitrogen_charge_for(self.leaching)
            extra_charge *= horizon

        legume_advantage = (
            extra_revenue + extra_subsidy - extra_seed - extra_charge
        )
        return COVER_CROP_LEGUME if legume_advantage >= 0 else COVER_CROP_NONLEGUME

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def get_practice_direct_cost_per_ha(
        self,
        field: str,
        value: int,
        cover_crop_type: int | None = None,
    ) -> float:
        """Get per-hectare direct cost for a single practice state.

        Costs are defined relative to the CA reference point (no-till, no cover crop):
        - tillage=0 (no-till): 0 (reference point)
        - tillage=1 (conventional): +cost (pays more than no-till)
        - cover_crop=0: 0 (no cover crop)
        - cover_crop=1: +cost (pays for cover crop)

        This encoding ensures correct cost differences when comparing bundles:
        conventional→no-till: 0 - (+cost) = -cost (saves money) ✓

        This prices a practice on its own. The cover crop's weed control credit
        is included here, since a cover crop is normally grown on a no-till
        field; where it is combined with tillage the credit is withdrawn by
        ``get_bundle_direct_cost_per_ha``, which sees both practices.

        Parameters
        ----------
        field : str
            Practice name ('tillage', 'cover_crop')
        value : int
            Practice state (0 or 1)
        cover_crop_type : int, optional
            Which cover crop is sown, since the seed cost of a legume is about
            twice that of a catch crop. Defaults to the type this farm would
            choose today.

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
            if value != 1:
                return 0.0
            if cover_crop_type is None:
                cover_crop_type = self.indicate_cover_crop_type()
            return costs.cover_crop.direct_for_type(cover_crop_type)
        else:
            # Residue handled separately (not per-ha)
            return 0.0

    def get_bundle_direct_cost_per_ha(
        self,
        bundle: ManagementBundle,
        cover_crop_type: int | None = None,
    ) -> float:
        """Per-hectare annual cost of a bundle's tillage and cover crop states.

        Sums the single-practice costs (see ``get_practice_direct_cost_per_ha``)
        and then corrects for the one interaction between them: the cover crop's
        weed control credit is only earned where weeds are not being controlled
        by cultivation anyway, i.e. under no-till.

        Parameters
        ----------
        bundle : ManagementBundle
            Practice combination to price.
        cover_crop_type : int, optional
            Which cover crop the bundle is priced with. Defaults to the crop
            in the ground for the bundle in force and to the one this farm
            would choose for any other, which is what makes a candidate
            costed against what it would actually cost.

        Returns
        -------
        float
            Per-hectare direct cost relative to the CA reference. Residue is
            not included; it is not a per-hectare cost (see
            ``get_residue_cost_for_bundle``).
        """
        if cover_crop_type is None:
            cover_crop_type = self.behaviour.cover_crop_type_for(bundle)

        total = (
            self.get_practice_direct_cost_per_ha("tillage", bundle.tillage)
            + self.get_practice_direct_cost_per_ha(
                "cover_crop", bundle.cover_crop, cover_crop_type
            )
        )

        # A tilled field controls its weeds by cultivation, so the cover crop
        # buys nothing there and the credit is taken back.
        if bundle.cover_crop == 1 and bundle.tillage == 1:
            total -= self.practice_costs.cover_crop.cost_components.get(
                WEED_CONTROL_COST, 0.0
            )

        return total

    def get_residue_cost_for_bundle(self, residue_on_field: int) -> float:
        """Annual cost of retaining residue, or income from selling it.

        A farm that retains gives up the straw it used to sell, so it carries
        the baseline opportunity cost every year. Leaving more or less straw
        than the baseline year does not change that: the commitment is to leave
        whatever grows.

        A farm that sells is paid the market value of what it actually has, so
        its cost is the shortfall against the baseline, negative in a good year
        and positive in a bad one.

        No subsidy here. Any payment for retaining is
        ``policy.practice_subsidies.residue_on_field``, which reaches capital
        as a transfer rather than as a discount on this cost.

        Parameters
        ----------
        residue_on_field : int
            1 if the farm committed to retaining, 0 if it sells. Pass the
            committed state, not the certified one; see
            TPB.committed_bundle for why the two can differ.

        Returns
        -------
        float
            Residue cost (positive) or income (negative).
        """
        if residue_on_field == 1:
            # Retention: the straw income given up, and nothing else.
            return self.baseline_residue_cost
        else:
            # Selling: income/loss relative to baseline (no subsidy)
            # Good year (current > baseline): negative cost = extra income
            # Bad year (current < baseline): positive cost = lost income
            current = self.compute_residue_opportunity_cost()
            return self.baseline_residue_cost - current

    def get_bundle_direct_costs(
        self,
        bundle: ManagementBundle,
        cover_crop_type: int | None = None,
    ) -> float:
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
            Total annual direct cost for the whole farm.
        """
        # Scale per-ha costs (tillage, cover_crop) to total farm cost
        total = (
            self.get_bundle_direct_cost_per_ha(bundle, cover_crop_type)
            * self.farm_size
        )

        # Residue: cost depends on retention state (see get_residue_cost_for_bundle)
        total += self.get_residue_cost_for_bundle(bundle.residue_on_field)

        return total

    def get_bundle_costs_difference(
        self,
        from_bundle: ManagementBundle,
        to_bundle: ManagementBundle,
        from_type: int | None = None,
        to_type: int | None = None,
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
        from_type, to_type : int, optional
            Cover crop sown under each bundle. Given separately because a
            switch between the two types is a cost change on its own, with the
            bundle unmoved.

        Returns
        -------
        float
            Cost difference: positive = to_bundle costs more than from_bundle.
        """
        # Per-hectare cost differences (tillage, cover_crop)
        # diff = to_cost - from_cost
        diff_per_ha = (
            self.get_bundle_direct_cost_per_ha(to_bundle, to_type)
            - self.get_bundle_direct_cost_per_ha(from_bundle, from_type)
        )

        # Scale per-ha costs to total farm cost
        diff_total = diff_per_ha * self.farm_size

        # Residue: use the frozen baseline cost for practice changes
        # (yield-driven income changes are in delta_revenue).
        # Cost is ALREADY total (from revenue × fraction), don't scale again.
        # Any retention payment is a policy transfer, not a cost, and reaches
        # capital through update_capital's transfers term instead.
        if from_bundle.residue_on_field != to_bundle.residue_on_field:
            if to_bundle.residue_on_field == 1:  # Started retaining
                diff_total += self.baseline_residue_cost
            else:  # Stopped retaining (selling residue)
                diff_total -= self.baseline_residue_cost

        return diff_total

    def get_practice_transition_cost(
        self,
        field: str,
        from_val: int,
        to_val: int,
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

        Returns
        -------
        float
            Transition cost for this single practice change, for the whole farm.
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

        return cost_per_ha * self.farm_size

    def get_bundle_transition_costs(
        self,
        current_bundle: ManagementBundle,
        target_bundle: ManagementBundle,
    ) -> float:
        """One-time transition cost for all practice changes in a bundle switch.

        See get_practice_transition_cost() for per-practice cost logic.

        Returns
        -------
        float
            Total transition cost for the whole farm.
        """
        return sum(
            self.get_practice_transition_cost(field, old, new)
            for field, old, new in zip(
                PRACTICE_FIELDS, current_bundle.value, target_bundle.value
            )
        )


    # =========================================================================
    # CAPITAL UPDATE
    # =========================================================================

    def update_capital(self):
        """Update the farmer's capital by this year's deviation from baseline.

        Capital moves by how far this farm departs from its own starting point,
        and by nothing else::

            K_{t+1} = K + Δrevenue - Δcosts + payments - charges

        Δrevenue is this year's revenue against the 2015-2025 pre-coupling
        average, and Δcosts is the cost of the current bundle against the bundle
        in place at the start, whose costs belong to the baseline and so cancel.
        A farm that neither gains nor spends more than it used to holds its stock
        steady; better yields or the running savings of no-till grow it, and
        worse yields or the added cost of a practice shrink it.

        Policy transfers are the same kind of quantity, and enter the same way.
        """

        # Cached for reuse, e.g. by the residue opportunity cost
        self._current_revenue = self.calculate_revenue()
        self._delta_revenue = self._current_revenue - self.baseline_revenue

        # Costs follow what the farm chose, not what it is certified as, so a
        # farm handed its residue label by realised litter cover is neither
        # charged for straw it still sells nor paid the retention subsidy.
        # The baseline crop is stated because it is frozen and nothing about
        # today's farm can recover it; the current one is left to resolve to
        # what is actually in the ground.
        self._delta_costs = self.get_bundle_costs_difference(
            self.baseline_bundle,
            self.behaviour.committed_bundle,
            from_type=self.baseline_cover_crop_type,
        )

        # Instruments are cash like any other: a payment for carbon put into
        # the soil, a charge for nitrogen let out of it, a payment for holding
        # a practice. All are zero unless a scenario turns them on, and all are
        # zero over the historic period (see
        # ManagementPerformanceTracker._load_history), because the record the
        # farm learned from was made without them.
        #
        # Unlike the two terms above they are absolute rather than deviations
        # from baseline, so they do not average out over a run. That is what
        # gives an instrument a persistent grip on the capital stock, in both
        # directions.
        # This year's straw value, kept for the same reason and because the
        # residue term of delta_costs is a difference against the frozen
        # baseline that no output could otherwise be resolved into its parts.
        self._residue_opportunity_cost = self.compute_residue_opportunity_cost()

        # Last year's one-offs are cleared here, before the decision step that
        # may charge new ones, so each year reports only its own.
        self._transition_cost = 0.0
        self._residue_sale_income = 0.0

        # Resolved once, here, and kept: the properties that report them read
        # these fields rather than the instrument again, so the output can be
        # reconciled against raw_profit whatever year the collector runs in.
        self._carbon_payment = self.carbon_payment_for(
            self.creditable_soilc_accumulation
        )
        self._practice_subsidy = self.practice_subsidy_for(
            self.behaviour.committed_bundle
        )
        self._nitrogen_charge = self.nitrogen_charge_for(self.leaching)
        self._transfers = (
            self._carbon_payment + self._practice_subsidy - self._nitrogen_charge
        )
        # The annual economic gain or loss against baseline, and the economic
        # metric the performance score reads.
        self._current_profit = self._delta_revenue - self._delta_costs + self._transfers

        self.capital = max(0.0, self.capital + self._current_profit)


    # =========================================================================
    # REVENUE CALCULATION
    # =========================================================================

    @property
    def others_cft(self):
        """The crop LPJmL grows on this cell's 'others' cropland.

        Under others_to_crop that band is a cereal stand rather than a residual
        category, temperate or tropical by the same latitude split LPJmL uses.
        """
        latitude = abs(float(self.cell.grid.lat.item()))
        return (
            OTHERS_CFT_TEMPERATE if latitude > OTHERS_TROPICS_LAT
            else OTHERS_CFT_TROPICAL
        )

    def _init_revenue_mapping(self):
        """Pre-compute band→price mappings for fast revenue calculation.

        Called once at init to avoid repeated string operations and xarray
        lookups during simulation. The mapping is constant for the simulation.

        Every cropped band should be priced. Practice costs use the full
        cropped area (``farm_size``); omitting a band zeros its revenue while
        still charging its cost share, which biases against cost-bearing
        practices. Bands the price file does not cover are therefore valued at
        the crops closest to them (``_price_for_band``), and only bands left
        unpriced while carrying a material share of the cell raise. LPJmL
        ``others`` maps to ``temperate cereals`` / ``maize`` by latitude
        (``others_to_crop``).
        """
        # Get band names from cftfrac (excluding NON_CROPS)
        cftfrac = self.get_from_earth("cftfrac", drop_band=NON_CROPS, time_idx=-1)
        bands = [str(b) for b in cftfrac.band.values]

        prices = self.pft_prices
        price_dim = "npft" if "npft" in prices.dims else "band"

        # others_to_crop: price 'others' as the cereal LPJmL grows at this lat
        aliases = {"others": self.others_cft}

        # Group bands by the price category that prices them, so a year of
        # revenue is a handful of vector sums instead of a lookup per band.
        category_to_indices = {}
        category_prices = {}
        substitutes = {}
        unpriced = []
        for i, band in enumerate(bands):
            # Strip 'rainfed ' or 'irrigated ' prefix
            if band.startswith("rainfed "):
                category = band[8:]
            elif band.startswith("irrigated "):
                category = band[10:]
            else:
                category = band

            match, price = _price_for_band(category, prices, price_dim, aliases)
            if match is None:
                unpriced.append((i, band))
                continue

            own = aliases.get(category.casefold(), category)
            if match.casefold() != own.casefold():
                substitutes[category] = match

            category_to_indices.setdefault(match, []).append(i)
            category_prices[match] = price

        self._report_price_substitutes(substitutes)
        self._check_unpriced_bands(unpriced, cftfrac, prices, price_dim)

        # Convert to numpy arrays for fast indexing
        self._revenue_groups = [
            (np.array(indices), category_prices[category])
            for category, indices in category_to_indices.items()
        ]

        # Cache cell area (constant)
        self._cell_area = self.cell.area.item()

    def _report_price_substitutes(self, substitutes):
        """Log once per country and crop that another price stands in."""
        for category, match in substitutes.items():
            key = (self.cell.country.code, category, match)
            if key in _LOGGED_PRICE_SUBSTITUTES:
                continue
            _LOGGED_PRICE_SUBSTITUTES.add(key)
            logger.warning(
                f"{key[0]}: FAO reports no producer price for '{category}' in "
                f"this country, so its harvest is valued at '{match}'."
            )

    def _check_unpriced_bands(self, unpriced, cftfrac, prices, price_dim):
        """Raise if bands without any usable price carry material cropland.

        LPJmL reports every crop band whether or not the cell grows it, so a
        band the price file cannot value is harmless while the cell leaves it
        empty. Once it is actually cropped its cost share is charged either
        way, and pricing it becomes a condition for the run to mean anything.
        """
        if not unpriced:
            return

        indices = [i for i, _ in unpriced]
        share = float(np.sum(cftfrac.values[indices]))
        min_fraction = self.model.config.coupled_config.materiality.min_crop_fraction
        band_names = sorted(band for _, band in unpriced)

        if share < min_fraction:
            logger.debug(
                f"cell {self.cell.grid.cell.item()}: unpriced band(s) "
                f"{band_names} left out of revenue, together {share:.2%} of "
                "the cell."
            )
            return

        raise ValueError(
            f"cell {self.cell.grid.cell.item()}: no producer price for cropped "
            f"band(s) {band_names}, together {share:.2%} of the cell. Available "
            f"price categories: {sorted(str(c) for c in prices[price_dim].values)}. "
            "Costs are charged on these bands, so leaving them unpriced "
            "understates revenue."
        )

    def _compute_baseline_revenue(self):
        """Compute baseline revenue as MEAN over entire historic period.

        Using the mean (not first year) ensures:
        - Robust baseline not skewed by outlier years
        - Capital trends reflect deviation from typical performance
        - No systematic bias if first year was unusually good/bad

        During simulation, we track DEVIATIONS from this baseline:
        - Higher yields → positive Δrevenue → capital grows faster
        - Lower yields → negative Δrevenue → capital grows slower

        Returns
        -------
        float
            Mean revenue over historic period (USD), raised to its materiality
            floor so it stays usable as a denominator.
        """
        from_earth = self.cell.from_earth
        has_history = hasattr(from_earth, "time") and len(from_earth.time) > 1

        if has_history:
            n_years = len(from_earth.time)
            revenues = []
            for i in range(n_years):
                revenues.append(self._compute_revenue_for_year(i))
            revenue = float(np.mean(revenues))
        else:
            # Single year: use that year's revenue
            revenue = self._compute_revenue_for_year(-1)

        # Revenue is a whole-farm quantity, so unlike the intensive
        # environmental metrics its floor has to scale with cropped area.
        # farm_size is crop area from the same from_earth cftfrac that
        # revenue itself is built on.
        materiality = self.model.config.coupled_config.materiality
        return max(
            revenue, materiality.baseline_floor_revenue_per_ha * self.farm_size
        )

    def _compute_baseline_metrics(self) -> None:
        """Compute historic-mean baselines for the environmental metrics.

        Sets ``baseline_yield``, ``baseline_soilc``, ``baseline_moisture`` and
        ``baseline_leaching`` as the mean over the historic period, mirroring
        ``_compute_baseline_revenue``. These are the self-referencing anchors the
        log-deviation rates are taken about; see farmer.log_rate.

        Metric definitions match the runtime properties so historic and
        coupled-run values live on the same scale:
        - yield: cft-fraction-weighted ``pft_harvestc``
        - soilc: top-layer ``soilc_agr_layer``
        - moisture: ``rootmoist_crop`` (``rootmoist_agr`` for older output)
        - leaching: ``leaching``, the nitrogen flux, every year of the record
          including the dry ones. A year that drained nothing lost nothing,
          which is a real zero and belongs in the mean.
        """
        from_earth = self.cell.from_earth
        has_history = hasattr(from_earth, "time") and len(from_earth.time) > 1

        if not has_history:
            self.baseline_yield = self.cropyield
            self.baseline_soilc = self.soilc
            self.baseline_moisture = self.root_moisture
            self.baseline_leaching = self.leaching
            return

        n_years = len(from_earth.time)
        moisture = moisture_var(from_earth)
        yields, soilcs, moistures, leachings = [], [], [], []
        for i in range(n_years):
            pft_harvestc = self.get_from_earth(
                "pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=i
            )
            cftfrac = self.get_from_earth(
                "cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=i
            )
            cropped = float(cftfrac.sum("band").item())
            yields.append(
                pft_harvestc.weighted(cftfrac).sum("band").item() / cropped
                if cropped > 0 else 0.0
            )
            soilcs.append(
                self.get_from_earth("soilc_agr_layer", as_scalar=True, band=0, time_idx=i)
            )
            moistures.append(
                self.get_from_earth(moisture, as_scalar=True, time_idx=i)
            )
            leachings.append(agricultural_leaching(self, time_idx=i))

        self.baseline_yield = float(np.mean(yields))
        self.baseline_soilc = float(np.mean(soilcs))
        self.baseline_moisture = float(np.mean(moistures))
        # Zero in a cell that never leached, which the offset in log_rate turns
        # into a rate of zero rather than a division by nothing.
        self.baseline_leaching = float(np.mean(leachings))

    def _read_leaching_state(self) -> None:
        """Read this year's leached nitrogen and runoff off the cell.

        Held as values for the year rather than read from the cell on demand,
        which is what keeps them the same vintage as yield, soil carbon,
        moisture and profit: the country performance store is built before any
        farmer has updated, so a cell read there would answer with the year the
        others have not reached yet.
        """
        self.leaching = self.cell_leaching
        self.runoff = self.cell_runoff

    def _read_negligible_levels(self) -> None:
        """Cache materiality offsets for ``log_rate`` on environmental metrics.

        Rates are ``ln((v+n)/(b+n))``. Same ``n`` must be used at runtime and in
        ``_load_history``, or the scoring ruler and the values it measures are
        on different scales. Bare ``ln(v/b)`` explodes when historic baselines
        are near zero (trivial physical change → large rate). See
        ``farmer.log_rate`` and config ``materiality.negligible_*``.
        """
        materiality = self.model.config.coupled_config.materiality
        self.negligible_yield = materiality.negligible_yield
        self.negligible_soilc = materiality.negligible_soilc
        self.negligible_moisture = materiality.negligible_moisture
        self.negligible_leaching = materiality.negligible_leaching

    def _compute_revenue_for_year(self, time_idx: int) -> float:
        """Compute revenue for a specific historic year.

        Used by ManagementPerformanceTracker to compute historic profit values.

        Parameters
        ----------
        time_idx : int
            Time index into historic data (0 = first year, -1 = last year).

        Returns
        -------
        float
            Revenue for that year (USD).
        """
        harvestc = self.get_from_earth(
            "pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=time_idx
        ).values
        cftfrac = self.get_from_earth(
            "cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=time_idx
        ).values

        production = harvestc * cftfrac * self._cell_area / (0.45 * 1e6)

        total_revenue = 0.0
        for indices, price in self._revenue_groups:
            total_revenue += production[indices].sum() * price

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
    # MAIN UPDATE METHOD
    # =========================================================================

    def update(self, t):
        """Update farmer state for current timestep.

        Update sequence:
        1. Call parent update (base farmer logic), read this year's levels
        2. Skip if control run (no CA dynamics)
        3. Update capital (depreciation, investment, profit)
        4. Skip TPB if capital below survival threshold
        5. Run TPB decision logic (includes affordability deselection)
        6. Apply practice transition if TPB threshold exceeded
        7. Reset observation years after evaluation

        Sets behaviour.transition_blocker to indicate why transition didn't happen.

        Parameters
        ----------
        t : int
            Current simulation year.
        """
        # Import blocker/driver constants here to avoid circular imports
        from inseeds.components.farming.ca_behaviour import (
            BLOCKER_NONE, BLOCKER_CONTROL_RUN, BLOCKER_CAPITAL_SURVIVAL,
            BLOCKER_TRANSITION_UNAFFORDABLE, BLOCKERS_WITHOUT_DECISION,
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
        # Step 4: Skip TPB if capital-constrained (survival mode)
        # -----------------------------------------------------------------
        if self.capital_constrained and self.capital < self.min_capital:
            # Survival mode suspends the decision, not the farming. The year is
            # still observed, because a farmer whose harvests went unrecorded
            # would stop existing for the model: the performance memory the
            # neighbours read would stay frozen at whatever it held when the
            # farm was last solvent, and old memories would never decay. The
            # counter keeps running too, so recovery is not followed by a wait.
            self.behaviour.observe()
            self.behaviour.transition_blocker = BLOCKER_CAPITAL_SURVIVAL
            self.behaviour.transition_driver = DRIVER_NONE
            self.behaviour.decrement_observation_years()
            return

        # -----------------------------------------------------------------
        # Step 5: Run TPB decision logic
        # -----------------------------------------------------------------
        self.behaviour.update()

        # -----------------------------------------------------------------
        # Step 6: Apply transition if TPB threshold exceeded
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

                    # Only apply if farmer can afford transition. Without
                    # capital constraints the cost is still charged - it is a
                    # real expense and belongs in profit - but it cannot block
                    # the change, on the assumption that it can be financed.
                    if self.capital >= cost or not self.capital_constrained:
                        # Deduct transition cost (the clamp only bites when
                        # unconstrained, where the stock may be outspent)
                        self.capital = max(0.0, self.capital - cost)
                        self._transition_cost = cost

                        # Special case: affordability-driven residue deselection
                        # gives cash from selling residue (income = opportunity
                        # cost). Only a farm that was actually retaining has any
                        # straw left to sell.
                        pathway = self.behaviour.target_pathway
                        if (pathway == "affordability" and
                            self.behaviour.residue_committed and
                            new_bundle.residue_on_field == 0):
                            residue_value = self.compute_residue_opportunity_cost()
                            self.capital += residue_value
                            self._residue_sale_income = residue_value

                        # Apply new practices
                        self.behaviour.commit_bundle(new_bundle)
                        # Clear blocker since transition succeeded
                        self.behaviour.transition_blocker = BLOCKER_NONE

                        # Set driver to indicate why transition succeeded
                        if pathway == "fallback":
                            self.behaviour.transition_driver = DRIVER_FALLBACK
                        elif pathway == "affordability":
                            self.behaviour.transition_driver = DRIVER_AFFORDABILITY_FORCED
                        elif pathway in ("local", "country", "cluster", "exploration"):
                            self.behaviour.set_tpb_component_driver(pathway)
                    else:
                        # Can't afford transition cost
                        self.behaviour.transition_blocker = BLOCKER_TRANSITION_UNAFFORDABLE
        else:
            # TPB score below threshold - identify which component is limiting
            self.behaviour.set_tpb_transition_blocker()

        # -----------------------------------------------------------------
        # Step 7: Reset observation years after a real decision
        # -----------------------------------------------------------------

        # Reset observation counter only after a real decision (accept or
        # decline a distinct target). BLOCKERS_WITHOUT_DECISION leave the
        # counter running so no-option years do not consume the cadence.
        if self.behaviour.transition_blocker not in BLOCKERS_WITHOUT_DECISION:
            self.behaviour.reset_observation_years()

        # A decision was due and the bundle came through it unchanged, so the
        # cover crop is the one practice left that can still answer a changed
        # price. A cleared blocker means a transition did happen, and there
        # apply_bundle has already chosen the crop.
        if (self.behaviour.evaluated_this_year
                and self.behaviour.transition_blocker != BLOCKER_NONE):
            self.behaviour.reconsider_cover_crop_type()
