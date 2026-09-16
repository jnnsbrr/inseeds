"""Conservation Agriculture practice bundles, costs, and performance tracking.

This module defines how farmers' agricultural practices are represented,
costed, tracked, and aggregated for social learning in the Conservation
Agriculture model.

Overview
--------
Conservation Agriculture (CA) is defined by three practices:

1. **No-till** (tillage=0): Minimal soil disturbance
2. **Cover crops** (cover_crop=1): Planting between main crops
3. **Residue retention** (residue_on_field=1): Leaving crop residues on soil

A practice bundle is a combination of these three. Each is binary, so there
are 2³ = 8 bundles. Tillage is inverted relative to the other two: 0 is the
CA state (no-till), while 0 on cover crop or residue is the conventional
state. Conventional farming is therefore ``(1, 0, 0)`` and full CA is
``(0, 1, 1)``, not "all off" versus "all on".

The bundle records only whether a cover crop is grown. Which crop - a
nitrogen-fixing legume or a non-legume catch crop - lives on the farmer
(``COVER_CROP_*``) because LPJmL needs it to pick the PFT. Memory slots
follow that split so the two are not averaged into one experience.

Key Concepts
------------
- **ManagementBundle**: Which three practices a farmer currently uses
- **ManagementCosts** / **PracticeCost**: Annual and one-time costs (USD/ha),
  GDP-scaled; cover-crop seed is typed
- **PolicySettings**: Carbon, nitrogen, and practice-subsidy instruments
- **ManagementPerformanceTracker**: Online trends of self-referenced rates
  since the last practice change
- **ManagementPerformance**: One remembered episode (bundle × cover-crop type)
- **ManagementPerformanceMemory**: A farmer's episodes, decayed and failure-capped
- **RegionManagementPerformance**: Winsorized mean of one bundle across farmers
- **RegionManagementPerformanceStore**: All bundles' performance in a region

Data Flow
---------
Individual Farmer
        │
        ▼
ManagementPerformanceTracker  ──► tracks trends over the current episode
        │
        ▼
ManagementPerformanceMemory   ──► remembers past (bundle, cover-crop) outcomes
        │
        ▼
RegionManagementPerformanceStore  ──► aggregates across farmers
        ├──► Country-level social learning
        └──► Cluster-level social learning (similar climates)

Example
-------
>>> # No-till with a cover crop but without residue retention:
>>> bundle = ManagementBundle.notill_covercrop
>>> print(bundle.label)
no-till + cover crop
>>> print(bundle.tillage, bundle.cover_crop, bundle.residue_on_field)
(0, 1, 0)
>>> print(ManagementBundle.conventional.value, ManagementBundle.conservation.value)
(1, 0, 0) (0, 1, 1)

See Also
--------
ca_behaviour : TPB decision model
ca_farmer : agent that holds a bundle, costs, and the tracker
ca_country : country-level prices, capital, and aggregation
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

import logging

from inseeds.components.farming.farmer import (
    AFT,
    NON_CROPS,
    agricultural_leaching,
    log_rate,
    moisture_var,
)
from inseeds.components.farming.metrics import (
    DEFAULT_CLIP_BOUNDS,
    METRIC_KEYS,
    METRICS,
    add_metric_properties,
    rates_of,
    scored_metrics,
    zero_metrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# When capital is low, farmers deselect practices in this order (most expensive first)
DESELECT_ORDER = ("cover_crop", "residue_on_field", "tillage")

# The three practice dimensions that define a management bundle
PRACTICE_FIELDS = ("tillage", "cover_crop", "residue_on_field")

# Which cover crop is grown, where the bundle records only whether one is. The
# type lives on the farmer because LPJmL needs it to pick the PFT, and the two
# are different enough - one fixes nitrogen, the other only holds on to what is
# already there - that the model prices them apart and remembers them apart.
COVER_CROP_NONE = 0
COVER_CROP_NONLEGUME = 1  # catch crop: scavenges nitrogen already in the soil
COVER_CROP_LEGUME = 2     # fixes atmospheric nitrogen

# Seed is the cost that separates the two, so it is configured once per type and
# exactly one of these applies. `PracticeCost.direct` leaves both out and
# `direct_for_type` adds the one that is actually being sown; summing them would
# charge a farm for a crop it did not plant.
SEED_COST_COMPONENTS = {
    COVER_CROP_NONLEGUME: "seeds_nonlegume",
    COVER_CROP_LEGUME: "seeds_legume",
}

# Clip bounds for per-farmer rates in region aggregates (Winsorized mean). Held
# in the metric registry, where each metric can carry its own, and re-exported
# here because that is where the model has always imported it from.
RATE_CLIP_BOUNDS = DEFAULT_CLIP_BOUNDS


# Human-readable labels and numeric IDs for each bundle
# Format: {internal_key: (numeric_id, display_label)}
_BUNDLE_METADATA: dict[str, tuple[int, str]] = {
    "notill": (0, "no-till"),
    "notill_residue": (1, "no-till + residue retention"),
    "notill_covercrop": (2, "no-till + cover crop"),
    "conservation": (3, "conservation agriculture"),
    "conventional": (4, "conventional farming"),
    "residue": (5, "residue retention"),
    "covercrop": (6, "cover crop"),
    "covercrop_residue": (7, "cover crop + residue retention"),
}


# =============================================================================
# UNIFIED PERFORMANCE SCORING
# =============================================================================

def _z(deviation: float, scale: float) -> float:
    """Normalise a deviation by cross-farmer spread (z-score units).

    Returns 0 if scale ≤ 0.
    """
    if scale <= 0:
        return 0.0
    return float(deviation / scale)


def _side_weights(own: float, other: float) -> tuple[float, float]:
    """Rescale two blend weights to sum to one, or to (0, 0) if both are zero.

    A side with nothing to say enters as zero, and rescaling then hands its
    weight to the side that does, rather than leaving the blend short.
    """
    total = own + other
    if total <= 0:
        return 0.0, 0.0
    return own / total, other / total


def compute_metric_score_inverted(
    level: float,
    trend: float,
    median_level: float,
    std_level: float,
    std_trend: float,
    planning_horizon: float,
    trend_weight: float,
) -> float:
    """Z-score metric score for lower-is-better variables (e.g. leaching).

    Inverts level and trend components relative to ``compute_metric_score``.
    Parameters match that function.
    """
    # Inverted level z-score: below median → score > 1.0
    level_score = 1.0 - _z(level - median_level, std_level)

    # Inverted trend, zero-centered: improving (negative) trend → score > 1.0
    trend_score = 1.0 - _z(trend * planning_horizon, std_trend)

    level_weight = 1.0 - trend_weight
    return level_weight * level_score + trend_weight * trend_score


def compute_metric_score(
    level: float,
    trend: float,
    median_level: float,
    std_level: float,
    std_trend: float,
    planning_horizon: float,
    trend_weight: float,
) -> float:
    """Per-metric performance score via robust z-normalisation.

    ``level_score = 1 + (level - median) / std_level``;
    ``trend_score = 1 + (trend × horizon) / std_trend``.
    Scales are MAD-based robust spreads from ``CAWorld.compute_reference_scales``.
    Score ≈ 1 at baseline level with zero trend; unbounded above/below.

    Parameters
    ----------
    level : float
        Self-referenced relative deviation (dimensionless, centred at 0).
    trend : float
        Annual change in that deviation (per year).
    median_level : float
        Z-score centre (0.0 for self-referenced rates).
    std_level : float
        Level scale (cross-farmer spread, floored at interannual std).
    std_trend : float
        Trend scale (robust MAD of projected trends).
    planning_horizon : float
        Years for trend projection (typically min_observation_years).
    trend_weight : float
        Weight on trend component; level weight = 1 - trend_weight.

    Returns
    -------
    float
        Combined level/trend score.
    """
    # Level z-score centered at 1.0 (using median for robustness)
    level_score = 1.0 + _z(level - median_level, std_level)

    # Trend zero-centered, scaled by population spread of projected trends
    trend_score = 1.0 + _z(trend * planning_horizon, std_trend)

    # Combine with configured weighting
    level_weight = 1.0 - trend_weight
    return level_weight * level_score + trend_weight * trend_score


def compute_performance_score(
    tracker_or_region: Any,
    farmer: Any,
) -> float:
    """Weighted performance score over the declared metrics.

    Which metrics those are, which way round each is scored and which config
    weights it answers to all come from ``metrics.METRICS``; this function only
    weighs them. Each is a relative deviation from the farm's own historic
    baseline — ``ln((v+n)/(b+n))`` (materiality offsets ``n``) and ``asinh``
    for profit. Per metric::

        score_m = 1 + rate/std_level + trend_weight * (trend*horizon)/std_trend

    with scales from ``CAWorld.compute_reference_scales`` (cross-farmer spread,
    floored at interannual std). Metrics declared lower-is-better, leaching
    today, are scored inverted. Same path for tracker, memory and region
    aggregates, which is why all three answer to ``mean_<metric>`` and
    ``<metric>_trend``.

    Parameters
    ----------
    tracker_or_region :
        Tracker, remembered performance, or region aggregate.
    farmer :
        Supplies AFT weights and ``world`` reference scales.

    Returns
    -------
    float
        Dimensionless weighted score (~1 at own baseline).

    Notes
    -----
    ``use_profit`` selects profit vs yield for the economic slot — only four of
    five ``weight_*`` config keys are active (see ``active_score_weights``).
    ``planning_horizon`` must match the trend-ruler horizon; it nearly cancels
    in the z-ratio and mainly affects PBC, not this score.

    See Also
    --------
    metrics.METRICS : the metrics this loops over
    compute_metric_score, CAWorld.compute_reference_scales, farmer.log_rate
    """
    # Same population rulers for every farmer this year
    scales = farmer.world.statistic.get("reference_scales")

    # Must match CAWorld trend-ruler horizon or trend z is mis-scaled
    planning_horizon = farmer.behaviour.projection_weight()

    # The economic metric is profit or physical yield, never both
    use_profit = farmer.model.config.coupled_config.tpb.use_profit

    total = 0.0
    for spec in scored_metrics(use_profit):
        score_metric = (
            compute_metric_score
            if spec.higher_is_better
            else compute_metric_score_inverted
        )
        # centre 0: rates already self-referenced
        score = score_metric(
            getattr(tracker_or_region, spec.mean_attr),
            getattr(tracker_or_region, spec.trend_attr),
            0.0,
            scales[spec.level_scale_key],
            scales[spec.trend_scale_key],
            planning_horizon,
            getattr(farmer, spec.trend_weight_key),
        )
        total += getattr(farmer, spec.weight_key) * score

    return total


def active_score_weights(coupled_config):
    """Score weights actually read by ``compute_performance_score``, per AFT.

    The two economic metrics share one slot, so one ``weight_*`` entry is
    always ignored: profit or yield per ``use_profit``.

    Parameters
    ----------
    coupled_config
        Config section with ``tpb`` and ``aftpar``.

    Returns
    -------
    dict
        ``{aft_name: {weight_name: value}}``.
    """
    use_profit = coupled_config.tpb.use_profit
    names = tuple(spec.weight_key for spec in scored_metrics(use_profit))
    return {
        aft.name: {
            name: float(getattr(getattr(coupled_config.aftpar, aft.name), name))
            for name in names
        }
        for aft in AFT
    }


def log_active_score_weights(coupled_config, tolerance=0.02):
    """Log active score weights; warn if sum deviates from 1.0.

    Weights should sum to 1 because the score is compared to a fixed 0.5
    threshold as an unnormalised weighted sum.

    Parameters
    ----------
    coupled_config
        Config section with ``tpb`` and ``aftpar``.
    tolerance : float, optional
        Allowed deviation from 1.0 before warning.
    """
    for aft_name, weights in active_score_weights(coupled_config).items():
        total = sum(weights.values())
        detail = ", ".join(
            f"{name.removeprefix('weight_')} {value:g}"
            for name, value in weights.items()
        )
        logger.info("Score weights in force (%s): %s | sum %g", aft_name, detail, total)
        if abs(total - 1.0) > tolerance:
            logger.warning(
                "Score weights for %s sum to %g, not 1.0. The score is an "
                "unnormalised weighted sum against a fixed 0.5 threshold, so "
                "this rescales every farmer's score.",
                aft_name,
                total,
            )


# =============================================================================
# COST STRUCTURES
# =============================================================================

def _scale_cost_value(value: Any, capital_ratio: float) -> float:
    """Scale a cost value based on capital ratio.

    If value is a list/tuple [min, max], interpolates based on capital_ratio.
    If value is a scalar, returns it unchanged.

    Parameters
    ----------
    value : float or list[float, float]
        Either a single value or [min, max] range.
    capital_ratio : float
        Ratio of farmer's capital to reference capital, clamped to [0, 1].

    Returns
    -------
    float
        Scaled cost value.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        min_val, max_val = float(value[0]), float(value[1])
        ratio = max(0.0, min(1.0, capital_ratio))  # Clamp to [0, 1]
        return min_val + (max_val - min_val) * ratio
    return float(value) if value else 0.0


@dataclass(frozen=True)
class PracticeCost:
    """Annual and transition costs for one practice (USD/ha).

    Attributes
    ----------
    transition : float
        One-time adoption cost (USD/ha).
    cost_components : dict[str, float]
        Named annual cost components (USD/ha/yr); summed by ``direct``.
    """

    transition: float
    cost_components: dict[str, float]

    @property
    def direct(self) -> float:
        """Annual cost that applies whichever variant of the practice is used.

        Excludes the per-type seed costs, which are mutually exclusive: see
        ``direct_for_type`` for the cover crop's full annual cost.
        """
        return sum(
            value
            for name, value in self.cost_components.items()
            if name not in SEED_COST_COMPONENTS.values()
        )

    def seed_cost(self, cover_crop_type: int | None) -> float:
        """Seed cost of the cover crop being sown, or zero if none is."""
        component = SEED_COST_COMPONENTS.get(int(cover_crop_type or 0))
        if component is None:
            return 0.0
        return self.cost_components.get(component, 0.0)

    def direct_for_type(self, cover_crop_type: int | None) -> float:
        """Annual cost including the seed for the cover crop actually sown."""
        return self.direct + self.seed_cost(cover_crop_type)

    @classmethod
    def from_config(cls, data: Any, capital_ratio: float = 1.0) -> "PracticeCost":
        """Build from config dict, optionally scaling [min, max] ranges.

        Parameters
        ----------
        data : dict
            ``transition`` plus ``costs`` dict or legacy ``direct``.
        capital_ratio : float
            Interpolation weight for ranged values (default 1.0).

        Returns
        -------
        PracticeCost
        """
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        elif not isinstance(data, dict):
            data = dict(data)

        transition = _scale_cost_value(data.get("transition", 0), capital_ratio)

        # New format: 'costs' dict with named components
        if "costs" in data:
            costs_raw = data["costs"]
            if hasattr(costs_raw, "to_dict"):
                costs_raw = costs_raw.to_dict()
            elif not isinstance(costs_raw, dict):
                costs_raw = dict(costs_raw)

            cost_components = {
                name: _scale_cost_value(value, capital_ratio)
                for name, value in costs_raw.items()
            }
        else:
            cost_components = {}

        return cls(transition=transition, cost_components=cost_components)


#: Stands in for "no end" when neither the policy block nor the LPJmL config
#: says when the run stops. Far enough out that no run reaches it, and finite
#: so the comparison in ``in_force`` needs no special case.
MAX_YEAR = 9999


def default_policy_span(config) -> tuple[int, int]:
    """Years instruments act over when the policy block does not say.

    The whole coupled run: from the first year inseeds is in the loop to the
    last year LPJmL simulates. Taken from the run rather than written down,
    because a literal year in the policy block is a second copy of something
    the simulation already knows, and the two go out of step silently. A
    ``start_year`` left at 2025 after the coupling was moved to 2030 would
    read as "in force from the start", which is harmless; moved the other way
    it would back-date an instrument into the period farms learned from,
    which is the one thing this must not do.

    Parameters
    ----------
    config : LpjmlConfig
        The run's LPJmL configuration.

    Returns
    -------
    tuple of int
        First and last year, both inclusive.
    """
    start = getattr(config, "start_coupling", None) or getattr(
        config, "firstyear", None
    )
    end = getattr(config, "lastyear", None)
    return (
        0 if start is None else int(start),
        MAX_YEAR if end is None else int(end),
    )


@dataclass(frozen=True, slots=True)
class PolicySettings:
    """Instrument rates in force, resolved for one country.

    Attributes
    ----------
    carbon_price : float
        USD per tonne of CO2 equivalent sequestered above baseline.
    nitrogen_charge : float
        USD per kg of nitrogen leached, already interpolated for this country.
    levy_free_concentration : float
        Nitrate concentration in mgN/l below which leaching is not charged.
    cover_crop_subsidy : float
        USD per hectare per year for holding a cover crop, already
        interpolated for this country. Positive is a payment to the farm.
        Paid whichever crop is sown.
    cover_crop_legume_subsidy : float
        USD per hectare per year on top of ``cover_crop_subsidy`` where the
        crop sown is a legume. An enhancement rather than a separate rate,
        which is the shape the real schemes take: EQIP 340 and the CAP
        eco-schemes post a base cover crop payment and pay more for a
        multi-species or legume mix. It is also the only per-hectare rate here
        that distinguishes the two crops, so it is what a scenario reaches for
        to steer the type rather than the practice.
    residue_subsidy : float
        USD per hectare per year for retaining residue, same convention.
    residue_subsidy_share : float
        Fraction of the farm's own straw opportunity cost paid for retaining,
        in [0, 1]. The alternative to ``residue_subsidy`` rather than an
        addition to it, and the form the real schemes take; see
        ``from_config``. Farm-specific, so the payment is worked out on the
        farmer and not by ``subsidy_for``.
    start_year, end_year : int
        Years the instruments act, both inclusive. ``end_year`` defaults to no
        end at all when the config leaves it out.
    windows : tuple of (str, int, int)
        Per-instrument overrides of that span, as ``(name, start, end)``. A
        tuple rather than a dict so the record stays hashable.
    """

    carbon_price: float
    nitrogen_charge: float
    levy_free_concentration: float
    cover_crop_subsidy: float
    cover_crop_legume_subsidy: float
    residue_subsidy: float
    residue_subsidy_share: float
    start_year: int
    end_year: int = MAX_YEAR
    windows: tuple[tuple[str, int, int], ...] = ()

    #: Attributes holding the rates for each instrument a window can name. The
    #: residue payment has two, a flat rate and a cost share, and a window over
    #: the instrument has to switch off whichever of them a scenario used.
    RATE_FIELDS = {
        "carbon_price": ("carbon_price",),
        "nitrogen_charge": ("nitrogen_charge",),
        "cover_crop": ("cover_crop_subsidy", "cover_crop_legume_subsidy"),
        "residue_on_field": ("residue_subsidy", "residue_subsidy_share"),
    }

    def window_for(self, name: str) -> tuple[int, int]:
        """Years the named instrument acts, falling back to the global span."""
        for entry, start, end in self.windows:
            if entry == name:
                return start, end
        return self.start_year, self.end_year

    def in_force(self, year: int) -> "PolicySettings":
        """This record with every instrument outside its window zeroed.

        Resolving the whole set at once, rather than testing the year at each
        point of use, is what keeps the bill and the expectation of it from
        disagreeing about whether a scheme is running - the same reason
        ``nitrogen_charge_for`` is one function.

        The allowance is not an instrument and is left alone: it only shapes a
        charge, and a charge outside its window is already zero.
        """
        zeroed = {}
        for name, fields in self.RATE_FIELDS.items():
            start, end = self.window_for(name)
            if not (start <= year <= end):
                zeroed.update({field: 0.0 for field in fields})
        if not zeroed:
            return self
        return replace(self, **zeroed)

    @property
    def any_in_force(self) -> bool:
        """Whether this resolves to an instrument at all.

        An allowance on its own is not one: it only shapes a charge that has
        to be turned on separately.
        """
        return bool(
            self.carbon_price
            or self.nitrogen_charge
            or self.cover_crop_subsidy
            or self.cover_crop_legume_subsidy
            or self.residue_subsidy
            or self.residue_subsidy_share
        )

    def subsidy_for(self, bundle, cover_crop_type=None) -> float:
        """Per-hectare annual payment for the practices in ``bundle``.

        The flat rates only. ``residue_subsidy_share`` is a fraction of a
        quantity only the farm knows, so it is added in
        ``Farmer.practice_subsidy_for`` and is not part of any per-hectare
        rate.

        Parameters
        ----------
        bundle : ManagementBundle
            The practices being paid for.
        cover_crop_type : int, optional
            Which crop is sown, where the bundle records only whether one is.
            The legume enhancement is paid on this and nothing else. Left out,
            only the base rate is paid, which is the right reading for a caller
            that has a bundle and no farm to ask - the enhancement is a
            property of the crop, not of the practice.
        """
        total = 0.0
        if bundle.cover_crop == 1:
            total += self.cover_crop_subsidy
            if cover_crop_type == COVER_CROP_LEGUME:
                total += self.cover_crop_legume_subsidy
        if bundle.residue_on_field == 1:
            total += self.residue_subsidy
        return total

    @classmethod
    def _read_windows(cls, raw: dict) -> tuple[tuple[str, int, int], ...]:
        """Parse ``policy.windows`` into ``(name, start, end)`` triples."""
        windows = raw.get("windows", {}) or {}
        if hasattr(windows, "to_dict"):
            windows = windows.to_dict()
        elif not isinstance(windows, dict):
            windows = dict(windows)

        parsed = []
        for name, span in windows.items():
            if name not in cls.RATE_FIELDS:
                raise ValueError(
                    f"policy.windows has no instrument called {name!r}. "
                    f"Known instruments: {sorted(cls.RATE_FIELDS)}."
                )
            try:
                start, end = span
            except (TypeError, ValueError):
                raise ValueError(
                    f"policy.windows.{name} must be a [start_year, end_year] "
                    f"pair, got {span!r}."
                ) from None
            if int(end) < int(start):
                raise ValueError(
                    f"policy.windows.{name} ends in {end} before it starts in "
                    f"{start}, so the instrument would never act. Leave it out "
                    "to switch it off instead of writing an empty window."
                )
            parsed.append((name, int(start), int(end)))
        return tuple(parsed)

    @staticmethod
    def _read_residue_share(subsidies: dict) -> float:
        """Parse ``practice_subsidies.residue_on_field_share``.

        A SINGLE NUMBER, not a [low, high] pair, and not scaled by GDP. A
        share is dimensionless, and the cost it applies to is the farm's own
        straw income, which already carries every local difference there is -
        the crop, the yield, the price, and how much straw that region takes
        off the field at all. Interpolating the share on income too would
        count the same gradient twice. The real schemes point the other way in
        any case: EQIP raises its share for historically underserved
        producers, so a share that rose with GDP would have the gradient
        backwards.
        """
        share = subsidies.get("residue_on_field_share", 0.0)
        if isinstance(share, (list, tuple)):
            raise ValueError(
                "policy.practice_subsidies.residue_on_field_share must be a "
                "single fraction, not a [low, high] range. It is applied to "
                "the farm's own straw opportunity cost, which already varies "
                "with local yields, prices and residue use, so a share that "
                "also varied by income would scale the same gradient twice."
            )
        share = float(share or 0.0)
        if not 0.0 <= share <= 1.0:
            raise ValueError(
                "policy.practice_subsidies.residue_on_field_share is a "
                f"fraction of the opportunity cost and must lie in [0, 1], "
                f"got {share}. Above 1 a farm would be paid more than the "
                "straw was worth, which makes retention profitable in itself "
                "rather than compensated."
            )
        return share

    @classmethod
    def from_config(
        cls,
        raw: Any,
        gdp_cost_ratio: float | None = None,
        default_span: tuple[int, int] = (0, MAX_YEAR),
    ):
        """Resolve the configured rates for one country's income level.

        Rates only. Whether they are in force in a given year is
        ``in_force``, which the farmer applies on every read.

        ``default_span`` is the window instruments take when the policy block
        gives no ``start_year`` or ``end_year``, which is how the coupled run's
        own extent becomes the default; see ``default_policy_span``.

        The nitrogen charge scales on the same GDP ratio as the practice costs,
        because nitrogen damage is local: it depends on the aquifer and on who
        drinks from it, so the rate that internalises it is set locally too.
        The flat practice subsidies scale for a different local reason - they
        come out of a national budget, so what a country pays tracks what it
        can afford. ``residue_on_field_share`` does not, being a share of a
        cost that is already local; see ``_read_residue_share``.

        The carbon price does not, and passing a range for it is an error
        rather than something to average. Carbon is a global externality, so a
        tonne is worth the same wherever it is put away; a price that varied by
        income would misprice it and would draw sequestration towards the
        countries paying most rather than the ones abating cheapest.
        """
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        carbon_price = raw.get("carbon_price", 0.0)
        if isinstance(carbon_price, (list, tuple)):
            raise ValueError(
                "policy.carbon_price must be a single global figure, not a "
                "[low, high] range: a tonne of CO2 is worth the same wherever "
                "it is sequestered, and a price that varied by income would "
                "move sequestration towards the richer country rather than "
                "the cheaper one."
            )

        ratio = 0.5 if gdp_cost_ratio is None else gdp_cost_ratio

        subsidies = raw.get("practice_subsidies", {}) or {}
        if hasattr(subsidies, "to_dict"):
            subsidies = subsidies.to_dict()
        elif not isinstance(subsidies, dict):
            subsidies = dict(subsidies)

        windows = cls._read_windows(raw)
        residue_flat = _scale_cost_value(
            subsidies.get("residue_on_field", 0.0), ratio
        )
        residue_share = cls._read_residue_share(subsidies)
        if residue_flat and residue_share:
            raise ValueError(
                "policy.practice_subsidies sets both residue_on_field and "
                "residue_on_field_share, which would pay a farm twice for one "
                "retention. They are two ways of writing the same instrument: "
                "a flat rate per hectare, or a share of the straw income the "
                "farm actually gives up. Set one and leave the other at zero."
            )

        return cls(
            carbon_price=float(carbon_price or 0.0),
            nitrogen_charge=_scale_cost_value(
                raw.get("nitrogen_charge", 0.0), ratio
            ),
            # A drinking-water standard, so the same figure everywhere and not
            # scaled by income: what is safe to drink does not depend on the
            # country's GDP.
            levy_free_concentration=float(
                raw.get("levy_free_concentration", 0.0) or 0.0
            ),
            cover_crop_subsidy=_scale_cost_value(
                subsidies.get("cover_crop", 0.0), ratio
            ),
            cover_crop_legume_subsidy=_scale_cost_value(
                subsidies.get("cover_crop_legume", 0.0), ratio
            ),
            residue_subsidy=residue_flat,
            residue_subsidy_share=residue_share,
            start_year=int(
                default_span[0]
                if raw.get("start_year") is None
                else raw["start_year"]
            ),
            end_year=int(
                default_span[1]
                if raw.get("end_year") is None
                else raw["end_year"]
            ),
            windows=windows,
        )


@dataclass(frozen=True, slots=True)
class ManagementCosts:
    """Direct and transition costs for tillage, cover crop, and residue.

    Attributes
    ----------
    tillage, cover_crop, residue_on_field : PracticeCost
        Per-practice cost structures (scaled by GDP at load).
    """

    tillage: PracticeCost
    cover_crop: PracticeCost
    residue_on_field: PracticeCost

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any] | Any,
        gdp_cost_ratio: float | None = None,
    ) -> ManagementCosts:
        """Build from config, scaling costs by GDP percentile ratio.

        Parameters
        ----------
        raw : dict
            Practice cost specifications.
        gdp_cost_ratio : float, optional
            Value in [0, 1] from country GDP; default 0.5.

        Returns
        -------
        ManagementCosts
        """
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        # Use GDP-based cost ratio, default to middle of range
        cost_ratio = gdp_cost_ratio if gdp_cost_ratio is not None else 0.5

        return cls(
            tillage=PracticeCost.from_config(raw.get("tillage", {}), cost_ratio),
            cover_crop=PracticeCost.from_config(raw.get("cover_crop", {}), cost_ratio),
            residue_on_field=PracticeCost.from_config(raw.get("residue_on_field", {}), cost_ratio),
        )


# =============================================================================
# PRACTICE BUNDLES
# =============================================================================

class ManagementBundle(Enum):
    """Triple of binary CA practices (8 combinations).

    Encoding (0 = CA state for tillage; conventional for others):
    - tillage: 0 no-till, 1 conventional
    - cover_crop: 0 none, 1 cover crop
    - residue_on_field: 0 baseline removal, 1 retention

    Full CA is ``(0, 1, 1)`` (conservation).
    """

    notill = (0, 0, 0)
    notill_residue = (0, 0, 1)
    notill_covercrop = (0, 1, 0)
    conservation = (0, 1, 1)
    conventional = (1, 0, 0)
    residue = (1, 0, 1)
    covercrop = (1, 1, 0)
    covercrop_residue = (1, 1, 1)

    def __new__(
        cls,
        tillage: int,
        cover_crop: int,
        residue_on_field: int,
    ) -> ManagementBundle:
        obj = object.__new__(cls)
        obj._value_ = (tillage, cover_crop, residue_on_field)
        return obj

    @property
    def key(self) -> str:
        """Internal identifier (notill, notill_residue, ...)."""
        return self._name_

    @property
    def tillage(self) -> int:
        return self.value[0]

    @property
    def cover_crop(self) -> int:
        return self.value[1]

    @property
    def residue_on_field(self) -> int:
        return self.value[2]

    @property
    def practices(self) -> tuple[str, str, str]:
        return PRACTICE_FIELDS

    @property
    def id(self) -> int:
        """Numeric bundle ID (0-7)."""
        return _BUNDLE_METADATA[self.key][0]

    @property
    def label(self) -> str:
        """Human-readable bundle name."""
        return _BUNDLE_METADATA[self.key][1]

    @classmethod
    def from_practices(
        cls,
        tillage: int,
        cover_crop: int,
        residue_on_field: int,
    ) -> ManagementBundle:
        """Return the bundle matching a practice triple."""
        return cls((tillage, cover_crop, residue_on_field))

    @classmethod
    def from_name(cls, name: str) -> ManagementBundle:
        """Return the bundle with the given human-readable name."""
        return cls(name)

    @classmethod
    def from_key(cls, key: str) -> ManagementBundle:
        """Return the bundle with the given internal identifier."""
        return cls(key)

    @classmethod
    def from_id(cls, bundle_id: int) -> ManagementBundle:
        """Return the bundle with the given numeric ID."""
        for bundle in cls:
            if bundle.id == bundle_id:
                return bundle
        raise ValueError(f"Unknown bundle ID: {bundle_id}")

    @classmethod
    def all_bundles(cls) -> tuple[ManagementBundle, ...]:
        """All 8 practice bundle members."""
        return tuple(cls)

    @classmethod
    def all_ids(cls) -> tuple[int, ...]:
        """All numeric bundle IDs in definition order."""
        return tuple(bundle.id for bundle in cls)

    @classmethod
    def all_keys(cls) -> tuple[str, ...]:
        """All internal bundle keys in definition order."""
        return tuple(bundle.key for bundle in cls)

    @classmethod
    def all_labels(cls) -> tuple[str, ...]:
        """All human-readable bundle labels in definition order."""
        return tuple(bundle.label for bundle in cls)

    def change_practices(
        self,
        *,
        tillage: int | None = None,
        cover_crop: int | None = None,
        residue_on_field: int | None = None,
    ) -> ManagementBundle:
        """Return the bundle with one or more practices changed."""
        current = self.value
        return type(self).from_practices(
            current[0] if tillage is None else tillage,
            current[1] if cover_crop is None else cover_crop,
            current[2] if residue_on_field is None else residue_on_field,
        )

    def similarity(self, bundle: ManagementBundle) -> float:
        """Fraction of matching practices between two bundles."""
        matches = sum(a == b for a, b in zip(self.value, bundle.value))
        return matches / len(PRACTICE_FIELDS)


# =============================================================================
# FARMER-LEVEL PERFORMANCE TRACKING
# =============================================================================

@dataclass
class ManagementPerformance:
    """Stored performance for one practice bundle (farmer memory).

    Holds mean levels and trends of self-referenced rates plus duration,
    last update year, and failure count for exploration limits.

    Attributes
    ----------
    level, trend : dict[str, float]
        Mean level and trend (per year) of every registered metric over the
        episode, keyed by metric. Named to match the tracker's properties of
        the same name, so a remembered episode and a live one read alike.
    duration : int
        Years on this bundle.
    last_updated : int
        Simulation year of last update (for decay).
    failure_count : int
        Poor-outcome count for this bundle.
    """

    # Self-referenced deviations, as the tracker reports them: dimensionless
    # and centred on zero, so no conversion is needed to score them.
    level: dict[str, float] = field(default_factory=zero_metrics)
    trend: dict[str, float] = field(default_factory=zero_metrics)

    duration: int = 0
    last_updated: int = 0

    failure_count: int = 0

    @classmethod
    def empty(cls) -> ManagementPerformance:
        return cls()

    @classmethod
    def from_tracker(
        cls,
        trend: dict[str, float],
        level: dict[str, float],
        duration: int,
        year: int,
        failure_count: int = 0,
    ) -> ManagementPerformance:
        """Create from tracker trend and level dictionaries."""
        return cls(
            level=dict(level),
            trend=dict(trend),
            duration=duration,
            last_updated=year,
            failure_count=failure_count,
        )

    @classmethod
    def from_trend(
        cls,
        trend: dict[str, float],
        duration: int,
        year: int,
        failure_count: int = 0,
    ) -> ManagementPerformance:
        return cls(
            trend=dict(trend),
            duration=duration,
            last_updated=year,
            failure_count=failure_count,
        )

    def update_from_tracker(
        self,
        trend: dict[str, float],
        level: dict[str, float],
        duration: int,
        year: int,
    ) -> None:
        """Refresh trends, levels and duration; preserve failure_count."""
        self.level = dict(level)
        self.trend = dict(trend)
        self.duration = duration
        self.last_updated = year

    def update_from_trend(
        self,
        trend: dict[str, float],
        duration: int,
        year: int,
    ) -> None:
        """Refresh trends and duration; preserve failure_count."""
        self.trend = dict(trend)
        self.duration = duration
        self.last_updated = year

    def clear(self) -> None:
        """Reset to neutral (no remembered experience)."""
        self.level = zero_metrics()
        self.trend = zero_metrics()
        self.duration = 0
        self.last_updated = 0
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1

    def __repr__(self) -> str:
        detail = ", ".join(
            f"{key}={self.level[key]:+.1%}, {key}_trend={self.trend[key]:+.2%}/yr"
            for key in METRIC_KEYS
        )
        return f"ManagementPerformance({detail}, n={self.duration}yr)"


# The unified scoring API, plus the ``level_``/``trend_`` spellings the rest of
# the model reads a remembered episode by. Generated rather than written out:
# a remembered episode holds the means of the same self-referenced rates a
# tracker does, so only the names differ.
add_metric_properties(
    ManagementPerformance,
    {
        "level_{key}": lambda self, key: self.level[key],
        "trend_{key}": lambda self, key: self.trend[key],
        "mean_{key}": lambda self, key: self.level[key],
        "{key}_trend": lambda self, key: self.trend[key],
    },
)


#: Returned for a slot the farm has no record of. Shared rather than allocated
#: per read: peer scans hit far more empty slots than full ones. Reads must not
#: mutate it, which is why every mutating path goes through ``_slot``.
_NO_EXPERIENCE = ManagementPerformance.empty()


class ManagementPerformanceMemory:
    """Per-episode performance memory for one farmer.

    An episode is a bundle *and* the cover crop grown under it. Two cover crops
    are two different practices - a legume fixes nitrogen, a catch crop takes up
    what is already there - and remembering both in one slot would average a
    farm's experience of each into an experience of neither. Bundles without a
    cover crop key on ``None``, so the bundle space itself stays binary and does
    not inflate with the type.

    Which type a slot means is resolved by ``type_for``, supplied by the
    behaviour: for the bundle being run it is the crop actually in the ground,
    and for a candidate it is the one this farm would plant. That keeps every
    caller talking in bundles while the slots stay typed.

    Memories older than ``memory_decay_years`` are cleared. Failure counts limit
    re-exploration of poor bundles.
    """

    def __init__(
        self,
        current_bundle: ManagementBundle,
        state: ManagementPerformanceTracker,
        year: int,
        type_for=None,
    ) -> None:
        self._type_for = type_for
        self._slots: dict[
            tuple[ManagementBundle, int | None], ManagementPerformance
        ] = {}

        if state.n > 1:
            self._slot(self.key(current_bundle)).update_from_tracker(
                state.trend, state.level, state.n, year
            )

    def key(self, bundle: ManagementBundle) -> tuple[ManagementBundle, int | None]:
        """The slot this farm's experience of ``bundle`` belongs in."""
        if not bundle.cover_crop:
            return (bundle, None)
        if self._type_for is None:
            return (bundle, None)
        return (bundle, self._type_for(bundle))

    def _slot(self, key) -> ManagementPerformance:
        """The stored slot, created empty if this is its first use."""
        if key not in self._slots:
            self._slots[key] = ManagementPerformance.empty()
        return self._slots[key]

    def get(self, key) -> ManagementPerformance:
        """Read a slot by its exact key, without creating it."""
        return self._slots.get(key, _NO_EXPERIENCE)

    def __getitem__(self, bundle: ManagementBundle) -> ManagementPerformance:
        """Read this farm's experience of ``bundle`` under the type it would use."""
        return self.get(self.key(bundle))

    def update_current(
        self,
        bundle: ManagementBundle,
        trend: dict[str, float],
        level: dict[str, float],
        n: int,
        year: int,
    ) -> None:
        """Refresh memory for the active bundle with latest trends and levels."""
        if n < 2:
            return
        self._slot(self.key(bundle)).update_from_tracker(trend, level, n, year)

    def record_performance(
        self,
        bundle: ManagementBundle,
        trend: dict[str, float],
        level: dict[str, float],
        n: int,
        year: int,
    ) -> None:
        """Store outcome when leaving a bundle; preserve failure count."""
        if n < 2:
            return
        key = self.key(bundle)
        self._slots[key] = ManagementPerformance.from_tracker(
            trend, level, n, year, failure_count=self.get(key).failure_count
        )

    def record_failure(self, bundle: ManagementBundle) -> None:
        self._slot(self.key(bundle)).record_failure()

    def decay(self, current_year: int, memory_decay_years: int) -> None:
        """Clear memories older than memory_decay_years."""
        for mem in self._slots.values():
            if mem.duration > 0 and current_year - mem.last_updated > memory_decay_years:
                mem.clear()


# =============================================================================
# REGIONAL AGGREGATION (for social learning)
# =============================================================================

@dataclass
class RegionManagementPerformance:
    """Regional mean performance for one practice bundle.

    Metrics are self-referenced relative deviations per farmer. Rates are
    Winsorized to ``RATE_CLIP_BOUNDS`` before summing. Sums (not means) are
    stored internally for mergeable aggregation; means via properties.

    Attributes
    ----------
    level_sums, trend_sums : dict[str, float]
        Summed level and trend rates across contributing farmers, keyed by
        metric. Sums rather than means so two aggregates can be pooled.
    count : int
        Number of farmers in the aggregate.
    """

    level_sums: dict[str, float] = field(default_factory=zero_metrics)
    trend_sums: dict[str, float] = field(default_factory=zero_metrics)
    count: int = 0

    @classmethod
    def empty(cls) -> RegionManagementPerformance:
        return cls()

    @classmethod
    def from_means(
        cls,
        level: dict[str, float],
        trend: dict[str, float],
        count: int,
    ) -> RegionManagementPerformance:
        """Build an aggregate that reports the given means over ``count`` farms."""
        return cls(
            level_sums={key: value * count for key, value in level.items()},
            trend_sums={key: value * count for key, value in trend.items()},
            count=count,
        )

    @property
    def n_farmers(self) -> int:
        return self.count

    @property
    def level(self) -> dict[str, float]:
        """Mean level of every metric, keyed by metric."""
        return {key: self._mean(self.level_sums[key]) for key in METRIC_KEYS}

    @property
    def trend(self) -> dict[str, float]:
        """Mean trend of every metric, keyed by metric."""
        return {key: self._mean(self.trend_sums[key]) for key in METRIC_KEYS}

    def _mean(self, total: float) -> float:
        return total / self.count if self.count else 0.0

    def add_observation(
        self,
        level: dict[str, float],
        trend: dict[str, float],
    ) -> None:
        """Add one farmer; clip rates to the metric's bounds before summing."""
        for key, spec in METRICS.items():
            lo, hi = spec.clip_bounds
            self.level_sums[key] += max(lo, min(hi, level[key]))
            self.trend_sums[key] += max(lo, min(hi, trend[key]))
        self.count += 1

    def merge(self, other: RegionManagementPerformance) -> None:
        """Add another aggregate's sums (cluster roll-up).

        Pooling, so the result is the mean over every farmer in the parts and a
        populous member of a cluster weighs more than a sparse one. That is
        what a cluster is: one population of farms sharing an agroecology, read
        as a whole. Blending across a border instead asks how far a farmer
        looks beyond it, which is a stated preference rather than a matter of
        sample size - see ``from_weighted_means``.
        """
        for key in METRIC_KEYS:
            self.level_sums[key] += other.level_sums[key]
            self.trend_sums[key] += other.trend_sums[key]
        self.count += other.count

    @classmethod
    def from_weighted_means(
        cls,
        own: RegionManagementPerformance,
        other: RegionManagementPerformance,
        *,
        own_weight: float,
        other_weight: float,
        n_farmers: int,
    ) -> RegionManagementPerformance:
        """Merge two aggregates using weights applied to their means.

        Weights on the means, not on the sums, so the configured own/neighbour
        split holds whatever the two populations are: how much attention a
        farmer pays across the border is a behavioural assumption, and pooling
        would silently replace it with the ratio of the farmer counts. Contrast
        ``merge``, which pools because a cluster is a single population.

        ``n_farmers`` only scales the sums so the result reports the blended
        means; it does not enter the blend.
        """
        if n_farmers <= 0:
            return cls.empty()

        def blend(own_means, other_means):
            return {
                key: own_weight * own_means[key] + other_weight * other_means[key]
                for key in METRIC_KEYS
            }

        return cls.from_means(
            blend(own.level, other.level),
            blend(own.trend, other.trend),
            n_farmers,
        )


# The unified scoring API. A region aggregate answers to the same names as a
# tracker and a remembered episode, which is what lets one scoring path serve
# all three.
add_metric_properties(
    RegionManagementPerformance,
    {
        "mean_{key}": lambda self, key: self._mean(self.level_sums[key]),
        "mean_{key}_trend": lambda self, key: self._mean(self.trend_sums[key]),
        "{key}_trend": lambda self, key: self._mean(self.trend_sums[key]),
        "{key}_sum": lambda self, key: self.level_sums[key],
        "{key}_trend_sum": lambda self, key: self.trend_sums[key],
    },
)


class RegionManagementPerformanceStore:
    """Performance and adoption data for all bundles in a region.

    Attributes
    ----------
    year : int
        Simulation year (-1 if unset).
    total_farmers : int
        Farmer count in region.
    bundle_counts : dict[ManagementBundle, float]
        Adoption counts per bundle (may be weighted floats after neighbour merge).
    country_codes : list[str]
        ISO codes included (cluster stores).
    """

    def __init__(
        self,
        year: int = -1,
        total_farmers: int = 0,
        bundle_counts: dict[ManagementBundle, float] | None = None,
        country_codes: list[str] | None = None,
    ) -> None:
        self.year = year
        self.total_farmers = total_farmers
        self.bundle_counts: dict[ManagementBundle, float] = bundle_counts or {}
        self.country_codes: list[str] = country_codes or []
        self._bundles: dict[ManagementBundle, RegionManagementPerformance] = {}

    def _get(self, bundle: ManagementBundle) -> RegionManagementPerformance:
        if bundle not in self._bundles:
            self._bundles[bundle] = RegionManagementPerformance.empty()
        return self._bundles[bundle]

    def get(self, bundle: ManagementBundle) -> RegionManagementPerformance:
        return self._bundles.get(bundle, RegionManagementPerformance.empty())

    def add_observation(self, bundle: ManagementBundle, farmer: Any) -> None:
        """Add farmer observation as self-referenced relative deviations.

        Level = current rates; trends from ``performance_tracker``.
        """
        self._get(bundle).add_observation(
            rates_of(farmer),
            farmer.behaviour.performance_tracker.trend,
        )

    def add_performance(
        self,
        bundle: ManagementBundle,
        performance: RegionManagementPerformance,
    ) -> None:
        self._get(bundle).merge(performance)

    def __getitem__(self, bundle: ManagementBundle) -> RegionManagementPerformance:
        return self.get(bundle)

    def items(self):
        return ((b, p) for b, p in self._bundles.items() if p.count > 0)

    def __repr__(self) -> str:
        n_bundles = len([b for b, p in self._bundles.items() if p.count > 0])
        return (
            f"RegionManagementPerformanceStore(year={self.year}, "
            f"farmers={self.total_farmers}, bundles={n_bundles})"
        )

    @property
    def bundle_n(self) -> dict[ManagementBundle, int]:
        return {bundle: perf.count for bundle, perf in self.items()}

    @property
    def farmer_n(self) -> int:
        return sum(self.bundle_n.values())

    def finalize_store(self) -> None:
        """Set bundle_counts and total_farmers from accumulated performance."""
        self.bundle_counts = {
            bundle: float(count) for bundle, count in self.bundle_n.items()
        }
        self.total_farmers = self.farmer_n

    def merge_store(self, other: RegionManagementPerformanceStore) -> None:
        """Roll up another region's store (cluster aggregation)."""
        for bundle, perf in other.items():
            self.add_performance(bundle, perf)

    def merge_with_neighbours(
        self,
        own_weight: float,
        neighbour_stores: list[RegionManagementPerformanceStore],
        neighbour_weight: float,
    ) -> RegionManagementPerformanceStore:
        """Blend country stats with neighbours' prior-year published stats.

        Performance (levels, trends) and adoption (bundle_counts) are merged
        separately per bundle, both as weighted means of the two sides rather
        than as weighted sums. Sums would make the weights depend on how many
        neighbours a country happens to have, since every neighbour store
        carries a full farmer count of its own: Germany with nine neighbours
        would count its own farms a ninth of what Portugal with one counts
        theirs, at the same configured weight.

        Neighbours enter as one pooled block, so a large neighbour speaks for
        more of it than a small one, and the block as a whole gets the
        configured weight. Where one side has no farmers at all the other
        carries the full weight.

        The farmer counts the merged store reports are the own farms plus one
        average neighbour's worth, for the same reason: they set how far social
        learning is shrunk towards neutral and how heavily the country weighs
        in its cluster's roll-up, and a country should not claim more evidence
        for happening to border more countries.
        """
        if not neighbour_stores or neighbour_weight <= 0:
            self.finalize_store()
            return self

        # Farmer counts per bundle in this country (integers, from aggregation).
        own_adoption_counts_by_bundle = self.bundle_n
        own_farmer_n = self.farmer_n

        # Union of all practice bundles seen in own country or any neighbour.
        all_bundles = set(self._bundles)
        for neighbour_store in neighbour_stores:
            all_bundles.update(neighbour_store._bundles)

        neighbour_farmer_n = sum(
            neighbour_store.total_farmers for neighbour_store in neighbour_stores
        )
        merged_store = RegionManagementPerformanceStore(year=self.year)
        merged_bundle_n: dict[ManagementBundle, float] = {}

        # Weights for the adoption shares, with the empty side's weight handed
        # to the other so a country whose farms are all unaccounted for still
        # reads its neighbours rather than a quarter of them.
        adoption_weights = _side_weights(
            own_weight if own_farmer_n > 0 else 0.0,
            neighbour_weight if neighbour_farmer_n > 0 else 0.0,
        )
        # Counts are shares of the farms behind the blend, so the ratio the
        # social norm reads gives the blended share back exactly.
        merged_farmer_n = int(
            own_farmer_n + neighbour_farmer_n / len(neighbour_stores)
        )

        for bundle in all_bundles:
            own_bundle_performance = self._bundles.get(
                bundle, RegionManagementPerformance.empty()
            )
            # Farmers contributing to own performance averages for this bundle.
            own_performance_n = own_bundle_performance.count
            # Farmers using this bundle (adoption count for social norm).
            own_adoption_count = own_adoption_counts_by_bundle.get(bundle, 0)

            # Pool neighbour performance sums across all neighbour countries.
            neighbour_bundle_performance = RegionManagementPerformance.empty()
            neighbour_performance_n = 0
            neighbour_reporting_n = 0
            # Pool neighbour adoption counts (may be float-weighted from last year).
            neighbour_bundle_n = 0.0
            for neighbour_store in neighbour_stores:
                neighbour_bundle_slice = neighbour_store.get(bundle)
                if neighbour_bundle_slice.count > 0:
                    neighbour_bundle_performance.merge(neighbour_bundle_slice)
                    neighbour_performance_n += neighbour_bundle_slice.count
                    neighbour_reporting_n += 1
                neighbour_bundle_n += neighbour_store.bundle_counts.get(
                    bundle, 0
                )

            # Farms behind the blended means: own plus one average neighbour of
            # those running the bundle, so identical neighbours added to the
            # block do not multiply the evidence.
            blended_n = int(
                own_performance_n
                + (
                    neighbour_performance_n / neighbour_reporting_n
                    if neighbour_reporting_n > 0
                    else 0.0
                )
            )

            # Performance merge: weighted mean of bundle-level means
            if own_performance_n > 0 and neighbour_performance_n > 0:
                merged_store._bundles[bundle] = (
                    RegionManagementPerformance.from_weighted_means(
                        own_bundle_performance,
                        neighbour_bundle_performance,
                        own_weight=own_weight,
                        other_weight=neighbour_weight,
                        n_farmers=blended_n,
                    )
                )
            elif own_performance_n > 0:
                merged_store._bundles[bundle] = own_bundle_performance
            elif neighbour_performance_n > 0:
                # Neighbours only: their pooled means, reported for the size of
                # the block rather than the sum over its members.
                merged_store._bundles[bundle] = (
                    RegionManagementPerformance.from_weighted_means(
                        neighbour_bundle_performance,
                        RegionManagementPerformance.empty(),
                        own_weight=1.0,
                        other_weight=0.0,
                        n_farmers=blended_n,
                    )
                )

            # Adoption merge: weighted mix of the shares running the bundle
            own_share = (
                own_adoption_count / own_farmer_n if own_farmer_n > 0 else 0.0
            )
            neighbour_share = (
                neighbour_bundle_n / neighbour_farmer_n
                if neighbour_farmer_n > 0
                else 0.0
            )
            merged_bundle_n[bundle] = merged_farmer_n * (
                adoption_weights[0] * own_share
                + adoption_weights[1] * neighbour_share
            )

        merged_store.bundle_counts = merged_bundle_n
        merged_store.total_farmers = merged_farmer_n
        return merged_store


# =============================================================================
# TREND CALCULATION (Linear Regression)
# =============================================================================

@dataclass
class ManagementPerformanceTracker:
    """Online linear regression of self-referenced metric rates.

    Accumulates running sums for mean levels and slopes without storing the
    full series. Resets on practice transition; prior episode stored in
    ``ManagementPerformanceMemory``. ``baseline_performance`` holds rates at
    last transition for fallback scoring with current reference scales.

    Attributes
    ----------
    t_start : int
        Year tracking began for current bundle.
    baseline_performance : ManagementPerformance | None
        Rate snapshot at last transition (fallback detection).
    n : int
        Observation count.
    """

    t_start: int
    # Rates at the last transition; scored on demand. Set on first update
    # (deferred init).
    baseline_performance: ManagementPerformance | None = None
    n: int = 0
    sum_t: float = 0.0
    sum_tt: float = 0.0
    # All metrics are stored as self-referenced relative deviations, so levels
    # are dimensionless and centred on zero. ``square_sums`` captures each
    # metric's interannual variability, used to calibrate the scoring scale
    # (see CAWorld.compute_reference_scales).
    sums: dict[str, float] = field(default_factory=zero_metrics)
    time_sums: dict[str, float] = field(default_factory=zero_metrics)
    square_sums: dict[str, float] = field(default_factory=zero_metrics)
    last_obs_year: int = -1

    @classmethod
    def from_farmer(cls, farmer: Any) -> ManagementPerformanceTracker:
        """Initialize from historic LPJmL output or current farmer snapshot."""
        t_start, history = cls._load_history(farmer)
        state = cls(t_start=t_start)
        state._ingest_history(history, farmer.model.lpjml.sim_year)
        return state

    @classmethod
    def reset_for_transition(
        cls,
        farmer: Any,
        current_year: int,
        baseline_performance: ManagementPerformance | None,
    ) -> ManagementPerformanceTracker:
        """Fresh state when switching to a new practice bundle."""
        rates = rates_of(farmer)
        return cls(
            t_start=current_year,
            baseline_performance=baseline_performance,
            n=1,
            # Time index of this single point is 0, so the time sums stay zero.
            sums=dict(rates),
            square_sums={key: value * value for key, value in rates.items()},
            last_obs_year=current_year,
        )

    @staticmethod
    def _load_history(farmer: Any) -> tuple[int, list[dict[str, float]]]:
        """Load historic time series of relative deviations for regression init.

        Every metric is expressed as a self-referenced deviation from the
        farmer's own historic baseline, so all five share one dimensionless
        scale:

        - yield, soilc, moisture, leaching: ``log_rate`` against the
          historic-mean baselines set by ``CAFarmer._compute_baseline_metrics``.
        - profit: ``asinh((revenue - baseline_revenue) / baseline_revenue)``.
          During the historic period delta_costs = 0 (same bundle as baseline),
          so this is a clean deviation from baseline revenue.

        These must be the exact transforms the farmer properties apply at
        runtime. The world calibrates the scoring scales from the interannual
        spread of this history and the same tracker then accumulates runtime
        observations, so a mismatch would measure a log-scale signal with a
        linear-scale ruler, with a bias that varies by metric and by site.
        """
        from_earth = farmer.cell.from_earth
        has_history = hasattr(from_earth, "time") and len(from_earth.time) > 1

        if has_history:
            try:
                t_start = int(farmer.model.config.outputyear)
            except AttributeError:
                t_start = farmer.model.lpjml.sim_year
                has_history = False

        if has_history:
            history = []
            n_years = len(from_earth.time)
            baseline_revenue = farmer.baseline_revenue
            moisture_name = moisture_var(from_earth)

            for i in range(n_years):
                pft_harvestc = farmer.get_from_earth(
                    "pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=i
                )
                cftfrac = farmer.get_from_earth(
                    "cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=i
                )
                cropped = float(cftfrac.sum("band").item())
                avg_yield = (
                    pft_harvestc.weighted(cftfrac).sum("band").item() / cropped
                    if cropped > 0 else 0.0
                )

                soilc = farmer.get_from_earth(
                    "soilc_agr_layer", as_scalar=True, band=0, time_idx=i
                )
                moisture = farmer.get_from_earth(
                    moisture_name, as_scalar=True, time_idx=i
                )
                leached = agricultural_leaching(farmer, time_idx=i)

                # Same quantity as ConservationAgricultureFarmer.profit,
                # written out: revenue/baseline - 1 is the revenue change over
                # baseline, and costs do not move over the historic period
                # because nobody has changed practice yet. Keep the two in step.
                #
                # No policy transfer either, whatever the scenario sets: these
                # are the years the farm learned from, and they were farmed
                # without an instrument. Back-dating one would put payments in
                # a memory that never received them, and peers would read a
                # practice as having paid before it was ever paid for.
                revenue_this_year = farmer._compute_revenue_for_year(i)
                profit = (
                    float(np.arcsinh(revenue_this_year / baseline_revenue - 1.0))
                    if baseline_revenue > 0 else 0.0
                )

                history.append({
                    "soilc": log_rate(
                        soilc, farmer.baseline_soilc, farmer.negligible_soilc
                    ),
                    "moisture": log_rate(
                        moisture, farmer.baseline_moisture, farmer.negligible_moisture
                    ),
                    "yield": log_rate(
                        avg_yield, farmer.baseline_yield, farmer.negligible_yield
                    ),
                    "leaching": log_rate(
                        leached, farmer.baseline_leaching, farmer.negligible_leaching
                    ),
                    "profit": profit,
                })
            return t_start, history

        # Fallback: single observation at baseline (all rates = 0)
        rates = rates_of(farmer)
        rates["profit"] = 0.0  # At baseline = no deviation
        return farmer.model.lpjml.sim_year, [rates]

    def _ingest_history(
        self,
        history: list[dict[str, float]],
        sim_year: int,
    ) -> None:
        """Populate regression accumulators from historic observations."""
        for i, obs in enumerate(history):
            self._add_point(i, obs)
        self.last_obs_year = sim_year

    def _add_point(self, t: int, rates: dict[str, float]) -> None:
        """Add one observation at time index t."""
        self.n += 1
        self.sum_t += t
        self.sum_tt += t * t
        for key in METRIC_KEYS:
            value = rates[key]
            self.sums[key] += value
            self.time_sums[key] += t * value
            self.square_sums[key] += value * value

    def add_observation(
        self,
        rates: dict[str, float],
        current_year: int,
    ) -> None:
        """Add current year's observation (skips if already recorded)."""
        if self.last_obs_year >= current_year:
            return
        self._add_point(self.n, rates)
        self.last_obs_year = current_year

    def _slope(self, sum_y: float, sum_ty: float) -> float:
        """Regression slope for one variable."""
        if self.n < 2:
            return 0.0

        denominator = self.n * self.sum_tt - self.sum_t * self.sum_t
        if abs(denominator) < 1e-10:
            return 0.0

        return (self.n * sum_ty - self.sum_t * sum_y) / denominator

    @property
    def trend(self) -> dict[str, float]:
        """Annual change in each metric's relative deviation from baseline.

        The tracker stores log (or asinh) rates, so these slopes are relative
        rates per year - e.g. soil carbon rising at +0.02/yr in log space - not
        absolute physical units. Comparability across metrics is handled later by
        CAWorld.compute_reference_scales, which divides by the cross-farmer
        spread of the projected trends.

        Returns
        -------
        dict
            Slope per metric, in relative-deviation units per year.
        """
        if self.n < 2:
            return zero_metrics()

        return {
            key: self._slope(self.sums[key], self.time_sums[key])
            for key in METRIC_KEYS
        }

    @property
    def level(self) -> dict[str, float]:
        """Absolute performance level (mean of all tracked metrics)."""
        return {key: self._mean(key) for key in METRIC_KEYS}

    def _mean(self, key: str) -> float:
        return self.sums[key] / self.n if self.n else 0.0

    def _std(self, key: str) -> float:
        """Interannual std of one metric's rate over the tracking period."""
        if self.n < 2:
            return 0.0
        mean = self.sums[key] / self.n
        variance = self.square_sums[key] / self.n - mean * mean
        return float(np.sqrt(variance)) if variance > 0 else 0.0

    @property
    def level_std(self) -> dict[str, float]:
        """Interannual std of each metric's relative deviation.

        Measures how much each metric fluctuates year-to-year for THIS farmer
        (weather/price-driven). Serves as the LOWER BOUND on the level ruler in
        CAWorld.compute_reference_scales: the cross-farmer spread of
        self-referenced rates is degenerate at init (every farmer sits on its own
        baseline, so all levels are ~0), so the within-farmer temporal spread
        carries the early years until the population has spread out. It cannot be
        the ruler on its own, because a stock's deviation from baseline outgrows
        its year-to-year wobble without bound.
        Returns 0.0 per metric if fewer than 2 observations.
        """
        return {key: self._std(key) for key in METRIC_KEYS}

    def __repr__(self) -> str:
        trend = self.trend
        level = self.level
        detail = ", ".join(
            f"{key}={level[key]:+.1%}({trend[key]:+.2%}/yr)" for key in METRIC_KEYS
        )
        return f"ManagementPerformanceTracker(n={self.n}, {detail})"


# The unified scoring API, plus the accumulator names the tracker was read by
# before its sums became dicts.
add_metric_properties(
    ManagementPerformanceTracker,
    {
        "mean_{key}": lambda self, key: self._mean(key),
        "{key}_trend": lambda self, key: self._slope(
            self.sums[key], self.time_sums[key]
        ),
        "sum_{key}": lambda self, key: self.sums[key],
        "sum_t_{key}": lambda self, key: self.time_sums[key],
        "sum_{key}_sq": lambda self, key: self.square_sums[key],
    },
)
