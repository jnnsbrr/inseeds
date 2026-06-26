"""Conservation Agriculture Management: Bundles, Costs, and Performance Tracking.

This module defines how farmers' agricultural practices are represented, tracked,
and aggregated for social learning in the Conservation Agriculture model.

Overview
--------
Conservation Agriculture (CA) is defined by three practices:
1. **No-till** (tillage=0): Minimal soil disturbance
2. **Cover crops** (cover_crop=1): Planting between main crops
3. **Residue retention** (residue_on_field=1): Leaving crop residues on soil

A "practice bundle" is a combination of these three practices. With each practice
being ON (1) or OFF (0), there are 2³ = 8 possible bundles, from "conventional"
(all OFF) to "full conservation agriculture" (all ON).

Key Concepts
------------
- **ManagementBundle**: Which practices a farmer currently uses (e.g., no-till + cover crop)
- **ManagementPerformance**: A farmer's remembered outcome (soil, yield, moisture trends)
- **RegionManagementPerformance**: Aggregated outcomes for one bundle across many farmers
- **RegionManagementPerformanceStore**: Collection of all bundles' performance in a region

Data Flow
---------
    Individual Farmer
           │
           ▼
    ManagementPerformanceTracker  ──► tracks trends over time
           │
           ▼
    ManagementPerformanceMemory   ──► remembers past bundle outcomes
           │
           ▼
    RegionManagementPerformanceStore  ──► aggregates across farmers
           │
           ├──► Country-level social learning
           │
           └──► Cluster-level social learning (similar climates)

Example
-------
>>> # A farmer practicing no-till with cover crops but no residue retention:
>>> bundle = ManagementBundle.notill_covercrop
>>> print(bundle.label)
'no-till + cover crop'
>>> print(bundle.tillage, bundle.cover_crop, bundle.residue_on_field)
(0, 1, 0)

See Also
--------
- ca_behaviour.py: TPB decision model using these data structures
- ca_country.py: Country-level aggregation
- ca_agroecology.py: Cluster-level aggregation across similar countries
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from inseeds.components.farming.farmer import NON_CROPS, sigmoid


# =============================================================================
# CONSTANTS
# =============================================================================

# When capital is low, farmers deselect practices in this order (most expensive first)
DESELECT_ORDER = ("cover_crop", "residue_on_field", "tillage")

# The three practice dimensions that define a management bundle
PRACTICE_FIELDS = ("tillage", "cover_crop", "residue_on_field")

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

# TODO: Consider implementing discounted future value approach for trends:
#   PV = trend × [1 - (1+r)^(-n)] / r
# where r is discount rate (~10-20%) and n is planning horizon.
# This would account for uncertainty about distant future gains.
# Current simple approach (trend × planning_horizon) is adequate for ABM
# and more interpretable. Discounting would reduce trend impact by ~25-40%.

def compute_metric_score(
    level: float,
    trend: float,
    scale: float,
    planning_horizon: float,
    trend_weight: float,
) -> float:
    """Compute performance score as projected endpoint normalized by fixed scale.

    Uses the "projected endpoint" approach: where will this metric be in
    `planning_horizon` years if current trend continues? This captures both
    current level AND trajectory in a single, decision-relevant value.

    The score is normalized by a FIXED world mean (from historic data) to make
    different metrics (yield ~50, soilC ~2500, moisture ~250) comparable.

    Parameters
    ----------
    level : float
        Current absolute metric value (e.g., yield in gC/m², soilc in gC/m²).
    trend : float
        Annual rate of change (same units as level, per year).
    scale : float
        Fixed scaling factor (world mean from historic data). Must be > 0.
        This is a CONSTANT for the entire simulation.
    planning_horizon : float
        Years over which to project trend impact (typically min_observation_years).
        Represents the farmer's decision-making horizon.
    trend_weight : float
        Weight for trend component (0-1). Level weight = 1 - trend_weight.

    Returns
    -------
    float
        Performance score as ratio to historic world mean.
        - Score = 1.0 means projected endpoint equals historic world mean
        - Score > 1.0 means above average (better)
        - Score < 1.0 means below average (worse)

    Formula
    -------
    projected_endpoint = level + trend × planning_horizon
    score = projected_endpoint / scale

    This can also be decomposed as:
    level_contrib = level / scale
    trend_contrib = (trend × planning_horizon) / scale
    score = level_weight × level_contrib + trend_weight × trend_contrib

    Examples
    --------
    - Yield: level=60, trend=2, scale=50, horizon=5
      → projected = 60 + 2×5 = 70 → score = 70/50 = 1.4 (40% above average)

    - SoilC: level=2000, trend=50, scale=2500, horizon=5
      → projected = 2000 + 50×5 = 2250 → score = 2250/2500 = 0.9 (10% below avg)
    """
    # Projected endpoint: where metric will be in planning_horizon years
    projected_trend = trend * planning_horizon

    # Normalize by fixed scale (world mean from historic data)
    level_contrib = level / scale
    trend_contrib = projected_trend / scale

    # Combine with configured weighting
    level_weight = 1.0 - trend_weight
    return level_weight * level_contrib + trend_weight * trend_contrib


def compute_performance_score(
    tracker_or_region: Any,
    farmer: Any,
) -> float:
    """Compute performance score using projected endpoint scaled by world means.

    This function provides a single, consistent scoring mechanism used across
    all comparison points: neighbor comparison, bundle selection, and attitude
    calculation. Uses FIXED world means from historic data as scaling factors.

    Parameters
    ----------
    tracker_or_region : ManagementPerformanceTracker or RegionManagementPerformance
        Performance data source to evaluate. Can be an individual farmer's
        tracker or aggregated regional statistics.
    farmer : Farmer
        Farmer whose weights (weight_yield, weight_soil, etc.) and trend
        weights (trend_weight_yield, etc.) are used for scoring. Also provides
        access to world.statistic for fixed reference scales.

    Returns
    -------
    float
        Dimensionless weighted performance score.
        - Score of ~1.0 means projected endpoint equals historic world mean
        - Score > 1.0 means above historic world mean (better)
        - Score < 1.0 means below historic world mean (worse)

    Notes
    -----
    The score combines three metrics (yield, soil carbon, moisture) with:
    1. Per-metric level/trend weighting (based on metric dynamics)
    2. Per-farmer importance weighting (based on AFT psychology)

    Uses FIXED WORLD MEANS from historic data as scaling factors:
    - Makes different metrics (yield ~50, soilC ~2500) comparable
    - Preserves absolute performance (higher level = higher score)
    - No dynamic normalization that could distort comparisons
    - A high-performing farmer at biophysical limits scores highly

    Formula: score = (level + trend × horizon) / historic_world_mean

    See Also
    --------
    compute_metric_score : Per-metric projected endpoint computation.
    CAWorld.compute_reference_scales : Where the fixed scales are computed.
    """
    # Get metrics from tracker being evaluated
    level_yield = tracker_or_region.mean_yield
    level_soilc = tracker_or_region.mean_soilc
    level_moisture = tracker_or_region.mean_moisture
    trend_yield = tracker_or_region.yield_trend
    trend_soilc = tracker_or_region.soilc_trend
    trend_moisture = tracker_or_region.moisture_trend

    # Get FIXED reference scales from world (computed once from historic data)
    scales = farmer.world.statistic.get("reference_scales")
    scale_yield = scales["yield"]
    scale_soilc = scales["soilc"]
    scale_moisture = scales["moisture"]

    # Planning horizon: farmer's decision-making window (min_observation_years)
    # Trends are projected over this period to make them comparable to levels
    planning_horizon = farmer.behaviour.get_aft_param("min_observation_years")

    # Compute per-metric scores as projected endpoint / scale
    score_yield = compute_metric_score(
        level_yield, trend_yield,
        scale_yield,
        planning_horizon,
        farmer.trend_weight_yield
    )
    score_soilc = compute_metric_score(
        level_soilc, trend_soilc,
        scale_soilc,
        planning_horizon,
        farmer.trend_weight_soil
    )
    score_moisture = compute_metric_score(
        level_moisture, trend_moisture,
        scale_moisture,
        planning_horizon,
        farmer.trend_weight_moisture
    )

    # Aggregate with farmer's importance weights
    return (
        farmer.weight_yield * score_yield
        + farmer.weight_soil * score_soilc
        + farmer.weight_moisture * score_moisture
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
    """Cost structure for a single agricultural practice.

    Each practice (tillage, cover crop, residue retention) has:
    - A one-time transition cost (equipment, training)
    - Multiple annual cost components (fuel, labor, seeds, herbicide, etc.)

    The generic cost_components dict allows adding any cost type in config
    without code changes. The `direct` property sums all components.

    Attributes
    ----------
    transition : float
        One-time switching cost (USD/ha). Paid once when adopting the practice.
        Example: New equipment, training, initial soil preparation.

    cost_components : dict[str, float]
        Named annual cost components (USD/ha/year). Summed for total direct cost.
        Example (conventional tillage): {"fuel": 45.0, "labor": 25.0, "herbicide": -35.0}
        Positive = higher cost vs no-till; negative = cost reduction vs no-till.

    Example
    -------
    >>> tillage_cost = PracticeCost(
    ...     transition=66.0,
    ...     cost_components={"fuel": 45.0, "labor": 25.0, "herbicide": -35.0}
    ... )
    >>> tillage_cost.direct  # Net cost of conventional tillage vs no-till
    35.0
    """

    transition: float
    cost_components: dict[str, float]

    @property
    def direct(self) -> float:
        """Total annual direct cost = sum of all cost components."""
        return sum(self.cost_components.values())

    @classmethod
    def from_config(cls, data: Any, capital_ratio: float = 1.0) -> "PracticeCost":
        """Create from configuration dictionary, optionally scaled by capital.

        Supports two config formats:
        1. New format with 'costs' dict containing named components
        2. Legacy format with single 'direct' value (backward compatible)

        Parameters
        ----------
        data : dict
            Configuration with 'transition' and either 'costs' dict or 'direct'.
            Values can be scalars or [min, max] ranges for capital scaling.
        capital_ratio : float
            Ratio of farmer's capital to reference (default 1.0 = reference level).
            Used to interpolate within [min, max] ranges.

        Returns
        -------
        PracticeCost
            Cost structure with scaled values.
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


@dataclass(frozen=True, slots=True)
class ManagementCosts:
    """Complete cost structure for all three CA practices.

    Groups the costs for tillage, cover crops, and residue retention together.
    Loaded from the model configuration file, scaled by farmer's capital intensity.

    Attributes
    ----------
    tillage : PracticeCost
        Costs for conventional tillage (tillage=1); negated when no-till (tillage=0)
    cover_crop : PracticeCost
        Costs for planting cover crops between main crop seasons
    residue_on_field : PracticeCost
        Costs for retaining crop residues on the field surface
    """

    tillage: PracticeCost
    cover_crop: PracticeCost
    residue_on_field: PracticeCost

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any] | Any,
        capital_per_ha: float | None = None,
        model: Any = None,
    ) -> ManagementCosts:
        """Create from configuration dictionary, scaled by capital intensity.

        Costs are scaled based on farmer's capital relative to a reference country
        (default: USA, where literature cost values originate).

        Parameters
        ----------
        raw : dict
            Configuration dictionary with practice cost specifications.
        capital_per_ha : float, optional
            Farmer's capital per hectare (USD/ha). If provided, costs are scaled.
        model : Model, optional
            Model instance to look up reference country's capital from FAO data.
            Required if reference_capital_country is specified in config.

        Returns
        -------
        ManagementCosts
            Cost structure with appropriately scaled values.
        """
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict):
            raw = dict(raw)

        # Get reference capital for scaling (default fallback ~ US level)
        reference_capital = 3000.0
        ref_country = raw.get("reference_capital_country")

        if ref_country and model is not None:
            # Look up reference country's capital from FAO data
            for country in model.world.countries:
                if country.code == ref_country:
                    reference_capital = country.initial_capital_per_ha
                    break

        # Compute capital ratio (clamped to [0, 1] in _scale_cost_value)
        if capital_per_ha is not None and reference_capital > 0:
            capital_ratio = capital_per_ha / reference_capital
        else:
            capital_ratio = 1.0  # Default to reference level

        return cls(
            tillage=PracticeCost.from_config(raw.get("tillage", {}), capital_ratio),
            cover_crop=PracticeCost.from_config(raw.get("cover_crop", {}), capital_ratio),
            residue_on_field=PracticeCost.from_config(raw.get("residue_on_field", {}), capital_ratio),
        )


# =============================================================================
# PRACTICE BUNDLES
# =============================================================================

class ManagementBundle(Enum):
    """A combination of agricultural practices used by a farmer.

    Conservation Agriculture (CA) involves three practices that can each be
    ON (1) or OFF (0), creating 8 possible "bundles":

    Practice Encoding
    -----------------
    - tillage:          0 = no-till (CA practice), 1 = conventional tillage
    - cover_crop:       0 = no cover crop,         1 = cover crop planted
    - residue_on_field: 0 = residues removed,      1 = residues retained (CA)

    Note: For tillage, 0 means the CA practice (no-till) is ACTIVE.
    This matches LPJmL's internal representation.

    The 8 Bundles
    -------------
    +------------------+----------+------------+---------+
    | Bundle Name      | Tillage  | Cover Crop | Residue |
    +==================+==========+============+=========+
    | notill           |    0     |     0      |    0    |
    | notill_residue   |    0     |     0      |    1    |
    | notill_covercrop |    0     |     1      |    0    |
    | conservation     |    0     |     1      |    1    | ← Full CA
    | conventional     |    1     |     0      |    0    | ← Baseline
    | residue          |    1     |     0      |    1    |
    | covercrop        |    1     |     1      |    0    |
    | covercrop_residue|    1     |     1      |    1    |
    +------------------+----------+------------+---------+

    Example
    -------
    >>> # Get a bundle by name
    >>> bundle = ManagementBundle.conservation
    >>> print(bundle.label)
    'conservation agriculture'

    >>> # Check individual practices
    >>> print(f"No-till: {bundle.tillage == 0}")
    No-till: True

    >>> # Find bundle from practice values
    >>> bundle = ManagementBundle.from_practices(tillage=0, cover_crop=1, residue_on_field=1)
    >>> print(bundle.key)
    'conservation'

    tillage:          0 = no-till (CA), 1 = conventional tillage
    cover_crop:       0 = none,         1 = cover crop
    residue_on_field: 0 = baseline,      1 = CA residue retention
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
    """A farmer's remembered experience with a specific practice bundle.

    When a farmer tries a practice bundle, they observe how their land responds
    over time. This class stores that memory, allowing farmers to recall past
    experiences when deciding whether to try a bundle again.

    Think of it like a farmer's mental note: "Last time I tried cover crops,
    my soil improved but yields dropped slightly over 3 years."

    Attributes
    ----------
    trend_soilc : float
        Remembered soil carbon trend (% change per year, can be negative).
        Positive = soil carbon was increasing.

    trend_moisture : float
        Remembered soil moisture trend (% change per year).
        Positive = moisture was increasing.

    trend_yield : float
        Remembered crop yield trend (% change per year).
        Positive = yields were improving.

    duration : int
        How many years the farmer used this bundle (sample size for trends).

    last_updated : int
        The simulation year when this memory was last updated.
        Used for memory decay (old memories fade).

    failure_count : int
        How many times this bundle led to poor outcomes.
        Used to avoid repeatedly trying bundles that failed.

    Example
    -------
    >>> # A farmer remembers that conservation agriculture improved their soil
    >>> # but had neutral effect on yields over 5 years:
    >>> memory = ManagementPerformance(
    ...     trend_soilc=0.02,      # 2% annual soil carbon increase
    ...     trend_yield=-0.005,    # 0.5% annual yield decrease
    ...     trend_moisture=0.01,   # 1% moisture increase
    ...     duration=5,
    ...     last_updated=2025,
    ... )
    """

    trend_soilc: float = 0.0
    trend_moisture: float = 0.0
    trend_yield: float = 0.0

    duration: int = 0
    last_updated: int = 0

    failure_count: int = 0

    @classmethod
    def empty(cls) -> ManagementPerformance:
        return cls()

    @classmethod
    def from_trend(
        cls,
        trend: dict[str, float],
        duration: int,
        year: int,
        failure_count: int = 0,
    ) -> ManagementPerformance:
        return cls(
            trend_soilc=trend["soilc"],
            trend_moisture=trend["moisture"],
            trend_yield=trend["yield"],
            duration=duration,
            last_updated=year,
            failure_count=failure_count,
        )

    def update_from_trend(
        self,
        trend: dict[str, float],
        duration: int,
        year: int,
    ) -> None:
        """Refresh trends and duration; preserve failure_count."""
        self.trend_soilc = trend["soilc"]
        self.trend_moisture = trend["moisture"]
        self.trend_yield = trend["yield"]
        self.duration = duration
        self.last_updated = year

    def clear(self) -> None:
        """Reset to neutral (no remembered experience)."""
        self.trend_soilc = 0.0
        self.trend_moisture = 0.0
        self.trend_yield = 0.0
        self.duration = 0
        self.last_updated = 0
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1

    def __repr__(self) -> str:
        return (
            f"ManagementPerformance(yield_level={self.yield_level:+.2%}, yield_trend={self.trend_yield:+.2%}/yr, "
            f"soilc_level={self.soilc_level:+.2%}, soilc_trend={self.trend_soilc:+.2%}/yr, n={self.duration}yr, "
            f"moisture_level={self.moisture_level:+.2%}, moisture_trend={self.trend_moisture:+.2%}/yr)"
        )


class ManagementPerformanceMemory:
    """A farmer's complete memory of all practice bundles they've tried.

    Farmers remember their experiences with different practice combinations.
    This class stores one ManagementPerformance for each of the 8 possible
    bundles, allowing farmers to recall: "What happened when I tried no-till?"

    This memory influences future decisions through social learning - farmers
    share their experiences with neighbors, and neighbors can learn from
    bundles they haven't personally tried.

    Memory Decay
    ------------
    Old memories fade over time (configurable via `memory_decay_years`).
    This reflects bounded rationality - farmers don't perfectly remember
    experiences from decades ago.

    Failure Tracking
    ----------------
    If a bundle leads to poor outcomes, the failure count increases.
    Farmers avoid bundles with multiple past failures (exploration limit).

    Example
    -------
    >>> # Access a farmer's memory of the conservation bundle:
    >>> memory = farmer.behaviour.performance_memory
    >>> ca_experience = memory[ManagementBundle.conservation]
    >>> print(f"Soil trend: {ca_experience.trend_soilc}")
    """

    def __init__(
        self,
        current_bundle: ManagementBundle,
        state: ManagementPerformanceTracker,
        year: int,
    ) -> None:
        for bundle in ManagementBundle:
            setattr(self, bundle.key, ManagementPerformance.empty())

        if state.n > 1:
            self._set(
                current_bundle,
                ManagementPerformance.from_trend(state.trend, state.n, year),
            )

    def _get(self, bundle: ManagementBundle) -> ManagementPerformance:
        return getattr(self, bundle.key)

    def _set(
        self, bundle: ManagementBundle, performance: ManagementPerformance
    ) -> None:
        setattr(self, bundle.key, performance)

    def __getitem__(self, bundle: ManagementBundle) -> ManagementPerformance:
        return self._get(bundle)

    def update_current(
        self,
        bundle: ManagementBundle,
        trend: dict[str, float],
        n: int,
        year: int,
    ) -> None:
        """Refresh memory for the active bundle with latest trends."""
        if n < 2:
            return
        self._get(bundle).update_from_trend(trend, n, year)

    def record_performance(
        self,
        bundle: ManagementBundle,
        trend: dict[str, float],
        n: int,
        year: int,
    ) -> None:
        """Store outcome when leaving a bundle; preserve failure count."""
        if n < 2:
            return
        failure_count = self._get(bundle).failure_count
        self._set(
            bundle,
            ManagementPerformance.from_trend(
                trend, n, year, failure_count=failure_count
            ),
        )

    def record_failure(self, bundle: ManagementBundle) -> None:
        self._get(bundle).record_failure()

    def decay(self, current_year: int, memory_decay_years: int) -> None:
        """Clear memories older than memory_decay_years."""
        for bundle in ManagementBundle:
            mem = getattr(self, bundle.key)
            if mem.duration > 0 and current_year - mem.last_updated > memory_decay_years:
                mem.clear()


# =============================================================================
# REGIONAL AGGREGATION (for social learning)
# =============================================================================

@dataclass
class RegionManagementPerformance:
    """Average performance of ONE practice bundle across many farmers in a region.

    When farmers learn from their country or agroecological cluster, they need
    to know: "How well does no-till work on average in my country?"

    This class aggregates performance data from all farmers using a specific
    bundle in a region (country or cluster).

    Why Store Sums?
    ---------------
    We store sums (not means) internally because:
    1. Sums can be combined when merging regions (country + neighbors)
    2. Means are computed on-demand via properties (mean_yield, etc.)
    3. This avoids precision loss when re-aggregating

    Attributes
    ----------
    yield_sum : float
        Sum of crop yields across all farmers (tonnes/ha × count).
    soilc_sum : float
        Sum of soil carbon across all farmers (kg C/m² × count).
    moisture_sum : float
        Sum of root zone moisture across all farmers (fraction × count).
    yield_trend_sum : float
        Sum of yield trends (% change/year × count).
    soilc_trend_sum : float
        Sum of soil carbon trends (% change/year × count).
    moisture_trend_sum : float
        Sum of moisture trends (% change/year × count).
    count : int
        Number of farmers contributing to these sums.

    Example
    -------
    >>> # Country-level mean for no-till bundle:
    >>> perf = country_store.get(ManagementBundle.notill)
    >>> print(f"Mean yield: {perf.mean_yield:.2f} t/ha")
    >>> print(f"Yield trend: {perf.mean_yield_trend:+.1%}/year")
    >>> print(f"Based on {perf.count} farmers")
    """

    yield_sum: float = 0.0
    soilc_sum: float = 0.0
    moisture_sum: float = 0.0
    yield_trend_sum: float = 0.0
    soilc_trend_sum: float = 0.0
    moisture_trend_sum: float = 0.0
    count: int = 0

    @classmethod
    def empty(cls) -> RegionManagementPerformance:
        return cls()

    @property
    def n_farmers(self) -> int:
        return self.count

    @property
    def mean_yield(self) -> float:
        """Mean yield across farmers in this region."""
        return self.yield_sum / self.count if self.count else 0.0

    @property
    def mean_soilc(self) -> float:
        """Mean soil carbon across farmers in this region."""
        return self.soilc_sum / self.count if self.count else 0.0

    @property
    def mean_moisture(self) -> float:
        """Mean root moisture across farmers in this region."""
        return self.moisture_sum / self.count if self.count else 0.0

    @property
    def mean_yield_trend(self) -> float:
        """Mean yield trend across farmers in this region."""
        return self.yield_trend_sum / self.count if self.count else 0.0

    @property
    def mean_soilc_trend(self) -> float:
        """Mean soil carbon trend across farmers in this region."""
        return self.soilc_trend_sum / self.count if self.count else 0.0

    @property
    def mean_moisture_trend(self) -> float:
        """Mean moisture trend across farmers in this region."""
        return self.moisture_trend_sum / self.count if self.count else 0.0

    # Trend aliases for unified scoring API
    @property
    def yield_trend(self) -> float:
        """Alias for mean_yield_trend (unified scoring API)."""
        return self.mean_yield_trend

    @property
    def soilc_trend(self) -> float:
        """Alias for mean_soilc_trend (unified scoring API)."""
        return self.mean_soilc_trend

    @property
    def moisture_trend(self) -> float:
        """Alias for mean_moisture_trend (unified scoring API)."""
        return self.mean_moisture_trend

    def add_observation(
        self,
        cropyield: float,
        soilc: float,
        moisture: float,
        trend: dict[str, float],
    ) -> None:
        self.yield_sum += cropyield
        self.soilc_sum += soilc
        self.moisture_sum += moisture
        self.yield_trend_sum += trend["yield"]
        self.soilc_trend_sum += trend["soilc"]
        self.moisture_trend_sum += trend["moisture"]
        self.count += 1

    def merge(self, other: RegionManagementPerformance) -> None:
        """Add another aggregate's sums (cluster roll-up)."""
        self.yield_sum += other.yield_sum
        self.soilc_sum += other.soilc_sum
        self.moisture_sum += other.moisture_sum
        self.yield_trend_sum += other.yield_trend_sum
        self.soilc_trend_sum += other.soilc_trend_sum
        self.moisture_trend_sum += other.moisture_trend_sum
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
        """Merge two aggregates using weights applied to their means."""
        if n_farmers <= 0:
            return cls.empty()

        m_yield = own_weight * own.mean_yield + other_weight * other.mean_yield
        m_soilc = own_weight * own.mean_soilc + other_weight * other.mean_soilc
        m_moisture = own_weight * own.mean_moisture + other_weight * other.mean_moisture
        m_yield_trend = (
            own_weight * own.mean_yield_trend + other_weight * other.mean_yield_trend
        )
        m_soilc_trend = (
            own_weight * own.mean_soilc_trend + other_weight * other.mean_soilc_trend
        )
        m_moisture_trend = (
            own_weight * own.mean_moisture_trend
            + other_weight * other.mean_moisture_trend
        )

        return cls(
            yield_sum=m_yield * n_farmers,
            soilc_sum=m_soilc * n_farmers,
            moisture_sum=m_moisture * n_farmers,
            yield_trend_sum=m_yield_trend * n_farmers,
            soilc_trend_sum=m_soilc_trend * n_farmers,
            moisture_trend_sum=m_moisture_trend * n_farmers,
            count=n_farmers,
        )

    def weighted_trend(self, farmer: Any) -> float:
        """Weighted sum of mean trends."""
        return (
            farmer.weight_yield * self.mean_yield_trend
            + farmer.weight_soil * self.mean_soilc_trend
            + farmer.weight_moisture * self.mean_moisture_trend
        )

    def weighted_level(self, farmer: Any) -> float:
        """Weighted sum of mean absolute values (level score)."""
        return (
            farmer.weight_yield * self.mean_yield
            + farmer.weight_soil * self.mean_soilc
            + farmer.weight_moisture * self.mean_moisture
        )

    def __repr__(self) -> str:
        return (
            f"RegionManagementPerformance(n={self.count}, "
            f"yield={self.mean_yield:.2f}, soilc={self.mean_soilc:.2f}, "
            f"yield_trend={self.mean_yield_trend:+.2%}/yr)"
        )


class RegionManagementPerformanceStore:
    """Collection of performance data for ALL bundles in a region.

    This is the main data structure for country-level and cluster-level
    social learning. It answers questions like:
    - "What is the average yield for farmers using no-till in my country?"
    - "What fraction of farmers in my cluster use conservation agriculture?"

    Data Flow
    ---------
    1. Each farmer's data is added via add_observation()
    2. finalize_store() computes bundle_counts and total_farmers
    3. For country-level: merge_with_neighbours() blends with neighbor countries
    4. For cluster-level: merge_store() combines multiple countries

    Attributes
    ----------
    year : int
        Simulation year this data represents (-1 if not set).

    total_farmers : int
        Total number of farmers in this region.

    bundle_counts : dict[ManagementBundle, float]
        How many farmers use each bundle. After neighbor merging, these
        may be weighted floats (e.g., 150.5) rather than integers.

    country_codes : list[str]
        ISO country codes included (for cluster-level stores).

    Example
    -------
    >>> # Get country-level performance store:
    >>> store = country.statistic.get("management_performance")
    >>>
    >>> # How does no-till perform on average?
    >>> notill_perf = store.get(ManagementBundle.notill)
    >>> print(f"Mean yield: {notill_perf.mean_yield:.2f} t/ha")
    >>>
    >>> # What fraction of farmers use no-till?
    >>> adoption_rate = store.bundle_counts.get(ManagementBundle.notill, 0) / store.total_farmers
    >>> print(f"Adoption rate: {adoption_rate:.1%}")
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
        self._get(bundle).add_observation(
            farmer.cropyield,
            farmer.soilc,
            farmer.root_moisture,
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
        """Merge this country's stats with neighbouring countries' published stats.

        Two quantities are merged separately per bundle:

        1. **Performance** (yield, soilc, moisture, trends) — weighted average of
           bundle-level averages, using farmer counts as sample sizes.
        2. **Adoption** (bundle_counts) — weighted merge of adoption shares used
           for social-norm calculations; may already be fractional from prior merges.

        Neighbour stats come from the previous year (world broadcast).
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
            # Pool neighbour adoption counts (may be float-weighted from last year).
            neighbour_bundle_n = 0.0
            for neighbour_store in neighbour_stores:
                neighbour_bundle_slice = neighbour_store.get(bundle)
                if neighbour_bundle_slice.count > 0:
                    neighbour_bundle_performance.merge(neighbour_bundle_slice)
                    neighbour_performance_n += neighbour_bundle_slice.count
                neighbour_bundle_n += neighbour_store.bundle_counts.get(
                    bundle, 0
                )

            # Performance merge: weighted mean of bundle-level means
            if own_performance_n > 0 and neighbour_performance_n > 0:
                merged_store._bundles[bundle] = (
                    RegionManagementPerformance.from_weighted_means(
                        own_bundle_performance,
                        neighbour_bundle_performance,
                        own_weight=own_weight,
                        other_weight=neighbour_weight,
                        n_farmers=(
                            own_performance_n
                            + neighbour_performance_n
                        ),
                    )
                )
            elif own_performance_n > 0:
                merged_store._bundles[bundle] = own_bundle_performance
            elif neighbour_performance_n > 0:
                merged_store._bundles[bundle] = neighbour_bundle_performance

            # Adoption merge: weighted mix of bundle adoption counts
            merged_bundle_n[bundle] = (
                own_weight * own_adoption_count
                + neighbour_weight * neighbour_bundle_n
            )

        merged_store.bundle_counts = merged_bundle_n
        merged_store.total_farmers = int(
            own_weight * own_farmer_n
            + neighbour_weight * neighbour_farmer_n
        )
        return merged_store


# =============================================================================
# TREND CALCULATION (Linear Regression)
# =============================================================================

@dataclass
class ManagementPerformanceTracker:
    """Tracks how a farmer's land responds to their current practices over time.

    This class computes TRENDS (is my soil improving or declining?) using
    linear regression, without storing the full time series.

    Why Trends Matter
    -----------------
    Farmers care not just about current yield, but whether things are getting
    better or worse. A trend of +2%/year in soil carbon means the practice
    is building long-term soil health.

    Mathematical Background
    -----------------------
    Uses "online" linear regression (streaming algorithm):

    For y = a + b×t:
        slope b = (n×Σ(t×y) - Σt×Σy) / (n×Σt² - (Σt)²)

    This only requires storing running sums, not the full history.
    The slope is then normalized by the mean to get % change per year.

    Reset on Transition
    -------------------
    When a farmer switches practices, the tracker resets to measure
    the NEW practice's effect. The old practice's performance is
    stored in ManagementPerformanceMemory.

    Attributes
    ----------
    t_start : int
        The year when tracking began (when current practice was adopted).

    baseline_trend : float
        Weighted performance trend at time of last transition. Used for fallback
        detection (if current trend drops below baseline for several years,
        farmer may revert to previous practice).

    n : int
        Number of observations (years of data).

    sum_soilc, sum_moisture, sum_yield : float
        Running sums of each metric (for computing means).

    sum_t_soilc, sum_t_moisture, sum_t_yield : float
        Running sums of t×metric (for computing slopes).

    Example
    -------
    >>> # After 5 years, check how no-till is affecting soil:
    >>> tracker = farmer.behaviour.performance_tracker
    >>> trends = tracker.trend
    >>> print(f"Soil carbon trend: {trends['soilc']:+.1%}/year")
    >>> print(f"Yield trend: {trends['yield']:+.1%}/year")
    """

    t_start: int
    baseline_trend: float = 0.0
    n: int = 0
    sum_t: float = 0.0
    sum_tt: float = 0.0
    sum_soilc: float = 0.0
    sum_t_soilc: float = 0.0
    sum_moisture: float = 0.0
    sum_t_moisture: float = 0.0
    sum_yield: float = 0.0
    sum_t_yield: float = 0.0
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
        baseline_trend: float,
    ) -> ManagementPerformanceTracker:
        """Fresh state when switching to a new practice bundle."""
        return cls(
            t_start=current_year,
            baseline_trend=baseline_trend,
            n=1,
            sum_soilc=farmer.soilc,
            sum_moisture=farmer.root_moisture,
            sum_yield=farmer.cropyield,
            last_obs_year=current_year,
        )

    @staticmethod
    def _load_history(farmer: Any) -> tuple[int, list[dict[str, float]]]:
        """Load historic time series for regression initialization."""
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
            for i in range(len(from_earth.time)):
                pft_harvestc = farmer.get_from_earth(
                    "pft_harvestc", as_scalar=False, drop_band=NON_CROPS, time_idx=i
                )
                cftfrac = farmer.get_from_earth(
                    "cftfrac", as_scalar=False, drop_band=NON_CROPS, time_idx=i
                )
                avg_yield = pft_harvestc.weighted(cftfrac).sum("band").item()
                history.append({
                    "soilc": farmer.get_from_earth(
                        "soilc_agr_layer", as_scalar=True, band=0, time_idx=i
                    ),
                    "moisture": farmer.get_from_earth(
                        "rootmoist_agr", as_scalar=True, time_idx=i
                    ),
                    "yield": avg_yield,
                })
            return t_start, history

        return farmer.model.lpjml.sim_year, [{
            "soilc": farmer.soilc,
            "moisture": farmer.root_moisture,
            "yield": farmer.cropyield,
        }]

    def _ingest_history(
        self,
        history: list[dict[str, float]],
        sim_year: int,
    ) -> None:
        """Populate regression accumulators from historic observations."""
        for i, obs in enumerate(history):
            self._add_point(i, obs["soilc"], obs["moisture"], obs["yield"])
        self.last_obs_year = sim_year

    def _add_point(
        self,
        t: int,
        soilc: float,
        moisture: float,
        cropyield: float,
    ) -> None:
        """Add one observation at time index t."""
        self.n += 1
        self.sum_t += t
        self.sum_tt += t * t
        self.sum_soilc += soilc
        self.sum_t_soilc += t * soilc
        self.sum_moisture += moisture
        self.sum_t_moisture += t * moisture
        self.sum_yield += cropyield
        self.sum_t_yield += t * cropyield

    def add_observation(
        self,
        soilc: float,
        moisture: float,
        cropyield: float,
        current_year: int,
    ) -> None:
        """Add current year's observation (skips if already recorded)."""
        if self.last_obs_year >= current_year:
            return
        self._add_point(self.n, soilc, moisture, cropyield)
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
        """Absolute annual change in soil, moisture, and yield.

        Returns raw regression slopes (absolute change per year), NOT normalized
        by mean. This is deliberate:

        1. Farmers observe and communicate absolute changes ("yield increased by
           500 kg/ha"), not percentages (Bandura 1977 social learning theory).

        2. The AFT-specific weights (weight_yield, weight_soil, weight_moisture)
           already capture subjective importance and implicitly handle scale
           differences between metrics.

        3. Percentage normalization produces values too small for meaningful
           differentiation in the sigmoid-based decision model.

        Returns
        -------
        dict
            {"soilc": slope, "moisture": slope, "yield": slope} in native units/year.
        """
        if self.n < 2:
            return {"soilc": 0.0, "moisture": 0.0, "yield": 0.0}

        slope_soilc = self._slope(self.sum_soilc, self.sum_t_soilc)
        slope_moisture = self._slope(self.sum_moisture, self.sum_t_moisture)
        slope_yield = self._slope(self.sum_yield, self.sum_t_yield)

        return {
            "soilc": slope_soilc,
            "moisture": slope_moisture,
            "yield": slope_yield,
        }

    @property
    def level(self) -> dict[str, float]:
        """Absolute performance level (mean of soil, moisture, and yield)."""
        return {
            "soilc": self.mean_soilc,
            "moisture": self.mean_moisture,
            "yield": self.mean_yield,
        }

    @property
    def mean_yield(self) -> float:
        """Average yield over tracking period."""
        return self.sum_yield / self.n if self.n else 0.0

    @property
    def mean_soilc(self) -> float:
        """Average soil carbon over tracking period."""
        return self.sum_soilc / self.n if self.n else 0.0

    @property
    def mean_moisture(self) -> float:
        """Average root moisture over tracking period."""
        return self.sum_moisture / self.n if self.n else 0.0

    @property
    def yield_trend(self) -> float:
        """Annual yield trend (absolute change per year)."""
        return self._slope(self.sum_yield, self.sum_t_yield)

    @property
    def soilc_trend(self) -> float:
        """Annual soil carbon trend (absolute change per year)."""
        return self._slope(self.sum_soilc, self.sum_t_soilc)

    @property
    def moisture_trend(self) -> float:
        """Annual moisture trend (absolute change per year)."""
        return self._slope(self.sum_moisture, self.sum_t_moisture)

    def weighted_trend(self, farmer: Any) -> float:
        """Combine soil, moisture, and yield TRENDS into a single trend score.

        This measures the RATE OF CHANGE - is performance improving or declining?
        Uses the farmer's outcome weights to represent overall trend direction.

        Note: This only considers trends, not absolute levels. A farmer with
        terrible yields but improving might score higher than one with good
        yields but declining.

        Parameters
        ----------
        farmer : Farmer
            Farmer providing the outcome weights.

        Returns
        -------
        float
            Weighted sum of trends (positive = improving, negative = declining).
        """
        return (
            farmer.weight_soil * self.soilc_trend
            + farmer.weight_moisture * self.moisture_trend
            + farmer.weight_yield * self.yield_trend
        )

    def weighted_level(self, farmer: Any) -> float:
        """Combine soil, moisture, and yield MEANS into a single level score.

        This measures ABSOLUTE PERFORMANCE - how good is the current state?
        Uses the farmer's outcome weights.

        Parameters
        ----------
        farmer : Farmer
            Farmer providing the outcome weights.

        Returns
        -------
        float
            Weighted sum of mean values.
        """
        return (
            farmer.weight_soil * self.mean_soilc
            + farmer.weight_moisture * self.mean_moisture
            + farmer.weight_yield * self.mean_yield
        )

    def __repr__(self) -> str:
        trend = self.trend
        level = self.level
        return (
            f"ManagementPerformanceTracker(n={self.n}, "
            f"yield_trend={trend['yield']:+.2f}/yr, yield_level={level['yield']:+.2f}, soilc_trend={trend['soilc']:+.2f}/yr, soilc_level={level['soilc']:+.2f}, moisture_trend={trend['moisture']:+.2f}/yr, moisture_level={level['moisture']:+.2f})"
        )