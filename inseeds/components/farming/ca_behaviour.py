"""Theory of Planned Behaviour (TPB) decision model for Conservation Agriculture.

This module implements how farmers decide whether to adopt or abandon
agricultural practices. It uses the Theory of Planned Behaviour framework
combined with social learning and adaptive management.

Each year, farmers may reconsider their practices based on:
1. **Attitude**: "Do I think this practice is good?" (own experience + social learning)
2. **Social Norm**: "Are my neighbors doing it?" (local + country + cluster)
3. **Perceived Behavioral Control (PBC)**: "Can I afford it?"

These combine into a TPB intention score. If it exceeds a threshold, the
farmer transitions to the new practice.

Decision Flow
-------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    Annual Update                            │
    ├─────────────────────────────────────────────────────────────┤
    │  observe()  record outcome and refresh memory (always)      │
    │  update()   decide, in this order:                          │
    │  1. Check if evaluation time (not every year)               │
    │  2. Forced deselection if practices no longer fit the budget│
    │  3. Check fallback (sustained decline → revert)             │
    │  4. Find target bundle:                                     │
    │     a. Local neighbor with better performance?              │
    │     b. Country-level best bundle?                           │
    │     c. Cluster-level best bundle?                           │
    │     d. Random exploration?                                  │
    │  5. Adjust for affordability                                │
    │  6. Compute TPB score                                       │
    │  7. If TPB > threshold → transition                         │
    └─────────────────────────────────────────────────────────────┘

Key Classes
-----------
DecisionModel
    Abstract base class for farmer decision models.
TPB
    Full TPB implementation with social learning at three spatial scales.

Transition Blockers and Drivers
-------------------------------
The model tracks WHY transitions happen (or don't). See BLOCKER_* and DRIVER_*
constants for the full list of codes. This enables analysis of adoption barriers.

Spatial Scales of Social Learning
---------------------------------
- **Local**: Direct neighbor comparison (highest weight)
- **Country**: Average performance across the country
- **Cluster**: Average across agroecologically similar countries

Scales can be switched off individually; their weight is then redistributed
to those left.

References
----------
.. [1] Ajzen, I. (1991). The theory of planned behavior. Organizational Behavior
       and Human Decision Processes, 50(2), 179-211.
.. [2] Bandura, A. (1977). Social Learning Theory. Prentice Hall.
.. [3] Holling, C.S. (1978). Adaptive Environmental Assessment and Management.

See Also
--------
ca_management : Practice bundles and performance tracking
ca_country : Country-level aggregation
ca_agroecology : Cluster-level aggregation
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from inseeds.components.farming.farmer import (
    annuity_factor,
    sigmoid,
    GC_M2_TO_TCO2_HA,
    NON_CROPS,
)
from inseeds.components.farming.ca_management import (
    COVER_CROP_LEGUME,
    COVER_CROP_NONLEGUME,
    DESELECT_ORDER,
    ManagementBundle,
    ManagementPerformance,
    ManagementPerformanceMemory,
    ManagementPerformanceTracker,
    PRACTICE_FIELDS,
    PerceptionNoise,
    ReportingNoise,
    RegionManagementPerformanceStore,
    compute_performance_score,
)
from inseeds.components.farming.metrics import METRICS, rates_of
from inseeds.components.farming.ca_world import LAND_COVARIATES


def _weighted_median(values, weights):
    """Median of values with each observation counted weights[i] times.

    Falls back to the unweighted median when no usable weight survives, so a
    degenerate weight vector cannot silently empty the estimate.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    usable = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not usable.any():
        finite = v[np.isfinite(v)]
        return float(np.median(finite)) if finite.size else 0.0

    v, w = v[usable], w[usable]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w)
    return float(v[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _country_performance_store(country) -> RegionManagementPerformanceStore | None:
    """Return country management performance store if available, None otherwise."""
    store = country.statistic.get("management_performance")
    if isinstance(store, RegionManagementPerformanceStore):
        return store
    return None


def redistribute_weights(
    weight_local: float,
    weight_country: float,
    weight_cluster: float,
    enable_local: bool = True,
    enable_country: bool = True,
    enable_cluster: bool = True,
) -> tuple[float, float, float]:
    """Redistribute weights among enabled spreading levels.

    When a level is disabled, its weight is redistributed proportionally
    to the enabled levels so the total remains the same. This ensures
    that disabling a level (e.g., cluster for single-country runs) doesn't
    drag down scores with neutral values.

    Parameters
    ----------
    weight_local : float
        Weight for local (neighbour) level.
    weight_country : float
        Weight for country level.
    weight_cluster : float
        Weight for agroecological cluster level.
    enable_local : bool
        Whether local spreading is enabled.
    enable_country : bool
        Whether country spreading is enabled.
    enable_cluster : bool
        Whether cluster spreading is enabled.

    Returns
    -------
    tuple[float, float, float]
        Redistributed weights (local, country, cluster).

    Example
    -------
    >>> redistribute_weights(0.7, 0.2, 0.1, True, True, False)
    (0.7778, 0.2222, 0.0)  # Cluster weight redistributed to local + country
    """
    weights = [
        (weight_local, enable_local),
        (weight_country, enable_country),
        (weight_cluster, enable_cluster),
    ]

    enabled_sum = sum(w for w, enabled in weights if enabled)

    if enabled_sum == 0:
        return (0.0, 0.0, 0.0)

    original_total = weight_local + weight_country + weight_cluster
    scale = original_total / enabled_sum

    return (
        weight_local * scale if enable_local else 0.0,
        weight_country * scale if enable_country else 0.0,
        weight_cluster * scale if enable_cluster else 0.0,
    )


# =============================================================================
# PRACTICE BUNDLE DEFINITIONS
# =============================================================================

# Practice bundle: (tillage, cover_crop, residue_on_field), each 0 or 1
#
# tillage:         0 = no-till (CA practice), 1 = conventional tillage
# cover_crop:      0 = no cover crop,         1 = cover crop planted
# residue:         0 = baseline removal,      1 = residue retained
#
# Full CA = (0, 1, 1): no-till + cover crops + residue retention

# Transition blocker codes - indicates WHY a proposed bundle was not adopted
# These follow the decision flow order (first blocker hit is the primary reason)
BLOCKER_NONE = 0                    # No blocker - transition happened or maintaining current
BLOCKER_OBSERVATION_YEARS = 1       # Observation period not complete (commitment period)
BLOCKER_FALLBACK_TRIGGERED = 2      # Reverting to previous bundle (adaptive management)
BLOCKER_NO_TARGET = 3               # No better neighbour + exploration didn't trigger
BLOCKER_TARGET_SAME = 4             # Target bundle same as current (already optimal)
BLOCKER_TARGET_UNAFFORDABLE = 5     # Target reduced to current due to cost
BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND = 6         # TPB low - own land attitude is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL = 7     # TPB low - LOCAL social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL = 8         # TPB low - LOCAL social norm is limiting
BLOCKER_TPB_LOW_PBC = 9             # TPB below threshold - PBC (cost affordability) is limiting
BLOCKER_TRANSITION_UNAFFORDABLE = 10 # Can't afford transition cost
BLOCKER_CAPITAL_SURVIVAL = 11       # Capital below survival threshold
BLOCKER_CONTROL_RUN = 12            # Control run - no CA dynamics
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY = 13  # TPB low - COUNTRY social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY = 14      # TPB low - COUNTRY social norm is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER = 15  # TPB low - CLUSTER social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER = 16      # TPB low - CLUSTER social norm is limiting
BLOCKER_AFFORDABILITY_FORCED = 17   # Practices deselected due to unaffordable direct costs

# Blockers that do not reset the observation counter. min_observation_years is
# the re-evaluation cadence after a real decision. observation_years / no_target /
# target_same mean no candidate was evaluated, so resetting would burn the full
# wait without a decision and throttle diffusion. Counter keeps running.
BLOCKERS_WITHOUT_DECISION = (
    BLOCKER_OBSERVATION_YEARS,
    BLOCKER_NO_TARGET,
    BLOCKER_TARGET_SAME,
)

BLOCKER_NAMES = {
    BLOCKER_NONE: "none",
    BLOCKER_FALLBACK_TRIGGERED: "fallback_triggered",
    BLOCKER_NO_TARGET: "no_target",
    BLOCKER_TARGET_SAME: "target_same",
    BLOCKER_TARGET_UNAFFORDABLE: "target_unaffordable",
    BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND: "tpb_low_attitude_own_land",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL: "tpb_low_attitude_social_local",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY: "tpb_low_attitude_social_country",
    BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER: "tpb_low_attitude_social_cluster",
    BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL: "tpb_low_social_norm_local",
    BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY: "tpb_low_social_norm_country",
    BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER: "tpb_low_social_norm_cluster",
    BLOCKER_TPB_LOW_PBC: "tpb_low_pbc",
    BLOCKER_TRANSITION_UNAFFORDABLE: "transition_unaffordable",
    BLOCKER_CAPITAL_SURVIVAL: "capital_survival",
    BLOCKER_CONTROL_RUN: "control_run",
    BLOCKER_OBSERVATION_YEARS: "observation_years",
    BLOCKER_AFFORDABILITY_FORCED: "affordability_forced",
}

# Transition driver codes - indicates WHY a transition succeeded (mirror of blocker)
# These identify which pathway and TPB component (with local/country distinction) that
# contributed most to enabling the transition. 3 pathways × 6 components = 18 codes.
DRIVER_NONE = 0                              # No transition - maintaining current bundle
DRIVER_FALLBACK = 1                          # Reverted to previous bundle (adaptive management)

# LOCAL pathway - learned from better-performing LOCAL neighbor
DRIVER_LOCAL_ATTITUDE_OWN_LAND = 2          # TPB passed, attitude_own_land was strongest
DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL = 3      # TPB passed, local social learning was strongest
DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY = 4    # TPB passed, country social learning was strongest
DRIVER_LOCAL_SOCIAL_NORM_LOCAL = 5          # TPB passed, local social norm was strongest
DRIVER_LOCAL_SOCIAL_NORM_COUNTRY = 6        # TPB passed, country social norm was strongest
DRIVER_LOCAL_PBC = 7                        # TPB passed, PBC (cost affordability) was strongest

# COUNTRY pathway - inspired by country-level best performer
DRIVER_COUNTRY_ATTITUDE_OWN_LAND = 8         # TPB passed, attitude_own_land was strongest
DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL = 9     # TPB passed, local social learning was strongest
DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY = 10  # TPB passed, country social learning was strongest
DRIVER_COUNTRY_SOCIAL_NORM_LOCAL = 11        # TPB passed, local social norm was strongest
DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY = 12      # TPB passed, country social norm was strongest
DRIVER_COUNTRY_PBC = 13                      # TPB passed, PBC (cost affordability) was strongest

# EXPLORATION pathway - discovered via random exploration
DRIVER_EXPLORATION_ATTITUDE_OWN_LAND = 14    # TPB passed, attitude_own_land was strongest
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL = 15  # TPB passed, local social learning was strongest
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY = 16  # TPB passed, country social learning was strongest
DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER = 20  # TPB passed, cluster social learning was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL = 17    # TPB passed, local social norm was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY = 18  # TPB passed, country social norm was strongest
DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER = 21  # TPB passed, cluster social norm was strongest
DRIVER_EXPLORATION_PBC = 19                  # TPB passed, PBC (cost affordability) was strongest

# Additional cluster-level drivers for LOCAL and COUNTRY pathways
DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER = 22    # TPB passed, cluster social learning was strongest (local pathway)
DRIVER_LOCAL_SOCIAL_NORM_CLUSTER = 23        # TPB passed, cluster social norm was strongest (local pathway)
DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER = 24  # TPB passed, cluster social learning was strongest (country pathway)
DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER = 25      # TPB passed, cluster social norm was strongest (country pathway)

# Forced transitions (bypass TPB)
DRIVER_AFFORDABILITY_FORCED = 26             # Practices deselected due to unaffordable direct costs

# Not a decision: the label followed the field (see reevaluate_residue_status)
DRIVER_RESIDUE_RECLASSIFIED = 27             # Litter cover crossed the CA threshold
DRIVER_COVER_CROP_RETYPED = 28               # Other cover crop now pays better

DRIVER_NAMES = {
    DRIVER_NONE: "none",
    DRIVER_FALLBACK: "fallback",
    # Social pathway (local neighbor inspiration)
    DRIVER_LOCAL_ATTITUDE_OWN_LAND: "local_attitude_own_land",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL: "local_attitude_social_local",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY: "local_attitude_social_country",
    DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER: "local_attitude_social_cluster",
    DRIVER_LOCAL_SOCIAL_NORM_LOCAL: "local_social_norm_local",
    DRIVER_LOCAL_SOCIAL_NORM_COUNTRY: "local_social_norm_country",
    DRIVER_LOCAL_SOCIAL_NORM_CLUSTER: "local_social_norm_cluster",
    DRIVER_LOCAL_PBC: "local_pbc",
    # Country pathway (country-level inspiration)
    DRIVER_COUNTRY_ATTITUDE_OWN_LAND: "country_attitude_own_land",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL: "country_attitude_social_local",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY: "country_attitude_social_country",
    DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER: "country_attitude_social_cluster",
    DRIVER_COUNTRY_SOCIAL_NORM_LOCAL: "country_social_norm_local",
    DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY: "country_social_norm_country",
    DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER: "country_social_norm_cluster",
    DRIVER_COUNTRY_PBC: "country_pbc",
    # Exploration pathway (random exploration)
    DRIVER_EXPLORATION_ATTITUDE_OWN_LAND: "exploration_attitude_own_land",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL: "exploration_attitude_social_local",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY: "exploration_attitude_social_country",
    DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER: "exploration_attitude_social_cluster",
    DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL: "exploration_social_norm_local",
    DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY: "exploration_social_norm_country",
    DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER: "exploration_social_norm_cluster",
    DRIVER_EXPLORATION_PBC: "exploration_pbc",
    # Forced transitions
    DRIVER_AFFORDABILITY_FORCED: "affordability_forced",
    # Not a decision
    DRIVER_RESIDUE_RECLASSIFIED: "residue_reclassified",
    DRIVER_COVER_CROP_RETYPED: "cover_crop_retyped",
}


# =============================================================================
# ABSTRACT DECISION MODEL BASE CLASS
# =============================================================================

class DecisionModel(ABC):
    """Abstract base for farmer decision models.

    Tracks:
    - Current practice bundle (tillage, cover_crop, residue)
    - Memory of past outcomes for each bundle tried
    - Current state snapshot for computing trends since last transition

    Subclasses implement specific decision logic (e.g., TPB).
    """

    def get_aft_param(self, param_name):
        """Get AFT parameter from agent, raising error if missing.

        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve from the agent.

        Returns
        -------
        value
            The parameter value.

        Raises
        ------
        AttributeError
            If the parameter is not defined on the farmer agent (missing from config).
        """
        if not hasattr(self.farmer, param_name):
            aft_name = getattr(self.farmer.aft, "name", "unknown")
            raise AttributeError(
                f"Missing AFT parameter '{param_name}' for AFT '{aft_name}'. "
                f"Add it to config.yaml under aftpar.{aft_name}.{param_name}"
            )
        return getattr(self.farmer, param_name)

    def __init__(self, farmer):
        """Initialize decision model from farmer's current practices.

        Parameters
        ----------
        farmer : ConservationAgricultureFarmer
            The farmer agent owning this behaviour.
        """
        self.farmer = farmer

        # Initialize transition blocker and driver to none
        self.transition_blocker = BLOCKER_NONE
        self.transition_driver = DRIVER_NONE

        # -----------------------------------------------------------------
        # Initialize current practice bundle from farmer agent state
        # -----------------------------------------------------------------
        # Bundle is a tuple: (tillage, cover_crop, residue_on_field)
        # Each element is 0 (practice OFF) or 1 (practice ON)
        #
        # Convention (matches LPJmL):
        #   tillage=0 means no tillage (no-till, CA practice)
        #   tillage=1 means tillage is ON (conventional)
        #   cover_crop=0 means no cover crop, >0 means cover crop planted
        #   residue: 1 if litter cover >= CA threshold (30%), 0 otherwise

        # Get CA cover threshold from config
        ca_threshold = farmer.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501

        self.practice_bundle = ManagementBundle.from_practices(
            int(farmer.tillage),
            1 if farmer.cover_crop > 0 else 0,
            1 if farmer.litter_cover >= ca_threshold else 0,
        )

        # True only after an explicit residue-retention decision (apply_bundle).
        # Under residue_status_by_cover, practice_bundle.residue may flip with
        # litter cover without commitment; costs/subsidies follow commitment.
        # Init: no transition yet → not committed, even if litter ≥ threshold.
        self.residue_committed = False

        # Year the current practice_bundle label began. Caps claimed experience
        # duration when residue_status_by_cover relabels without a TPB decision.
        self.bundle_since = farmer.model.lpjml.sim_year

        # Proposed bundle for potential transition (set by update())
        self.proposed_bundle = None

        # Previous bundle (for fallback mechanism)
        self.previous_bundle = None

        # Counter for consecutive years of declining performance
        self.decline_years = 0

        # Online regression state for trend computation since last transition
        self.performance_tracker = ManagementPerformanceTracker.from_farmer(farmer)

        # -----------------------------------------------------------------
        # Initialize management bundle memory
        # -----------------------------------------------------------------
        # Farmers remembers performance from each bundle they've tried
        # Memory includes: trends, duration used, when last updated, failure count

        current_year = self.farmer.model.lpjml.sim_year
        self.performance_memory = ManagementPerformanceMemory(
            self.practice_bundle,
            self.performance_tracker,
            current_year,
            type_for=self.cover_crop_type_for,
        )

        # baseline_performance will be set on first update (avoids init-order
        # issues with reference_scales not being computed yet)

        # Own score for the current year, filled on first read (see own_score)
        self._own_score = None
        self._own_score_year = None

        # How this farm's account of each comparison departs from its record,
        # drawn on first telling and kept thereafter so the story does not
        # change with the listener or the year. Empty and untouched unless
        # tpb.reporting_noise is on; see report_error. One entry per comparison
        # the farm is actually asked about, which is a handful of bundle pairs.
        self._report_errors = {}

    # -------------------------------------------------------------------------
    # Properties for external access
    # -------------------------------------------------------------------------

    @property
    def practice_bundle_id(self):
        """Current practice bundle as numeric ID (0-7).

        This property is settable for Dask actor synchronization.
        Setting it converts the ID back to a ManagementBundle enum.
        """
        return self.practice_bundle.id if self.practice_bundle else -1

    @practice_bundle_id.setter
    def practice_bundle_id(self, value):
        """Set practice bundle from numeric ID.

        Used by Dask actor sync to restore bundle state on driver.
        """
        if value is not None and value >= 0:
            self.practice_bundle = ManagementBundle.from_id(int(value))

    @property
    def committed_bundle(self):
        """Bundle used for costs, policy payments and LPJmL residue input.

        Differs from ``practice_bundle`` only in the residue flag, and only when
        ``residue_status_by_cover`` is true: the certified label follows litter
        cover (FAO threshold), while costs follow ``residue_committed`` (explicit
        retention decision). Uncommitted farms above the threshold keep baseline
        straw sales, so they pay no opportunity cost and draw no retention
        payment from ``policy.practice_subsidies`` - having enrolled in nothing.
        """
        if self.residue_committed == bool(self.practice_bundle.residue_on_field):
            return self.practice_bundle
        return self.practice_bundle.change_practices(
            residue_on_field=int(self.residue_committed)
        )

    @property
    def proposed_bundle_id(self):
        """Proposed bundle as numeric ID (-1 if none)."""
        return self.proposed_bundle.id if self.proposed_bundle else -1

    @property
    def transition_blocker(self):
        """Primary reason why proposed bundle was not adopted (numeric code).
        
        See BLOCKER_* constants for codes. 0 = no blocker (transition happened
        or no transition needed).
        """
        return getattr(self, '_transition_blocker', BLOCKER_NONE)

    @transition_blocker.setter
    def transition_blocker(self, value):
        """Set transition_blocker to a specific value.
        Parameters
        ----------
        value : int
            The blocker value to set.
        """
        self._transition_blocker = value
        if value not in BLOCKER_NAMES.keys():
            raise ValueError(f"Invalid transition blocker value: {value}. Must be one of: {BLOCKER_NAMES.keys()}")

    @property
    def transition_driver(self):
        """Primary reason why a transition succeeded (numeric code).
        
        See DRIVER_* constants for codes. 0 = no transition (maintaining current).
        This is the mirror of transition_blocker - identifies which pathway and
        TPB component enabled the transition when it succeeds.
        """
        return getattr(self, '_transition_driver', DRIVER_NONE)

    @transition_driver.setter
    def transition_driver(self, value):
        """Set transition_driver to a specific value.
        Parameters
        ----------
        value : int
            The driver value to set.
        """
        self._transition_driver = value
        if value not in DRIVER_NAMES.keys():
            raise ValueError(f"Invalid transition driver value: {value}. Must be one of: {DRIVER_NAMES.keys()}")

    # -------------------------------------------------------------------------
    # Memory management
    # -------------------------------------------------------------------------

    def record_transition(self, new_bundle):
        """Store outcome of current bundle and prepare for transition to new_bundle.

        Called when farmer commits to transitioning practices. Records the
        performance of the outgoing bundle for future reference.

        Parameters
        ----------
        new_bundle : tuple of (int, int, int)
            The new (tillage, cover_crop, residue_on_field) bundle.
        """
        current_year = self.farmer.model.lpjml.sim_year
        n_obs = self.performance_tracker.n

        # -----------------------------------------------------------------
        # Store outcome of outgoing bundle (if used for at least 2 years)
        # -----------------------------------------------------------------
        if n_obs > 1:
            self.performance_memory.record_performance(
                self.practice_bundle,
                self.performance_tracker.trend,
                self.performance_tracker.level,
                n_obs,
                current_year,
            )

        # -----------------------------------------------------------------
        # Prepare for new bundle: reset regression state
        # -----------------------------------------------------------------

        # Remember previous bundle (for potential fallback)
        self.previous_bundle = self.practice_bundle

        # Capture the rates before resetting (for fallback comparison). This is
        # the performance level we expect to maintain or exceed; it is scored on
        # demand so that it always shares a ruler with whatever it is compared to.
        baseline = self._snapshot_performance() if n_obs > 1 else None

        # Reset regression accumulators for new bundle
        self.performance_tracker = ManagementPerformanceTracker.reset_for_transition(
            farmer=self.farmer,
            current_year=current_year,
            baseline_performance=baseline
        )

        # The tracker this year's score was built from is gone, so the memo
        # cannot stand even though the year has not turned.
        self._own_score_year = None

        # Transition to new bundle
        self.practice_bundle = new_bundle
        self.bundle_since = current_year

        # Reset decline counter
        self.decline_years = 0

    # -------------------------------------------------------------------------
    # Abstract methods (implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    def update(self):
        """Update decision state (compute proposed bundle, TPB scores)."""
        pass

    @abstractmethod
    def should_transition(self):
        """Return True if farmer should transition to proposed bundle."""
        pass


# =============================================================================
# TPB (THEORY OF PLANNED BEHAVIOUR) IMPLEMENTATION
# =============================================================================

class TPB(DecisionModel):
    """TPB decision model with social learning and fallback.

    Implements Ajzen (1991) Theory of Planned Behaviour:
    - Attitude: evaluation of the behavior (own experience + social learning)
    - Social Norm: perceived social pressure (what neighbours do)
    - PBC: Perceived Behavioral Control (can I afford it?)

    Extended with:
    - Social learning from neighbours (Bandura 1977)
    - Fallback mechanism for adaptive management (Holling 1978)
    - Random exploration for innovation diffusion

    The TPB intention formula used is multiplicative:
        TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

    This means PBC acts as a gate: if farmer cannot afford the practice
    (PBC → 0), intention is blocked regardless of attitude/norms.
    """

    def __init__(self, farmer):
        """Initialize TPB model with zero scores."""
        super().__init__(farmer)

        # TPB components (updated each timestep)
        self._tpb = 0.0          # Overall intention score
        self._tpb_bypassed = False  # Change forced, so no intention was formed
        self._attitude = 0.0     # Attitude toward behavior (combined)
        self._attitude_own_land = 0.0    # Attitude from own land performance
        self._attitude_social_learning = 0.0  # Attitude from social learning
        self._social_norm = 0.0  # Subjective norm
        self._pbc = 0.0          # Perceived behavioral control

        # Observation years counter: randomize initial timing to desynchronize farmers
        # (avoids artificial waves of simultaneous evaluation)
        min_obs = self.get_aft_param("min_observation_years")
        self._observation_years = np.random.randint(1, min_obs)

        # Litter cover of the recent past, which is what the residue label is
        # certified against. A window rather than the current year, so weather
        # cannot flip the label; see reevaluate_residue_status.
        self._litter_cover_record = deque(maxlen=min_obs)

        # Whether the cadence elapsed this year, so the farm was due a decision.
        # Read after the transition step, where the counter has already been
        # decremented or reset and can no longer answer the question.
        self.evaluated_this_year = False

        # Seasons the cover crop now in the ground has been held. The type
        # answers to its own clock, longer than the bundle cadence: the legume
        # is bought for a yield difference of one to two per cent, against a
        # seed premium of much the same size, so the comparison turns on peer
        # records that only mean anything once episodes are long enough to
        # average a few seasons of weather. Re-opening the choice every three
        # to five years fragments those records across the two crops and leaves
        # the estimator reading noise.
        self._cover_crop_type_years = 0

    # -------------------------------------------------------------------------
    # Properties for external access to TPB components
    # -------------------------------------------------------------------------
    def projection_weight(self):
        """Discounted benefit-years this farmer counts when judging a switch.

        Kept separate from min_observation_years, which is only the re-evaluation
        cadence. Tying the two together meant a longer investment view could not
        be given to a farmer without also making them reconsider less often, and
        those are unrelated traits.

        Note where this does and does not bite. In PBC it scales an annual cost
        and revenue stream against baseline revenue, so a longer horizon has a
        real effect. In the performance score it multiplies a trend that is then
        divided by the cross-farmer spread of the same projected trends, so the
        multiplier very nearly cancels and the horizon is close to irrelevant
        there. The trend's influence is set by the trend weights, not by how far
        ahead the farmer looks.
        """
        return annuity_factor(
            self.get_aft_param("projection_horizon"),
            self.get_aft_param("discount_rate"),
        )

    @property
    def tpb(self):
        """Overall TPB intention score, 0.0 when none was formed this year.

        Zero covers both farmers who did not evaluate - the observation cadence,
        no distinct target - and the emergency paths that transition without an
        intention. Those cases are told apart by transition_blocker and
        transition_driver, which is where the reason belongs; what matters here
        is that no forced transition is reported as a high willingness to change.
        """
        return self._tpb

    @property
    def attitude(self):
        """Attitude component of TPB (combined)."""
        return self._attitude

    @property
    def attitude_own_land(self):
        """Attitude from own land performance."""
        return self._attitude_own_land

    @property
    def attitude_social_learning(self):
        """Attitude from social learning."""
        return self._attitude_social_learning

    @property
    def social_norm(self):
        """Social norm component of TPB."""
        return self._social_norm

    @property
    def pbc(self):
        """Perceived behavioral control component of TPB."""
        return self._pbc

    # =========================================================================
    # EVALUATION TIMING
    # =========================================================================

    def should_evaluate(self) -> bool:
        """Check if farmer should evaluate practice transition this year.

        Farmers don't reconsider practices every year. They evaluate when
        the observation period has ended (randomized around min_observation_years).

        Returns
        -------
        bool
            True if farmer should run TPB evaluation this year.
        """
        return self._observation_years <= 0

    def reset_observation_years(self):
        """Reset observation years after a transition decision (transition or stay).

        Uses normal distribution around min_observation_years (mean=min_obs,
        std=min_obs/2) to create heterogeneity in re-evaluation timing.
        This reflects that some farmers re-evaluate sooner (more proactive)
        while others wait longer (more conservative). Minimum is 1 year.

        Called after TPB evaluation completes, regardless of whether transition
        happened.
        """
        min_obs = self.get_aft_param("min_observation_years")
        if min_obs > 0:
            # Normal distribution around min_obs, minimum 1 year
            self._observation_years = max(
                1, int(np.random.normal(min_obs, min_obs / 2))
            )
        else:
            self._observation_years = 0

    def decrement_observation_years(self):
        """Decrement the observation years counter by one year.

        Called each year when farmer doesn't evaluate.
        """
        if self._observation_years > 0:
            self._observation_years -= 1

    # =========================================================================
    # MAIN UPDATE LOGIC
    # =========================================================================

    def reevaluate_residue_status(self):
        """Set the residue component of the bundle from realised litter cover.

        Under practice_dimensions.residue.residue_status_by_cover, the residue
        component is a measurement rather than an intention: a farm counts as
        retaining residue when its litter cover reaches the FAO conservation
        agriculture threshold of 30%. Reclassification is symmetric in both
        directions.

        Measured as the median of the recent window rather than off the current
        year, and taken on the re-evaluation cadence rather than annually.
        Litter cover moves with the harvest, so a single thin year would
        otherwise decertify a farm that is retaining and one fat year certify a
        farm that is not - and each flip re-keys the slot the memory records
        under and redates the label, so weather alone could keep a farm from
        ever accumulating experience of anything. The median asks whether the
        field was covered in most years of the period, which is what a
        persistent practice means; a mean would let one exceptional harvest
        carry the average across the line.

        With the flag off, the component stays as the farmer set it and residue
        certification is left to post-processing, which can combine the reported
        bundle with the reported litter cover.

        Crossing the threshold is a change of label, not of management, so it
        leaves residue_committed alone: the farm keeps selling straw at the
        baseline rate, and keeps paying and earning accordingly. It is reported
        as its own driver so it cannot be mistaken for a decision, and it dates
        the new label so that the experience the farm claims with it starts here
        rather than being backdated to the last real transition.
        """
        residue_config = self.farmer.model.config.coupled_config.practice_dimensions.residue
        if not residue_config.residue_status_by_cover:
            return
        if not self._litter_cover_record:
            return

        cover = float(np.median(self._litter_cover_record))
        retains = cover >= residue_config.ca_cover_threshold
        if retains != bool(self.practice_bundle.residue_on_field):
            self.practice_bundle = self.practice_bundle.change_practices(
                residue_on_field=int(retains)
            )
            self.bundle_since = self.farmer.model.lpjml.sim_year
            self.transition_driver = DRIVER_RESIDUE_RECLASSIFIED

    def cover_crop_type_for(self, bundle):
        """Which cover crop this farm grows, or would grow, under ``bundle``.

        For the bundle in force that is the crop actually in the ground, which
        holds for the whole episode because only a decision changes it; for a
        candidate it is the one the farm would plant. None where the bundle has
        no cover crop, which is what keeps those slots on a single key.

        Certified and committed both count as in force. They differ only in the
        residue flag, which says nothing about what was sown, so pricing the
        committed bundle on the preference would bill a farm for a crop it did
        not plant whenever the residue label had come apart from the decision.
        """
        if not bundle.cover_crop:
            return None
        if self.farmer.cover_crop and bundle in (
            self.practice_bundle,
            self.committed_bundle,
        ):
            return int(self.farmer.cover_crop)
        return int(self.farmer.indicate_cover_crop_type())

    def commit_bundle(self, new_bundle):
        """Move the farm onto ``new_bundle``: file the old episode, then farm it.

        The order is the point of having this in one place. ``record_transition``
        files the outgoing episode, and the slot it goes in is keyed on the
        cover crop in the ground, which ``apply_bundle`` is about to replace.
        Applying first would file the legume's years under the catch crop's
        name, in the very slots the type choice reads back.

        Recording first also means ``apply_bundle`` picks the crop with the new
        bundle already in force, so the choice is priced against the practices
        the farm is moving to rather than the ones it is leaving.
        """
        self.record_transition(new_bundle)
        self.apply_bundle(new_bundle)

    def reconsider_cover_crop_type(self):
        """Re-open the crop choice at a decision point that left the bundle alone.

        The bundle is re-opened at every decision by the peer scan, but the crop
        is not one of its fields, so a farm whose neighbours run the same bundle
        with the other crop reads that as ``TARGET_SAME`` and stops. Without
        this the crop would be settled once and for all at the last bundle
        change, and a farm that had reached a bundle it was happy with would go
        on sowing against a charge introduced afterwards.

        On the cadence only, never mid-year, so the crop answers to the same
        clock as every other practice. And only where the bundle came through
        the decision unchanged: if it changed, ``apply_bundle`` has just chosen
        the crop, and asking again would immediately undo its exploration draw.

        A new crop is a new practice, so this ends the episode through the
        ordinary transition path rather than a private one - the outgoing record
        is filed under the crop that earned it and the tracker starts again.
        """
        if not self.practice_bundle.cover_crop:
            return

        sown = int(self.farmer.cover_crop or 0)
        preferred = int(self.farmer.indicate_cover_crop_type())
        if not sown or preferred == sown:
            return

        # The crop is held for its own term, which outlasts the bundle cadence.
        # Without this the type turns over every three to five years on a
        # comparison whose two sides differ by about a per cent of revenue, so
        # it tracks the noise in the peer records rather than the difference
        # between the crops.
        if self._cover_crop_type_years < self.get_aft_param(
            "cover_crop_type_commitment_years"
        ):
            return

        # Filed before the crop changes: cover_crop_type_for reads what is in
        # the ground, so the outgoing record would otherwise be keyed on the
        # crop that replaced it.
        self.record_transition(self.practice_bundle)
        self.farmer.cover_crop = preferred
        self.farmer.set_lpjml(attribute="cover_crop")
        self._cover_crop_type_years = 0
        self.transition_driver = DRIVER_COVER_CROP_RETYPED

        # A practice changed, so the commitment period starts over, exactly as
        # it would after a bundle transition.
        self.reset_observation_years()

    def observe(self):
        """Record this year's outcome, whether or not a decision follows.

        This year's litter cover is filed for the residue certification, the
        year's five rates enter the regression accumulators, and the bundle
        memory that neighbours read is refreshed and then decayed.

        Separated from the decision because observing is not deciding. A farmer
        too poor to change practice still farms, still sees the harvest, and is
        still watched by neighbours, so every caller must call this even when it
        skips the decision, and nothing here may presume a decision follows.
        """
        # Reset blocker, driver, and pathway at start of each update
        self.transition_blocker = BLOCKER_NONE
        self.transition_driver = DRIVER_NONE
        self.target_pathway = None
        self._reset_tpb_components()

        # -----------------------------------------------------------------
        # Step 0: File this year's cover for the residue label
        # -----------------------------------------------------------------
        # Recorded every year, but read only on the cadence, so the label
        # certifies against a run of seasons rather than the latest one.
        self._litter_cover_record.append(self.farmer.litter_cover)

        # Ages of the cover crop in place. Only while one is there: a farm with no
        # cover crop is not serving a commitment, and must be free to pick a
        # type the moment it adopts one.
        if self.farmer.cover_crop:
            self._cover_crop_type_years += 1
        else:
            self._cover_crop_type_years = 0

        # -----------------------------------------------------------------
        # Step 1: Add current year's observation to regression
        # -----------------------------------------------------------------
        # Every metric is fed as a self-referenced relative deviation, so they
        # are dimensionless and centred at zero.
        self.performance_tracker.add_observation(
            rates_of(self.farmer),
            self.farmer.model.lpjml.sim_year,
            noise=PerceptionNoise.from_config(
                self.farmer.model.config.coupled_config.tpb
            ),
        )

        # The tracker has moved, so any score cached before this point is stale.
        self._own_score_year = None

        # Initialize the baseline snapshot on first update (deferred from init to
        # avoid chicken-and-egg with reference_scales)
        if (
            self.performance_tracker.baseline_performance is None
            and self.performance_tracker.n > 1
        ):
            self.performance_tracker.baseline_performance = self._snapshot_performance()

        # -----------------------------------------------------------------
        # Step 2: Update performance memory with current trends and level
        # -----------------------------------------------------------------
        # Keep performance memory up-to-date so neighbours see current performance
        # The years of experience claimed for the bundle cannot exceed the years
        # the label has been held: a transition resets the tracker and the two
        # agree, but a relabel does not, and the accumulated record must not be
        # presented as decades of experience with a label acquired this year.
        year = self.farmer.model.lpjml.sim_year
        self.performance_memory.update_current(
            self.practice_bundle,
            self.performance_tracker.trend,
            self.performance_tracker.level,
            min(self.performance_tracker.n, year - self.bundle_since + 1),
            year,
        )

        # -----------------------------------------------------------------
        # Step 3: Decay old memories
        # -----------------------------------------------------------------
        self.performance_memory.decay(
            self.farmer.model.lpjml.sim_year,
            self.get_aft_param("memory_decay_years"),
        )

    def _reset_tpb_components(self):
        """Zero the reported TPB components so a stale year cannot be read.

        Anything not computed this year must report 0.0 rather than last year's
        value, whether the reason is the observation-year cadence or an
        emergency path that bypasses the evaluation entirely.
        """
        self._tpb = 0.0
        self._tpb_bypassed = False
        self._attitude = 0.0
        self._attitude_own_land = 0.0
        self._attitude_social_learning = 0.0
        self._attitude_social_learning_local = 0.0
        self._attitude_social_learning_country = 0.0
        self._attitude_social_learning_cluster = 0.0
        self._social_norm = 0.0
        self._social_norm_local = 0.0
        self._social_norm_country = 0.0
        self._social_norm_cluster = 0.0
        self._pbc = 0.0

    def update(self):
        """Observe this year, then decide whether to propose a new bundle.

        Decision flow, once observe() has recorded the year:
        1. Has the re-evaluation cadence elapsed?
        2. Have running costs outgrown the capital, forcing a deselection?
        3. Has performance declined long enough to fall back?
        4. Which bundle to target: the best-performing peer's, or a random one?
        5. Trim that target to what can be afforded.
        6. Score it with TPB.

        Steps 2 and 3 are forced changes that bypass the TPB evaluation, so they
        return with the bypass flag set rather than an intention score.

        Sets transition_blocker to say why no transition happened, and
        transition_driver to say what enabled one that did.
        """
        self.observe()

        # Step 1: Require minimum observation years before transitioning
        # Farmers record observations every year (above), but only evaluate
        # TPB once the observation period is complete. This avoids noisy
        # decisions based on single-year fluctuations.
        # After a transition, the counter resets via reset_observation_years().

        self.evaluated_this_year = self.should_evaluate()
        if not self.evaluated_this_year:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_OBSERVATION_YEARS
            self.decrement_observation_years()
            return

        # The residue label is a measurement, but it is taken when the farm is
        # being looked at rather than continuously, so it sits on the same clock
        # as the bundle. Before the target search, which reads practice_bundle.
        self.reevaluate_residue_status()

        # -------------------------------------------------------------------------
        # Step 2: Check affordability deselection (only if capital constrained)
        # -------------------------------------------------------------------------
        # If current practices cost more than available capital, propose
        # a bundle with costly practices deselected (survival mechanism)
        if self.check_affordability_deselection():
            # Deselection sets proposed_bundle and the bypass flag internally
            self.transition_blocker = BLOCKER_AFFORDABILITY_FORCED
            self.target_pathway = "affordability"
            return

        # -------------------------------------------------------------------------
        # Step 3: Check fallback condition
        # -------------------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self.check_fallback():
            # Fallback sets proposed_bundle and the bypass flag internally
            self.transition_blocker = BLOCKER_FALLBACK_TRIGGERED
            self.target_pathway = "fallback"  # Track pathway for transition_driver
            return

        # -------------------------------------------------------------------------
        # Step 4: Find target bundle (neighbour imitation or exploration)
        # -------------------------------------------------------------------------

        # First, try to imitate best-performing neighbour (local)
        target_bundle = self.most_promising_bundle_local()
        self.target_pathway = "local"

        # If no better local neighbour, try country-level inspiration
        if target_bundle is None:
            target_bundle = self.most_promising_bundle_country()
            self.target_pathway = "country"

        # If no country inspiration, try cluster-level (agroecologically similar countries)
        if target_bundle is None:
            target_bundle = self.most_promising_bundle_cluster()
            self.target_pathway = "cluster"

        # If no cluster inspiration, maybe explore randomly
        if target_bundle is None:
            target_bundle = self.maybe_explore_bundle()
            self.target_pathway = "exploration"

        # No change proposed
        if target_bundle is None:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_NO_TARGET
            return

        if target_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_TARGET_SAME
            return

        # -------------------------------------------------------------------------
        # Step 5: Adjust for affordability (only if capital constrained)
        # -------------------------------------------------------------------------
        # If farmer can't afford full target bundle, find affordable subset
        affordable_bundle = self.affordable_bundle(target_bundle)

        # If affordable bundle is same as current, no change
        if affordable_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self.transition_blocker = BLOCKER_TARGET_UNAFFORDABLE
            return

        # -------------------------------------------------------------------------
        # Step 6: Compute TPB scores for proposed bundle
        # -------------------------------------------------------------------------
        self.proposed_bundle = affordable_bundle
        self.compute_tpb_for_bundle(affordable_bundle)

    # =========================================================================
    # AFFORDABILITY DESELECTION (Survival Mechanism)
    # =========================================================================

    def check_affordability_deselection(self):
        """Check if practices must be deselected due to unaffordable costs.

        When capital is too low to sustain current practices, propose a bundle
        with practices deselected in order of cost (most expensive first) until
        direct costs are within the affordable range.

        Like fallback, this is an emergency response that bypasses TPB, because
        the farmer has no choice but to abandon costly practices to survive.
        However, the transition still goes through the normal pathway for proper
        recording.

        Does nothing when farm_economics.capital_constraints is off, since a
        farmer who can always borrow is never forced to abandon a practice for
        lack of cash.

        Returns
        -------
        bool
            True if deselection is needed (sets proposed_bundle internally).
        """
        farmer = self.farmer
        if not farmer.capital_constrained:
            return False

        available_capital = farmer.capital - farmer.min_capital

        # Deselect only committed practices (cost-bearing). Certified residue
        # from litter cover alone has no direct cost to shed.
        bundle = self.committed_bundle
        current_direct_costs = farmer.get_bundle_direct_costs(bundle=bundle)

        # No action needed if costs are within budget
        if current_direct_costs <= available_capital:
            return False

        # Find affordable bundle by deselecting practices
        for practice_name in DESELECT_ORDER:
            if getattr(bundle, practice_name) == 0:
                continue

            # Try deselecting this practice
            new_bundle = bundle.change_practices(**{practice_name: 0})
            new_costs = farmer.get_bundle_direct_costs(bundle=new_bundle)

            # Only deselect if it actually reduces costs (or provides savings)
            if new_costs < current_direct_costs:
                bundle = new_bundle
                current_direct_costs = new_costs

                # Check if costs are now within budget
                if current_direct_costs <= available_capital:
                    break

        # No change needed if bundle didn't change
        if bundle == self.committed_bundle:
            return False

        # Propose the deselected bundle
        self.proposed_bundle = bundle

        # Bypass TPB: this is an emergency survival response, not a planned
        # behaviour change, so no intention was formed.
        self._tpb_bypassed = True

        return True

    # =========================================================================
    # BASELINE ANCHOR
    # =========================================================================

    def _snapshot_performance(self):
        """Freeze the tracker's current rates as a scoreable snapshot.

        Rates, not the score they currently produce: the scoring rulers move
        between years, so a score frozen today would not be comparable to one
        computed later. See ManagementPerformanceTracker.baseline_performance.
        """
        return ManagementPerformance.from_tracker(
            self.performance_tracker.trend,
            self.performance_tracker.level,
            self.performance_tracker.n,
            self.farmer.model.lpjml.sim_year,
        )

    def baseline_score(self):
        """Score the baseline snapshot with the CURRENT rulers.

        Returns None while no anchor exists yet. Scoring on demand is what keeps
        the comparison honest: both the baseline and the score it is compared to
        are divided by the same rulers, so only the farm can move the difference.
        """
        baseline = self.performance_tracker.baseline_performance
        if baseline is None:
            return None
        return compute_performance_score(baseline, self.farmer)

    def own_score(self):
        """This farm's current performance score, computed once per year.

        Bundle-independent, so it is the same number for every candidate a
        farmer weighs in a year, yet it is read by all four attitude channels
        and by the fallback and exploration checks. Without memoising, those
        reads rebuild the same five z-scores once per candidate bundle.

        Keyed on the simulation year because both inputs move annually and
        together: the tracker gains its observation at the start of the update,
        and the rulers are recalibrated once a year. Same idiom as the country
        effect cache.
        """
        year = self.farmer.model.lpjml.sim_year
        if self._own_score_year != year:
            self._own_score = compute_performance_score(
                self.performance_tracker, self.farmer
            )
            self._own_score_year = year
        return self._own_score

    # =========================================================================
    # FALLBACK MECHANISM (Adaptive Management)
    # =========================================================================

    def check_fallback(self):
        """Propose revert to ``previous_bundle`` after sustained own-farm decline.

        Trigger: ``own_score()`` below ``baseline_score()`` (performance at the
        last transition) for ``fallback_years`` consecutive years. One year at
        or above the anchor resets the counter. Comparison is own-farm only —
        peer-informed exit uses the normal re-evaluation path.

        Background decline can look like practice failure; two bounds limit that:
        consecutive years are required (interannual noise often resets the
        count), and the anchor is rescored with current rulers so a widening
        cross-farmer spread does not freeze an ever-harder bar.

        Returns
        -------
        bool
            True if fallback triggers (sets ``proposed_bundle``).
        """
        # Can't fall back if no previous bundle
        if self.previous_bundle is None:
            return False

        # -------------------------------------------------------------------------
        # Compare current performance to the state left behind at the switch
        # -------------------------------------------------------------------------
        reference = self.baseline_score()
        if reference is None:
            return False  # Nothing to compare against

        # Track consecutive years of being worse off than before the switch
        if self.own_score() < reference:
            self.decline_years += 1
        else:
            self.decline_years = 0  # Reset if we are ahead again

        # -------------------------------------------------------------------------
        # Trigger fallback after sustained decline
        # -------------------------------------------------------------------------
        fallback_years = self.get_aft_param("fallback_years")
        if self.decline_years >= fallback_years:
            # Mark current bundle as "failed" (reduces future exploration)
            self.performance_memory.record_failure(self.practice_bundle)

            # Propose reverting to previous bundle
            self.proposed_bundle = self.previous_bundle

            # Fallback bypasses TPB: this is an emergency response, not a
            # planned behaviour change, so no intention was formed and the TPB
            # components stay at zero.
            self._tpb_bypassed = True

            return True

        return False

    # =========================================================================
    # RANDOM EXPLORATION (Innovation Diffusion)
    # =========================================================================

    def explores_cover_crop_type(self):
        """Whether this farm sows the cover crop it did not prefer.

        The type decision is an economic comparison between two crops, and it
        can only be made on peers who have grown each. Left to itself the
        cheaper seed wins wherever there is no evidence, no farm sows the
        other, and the absence of evidence becomes permanent - the estimator
        would be reading a record its own answer had written.

        So the same impulse that has farms try unfamiliar bundles has them try
        the other cover crop, drawn once, by ``apply_bundle``, as the crop goes
        in the ground.

        On its own rate rather than the bundle's. Bundle exploration picks one
        of eight, so a fifteen per cent draw perturbs a farm's trajectory
        slightly. Here the draw is over two, so the same rate would have
        fifteen to thirty per cent of farms permanently growing the crop they
        judged worse - not a probe but a standing randomisation, and a large
        one next to a legume advantage worth about a per cent of revenue.

        Draws its own random number rather than reusing the bundle draw,
        because a farm that has just changed bundle is not thereby more likely
        to change crop as well.

        Returns
        -------
        bool
            True if the farm should sow the crop it did not prefer.
        """
        return bool(
            np.random.random()
            < self.get_aft_param("cover_crop_type_exploration_prob")
        )

    def maybe_explore_bundle(self):
        """Randomly explore a new bundle (innovation diffusion).

        Some farmers try new practices without neighbour influence:
        - "Pioneers" explore more frequently
        - Poor performers explore more (searching for better options)
        - Bundles that failed repeatedly are excluded, and bundles already tried
          are drawn less often than untried ones
        - Bundles closer to the current one are drawn more often

        Returns
        -------
        ManagementBundle or None
            Random valid bundle to try, or None if no exploration.
        """
        # Determine exploration probability

        # Base probability depends on farmer type (from config)
        # Pioneers (innovators) are more willing to experiment
        base_prob = self.get_aft_param("exploration_base_prob")

        # Poor performers explore more (searching for better options)
        # Score below baseline means declining from when we started this practice
        baseline = self.baseline_score()
        if baseline is not None and self.own_score() < baseline:
            base_prob *= self.get_aft_param("poor_performance_multiplier")

        # Cap exploration probability
        explore_prob = min(base_prob, self.get_aft_param("max_exploration_prob"))

        # -----------------------------------------------------------------
        # Random draw: explore or not?
        # -----------------------------------------------------------------
        if np.random.random() > explore_prob:
            return None  # No exploration this year

        # -----------------------------------------------------------------
        # Select valid bundle to explore (biased toward incremental changes)
        # -----------------------------------------------------------------
        exploration = self.farmer.model.config.coupled_config.exploration
        max_failures = exploration.max_failures

        # Generate all possible bundles with their "distance" from current
        current = self.practice_bundle
        valid = []
        for b in ManagementBundle.all_bundles():
            if b == current:
                continue
            if self.performance_memory[b].failure_count >= max_failures:
                continue
            # Distance = number of practice changes (0-3)
            dist = (
                int(b.tillage != current.tillage)
                + int(b.cover_crop != current.cover_crop)
                + int(b.residue_on_field != current.residue_on_field)
            )
            valid.append((b, dist))

        if not valid:
            return None

        # Weight by inverse distance: 1-change bundles are 3x more likely than 3-change
        # weights: dist=1 -> 3, dist=2 -> 2, dist=3 -> 1
        # Additionally penalize bundles already tried (in memory) - favor novel options
        weights = []
        for b, dist in valid:
            w = 4 - dist  # base weight from distance
            if self.performance_memory[b].duration > 0:
                w *= 0.25  # already tried -> 4x less likely
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        idx = np.random.choice(len(valid), p=weights)
        return valid[idx][0]

    # =========================================================================
    # TRANSITION DECISION
    # =========================================================================

    def should_transition(self):
        """Determine if farmer should transition to proposed bundle.

        Compares the TPB intention score to the threshold, unless the change was
        forced by an emergency path (affordability deselection or fallback),
        which proceeds regardless because no intention was formed to compare.
        Those paths set _tpb_bypassed rather than a high intention, so a forced
        retreat is not reported as the strongest possible willingness to change.

        Flip-flopping is prevented by min_observation_years: after a transition
        the performance tracker resets, so a farmer needs a commitment period of
        new harvests before reconsidering.

        The threshold is the same for every AFT; behavioural differences between
        them come from the score weights and from PBC.

        Returns
        -------
        bool
            True if the change was forced, or TPB exceeds the threshold.
        """
        if self._tpb_bypassed:
            return True

        tpb_config = self.farmer.model.config.coupled_config.tpb
        threshold = tpb_config.transition_threshold

        # bool(): the scores are numpy floats wherever they come from cell
        # data, and a numpy bool is not a bool to callers that test its type.
        return bool(self._tpb > threshold)

    def _get_enabled_spreading_levels(self):
        """Get which spreading levels are enabled from config.

        Returns
        -------
        tuple[bool, bool, bool]
            (enable_local, enable_country, enable_cluster)
        """
        tpb = self.farmer.model.config.coupled_config.tpb
        return tpb.enable_local, tpb.enable_country, tpb.enable_cluster


    def set_tpb_transition_blocker(self):
        """Set transition_blocker to indicate which TPB component is most limiting.

        Uses a simple recursive approach:
        1. First identify which main component (attitude, social_norm, pbc) is lowest
        2. Then drill down into that component's sub-parts to identify the specific blocker

        Only compares enabled spreading levels (respects tpb config).
        """
        if self.proposed_bundle is None:
            return

        # Get enabled levels to avoid blaming disabled components
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

        # Level 1: Which main component is the blocker?
        main_components = {
            'attitude': self._attitude,
            'social_norm': self._social_norm,
            'pbc': self._pbc,
        }
        main_blocker = min(main_components, key=main_components.get)

        # Level 2: Drill down into sub-components
        if main_blocker == 'attitude':
            # Compare own_land vs social_learning
            if self._attitude_own_land <= self._attitude_social_learning:
                self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND
            else:
                # Compare only ENABLED social learning levels
                social_components = {}
                if enable_local:
                    social_components['local'] = self._attitude_social_learning_local
                if enable_country:
                    social_components['country'] = self._attitude_social_learning_country
                if enable_cluster:
                    social_components['cluster'] = self._attitude_social_learning_cluster

                if social_components:
                    min_social = min(social_components, key=social_components.get)
                    if min_social == 'local':
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL
                    elif min_social == 'country':
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY
                    else:
                        self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER
                else:
                    # Fallback if no social levels enabled (shouldn't happen)
                    self.transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND

        elif main_blocker == 'social_norm':
            # Compare only ENABLED social norm levels
            norm_components = {}
            if enable_local:
                norm_components['local'] = self._social_norm_local
            if enable_country:
                norm_components['country'] = self._social_norm_country
            if enable_cluster:
                norm_components['cluster'] = self._social_norm_cluster

            if norm_components:
                min_norm = min(norm_components, key=norm_components.get)
                if min_norm == 'local':
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL
                elif min_norm == 'country':
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY
                else:
                    self.transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER
            else:
                # Fallback if no social levels enabled (shouldn't happen)
                self.transition_blocker = BLOCKER_TPB_LOW_PBC

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            self.transition_blocker = BLOCKER_TPB_LOW_PBC

    def set_tpb_component_driver(self, pathway: str):
        """Set transition_driver to indicate which TPB component enabled the transition.

        Uses a simple recursive approach (mirror of set_tpb_transition_blocker):
        1. First identify which main component (attitude, social_norm, pbc) is highest
        2. Then drill down into that component's sub-parts to identify the specific driver

        Only compares enabled spreading levels (respects tpb config).

        Parameters
        ----------
        pathway : str
            One of "local" (learned from local neighbor), "country" (inspired
            by country-level data), "cluster" (agroecological cluster learning),
            or "exploration" (random exploration).
        """
        if self.proposed_bundle is None:
            return

        # Get enabled levels to avoid crediting disabled components
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

        # Level 1: Which main component is the strongest driver?
        main_components = {
            'attitude': self._attitude,
            'social_norm': self._social_norm,
            'pbc': self._pbc,
        }
        main_driver = max(main_components, key=main_components.get)

        # Level 2: Drill down into sub-components
        if main_driver == 'attitude':
            # Compare own_land vs social_learning
            if self._attitude_own_land >= self._attitude_social_learning:
                sub_driver = 'attitude_own_land'
            else:
                # Compare only ENABLED social learning levels
                social_components = {}
                if enable_local:
                    social_components['local'] = self._attitude_social_learning_local
                if enable_country:
                    social_components['country'] = self._attitude_social_learning_country
                if enable_cluster:
                    social_components['cluster'] = self._attitude_social_learning_cluster

                if social_components:
                    max_social = max(social_components, key=social_components.get)
                    if max_social == 'local':
                        sub_driver = 'attitude_social_local'
                    elif max_social == 'country':
                        sub_driver = 'attitude_social_country'
                    else:
                        sub_driver = 'attitude_social_cluster'
                else:
                    # Fallback if no social levels enabled
                    sub_driver = 'attitude_own_land'

        elif main_driver == 'social_norm':
            # Compare only ENABLED social norm levels
            norm_components = {}
            if enable_local:
                norm_components['local'] = self._social_norm_local
            if enable_country:
                norm_components['country'] = self._social_norm_country
            if enable_cluster:
                norm_components['cluster'] = self._social_norm_cluster

            if norm_components:
                max_norm = max(norm_components, key=norm_components.get)
                if max_norm == 'local':
                    sub_driver = 'social_norm_local'
                elif max_norm == 'country':
                    sub_driver = 'social_norm_country'
                else:
                    sub_driver = 'social_norm_cluster'
            else:
                # Fallback if no social levels enabled
                sub_driver = 'pbc'

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            sub_driver = 'pbc'

        # Map based on pathway and sub-component
        # Pathways: "local" (learned from local neighbor), "country" (inspired by
        # country-level stats), "cluster" (agroecological cluster), "exploration"
        driver_maps = {
            "local": {
                'attitude_own_land': DRIVER_LOCAL_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_LOCAL_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_LOCAL_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_LOCAL_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_LOCAL_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_LOCAL_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_LOCAL_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_LOCAL_PBC,
            },
            "country": {
                'attitude_own_land': DRIVER_COUNTRY_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_COUNTRY_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_COUNTRY_PBC,
            },
            "cluster": {
                'attitude_own_land': DRIVER_COUNTRY_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_COUNTRY_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_COUNTRY_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_COUNTRY_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_COUNTRY_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_COUNTRY_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_COUNTRY_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_COUNTRY_PBC,
            },
            "exploration": {
                'attitude_own_land': DRIVER_EXPLORATION_ATTITUDE_OWN_LAND,
                'attitude_social_local': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_LOCAL,
                'attitude_social_country': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_COUNTRY,
                'attitude_social_cluster': DRIVER_EXPLORATION_ATTITUDE_SOCIAL_CLUSTER,
                'social_norm_local': DRIVER_EXPLORATION_SOCIAL_NORM_LOCAL,
                'social_norm_country': DRIVER_EXPLORATION_SOCIAL_NORM_COUNTRY,
                'social_norm_cluster': DRIVER_EXPLORATION_SOCIAL_NORM_CLUSTER,
                'pbc': DRIVER_EXPLORATION_PBC,
            },
        }

        driver_map = driver_maps.get(pathway, driver_maps["exploration"])
        self.transition_driver = driver_map[sub_driver]

    # =========================================================================
    # APPLY BUNDLE TO AGENT
    # =========================================================================

    def apply_bundle(self, bundle):
        """Apply practice bundle to farmer and push to LPJmL.

        Sets farmer's tillage, cover_crop, and residue_on_field attributes
        and sends updates to the coupled LPJmL model.

        Parameters
        ----------
        bundle : Bundle
            Practice bundle (tillage, cover_crop, residue_on_field), each 0 or 1.
        """
        # Tillage: 0 = no-till (CA), 1 = conventional tillage
        # Directly maps to LPJmL's with_tillage
        self.farmer.tillage = bundle.tillage

        # -----------------------------------------------------------------
        # Cover crop: 0 = none, 1 = non-legume (catch crop), 2 = legume.
        # -----------------------------------------------------------------

        # Settled here and nowhere else, so the crop holds for as long as the
        # bundle does and the episode the tracker measures is one crop
        # throughout. Occasionally the farm sows the other one to keep the
        # comparison fed; see explores_cover_crop_type.
        if bundle.cover_crop == 1:
            choice = int(self.farmer.indicate_cover_crop_type())
            if self.explores_cover_crop_type():
                choice = (
                    COVER_CROP_NONLEGUME
                    if choice == COVER_CROP_LEGUME
                    else COVER_CROP_LEGUME
                )
            # Only a different crop starts a new term. A farm that changes
            # bundle without changing crop keeps the seasons it has already
            # served, so moving between bundles cannot be used to shorten the
            # commitment.
            if choice != int(self.farmer.cover_crop or 0):
                self._cover_crop_type_years = 0
            self.farmer.cover_crop = choice
        else:
            self.farmer.cover_crop = 0
            self._cover_crop_type_years = 0

        # -----------------------------------------------------------------
        # Residue retention: baseline value, 1 = retain all residue
        # -----------------------------------------------------------------

        # Residue: apply_bundle is the only commitment path. Retain → leave all
        # residue for LPJmL (fraction 1). Else → baseline removal rate.
        # Relabel via residue_status_by_cover does not set residue_committed.
        self.residue_committed = bundle.residue_on_field == 1
        if self.residue_committed:
            self.farmer.residue_on_field = 1.0
        else:
            self.farmer.residue_on_field = self.farmer.residue_baseline

        # -----------------------------------------------------------------
        # Push changes to LPJmL
        # -----------------------------------------------------------------
        for attr in ["tillage", "cover_crop", "residue_on_field"]:
            self.farmer.set_lpjml(attribute=attr)

    # =========================================================================
    # COMPARISON METHODS
    # =========================================================================

    def most_promising_bundle_local(self):
        """Find best-performing neighbour's bundle (if better than self).

        Social learning: farmers observe neighbours and may imitate those
        who are performing better (Bandura 1977).

        Returns
        -------
        tuple or None
            Best neighbour's bundle, or None if no neighbour is better.
        """
        best_neighbour = None
        best_gap = 0.0

        for neighbour in self.farmer.neighbourhood:
            # Skip neighbours who aren't performing better
            if not self.is_better_performing(neighbour):
                continue

            # Track neighbour with largest performance gap
            gap = self.performance_gap(neighbour)
            if gap > best_gap:
                best_gap = gap
                best_neighbour = neighbour

        if best_neighbour is None:
            return None

        return best_neighbour.behaviour.practice_bundle

    def _best_bundle_in_store(self, store):
        """Highest-scoring bundle in a region store, if it beats the farm's own.

        Shared by the country and cluster searches, which put the same question
        to different populations and differ only in which store they read.

        Parameters
        ----------
        store : RegionManagementPerformanceStore or None
            Aggregate performance by bundle for the region.

        Returns
        -------
        ManagementBundle or None
            None when the region is unknown or nothing on offer scores better
            than what the farm already runs.
        """
        if store is None:
            return None

        best_bundle = None
        best_score = self.own_score()  # Must beat our current score

        for bundle, perf in store.items():
            if bundle == self.practice_bundle:
                continue
            bundle_score = compute_performance_score(perf, self.farmer)
            if bundle_score > best_score:
                best_score = bundle_score
                best_bundle = bundle

        return best_bundle

    def most_promising_bundle_country(self):
        """Find best-performing bundle at COUNTRY level (if better than current).

        Non-local social learning: when no local neighbour is better, farmers
        may look to successful practices used elsewhere in their country.
        Country stats are already merged with neighbouring countries (configured
        via neighbour_country_weight) at the country level.

        Returns
        -------
        ManagementBundle or None
            Best country-level bundle, or None if no bundle performs better
            than current practice or if country data is unavailable.
        """
        return self._best_bundle_in_store(
            _country_performance_store(self.farmer.cell.country)
        )

    def most_promising_bundle_cluster(self):
        """Find best-performing bundle at CLUSTER level (if better than current).

        Tele-coupled social learning: when no local or country-level neighbour
        is better, farmers may look to successful practices in agroecologically
        similar countries (same temperature, precipitation, PET patterns).

        Returns
        -------
        ManagementBundle or None
            Best cluster-level bundle, or None if no bundle performs better
            than current practice or if cluster data is unavailable.
        """
        return self._best_bundle_in_store(self._cluster_performance_store())

    def performance_gap(self, neighbour):
        """Compute positive performance score difference (neighbour - self).

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        float
            Positive score gap (0 if neighbour has lower or equal score).
        """
        their_score = compute_performance_score(
            neighbour.behaviour.performance_tracker, self.farmer
        )
        # Return positive gap only (0 if neighbor is not better)
        return max(0.0, their_score - self.own_score())

    def is_better_performing(self, neighbour):
        """Check if neighbour is performing better than self.

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        bool
            True if neighbour's unified score exceeds self's score.
        """
        return self.performance_gap(neighbour) > 0.0

    # =========================================================================
    # CROP SIMILARITY
    # =========================================================================

    def crop_similarity(self, neighbour):
        """Check if neighbour grows the same dominant crop at similar scale.

        Fast O(n) comparison using only argmax (no sorting).
        Farmers with similar crop portfolios face similar conditions,
        making their experiences more relevant for social learning.

        Parameters
        ----------
        neighbour : Farmer
            Neighbour farmer to compare with.

        Returns
        -------
        float
            1.0: Same dominant crop with similar share (within 50% relative diff)
            0.5: Same dominant crop but different share
            0.0: Different dominant crops or neighbour doesn't grow it
        """
        # Get crop fractions via farmer's get_from_earth (handles multi-year data)
        # Exclude managed grassland - it's not a crop for similarity comparison
        cft_self = self.farmer.get_from_earth(
            "cftfrac", drop_band=NON_CROPS, time_idx=-1
        ).values.flatten()
        cft_neighbour = neighbour.get_from_earth(
            "cftfrac", drop_band=NON_CROPS, time_idx=-1
        ).values.flatten()

        # Find own dominant crop (single argmax - very fast)
        dominant_idx = np.argmax(cft_self)
        share_self = cft_self[dominant_idx]

        # Check if neighbour grows the same crop significantly
        share_neighbour = cft_neighbour[dominant_idx]

        if share_neighbour < 0.01:
            return 0.0  # Neighbour doesn't grow this crop

        # Check if shares are similar (within 50% relative difference)
        max_share = max(share_self, share_neighbour)
        relative_diff = abs(share_self - share_neighbour) / max_share

        if relative_diff < 0.5:
            return 1.0  # Same crop, similar share
        else:
            return 0.5  # Same crop, different share

    def total_similarity(self, new_bundle, neighbour):
        """Combined bundle + crop similarity for social learning.

        Weighted average of practice bundle similarity and crop portfolio
        similarity. Allows tuning relative importance via config.

        Parameters
        ----------
        new_bundle : tuple
            Practice bundle being evaluated.
        neighbour : Farmer
            Neighbour farmer to compare with.

        Returns
        -------
        float
            Combined similarity in [0, 1].
        """
        bundle_sim = new_bundle.similarity(neighbour.behaviour.practice_bundle)
        crop_sim = self.crop_similarity(neighbour)

        # Weights from config (default: 60% bundle, 40% crop)
        w_bundle = self.get_aft_param("weight_bundle_similarity")
        w_crop = self.get_aft_param("weight_crop_similarity")

        return w_bundle * bundle_sim + w_crop * crop_sim

    def _land_similarity_config(self):
        """The ``tpb.land_similarity`` block, or ``None`` if it is off or absent.

        On, it reaches every comparison the estimator serves, because the
        objection is to the estimator and not to any one practice. Absent as
        well as off, because a configuration written before this setting
        existed must keep running and must reproduce its old numbers.
        """
        tpb = self.farmer.model.config.coupled_config.tpb
        settings = getattr(tpb, "land_similarity", None)
        if settings is None or not getattr(settings, "enabled", False):
            return None
        return settings

    def land_similarity(self, peer, settings):
        """How well ``peer``'s land stands in for this farm's, in [floor, 1].

        Distance between the two farms in the baseline covariates of
        ``LAND_COVARIATES``, each divided by its cross-farmer spread so
        nitrogen mineralisation and root-zone moisture are comparable before
        being summed::

            similarity = floor + (1 - floor) / (1 + d2)
            d2         = mean_k ((own_k - peer_k) / scale_k)^2

        The mean rather than the sum over covariates, so adding a covariate
        changes what is measured and not how sharply it discriminates.

        There is no width parameter because the covariates already supply one:
        distances are in units of the population's own robust spread, so a peer
        one spread away in every covariate is halfway discounted by
        construction, and the kernel sharpens or relaxes with the dispersion of
        the land rather than with a number someone chose. The rational form
        rather than a Gaussian for its heavier tail: with a pool that can be
        three peers, a distant one should be discounted and not erased.

        ``floor`` is the only setting, and it is the strength of the whole
        correction: 0 lets a distant peer be discarded outright, 1 makes the
        kernel inert and recovers the unweighted estimator.

        Returns
        -------
        float
            Weight multiplier for this peer's contribution.
        """
        scales = self.farmer.world.statistic.get("land_scales")
        if not scales:
            # Scales not computed yet (farmers still initialising); a uniform
            # weight reproduces the unweighted estimator rather than guessing.
            return 1.0

        floor = float(getattr(settings, "floor", 0.0))

        squared = []
        for name in LAND_COVARIATES:
            scale = scales.get(name)
            if not scale:
                continue
            own = getattr(self.farmer, name, None)
            theirs = getattr(peer, name, None)
            if own is None or theirs is None:
                continue
            if not (np.isfinite(own) and np.isfinite(theirs)):
                continue
            squared.append(((own - theirs) / scale) ** 2)

        if not squared:
            return 1.0

        distance = float(np.mean(squared))
        return floor + (1.0 - floor) / (1.0 + distance)

    def _shrink_to_neutral(self, score, evidence):
        """Shrink a [0,1] component score toward 0.5 when evidence is thin.

        Without this, one farm-year and a large sample enter the TPB sum with
        equal authority (e.g. a single early adopter dominating a country
        aggregate). Linear ramp: full weight at ``confidence_years`` observations
        (years or contributing farms — both count as independent observations).
        Pioneers use a shorter ``confidence_years``.

        Parameters
        ----------
        score : float
            Component in [0, 1].
        evidence : float
            Observation years or contributing farms behind the score.

        Returns
        -------
        float
            ``0.5 + (evidence/confidence_years clipped to 1) * (score - 0.5)``.
        """
        full = self.get_aft_param("confidence_years")
        if full <= 0:
            return score
        confidence = min(1.0, max(0.0, evidence / full))
        return 0.5 + confidence * (score - 0.5)

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Own Land)
    # =========================================================================

    def compute_attitude_own_land(self, new_bundle):
        """Own-land attitude toward ``new_bundle`` in [0, 1].

        Compares a reference performance score to current ``own_score()``;
        ``sigmoid(reference - own)`` raises attitude when the reference is
        better (motivates change).

        Reference choice
        ----------------
        1. If ``performance_memory`` has duration > 0 for ``new_bundle``: use
           that remembered score (direct experience of the candidate). Floor
           0.0 — a tried bundle may argue for or against switching.
        2. Otherwise: use ``baseline_score()`` at the last transition (status
           quo dissatisfaction). Bundle-independent. Floored at 0.5 so good
           current performance does not veto an untried practice; only
           degradation below the pre-transition anchor raises attitude.

        Peer evidence is excluded here (belongs in social learning); including
        it would double-count. Short evidence is shrunk toward 0.5 via
        ``_shrink_to_neutral``. Baseline is stored as rates and rescored each
        year so scores stay comparable under the current population rulers.

        Parameters
        ----------
        new_bundle : ManagementBundle
            Candidate bundle.

        Returns
        -------
        float
            Attitude in [0, 1].
        """
        remembered = self.performance_memory[new_bundle]
        if remembered.duration > 0:
            reference = compute_performance_score(remembered, self.farmer)
            evidence = remembered.duration
            floor = 0.0  # tried bundle: two-sided
        else:
            reference = self.baseline_score()
            if reference is None:
                return 0.5  # no pre-transition anchor yet
            evidence = self.performance_tracker.n
            floor = 0.5  # untried: dissatisfaction only, not a veto when thriving

        # Positive when reference beats current performance → higher attitude
        return self._shrink_to_neutral(
            max(floor, sigmoid(reference - self.own_score())), evidence
        )

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def compute_attitude_social_learning_local(self, new_bundle):
        """Local social-learning attitude from neighbours on ``new_bundle``.

        Pairwise score gaps vs ``own_score()``, weighted by bundle+crop
        similarity (homophily). Normalised by similarity sum only; neighbour
        confidence (``n / confidence_years``) stays in the numerator so thin
        experience attenuates the signal toward 0.5 — limits early-cascade
        overweighting of recent adopters. Region channels average aggregates
        and cannot apply pairwise similarity.

        References: Festinger (1954); McPherson et al. (2001); Boyd & Richerson
        (1985); van Ittersum et al. (2013).

        Parameters
        ----------
        new_bundle : ManagementBundle
            Candidate bundle (used for similarity to each neighbour's practice).

        Returns
        -------
        float
            Attitude in [0, 1]; 0.5 if no usable neighbours.
        """
        if not self.farmer.neighbourhood:
            return 0.5  # Neutral without neighbours

        confidence_years = self.get_aft_param("confidence_years")
        my_score = self.own_score()

        weighted_score_diff = 0.0
        similarity_weight = 0.0

        for neighbour in self.farmer.neighbourhood:
            # How similar is neighbour? (bundle + crop similarity)
            similarity = self.total_similarity(new_bundle, neighbour)
            if similarity == 0:
                continue  # Skip completely different neighbours

            n_tracker = neighbour.behaviour.performance_tracker

            # Confidence: more observations -> more reliable information
            confidence = (
                min(1.0, n_tracker.n / confidence_years)
                if confidence_years > 0
                else 1.0
            )

            # Score difference: positive if neighbour is better
            score_diff = (
                compute_performance_score(n_tracker, self.farmer) - my_score
            )

            # similarity selects; confidence scales magnitude (not cancelled)
            weighted_score_diff += similarity * confidence * score_diff
            similarity_weight += similarity

        if similarity_weight == 0:
            return 0.5  # Neutral if no relevant neighbours

        # Similarity normalises out, confidence does not (see above).
        avg_score_diff = weighted_score_diff / similarity_weight

        # Sigmoid maps to (0, 1) attitude score. The score difference is already
        # in units of the cross-farmer spread, so it needs no extra scaling.
        return sigmoid(avg_score_diff)

    def _cluster_performance_store(self):
        """Aggregate performance store for this farm's agroecological cluster.

        Reached through country._world, which resolves on both the driver and
        the workers. Returns None when the country belongs to no cluster or the
        cluster carries no store yet, so callers get one "nothing to learn from
        here" answer rather than three separate guards.
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)
        if cluster_id < 0:
            return None

        cluster_stats = country._world.statistic.get(
            "cluster_management_performance", {}
        )
        store = cluster_stats.get(cluster_id)
        return store if isinstance(store, RegionManagementPerformanceStore) else None

    def _attitude_from_region(self, store, new_bundle):
        """Attitude from a region's aggregate experience with a bundle.

        The country and cluster channels ask one question of two different
        populations - how does this bundle's aggregate performance compare to
        mine - so they share the arithmetic and differ only in which store they
        read. Neither weights by similarity: the aggregate is already grouped by
        bundle, which makes it O(1) rather than the O(n²) of pairwise local
        comparison.

        Shrinking by the number of contributing farms is what keeps a region
        that has three adopters from speaking like one that has three hundred.

        Parameters
        ----------
        store : RegionManagementPerformanceStore or None
            Aggregate performance by bundle for the region.
        new_bundle : ManagementBundle
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1], neutral when the region says nothing.
        """
        if store is None:
            return 0.5

        perf = store.get(new_bundle)
        if perf.count == 0:
            return 0.5  # No farm in this region runs the bundle

        # Positive when the bundle outperforms this farm.
        score_diff = compute_performance_score(perf, self.farmer) - self.own_score()
        return self._shrink_to_neutral(sigmoid(score_diff), perf.count)

    def compute_attitude_social_learning_country(self, new_bundle):
        """Compute attitude from COUNTRY-LEVEL bundle performance.

        Compares own performance against the aggregate of farmers in the same
        country running the proposed bundle. See _attitude_from_region for the
        shared arithmetic.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies
        - Yield gap analysis (van Ittersum et al. 2013): normalized comparisons

        Parameters
        ----------
        new_bundle : ManagementBundle
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        store = _country_performance_store(self.farmer.cell.country)
        return self._attitude_from_region(store, new_bundle)

    def compute_attitude_social_learning_cluster(self, new_bundle):
        """Compute attitude from AGROECOLOGICAL CLUSTER-level bundle performance.

        Cross-border social learning: farmers learn from countries with similar
        agroecological conditions, using cluster aggregates computed from peer
        countries. See _attitude_from_region for the shared arithmetic.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies
        - Yield gap analysis (van Ittersum et al. 2013): normalized comparisons

        Parameters
        ----------
        new_bundle : ManagementBundle
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        return self._attitude_from_region(
            self._cluster_performance_store(), new_bundle
        )

    # =========================================================================
    # TPB COMPONENT: SOCIAL NORM
    # =========================================================================

    def _social_norm_from_region(self, store, new_bundle, threshold_param):
        """Descriptive norm from a region's share of farmers running a bundle.

        Shared by the country and cluster channels: both ask what fraction of a
        population runs the bundle and compare it to the adoption level at which
        it starts to feel normative (Granovetter 1978). Thresholds are
        AFT-specific, pioneers responding to a smaller share, and are set higher
        than the local threshold because a regional statistic is more abstract
        than watching the neighbours.

        Unlike the local norm there is no similarity weighting, since a regional
        aggregate carries no information about who its farmers are.

        Parameters
        ----------
        store : RegionManagementPerformanceStore or None
            Aggregate performance and bundle counts for the region.
        new_bundle : ManagementBundle
            Bundle being evaluated.
        threshold_param : str
            Name of the AFT parameter holding this region's norm threshold.

        Returns
        -------
        float
            Social norm score in [0, 1], neutral when the region says nothing.
        """
        # A single farm is its own region, and cannot tell anyone what is normal.
        if store is None or store.total_farmers < 2:
            return 0.5

        uptake = store.bundle_counts.get(new_bundle, 0) / store.total_farmers
        return sigmoid(uptake - self.get_aft_param(threshold_param))

    def compute_social_norm_local(self, new_bundle):
        """Compute social norm based on LOCAL neighbourhood practice distribution.

        Social norm reflects "what others are doing" (descriptive norm), so the
        quantity has to be local uptake of the bundle, on the same share-of-
        neighbours scale the AFT threshold is calibrated for. Bundle similarity
        supplies that uptake in graded form, since a neighbour already running
        no-till but no cover crop counts partially towards the full bundle
        feeling normal.

        Crop similarity enters as the WEIGHT on each neighbour rather than as a
        second addend: it says whose behaviour is comparable, because a wheat
        grower learns little from a rice grower's choices. Added instead of
        weighted it would substitute for uptake, letting a farmer whose
        neighbours merely share a crop feel more normed towards conservation
        agriculture than one who has neighbours actually practising it.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        if not self.farmer.neighbourhood:
            return 0.5  # Neutral without neighbours

        # Graded local uptake of the bundle, weighted by whose practice is
        # comparable. Neighbours growing a different dominant crop carry zero
        # weight, matching how they are skipped in social learning.
        weighted_uptake = 0.0
        total_weight = 0.0
        for neighbour in self.farmer.neighbourhood:
            relevance = self.crop_similarity(neighbour)
            if relevance == 0:
                continue
            uptake = new_bundle.similarity(neighbour.behaviour.practice_bundle)
            weighted_uptake += relevance * uptake
            total_weight += relevance

        if total_weight == 0:
            return 0.5  # No comparable neighbours, so the norm says nothing

        local_uptake = weighted_uptake / total_weight

        # Shifted sigmoid: adoption threshold acts as the neutrality point
        # (Granovetter 1978 heterogeneous-threshold diffusion model).
        # Below threshold -> drag, above -> boost, smoothly transitioning.
        # Threshold is AFT-specific: pioneers feel "normed" at lower adoption,
        # traditionalists require broader local uptake to feel normative.
        threshold = self.get_aft_param("threshold_social_norm_local")
        return sigmoid(local_uptake - threshold)

    def compute_social_norm_country(self, new_bundle):
        """Compute social norm based on COUNTRY-LEVEL practice distribution.

        Uses cached country statistics for O(1) lookup. This reflects
        "what farmers in my country are doing" as a broader social influence.
        Country stats are already merged with neighbouring countries
        (configured via neighbour_country_weight) at the country level.

        Follows the simple approach from tillage_farmer.py: fraction of
        farmers using the practice → sigmoid transformation.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        return self._social_norm_from_region(
            _country_performance_store(self.farmer.cell.country),
            new_bundle,
            "threshold_social_norm_country",
        )

    def compute_social_norm_cluster(self, new_bundle):
        """Compute social norm from AGROECOLOGICAL CLUSTER-level practice distribution.

        Tele-coupled social norm: farmers are influenced by adoption patterns
        in countries with similar agroecological conditions. This captures how
        agricultural practices diffuse across countries within similar
        agro-ecological zones.

        Uses aggregated cluster statistics from world.statistic (computed from
        peer countries in the same agroecological cluster).

        Parameters
        ----------
        new_bundle : ManagementBundle
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        return self._social_norm_from_region(
            self._cluster_performance_store(),
            new_bundle,
            "threshold_social_norm_cluster",
        )

    # =========================================================================
    # AFFORDABILITY ADJUSTMENTMENT BASED ON CAPITAL
    # =========================================================================

    def affordable_bundle(self, target_bundle):
        """Find affordable subset of target bundle.

        If farmer can't afford full target bundle, add changes cheapest-first
        to maximize what can be adopted within capital constraints.

        Returns the target untouched when farm_economics.capital_constraints is
        off: the transition cost is then assumed financeable, and whether it is
        worth financing is PBC's question, not this one.

        See farmer.get_practice_transition_cost() for per-practice cost logic.

        Parameters
        ----------
        target_bundle : Bundle
            Desired practice bundle.

        Returns
        -------
        Bundle
            Affordable bundle (may equal current if nothing affordable).
        """
        if not self.farmer.capital_constrained:
            return target_bundle

        current = self.practice_bundle
        total_cost = self.farmer.get_bundle_transition_costs(
            self.practice_bundle,
            target_bundle,
        )

        # Check if full target is affordable
        if self.farmer.capital >= total_cost:
            return target_bundle

        # Build affordable subset: add changes cheapest-first
        changes = []
        for field in PRACTICE_FIELDS:
            old_val = getattr(current, field)
            new_val = getattr(target_bundle, field)

            if old_val == new_val:
                continue

            cost = self.farmer.get_practice_transition_cost(field, old_val, new_val)
            changes.append((field, new_val, cost))

        # Sort by cost (cheapest first)
        changes.sort(key=lambda x: x[2])

        # Greedily add affordable changes
        result = current
        remaining_capital = self.farmer.capital

        for field, new_val, cost in changes:
            if remaining_capital >= cost:
                result = result.change_practices(**{field: new_val})
                remaining_capital -= cost

        return result

    # =========================================================================
    # TPB COMPONENT: PERCEIVED BEHAVIORAL CONTROL (PBC)
    # =========================================================================

    def anticipated_relative_yield_change(
        self, current_bundle, target_bundle, min_peers=3
    ):
        """Expected fractional yield change from switching bundles (0.1 = +10%).

        The yield case of ``anticipated_change``, kept as its own name because
        it is what PBC asks for: the revenue side of the switch.
        """
        return self.anticipated_change(
            "yield", current_bundle, target_bundle, min_peers=min_peers
        )

    def anticipated_change(
        self, metric, current_bundle, target_bundle, min_peers=3
    ):
        """Expected fractional change in one metric from switching bundles.

        From peers that have run *both* bundles: within-peer difference of the
        remembered rates of ``metric``, then experience-weighted median.
        Within-peer differencing removes site confounding (cohort averages would
        compare farms, not practices). Search order: local then country
        (homophily; first level with ``min_peers`` dual-experience farms).

        The same estimator serves every metric because they are all stored the
        same way, as self-referenced log deviations. What differs is what the
        answer is used for: yield feeds PBC as a revenue change, and a priced
        metric would need converting to its physical unit first, which a log
        deviation cannot do on its own (see ``metrics.MetricSpec``).

        Feeds PBC only (financial feasibility). Desirability uses social
        learning; routing the same peer evidence through both would double-count.

        Notes
        -----
        Remembered levels are log deviations, so differences are log ratios;
        ``expm1`` converts to a fraction. Aggregation in log space is the
        geometric mean of peer ratios. Experience weight =
        ``min(dur_current, dur_target, max_experience_years)`` — soil benefits
        saturate ~15 yr; longer records add climate/LU confounding.

        Parameters
        ----------
        metric : str
            Key in ``metrics.METRICS``.
        current_bundle, target_bundle : ManagementBundle
        min_peers : int
            Minimum dual-experience peers before a spatial level is used.

        Returns
        -------
        float
            Fractional change in the metric, or 0.0 if insufficient peers.
        """
        if current_bundle == target_bundle:
            return 0.0

        # Peers are read at the cover crop this farm would grow, not the one
        # they grew. A legume's record says nothing about a catch crop.
        return self.anticipated_change_between(
            metric,
            self.performance_memory.key(current_bundle),
            self.performance_memory.key(target_bundle),
            min_peers,
        )

    def anticipated_change_between(self, metric, current_key, target_key, min_peers=3):
        """The estimator proper, on two explicit ``(bundle, type)`` keys.

        Separated from ``anticipated_change`` because not every comparison the
        model makes is between two bundles. Choosing which cover crop to sow
        holds the bundle still and moves only the type, and that comparison
        needs the same within-peer differencing, the same experience weighting
        and the same country cache as any other.

        Under ``tpb.land_similarity.enabled`` each peer is additionally
        weighted by how well its land stands in for this farm's. Without that
        the estimate is a property of the pool and not of the asker: peers are
        differenced within themselves, which removes site confounding from each
        peer's own record, but nothing then asks whether that site resembles the
        receiving one. A farm whose soil already mineralises plenty of nitrogen,
        where a legume's fixation is worth least, reads the same fraction as a
        farm where it is worth most.

        Parameters
        ----------
        metric : str
            Key in ``metrics.METRICS``.
        current_key, target_key : tuple
            ``(bundle, cover_crop_type)`` pairs to difference between.
        min_peers : int
            Minimum dual-experience peers before a spatial level is used.

        Returns
        -------
        float
            Fractional change in the metric, or 0.0 if insufficient peers.
        """
        if current_key == target_key:
            return 0.0

        effect = self._peer_effect(metric, current_key, target_key, min_peers)
        return 0.0 if effect is None else float(np.expm1(effect))

    def anticipated_trend_change(
        self, metric, current_bundle, target_bundle, min_peers=3
    ):
        """Expected change in one metric's yearly trend from switching bundles.

        The same peers, weights and search order as ``anticipated_change``,
        differencing each peer's remembered trend on the two bundles instead of
        its remembered level. A level answers how far above its own baseline a
        practice holds a farm; a trend answers which way the metric is moving
        under it, and the two can disagree in sign. On land losing soil carbon
        under every management, a peer that slowed the loss from -8 to -2
        gC/m2/yr still sits lower in its second episode than in its first, so
        its level difference reads as a loss while its trend difference reads
        as the 6 gC/m2/yr gained against the old path - which is what a
        payment on the carbon flow is written on.

        Reporting noise is not applied: it is scaled on the interannual spread
        of levels and has no counterpart for a slope.

        Returns
        -------
        float
            Change in the trend in log deviation per year, or 0.0 if
            insufficient peers.
        """
        if current_bundle == target_bundle:
            return 0.0

        effect = self._peer_effect(
            metric,
            self.performance_memory.key(current_bundle),
            self.performance_memory.key(target_bundle),
            min_peers,
            quantity="trend",
        )
        return 0.0 if effect is None else float(effect)

    def _peer_effect(
        self, metric, current_key, target_key, min_peers, quantity="level"
    ):
        """Peer median of ``quantity`` differences in log space, or None."""
        spec = METRICS[metric]
        enable_local, enable_country, _ = self._get_enabled_spreading_levels()
        tpb = self.farmer.model.config.coupled_config.tpb
        max_experience = tpb.max_experience_years
        land = self._land_similarity_config()
        reporting = (
            ReportingNoise.from_config(tpb) if quantity == "level" else None
        )
        effect = None

        # Local first: closer climates / soils transfer better
        if enable_local and self.farmer.neighbourhood:
            neighbours = [f for f in self.farmer.neighbourhood if f is not self.farmer]
            _, deltas, weights, peers = self._collect_effects(
                metric,
                neighbours,
                current_key,
                target_key,
                max_experience,
                reporting=reporting,
                quantity=quantity,
            )
            if len(deltas) >= min_peers:
                # min_peers is counted before similarity weighting, so turning
                # the weighting on cannot change which spatial level is used:
                # it reweights the pool, it does not shrink it.
                effect = self._reduce_effects(
                    spec,
                    deltas,
                    weights,
                    similarity=self._similarity_weights(peers, land),
                )

        if effect is None and enable_country:
            index, deltas, weights, peers, reduced = self._country_effects(
                metric, current_key, target_key, max_experience, quantity
            )
            own = index.get(id(self.farmer))
            if len(index) - (1 if own is not None else 0) >= min_peers:
                # Cached country median includes self if dual-experienced;
                # drop own index so the farm does not learn from itself.
                if own is None and land is None:
                    effect = reduced
                else:
                    effect = self._reduce_effects(
                        spec,
                        deltas,
                        weights,
                        skip=own,
                        similarity=self._similarity_weights(peers, land),
                    )

        return effect

    def _similarity_weights(self, peers, settings):
        """Per-peer land similarity, or ``None`` when the weighting is off.

        ``None`` rather than a vector of ones so that the unweighted path stays
        bit-identical to the model as it was specified before this setting
        existed.
        """
        if settings is None:
            return None
        return [self.land_similarity(peer, settings) for peer in peers]

    def anticipated_absolute_change(self, metric, current_bundle, target_bundle):
        """Expected change in one metric in its own physical unit.

        The relative estimate scaled by what the farm actually has::

            delta_absolute = baseline_physical * expm1(delta_log)

        An instrument is owed a quantity, not a ratio: a carbon payment is on
        tonnes and a nitrogen charge on kilograms, and the same 5% means a
        different amount on a carbon-rich soil than on a poor one. Converting
        here rather than at each call site keeps the baseline and the deviation
        from being paired up differently in three places.

        The conversion is against the farm's own baseline because that is what
        the deviation was taken about. It is a first-order estimate: the peer
        median is a ratio observed on other farms, applied to this farm's stock.

        Parameters
        ----------
        metric : str
            Key in ``metrics.METRICS``. Must be one with a physical counterpart.
        current_bundle, target_bundle : ManagementBundle

        Returns
        -------
        float
            Change in ``spec.physical_unit``, positive meaning more of it.

        Raises
        ------
        ValueError
            If the metric has no physical counterpart to be measured in.
        """
        spec = METRICS[metric]
        if not spec.priceable:
            raise ValueError(
                f"metric {metric!r} has no physical counterpart, so there is "
                "nothing to express an absolute change in. Give it a "
                "physical_attr in metrics.METRICS first."
            )

        relative = self.anticipated_change(metric, current_bundle, target_bundle)
        baseline = getattr(self.farmer, spec.physical_baseline_attr)
        return float(baseline) * relative

    def anticipated_carbon_credit_change(self, new_bundle):
        """Expected change in the yearly carbon credit from a switch, gC/m2/yr.

        The payment is settled on the slope of the farm's own soil carbon, so
        that is what the expectation is formed on: the farm's current slope,
        moved by however much peers' slopes moved when they made the same
        switch (``anticipated_trend_change``), credited on the instrument's own
        basis by ``creditable_soilc_accumulation_at``, less what the farm is
        credited on its current slope::

            s_after = s_own + delta_trend
            gain    = credit(baseline * expm1(s_after), new_bundle)
                      - credit(baseline * expm1(s_own), current_bundle)

        Under ``stock_gain`` this pays only where the stock would actually
        rise; under an avoided-loss basis it pays for the departure from the
        counterfactual slope, which for a farm still on conventional practice
        is its own current slope - so slowing a loss reads as the credit it
        will settle as.

        Reading the peers' level difference instead, as this once did, priced
        neither basis. On land losing carbon a peer that slowed the loss still
        holds less carbon in its second episode than in its first, so the level
        difference reads as a loss and the farm expected nothing even where the
        avoided-loss scheme pays.

        The slope a farm is credited on is fitted over its current episode, so
        a new practice draws little for its first couple of years. That lag is
        not discounted out here.
        """
        farmer = self.farmer
        own = self.performance_tracker.trend["soilc"]
        delta = self.anticipated_trend_change(
            "soilc", self.practice_bundle, new_bundle
        )
        baseline = farmer.baseline_soilc
        before = farmer.creditable_soilc_accumulation_at(
            float(baseline * np.expm1(own)), self.committed_bundle
        )
        after = farmer.creditable_soilc_accumulation_at(
            float(baseline * np.expm1(own + delta)), new_bundle
        )
        return after - before

    def anticipated_policy_transfer(self, new_bundle, benefit_years):
        """Cash a switch is expected to draw from policy, over the horizon.

        Positive is money in. Zero when no instrument is in force, which is why
        a run without one is unchanged by any of this.

        Without this the instruments would reach the farm only through the
        outturn, and a farm would have to adopt a practice before discovering
        it was being paid for it. Peers already know: they have been running it
        under the same instrument, and the estimator reads their record.

        Parameters
        ----------
        new_bundle : ManagementBundle
            The bundle being weighed.
        benefit_years : float
            Discounted years the farm projects the switch over.

        Returns
        -------
        float
            Expected net transfer over the planning horizon, USD per farm.
        """
        settings = self.farmer.policy
        if settings is None:
            return 0.0

        farm_size = self.farmer.farm_size
        transfer = 0.0

        # The two payments a no-stacking rule has to choose between, kept
        # apart until the end. The rule picks the larger payment, which is a
        # question about levels and not about the deltas they are used as:
        # the difference between two maxima is not the maximum of two
        # differences.
        carbon_gain = 0.0
        practice_delta = 0.0

        if settings.carbon_price:
            # Paid every year on the flow, so the horizon applies.
            carbon_gain = (
                self.anticipated_carbon_credit_change(new_bundle)
                * GC_M2_TO_TCO2_HA
                * settings.carbon_price
                * farm_size
                * benefit_years
            )

        if settings.nitrogen_charge:
            # Leaching is a flow, so this is a change in the annual loss and
            # the horizon does apply. A negative delta is less nitrogen lost,
            # so a smaller bill every year, so cash in.
            #
            # Priced as the difference between two bills rather than as the
            # delta times a rate, because an allowance makes the charge kinked:
            # a farm already under the threshold saves nothing by going lower,
            # and one above it saves only down as far as the threshold. Reading
            # the rate straight off would promise both of them money that the
            # tax office would never hand over.
            delta_flux = self.anticipated_absolute_change(
                "leaching", self.practice_bundle, new_bundle
            )
            saving = self.farmer.nitrogen_charge_for(
                self.farmer.leaching
            ) - self.farmer.nitrogen_charge_for(self.farmer.leaching + delta_flux)
            transfer += saving * benefit_years

        if (
            settings.cover_crop_subsidy
            or settings.residue_subsidy
            or settings.residue_subsidy_share
        ):
            # The one term here the farm does not need peers to tell it about:
            # the rate is posted, so the payment for a bundle is known exactly
            # rather than estimated from what neighbours have earned. A flow,
            # paid for every year the practice is held, so the horizon applies.
            practice_delta = (
                self.farmer.practice_subsidy_for(new_bundle)
                - self.farmer.practice_subsidy_for(self.practice_bundle)
            ) * benefit_years

        if not settings.no_stacking:
            # Both payments, summed, with the arithmetic left exactly as it
            # was so a run without the rule is unchanged to the bit.
            return transfer + carbon_gain + practice_delta

        # Only the larger is paid, so the gain is what the farm would draw
        # after the switch less what it draws now, each side resolved by the
        # rule. Levels as horizon totals, like the deltas above.
        carbon_now = self.farmer.carbon_payment * benefit_years
        practice_now = (
            self.farmer.practice_subsidy_for(self.practice_bundle) * benefit_years
        )
        after = sum(settings.unstacked(carbon_now + carbon_gain,
                                       practice_now + practice_delta))
        before = sum(settings.unstacked(carbon_now, practice_now))

        return transfer + after - before

    def report_error(self, metric, current_key, target_key, noise):
        """How this farm's account of a comparison departs from its own record.

        In the same log units as the delta it is added to, scaled on the farm's
        own interannual spread so that ``reporting_noise.sigma`` and
        ``perception_noise.sigma`` mean the same thing and can be read against
        each other: a reporting sigma of four says what a farm passes on is
        four times as far off as what it sees itself, which is roughly the
        ratio the measurement literature reports between instrumented
        comparison and self-reporting.

        The random part is drawn once per comparison and kept, so the farm is
        consistent from neighbour to neighbour and from year to year. The bias
        is not cached: it tracks whichever practice the farm runs *now*, so a
        farm that switches starts talking up the other one.
        """
        spread = self.performance_tracker.clean_interannual_std[metric]
        if spread <= 0.0:
            return 0.0

        scale = noise.observability.get(metric, 1.0)
        key = (metric, current_key, target_key)
        drawn = self._report_errors.get(key)
        if drawn is None:
            drawn = noise.sigma * scale * np.random.normal()
            self._report_errors[key] = drawn

        # Positive favours the target, negative the practice being left.
        favour = 0.0
        if target_key[0] == self.practice_bundle:
            favour = 1.0
        elif current_key[0] == self.practice_bundle:
            favour = -1.0

        return spread * (drawn + noise.bias * scale * favour)

    @staticmethod
    def _collect_effects(
        metric,
        farmers,
        current_key,
        target_key,
        max_experience,
        reporting=None,
        quantity="level",
    ):
        """Within-peer deltas in one metric between two episodes, with weights.

        ``current_key`` and ``target_key`` are ``(bundle, cover_crop_type)``
        pairs resolved by the *asking* farm, so a peer only contributes where it
        has run the same practice this farm is weighing, cover crop and all.

        Skips peers missing either episode. Weight =
        ``min(duration_current, duration_target, max_experience)``: thinner
        record limits the comparison; ``max_experience_years`` caps so longer
        histories do not overweight climate/LU confounding after yield/soil
        effects saturate (~15 yr).

        Returns
        -------
        contributor_index : dict
            ``id(farmer) →`` index in deltas/weights (for dropping self).
        deltas, weights : list
            Log-rate differences and experience weights.
        peers : list
            The contributing farmers, in the order of ``deltas``, so a caller
            can weight them by how well their land stands in for its own. The
            scan itself is independent of who is asking, which is what lets the
            country level stay cached while the weighting does not.

        With ``reporting``, each delta is what the peer says rather than what
        its record holds. That still leaves the scan independent of who is
        asking - the distortion belongs to the peer telling the story, not to
        whoever is listening - so the country cache remains valid.

        ``quantity`` names the episode attribute differenced, ``"level"`` or
        ``"trend"``.
        """
        contributor_index = {}
        deltas = []
        weights = []
        peers = []

        for f in farmers:
            if not hasattr(f, "behaviour") or not hasattr(
                f.behaviour, "performance_memory"
            ):
                continue

            mem = f.behaviour.performance_memory
            current_exp = mem.get(current_key)
            target_exp = mem.get(target_key)

            # Both must have experience (duration > 0)
            if current_exp.duration <= 0 or target_exp.duration <= 0:
                continue

            contributor_index[id(f)] = len(deltas)
            delta = (
                getattr(target_exp, quantity)[metric]
                - getattr(current_exp, quantity)[metric]
            )
            if reporting is not None:
                delta += f.behaviour.report_error(
                    metric, current_key, target_key, reporting
                )
            deltas.append(delta)
            weights.append(
                min(current_exp.duration, target_exp.duration, max_experience)
            )
            peers.append(f)

        return contributor_index, deltas, weights, peers

    @staticmethod
    def _reduce_effects(spec, deltas, weights, skip=None, similarity=None):
        """Experience-weighted median of the per-peer differences.

        Median rather than mean because the pool can be as small as three peers,
        where one distorted memory would otherwise set the estimate on its own;
        clipping to the metric's bounds guards these rates as the region
        aggregates do. Weighting counts peers who have run the practice long
        enough for its benefits to show for more than recent converts; see
        _collect_effects.

        ``similarity``, when given, multiplies each experience weight by how
        well that peer's land stands in for the asking farm's, so a long record
        on land that does not resemble this farm's no longer outvotes a shorter
        one that does. It scales the existing weight rather than replacing it:
        both questions, how good the record is and how relevant it is, bear on
        the same estimate.
        """
        if similarity is not None:
            weights = [w * s for w, s in zip(weights, similarity)]

        if skip is not None:
            deltas = [v for i, v in enumerate(deltas) if i != skip]
            weights = [w for i, w in enumerate(weights) if i != skip]

        lo, hi = spec.clip_bounds
        return _weighted_median(np.clip(deltas, lo, hi), weights)

    def _country_effects(
        self, metric, current_key, target_key, max_experience, quantity="level"
    ):
        """Country-wide effects on one metric this year, computed once and shared.

        Every farmer in a country scans the same peers and reaches the same
        answer, apart from leaving itself out, so the scan is done once and
        shared. This is purely a saving: the estimate is the same over the same
        peers, and a contributing farmer is still removed from its own. Early in
        a run it is the dominant cost, when local peers are too few and the
        country level is always the one taken.

        The metric, the cover crop types and the experience cap are all in the
        key. The metric because the same bundle pair has a different answer for
        each - a carbon estimate reading the yield entry would be wrong and
        would look right. The types because two farms weighing the same bundle
        pair with different cover crops in mind are reading different peers.
        The cap because it is one value for all farmer types today, and a
        per-AFT cap would otherwise silently share one type's answer with
        another. The quantity because a level difference and a trend difference
        on the same pair are different answers.

        What is cached is the scan, not the answer. ``reduced`` is the estimate
        every farm would reach from an unweighted pool and is still shared; with
        land similarity on, the deltas and the peers behind them are what get
        re-used, and each farm reduces them against its own land.
        """
        country = self.farmer.cell.country
        year = self.farmer.model.lpjml.sim_year

        # Memories move every year, so last year's scan is stale.
        if getattr(country, "_effect_cache_year", None) != year:
            country._effect_cache = {}
            country._effect_cache_year = year

        key = (metric, current_key, target_key, max_experience, quantity)
        entry = country._effect_cache.get(key)
        if entry is None:
            reporting = None
            if quantity == "level":
                reporting = ReportingNoise.from_config(
                    self.farmer.model.config.coupled_config.tpb
                )
            index, deltas, weights, peers = self._collect_effects(
                metric,
                country.farmers,
                current_key,
                target_key,
                max_experience,
                reporting=reporting,
                quantity=quantity,
            )
            spec = METRICS[metric]
            reduced = self._reduce_effects(spec, deltas, weights) if deltas else None
            entry = (index, deltas, weights, peers, reduced)
            country._effect_cache[key] = entry

        return entry

    def pbc_for_bundle(self, new_bundle):
        """Compute Perceived Behavioural Control for a bundle.

        PBC answers "can I actually do this?", and falls as the net financial
        impact of the switch grows against the income available to absorb it::

            PBC   = pbc_base * 2 * sigmoid(-ratio),  capped at 1
            ratio = net_financial_impact / (baseline_revenue * benefit_years)

        The net impact is the one-time transition cost, plus the change in annual
        direct costs and minus the anticipated revenue gain, the latter two over
        the discounted benefit-years the farmer projects. It is compared against
        income over that same view, so the ratio is dimensionless and answers
        whether the switch is affordable across the period it pays back over.

        2 * sigmoid(-ratio) is 1 - tanh(ratio), which anchors a cost-neutral
        switch at the AFT base and lets the base move both ways: a costly switch
        (ratio >> 0) drives PBC to 0, a profitable one (ratio < 0) raises it
        above the base until the cap. It is bounded and monotone throughout, so
        a very large anticipated gain cannot flip its sign.

        pbc_base carries the difference in risk tolerance between farmer types,
        and is the only place that difference enters PBC.

        References
        ----------
        Li, X., Dai, J., Zhu, X., Li, J., He, J., Huang, Y., ... & Shen, Q. (2023).
        Mechanism of attitude, subjective norms, and perceived behavioral control
        influence the green development behavior of construction enterprises.
        Humanities and Social Sciences Communications, 10(1), 266.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            PBC score in [0, 1].
        """
        # Calculate cost impact of transitioning

        # One-time transition costs
        transition_costs = self.farmer.get_bundle_transition_costs(
            current_bundle=self.practice_bundle,
            target_bundle=new_bundle,
        )

        # Change in annual direct costs (new bundle vs current bundle)
        # Positive = new bundle costs more, Negative = new bundle saves money.
        # Measured against the costs the farm actually bears, so a farm holding
        # a residue label it never bought sees the true added cost of committing.
        delta_direct_costs = (
            self.farmer.get_bundle_direct_costs(bundle=new_bundle)
            - self.farmer.get_bundle_direct_costs(bundle=self.committed_bundle)
        )

        # Calculate anticipated revenue change from yield difference
        # Peer experience gives the anticipated yield change as a fraction of
        # own yield. Applying that fraction to baseline revenue turns it into a
        # whole-farm USD change on the SAME footing as baseline_revenue and the
        # cost terms below, and implicitly carries the farmer's real crop/price
        # mix and area (already embedded in baseline_revenue), avoiding a
        # separate gC->tonnes and area conversion and a plain price average.
        relative_yield_change = self.anticipated_relative_yield_change(
            self.practice_bundle, new_bundle
        )
        delta_revenue = relative_yield_change * self.farmer.baseline_revenue  # $/farm/yr

        # -----------------------------------------------------------------
        # Total financial impact over planning horizon
        # -----------------------------------------------------------------

        # Positive = net outflow (costs exceed revenue gain)
        # Negative = net inflow (revenue gain exceeds costs)
        # Discounted benefit-years, not the re-evaluation cadence: the recurring
        # cost and the revenue gain are both annual streams over the planning
        # view, while the transition cost is paid once and is not discounted.
        benefit_years = self.projection_weight()
        net_financial_impact = (
            transition_costs
            + delta_direct_costs * benefit_years
            - delta_revenue * benefit_years  # Revenue is subtracted (inflow)
            - self.anticipated_policy_transfer(new_bundle, benefit_years)
        )

        # -----------------------------------------------------------------
        # PBC: smooth, bounded affordability factor from financial impact
        # -----------------------------------------------------------------

        # Numerator and denominator both scale with benefit_years, so the
        # horizon cancels out of the recurring terms and survives only on the
        # one-time transition cost. A longer view amortises the switch over more
        # years rather than inflating the future gain. The floor of 1.0 guards
        # against a cell with no crops and hence no baseline revenue.
        revenue_over_horizon = max(self.farmer.baseline_revenue * benefit_years, 1.0)
        ratio = net_financial_impact / revenue_over_horizon
        financial_factor = 2.0 * sigmoid(-ratio)

        return max(0.0, min(1.0, self.farmer.pbc_base * financial_factor))  # Cap to [0, 1]


    # =========================================================================
    # COMPUTE FULL TPB SCORE
    # =========================================================================

    def compute_tpb_for_bundle(self, new_bundle):
        """Compute all TPB components and overall intention for a bundle.

        TPB formula (multiplicative):
            TPB = (w_att × Attitude + w_norm × SocialNorm) × PBC

        This means PBC acts as a gate: low PBC blocks adoption regardless
        of positive attitude/norms.

        Components are computed at local (neighbour), country, and agroecological
        cluster levels, then combined with configurable weights. Levels can be
        disabled via config (tpb), and their weights are redistributed
        to enabled levels.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.
        """
        # -----------------------------------------------------------------
        # Get spreading level config (which levels are enabled)
        # -----------------------------------------------------------------
        enable_local, enable_country, enable_cluster = self._get_enabled_spreading_levels()

        # -----------------------------------------------------------------
        # Redistribute weights for attitude social learning
        # -----------------------------------------------------------------
        raw_att_local = self.farmer.weight_attitude_local
        raw_att_country = self.farmer.weight_attitude_country
        raw_att_cluster = self.farmer.weight_attitude_cluster

        att_w_local, att_w_country, att_w_cluster = redistribute_weights(
            raw_att_local, raw_att_country, raw_att_cluster,
            enable_local, enable_country, enable_cluster,
        )

        # -----------------------------------------------------------------
        # Redistribute weights for social norm
        # -----------------------------------------------------------------
        raw_norm_local = self.farmer.weight_social_norm_local
        raw_norm_country = self.farmer.weight_social_norm_country
        raw_norm_cluster = self.farmer.weight_social_norm_cluster

        norm_w_local, norm_w_country, norm_w_cluster = redistribute_weights(
            raw_norm_local, raw_norm_country, raw_norm_cluster,
            enable_local, enable_country, enable_cluster,
        )

        # -----------------------------------------------------------------
        # Attitude: own experience + social learning (local + country + cluster)
        # -----------------------------------------------------------------
        # Own land attitude (always computed - it's your own observation)
        self._attitude_own_land = self.compute_attitude_own_land(new_bundle)

        # Social learning: compute only enabled levels
        self._attitude_social_learning_local = (
            self.compute_attitude_social_learning_local(new_bundle)
            if enable_local else 0.0
        )
        self._attitude_social_learning_country = (
            self.compute_attitude_social_learning_country(new_bundle)
            if enable_country else 0.0
        )
        self._attitude_social_learning_cluster = (
            self.compute_attitude_social_learning_cluster(new_bundle)
            if enable_cluster else 0.0
        )

        # Combine with redistributed weights
        self._attitude_social_learning = (
            att_w_local * self._attitude_social_learning_local
            + att_w_country * self._attitude_social_learning_country
            + att_w_cluster * self._attitude_social_learning_cluster
        )

        # Full attitude: own land + combined social learning
        self._attitude = (
            self.farmer.weight_own_land * self._attitude_own_land
            + self.farmer.weight_social_learning * self._attitude_social_learning
        )

        # -----------------------------------------------------------------
        # Social Norm: local (neighbours) + country-level + agroecological cluster
        # -----------------------------------------------------------------
        # Compute only enabled levels
        self._social_norm_local = (
            self.compute_social_norm_local(new_bundle)
            if enable_local else 0.0
        )
        self._social_norm_country = (
            self.compute_social_norm_country(new_bundle)
            if enable_country else 0.0
        )
        self._social_norm_cluster = (
            self.compute_social_norm_cluster(new_bundle)
            if enable_cluster else 0.0
        )

        # Combine with redistributed weights
        self._social_norm = (
            norm_w_local * self._social_norm_local
            + norm_w_country * self._social_norm_country
            + norm_w_cluster * self._social_norm_cluster
        )

        # -----------------------------------------------------------------
        # PBC: can I afford this?
        # -----------------------------------------------------------------
        self._pbc = self.pbc_for_bundle(new_bundle)

        # -----------------------------------------------------------------
        # TPB Intention: combine components
        # -----------------------------------------------------------------
        # Multiplicative formula: PBC gates the attitude/norm contribution
        self._tpb = (
            self.farmer.weight_attitude * self._attitude
            + self.farmer.weight_norm * self._social_norm
        ) * self._pbc
