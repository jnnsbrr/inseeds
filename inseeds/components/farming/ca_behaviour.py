"""Theory of Planned Behaviour (TPB) decision model for Conservation Agriculture.

This module implements how farmers decide whether to adopt or abandon
agricultural practices. It uses the Theory of Planned Behaviour framework
combined with social learning and adaptive management.

Overview
--------
Each year, farmers may reconsider their practices based on:
1. **Attitude**: "Do I think this practice is good?" (own experience + social learning)
2. **Social Norm**: "Are my neighbors doing it?" (local + country + cluster)
3. **Perceived Behavioral Control (PBC)**: "Can I afford it?"

These combine into a TPB intention score. If it exceeds a threshold, the
farmer transitions to the new practice.

Decision Flow
-------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    Annual Update                             │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Check if evaluation time (not every year)               │
    │  2. Check fallback (sustained decline → revert)             │
    │  3. Find target bundle:                                     │
    │     a. Local neighbor with better performance?              │
    │     b. Country-level best bundle?                           │
    │     c. Cluster-level best bundle?                           │
    │     d. Random exploration?                                  │
    │  4. Adjust for affordability                                │
    │  5. Compute TPB score                                       │
    │  6. If TPB > threshold → transition                         │
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
import logging

import numpy as np

from inseeds.components.farming.farmer import sigmoid, NON_CROPS

logger = logging.getLogger(__name__)
from inseeds.components.farming.ca_management import (
    ManagementBundle,
    ManagementPerformanceMemory,
    ManagementPerformanceTracker,
    PRACTICE_FIELDS,
    RegionManagementPerformanceStore,
)


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
BLOCKER_MIN_OBS_YEARS = 1           # Not enough observation years yet
BLOCKER_FALLBACK_TRIGGERED = 2      # Reverting to previous bundle (adaptive management)
BLOCKER_NO_TARGET = 3               # No better neighbour + exploration didn't trigger
BLOCKER_TARGET_SAME = 4             # Target bundle same as current (already optimal)
BLOCKER_TARGET_UNAFFORDABLE = 5     # Target reduced to current due to cost
BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND = 6         # TPB low - own land attitude is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL = 7     # TPB low - LOCAL social learning is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY = 14  # TPB low - COUNTRY social learning is limiting
BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER = 16  # TPB low - CLUSTER social learning is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL = 8         # TPB low - LOCAL social norm is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY = 15      # TPB low - COUNTRY social norm is limiting
BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER = 17      # TPB low - CLUSTER social norm is limiting
BLOCKER_TPB_LOW_PBC = 9             # TPB below threshold - PBC (cost affordability) is limiting
BLOCKER_TRANSITION_UNAFFORDABLE = 10 # Can't afford transition cost
BLOCKER_CAPITAL_SURVIVAL = 11       # Capital below survival threshold
BLOCKER_CONTROL_RUN = 12            # Control run - no CA dynamics
BLOCKER_EVALUATION_TIME = 13        # Not yet time to re-evaluate (commitment period)
BLOCKER_AFFORDABILITY_FORCED = 18   # Practices deselected due to unaffordable direct costs

BLOCKER_NAMES = {
    BLOCKER_NONE: "none",
    BLOCKER_MIN_OBS_YEARS: "min_obs_years",
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
    BLOCKER_EVALUATION_TIME: "evaluation_time",
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
            1 if farmer.cover_crop > 0 else 0, # TODO: change to cover_crop type: 0 = none, 1 = catch crop, 2 = legume
            1 if farmer.litter_cover >= ca_threshold else 0,
        )

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
        )

        # Initialize baseline score for fallback comparison
        if self.performance_tracker.n > 1:
            self.performance_tracker.baseline_score = (
                self.performance_tracker.weighted_score(self.farmer)
            )

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
    def proposed_bundle_label(self):
        """Proposed bundle as human-readable name."""
        return self.proposed_bundle.label if self.proposed_bundle else ""

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

    @property
    def transition_blocker_name(self):
        """Human-readable name of transition blocker."""
        return BLOCKER_NAMES.get(self.transition_blocker, "unknown")

    @property
    def transition_driver(self):
        """Primary reason why a transition succeeded (numeric code).
        
        See DRIVER_* constants for codes. 0 = no transition (maintaining current).
        This is the mirror of transition_blocker - identifies which pathway and
        TPB component enabled the transition when it succeeds.
        """
        return getattr(self, '_transition_driver', DRIVER_NONE)

    @property
    def transition_driver_name(self):
        """Human-readable name of transition driver."""
        return DRIVER_NAMES.get(self.transition_driver, "unknown")

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
                n_obs,
                current_year,
            )

        # -----------------------------------------------------------------
        # Prepare for new bundle: reset regression state
        # -----------------------------------------------------------------

        # Remember previous bundle (for potential fallback)
        self.previous_bundle = self.practice_bundle

        # Capture baseline score before resetting (for fallback comparison)
        # This is the performance level we expect to maintain or exceed
        baseline = (
            self.performance_tracker.weighted_score(self.farmer)
            if n_obs > 1 else 0.0
        )

        # Reset regression accumulators for new bundle
        self.performance_tracker = ManagementPerformanceTracker.reset_for_transition(
            farmer=self.farmer,
            current_year=current_year,
            baseline_score=baseline
        )

        # Transition to new bundle
        self.practice_bundle = new_bundle

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
        self._attitude = 0.0     # Attitude toward behavior (combined)
        self._attitude_own_land = 0.0    # Attitude from own land performance
        self._attitude_social_learning = 0.0  # Attitude from social learning
        self._social_norm = 0.0  # Subjective norm
        self._pbc = 0.0          # Perceived behavioral control

        # Evaluation time: farmers don't reconsider every year
        # Randomize initial evaluation time to desynchronize farmers
        # (avoids artificial waves of simultaneous evaluation)
        interval = self.farmer.model.config.coupled_config.tpb_thresholds.evaluation_interval
        self._years_until_evaluation = np.random.randint(0, interval)

    # -------------------------------------------------------------------------
    # Properties for external access to TPB components
    # -------------------------------------------------------------------------

    @property
    def tpb(self):
        """Overall TPB intention score."""
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
        the time period has ended (randomized around evaluation_interval).

        This saves computation and is more realistic - farmers commit to
        observing results before reconsidering.

        Returns
        -------
        bool
            True if farmer should run TPB evaluation this year.
        """
        return self._years_until_evaluation <= 0

    def reset_evaluation_time(self):
        """Reset evaluation time after a transition decision (transition or stay).

        Uses normal distribution around evaluation_interval (mean=interval,
        std=interval/2) to create heterogeneity in re-evaluation timing.
        This reflects that some farmers re-evaluate sooner (more proactive)
        while others wait longer (more conservative). Minimum is 1 year.

        Called after TPB evaluation completes, regardless of whether transition
        happened.
        """
        interval = self.farmer.model.config.coupled_config.tpb_thresholds.evaluation_interval
        if interval > 0:
            # Normal distribution around interval, minimum 1 year
            self._years_until_evaluation = max(
                1, int(np.random.normal(interval, interval / 2))
            )
        else:
            self._years_until_evaluation = 0

    def decrement_evaluation_time(self):
        """Decrement the evaluation time counter by one year.

        Called each year when farmer doesn't evaluate.
        """
        if self._years_until_evaluation > 0:
            self._years_until_evaluation -= 1

    # =========================================================================
    # MAIN UPDATE LOGIC
    # =========================================================================

    def reevaluate_residue_status(self):
        """Re-evaluate residue component of bundle based on actual litter cover.

        Farmers may organically cross the CA residue threshold (30% soil cover)
        through their management practices. This method "certifies" or
        "de-certifies" their CA residue status based on actual outcomes.

        - If litter_cover >= threshold AND bundle residue = 0 → upgrade to 1
        - If litter_cover < threshold AND bundle residue = 1 → downgrade to 0

        This ensures the bundle reflects actual field conditions, not just
        intended practices.
        """
        ca_threshold = self.farmer.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501
        current_residue = self.practice_bundle.residue_on_field
        litter_cover = self.farmer.litter_cover

        # Check if status should change
        if litter_cover >= ca_threshold and current_residue == 0:
            self.practice_bundle = self.practice_bundle.change_practices(
                residue_on_field=1
            )
        elif litter_cover < ca_threshold and current_residue == 1:
            self.practice_bundle = self.practice_bundle.change_practices(
                residue_on_field=0
            )

    def update(self):
        """Compute proposed bundle and TPB scores for this timestep.

        Decision flow:
        0. Re-evaluate residue status based on actual litter cover
        1. Add current observation to regression accumulators
        2. Update performance memory with current trends (for neighbour visibility)
        3. Decay old memories (bounded rationality)
        4. Check if minimum observation period has passed
        5. Check fallback condition (sustained decline → revert)
        6. Find best-performing neighbour's bundle OR explore randomly
        7. Adjust target bundle for affordability
        8. Compute TPB scores for the proposed bundle
        
        Sets _transition_blocker to indicate why transition didn't happen (if applicable).
        Sets _transition_driver to indicate why transition succeeded (if applicable).
        """
        # Reset blocker, driver, and pathway at start of each update
        self._transition_blocker = BLOCKER_NONE
        self._transition_driver = DRIVER_NONE
        self.target_pathway = None

        # -----------------------------------------------------------------
        # Step 0: Re-evaluate residue status based on actual litter cover
        # -----------------------------------------------------------------
        # Farmers may cross CA threshold organically - certify/de-certify
        self.reevaluate_residue_status()

        # -----------------------------------------------------------------
        # Step 1: Add current year's observation to regression
        # -----------------------------------------------------------------
        self.performance_tracker.add_observation(
            self.farmer.soilc,
            self.farmer.root_moisture,
            self.farmer.cropyield,
            self.farmer.model.lpjml.sim_year,
        )

        # -----------------------------------------------------------------
        # Step 2: Update performance memory with current trends
        # -----------------------------------------------------------------
        # Keep performance memory up-to-date so neighbours see current performance
        self.performance_memory.update_current(
            self.practice_bundle,
            self.performance_tracker.trend,
            self.performance_tracker.n,
            self.farmer.model.lpjml.sim_year,
        )

        # -----------------------------------------------------------------
        # Step 3: Decay old memories
        # -----------------------------------------------------------------
        self.performance_memory.decay(
            self.farmer.model.lpjml.sim_year,
            self.get_aft_param("memory_decay_years"),
        )

        # -----------------------------------------------------------------
        # Step 4: Require minimum observation years before transitioning
        # -----------------------------------------------------------------
        # Avoids noisy decisions based on single-year fluctuations
        # Typical value: 3 years (allows trends to stabilize)
        # With historic data initialization, farmers start with enough history

        n_obs = self.performance_tracker.n
        min_obs = self.get_aft_param("min_observation_years")

        if n_obs < min_obs:
            self._tpb = 0.0
            self.proposed_bundle = None
            self._transition_blocker = BLOCKER_MIN_OBS_YEARS
            return

        # -----------------------------------------------------------------
        # Step 5: Check fallback condition
        # -----------------------------------------------------------------
        # If performance has declined for FALLBACK_YEARS consecutive years,
        # propose reverting to the previous bundle (adaptive management)

        if self.check_fallback():
            # Fallback sets proposed_bundle and _tpb internally
            # Blocker will be set by should_transition() if TPB too low
            self._transition_blocker = BLOCKER_FALLBACK_TRIGGERED
            self.target_pathway = "fallback"  # Track pathway for transition_driver
            return

        # -----------------------------------------------------------------
        # Step 6: Find target bundle (neighbour imitation or exploration)
        # -----------------------------------------------------------------

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
            self._transition_blocker = BLOCKER_NO_TARGET
            return

        if target_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self._transition_blocker = BLOCKER_TARGET_SAME
            return

        # -----------------------------------------------------------------
        # Step 7: Adjust for affordability
        # -----------------------------------------------------------------
        # If farmer can't afford full target bundle, find affordable subset

        affordable_bundle = self.affordable_bundle(target_bundle)

        # If affordable bundle is same as current, no change
        if affordable_bundle == self.practice_bundle:
            self._tpb = 0.0
            self.proposed_bundle = None
            self._transition_blocker = BLOCKER_TARGET_UNAFFORDABLE
            return

        # -----------------------------------------------------------------
        # Step 8: Compute TPB scores for proposed bundle
        # -----------------------------------------------------------------
        self.proposed_bundle = affordable_bundle
        self.compute_tpb_for_bundle(affordable_bundle)

    # =========================================================================
    # FALLBACK MECHANISM (Adaptive Management)
    # =========================================================================

    def check_fallback(self):
        """Check if farmer should revert to previous bundle due to sustained decline.

        Implements adaptive management: if outcomes have declined for
        fallback_years consecutive years, propose reverting to the
        previous practice bundle.

        Returns
        -------
        bool
            True if fallback is triggered (sets proposed_bundle internally).
        """
        # Can't fall back if no previous bundle
        if self.previous_bundle is None:
            return False

        # Allow grace period for new practices to show effects
        # Grace period = min_observation_years (reuse existing param)
        n_obs = self.performance_tracker.n
        grace_period = self.get_aft_param("min_observation_years")
        if n_obs < grace_period:
            return False

        # -----------------------------------------------------------------
        # Compare current performance to baseline at transition time
        # -----------------------------------------------------------------
        # baseline_score captures the trend score at the time of transition.
        # If current score is worse than baseline, we're declining.
        baseline = self.performance_tracker.baseline_score
        current_score = self.performance_tracker.weighted_score(self.farmer)

        # Track consecutive years of decline
        if current_score < baseline:
            self.decline_years += 1
        else:
            self.decline_years = 0  # Reset if performance improves

        # -----------------------------------------------------------------
        # Trigger fallback after sustained decline
        # -----------------------------------------------------------------
        fallback_years = self.get_aft_param("fallback_years")
        if self.decline_years >= fallback_years:
            # Mark current bundle as "failed" (reduces future exploration)
            self.performance_memory.record_failure(self.practice_bundle)

            # Propose reverting to previous bundle
            self.proposed_bundle = self.previous_bundle

            # Fallback bypasses TPB: this is an emergency response, not a
            # planned behavior change. Set intention directly to 1.0.
            # TPB factors are left unchanged (not meaningful for fallback).
            self._tpb = 1.0

            return True

        return False

    # =========================================================================
    # RANDOM EXPLORATION (Innovation Diffusion)
    # =========================================================================

    def maybe_explore_bundle(self):
        """Randomly explore a new bundle (innovation diffusion).

        Some farmers try new practices without neighbour influence:
        - "Pioneers" explore more frequently
        - Poor performers explore more (searching for better options)
        - Longer experience increases willingness to try new things
        - Bundles that failed before are avoided

        Returns
        -------
        tuple or None
            Random valid bundle to try, or None if no exploration.
        """
        # -----------------------------------------------------------------
        # Determine exploration probability
        # -----------------------------------------------------------------

        # Base probability depends on farmer type (from config)
        # Pioneers (innovators) are more willing to experiment
        base_prob = self.get_aft_param("exploration_base_prob")

        # Poor performers explore more (searching for better options)
        # With relative trends: negative = declining, positive = improving
        # Threshold is in relative terms (e.g., 0 = any decline, -0.02 = >2% decline)
        current_score = self.performance_tracker.weighted_score(self.farmer)
        poor_performance_threshold = self.get_aft_param("poor_performance_threshold")
        poor_performance_multiplier = self.get_aft_param("poor_performance_multiplier")
        if current_score < poor_performance_threshold:
            base_prob *= poor_performance_multiplier

        # Experience affects willingness to explore (smooth ramp based on confidence_years)
        # Early: 0.5x exploration (cautious), Experienced: 1.5x exploration (confident)
        n_obs = self.performance_tracker.n
        confidence_years = self.get_aft_param("confidence_years")
        experience_factor = min(1.0, n_obs / confidence_years)
        exploration_modifier = 0.5 + experience_factor * 1.0  # Ramps from 0.5 to 1.5
        base_prob *= exploration_modifier

        # Cap exploration probability (configurable)
        max_exploration_prob = self.get_aft_param("max_exploration_prob")
        explore_prob = min(base_prob, max_exploration_prob)

        # -----------------------------------------------------------------
        # Random draw: explore or not?
        # -----------------------------------------------------------------
        if np.random.random() > explore_prob:
            return None  # No exploration this year

        # -----------------------------------------------------------------
        # Select valid bundle to explore
        # -----------------------------------------------------------------
        exploration = self.farmer.model.config.coupled_config.exploration
        max_failures = getattr(exploration, "max_failures", 2)

        # Generate all possible bundles
        valid = [
            b for b in ManagementBundle.all_bundles()
            if b != self.practice_bundle
            and self.performance_memory[b].failure_count < max_failures
        ]

        if not valid:
            return None

        # Random selection from valid bundles
        return valid[np.random.randint(len(valid))]

    # =========================================================================
    # TRANSITION DECISION
    # =========================================================================

    def should_transition(self):
        """Determine if farmer should transition to proposed bundle.

        Compares TPB intention score to threshold. Higher threshold
        when reverting (to avoid oscillation).

        Thresholds are non-AFT-specific (from config.tpb_thresholds).
        Behavioral differences between AFTs come from weights and PBC.

        Returns
        -------
        bool
            True if TPB exceeds threshold and transition should occur.
        """
        # Get thresholds from config (non-AFT-specific)
        tpb_config = self.farmer.model.config.coupled_config.tpb_thresholds
        transition_threshold = tpb_config.transition_threshold
        revert_threshold = tpb_config.revert_threshold

        # Higher threshold for reverting (avoid flip-flopping)
        if self.proposed_bundle == self.previous_bundle:
            threshold = revert_threshold
        else:
            threshold = transition_threshold

        return self._tpb > threshold

    def _get_enabled_spreading_levels(self):
        """Get which spreading levels are enabled from config.

        Returns
        -------
        tuple[bool, bool, bool]
            (enable_local, enable_country, enable_cluster)
        """
        spreading_config = getattr(
            self.farmer.model.config.coupled_config, "spreading_levels", None
        )
        enable_local = getattr(spreading_config, "enable_local", True) if spreading_config else True
        enable_country = getattr(spreading_config, "enable_country", True) if spreading_config else True
        enable_cluster = getattr(spreading_config, "enable_cluster", True) if spreading_config else True
        return enable_local, enable_country, enable_cluster

    def set_tpb_transition_blocker(self):
        """Set transition_blocker to indicate which TPB component is most limiting.

        Uses a simple recursive approach:
        1. First identify which main component (attitude, social_norm, pbc) is lowest
        2. Then drill down into that component's sub-parts to identify the specific blocker

        Only compares enabled spreading levels (respects spreading_levels config).
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
                self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND
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
                        self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_LOCAL
                    elif min_social == 'country':
                        self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_COUNTRY
                    else:
                        self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_SOCIAL_CLUSTER
                else:
                    # Fallback if no social levels enabled (shouldn't happen)
                    self._transition_blocker = BLOCKER_TPB_LOW_ATTITUDE_OWN_LAND

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
                    self._transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_LOCAL
                elif min_norm == 'country':
                    self._transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_COUNTRY
                else:
                    self._transition_blocker = BLOCKER_TPB_LOW_SOCIAL_NORM_CLUSTER
            else:
                # Fallback if no social levels enabled (shouldn't happen)
                self._transition_blocker = BLOCKER_TPB_LOW_PBC

        else:  # pbc
            # PBC is driven by cost affordability (pbc_base captures AFT risk differences)
            self._transition_blocker = BLOCKER_TPB_LOW_PBC

    def set_tpb_component_driver(self, pathway: str):
        """Set transition_driver to indicate which TPB component enabled the transition.

        Uses a simple recursive approach (mirror of set_tpb_transition_blocker):
        1. First identify which main component (attitude, social_norm, pbc) is highest
        2. Then drill down into that component's sub-parts to identify the specific driver

        Only compares enabled spreading levels (respects spreading_levels config).

        Parameters
        ----------
        pathway : str
            One of "social" (learned from local neighbor), "country" (inspired
            by country-level data), or "exploration" (random exploration).
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
        driver_maps = {
            "social": {
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
        self._transition_driver = driver_map[sub_driver]

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
        # -----------------------------------------------------------------
        # Tillage: 0 = no-till (CA), 1 = conventional tillage
        # Directly maps to LPJmL's with_tillage
        # -----------------------------------------------------------------
        self.farmer.tillage = bundle.tillage

        # -----------------------------------------------------------------
        # Cover crop: 0 = none, 1/2 = type based on conditions
        # -----------------------------------------------------------------
        if bundle.cover_crop == 1:
            # Determine cover crop type based on environmental conditions
            # 1 = non-legume (catch crop), 2 = legume (N-fixing)
            self.farmer.cover_crop = self.farmer.indicate_cover_crop_type()
        else:
            self.farmer.cover_crop = 0

        # -----------------------------------------------------------------
        # Residue retention: 0 = baseline, 1 = retain to reach CA threshold
        # -----------------------------------------------------------------
        # CA threshold from config (default 30% soil cover per FAO definition)
        # If already at/above threshold, maintain current level
        # If below, increase retention to reach threshold (if affordable)
        ca_threshold = self.farmer.model.config.coupled_config.practice_dimensions.residue.ca_cover_threshold  # noqa: E501

        if bundle.residue_on_field == 1:
            # Want CA residue retention
            if self.farmer.litter_cover < ca_threshold:
                # Below threshold - need to increase retention if affordable
                opp_cost = self.farmer.residue_opportunity_cost
                can_afford = opp_cost <= 0 or self.farmer.capital >= opp_cost
                if can_afford:
                    self.farmer.residue_on_field = 1.0
            # If already at threshold, keep current residue_on_field unchanged
        else:
            # Not pursuing CA residue retention - use baseline
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

    def most_promising_bundle_country(self):
        """Find best-performing bundle at COUNTRY level (if better than current).

        Non-local social learning: when no local neighbour is better, farmers
        may look to successful practices used elsewhere in their country.
        Country stats are already merged with neighbouring countries (configured
        via neighbour_country_weight) at the country level.

        Uses cached country statistics for O(1) lookup.

        Returns
        -------
        tuple or None
            Best country-level bundle, or None if no bundle performs better
            than current practice or if country data is unavailable.
        """
        store = _country_performance_store(self.farmer.cell.country)
        if store is None:
            return None

        # Get own weighted score from current trend
        own_score = self.performance_tracker.weighted_score(self.farmer)

        # Find best performing bundle at country level
        best_bundle = None
        best_score = own_score  # Must beat our current performance

        for bundle, perf in store.items():
            # Skip our own bundle
            if bundle == self.practice_bundle:
                continue

            country_score = perf.weighted_trend_score(self.farmer)

            if country_score > best_score:
                best_score = country_score
                best_bundle = bundle

        return best_bundle

    def most_promising_bundle_cluster(self):
        """Find best-performing bundle at CLUSTER level (if better than current).

        Tele-coupled social learning: when no local or country-level neighbour
        is better, farmers may look to successful practices in agroecologically
        similar countries (same temperature, precipitation, PET patterns).

        Uses cached cluster statistics from world.statistic for O(1) lookup.

        Returns
        -------
        tuple or None
            Best cluster-level bundle, or None if no bundle performs better
            than current practice or if cluster data is unavailable.
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return None

        # Access via country._world which works on both driver and workers
        cluster_stats = self.farmer.cell.country._world.statistic.get(
            "cluster_management_performance", {}
        )
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return None

        # Get own weighted score from current trend
        own_score = self.performance_tracker.weighted_score(self.farmer)

        # Find best performing bundle at cluster level
        best_bundle = None
        best_score = own_score  # Must beat our current performance

        for bundle, perf in store.items():
            # Skip our own bundle
            if bundle == self.practice_bundle:
                continue

            cluster_score = perf.weighted_trend_score(self.farmer)

            if cluster_score > best_score:
                best_score = cluster_score
                best_bundle = bundle

        return best_bundle

    def is_better_performing(self, neighbour):
        """Check if neighbour has higher weighted score than self.

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        bool
            True if neighbour's score exceeds self's score.
        """
        neighbour_score = neighbour.behaviour.performance_tracker.weighted_score(
            self.farmer
        )
        own_score = self.performance_tracker.weighted_score(self.farmer)
        return neighbour_score > own_score

    def performance_gap(self, neighbour):
        """Compute positive performance difference (neighbour - self).

        Parameters
        ----------
        neighbour : Farmer
            neighbouring farmer to compare.

        Returns
        -------
        float
            Positive gap (0 if neighbour is worse).
        """
        neighbour_score = neighbour.behaviour.performance_tracker.weighted_score(
            self.farmer
        )
        own_score = self.performance_tracker.weighted_score(self.farmer)
        return max(0.0, neighbour_score - own_score)

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
            "cftfrac", drop_band=NON_CROPS
        ).values.flatten()
        cft_neighbour = neighbour.get_from_earth(
            "cftfrac", drop_band=NON_CROPS
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

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Own Land)
    # =========================================================================

    def compute_attitude_own_land(self):
        """Compute attitude from own land performance trends.

        "Am I doing poorly with my current practices?"
        Based on performance_tracker.trend (regression over all observations).
        Declining performance → high attitude → more willing to transition
        Improving performance → low attitude → less willing to transition

        Structure matches old tillage_farmer.py:
        - Weighted sum of individual trend components
        - Negate (so decline → positive)
        - Final sigmoid

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        trend = self.performance_tracker.trend
        raw_own = (
            self.farmer.weight_yield * (-trend["yield"])
            + self.farmer.weight_soil * (-trend["soilc"])
            + self.farmer.weight_moisture * (-trend["moisture"])
        )
        return sigmoid(raw_own)

    # =========================================================================
    # TPB COMPONENT: ATTITUDE (Social Learning)
    # =========================================================================

    def compute_attitude_social_learning_local(self, new_bundle):
        """Compute attitude from LOCAL neighbours using the proposed bundle.

        Evaluates neighbours based on observable outcome differences, weighted
        by similarity and confidence. Neighbours declining MORE than the farmer
        are filtered out (relative comparison).

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Homophily (McPherson et al. 2001): similarity-weighted learning
        - Adaptive learning (Boyd & Richerson 1985): filter out worse performers

        The relative slope comparison handles contexts like post-land-use-change
        (e.g., Paraguay) where ALL farmers are declining but CA declines slower
        than conventional. Using an absolute threshold would filter out everyone.

        This is the LOCAL component - uses direct neighbour comparisons with
        full similarity weighting. See compute_attitude_social_learning_country()
        for the country-level component.

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        if not self.farmer.neighbourhood:
            return 0.5  # Neutral without neighbours

        # Accumulate weighted comparisons
        weighted_yield = 0.0
        weighted_soil = 0.0
        weighted_moisture = 0.0
        total_weight = 0.0

        # Get parameters from config
        confidence_years = self.get_aft_param("confidence_years")

        # My current slope (for relative comparison)
        my_slope = self.performance_tracker.weighted_score(self.farmer)

        # My current absolute values (avoid division by zero)
        my_yield = max(self.farmer.cropyield, 1e-6)
        my_soil = max(self.farmer.soilc, 1e-6)
        my_moisture = max(self.farmer.root_moisture, 1e-6)

        for neighbour in self.farmer.neighbourhood:
            # How similar is neighbour? (bundle + crop similarity)
            similarity = self.total_similarity(new_bundle, neighbour)

            if similarity == 0:
                continue  # Skip completely different neighbours

            # -----------------------------------------------------------------
            # Filter: skip neighbours declining MORE than me (relative comparison)
            # -----------------------------------------------------------------
            # This handles degrading contexts (e.g., Paraguay post-land-use-change)
            # where everyone is declining but CA declines slower than conventional.
            # Neighbours performing BETTER than me (higher slope) are included.
            neighbour_slope = neighbour.behaviour.performance_tracker.weighted_score(
                self.farmer
            )
            if neighbour_slope < my_slope:
                continue  # Skip neighbour declining more than me

            # Confidence: more observations → more reliable information
            n_obs = neighbour.behaviour.performance_tracker.n
            confidence = min(1.0, n_obs / confidence_years)

            # -----------------------------------------------------------------
            # Absolute comparisons (ratio - 1)
            # -----------------------------------------------------------------
            # Positive if neighbour is better, negative if worse
            # Based on social comparison theory: farmers evaluate relative to self
            yield_cmp = neighbour.cropyield / my_yield - 1
            soil_cmp = neighbour.soilc / my_soil - 1
            moisture_cmp = neighbour.root_moisture / my_moisture - 1

            # Weight = similarity × confidence
            weight = similarity * confidence

            # Accumulate (no slope multiplication - keeps stable performers influential)
            weighted_yield += weight * yield_cmp
            weighted_soil += weight * soil_cmp
            weighted_moisture += weight * moisture_cmp
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no relevant neighbours

        # Normalize by total weight
        avg_yield = weighted_yield / total_weight
        avg_soil = weighted_soil / total_weight
        avg_moisture = weighted_moisture / total_weight

        # Weighted sum of comparisons (multi-attribute utility theory)
        raw_score = (
            self.farmer.weight_yield * avg_yield
            + self.farmer.weight_soil * avg_soil
            + self.farmer.weight_moisture * avg_moisture
        )

        # Sigmoid maps to (0, 1) attitude score
        return sigmoid(raw_score)

    def compute_attitude_social_learning_country(self, new_bundle):
        """Compute attitude from COUNTRY-LEVEL bundle performance.

        Uses cached country statistics to compare own performance against
        average performance of farmers using the proposed bundle across
        the entire country. Bundles with average slope worse than the farmer's
        own slope are filtered out (relative comparison).

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies

        The relative slope comparison handles degrading contexts where all practices
        decline but some decline slower than others.

        No similarity weighting at country level - uses simpler bundle-based
        grouping for computational efficiency (O(1) vs O(n²)).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        store = _country_performance_store(self.farmer.cell.country)
        if store is None:
            return 0.5  # Neutral if no country data

        perf = store.get(new_bundle)
        if perf.count == 0:
            return 0.5  # Neutral if no data for this bundle

        # -----------------------------------------------------------------
        # Filter: skip bundles performing worse than me (relative comparison)
        # -----------------------------------------------------------------
        # My slope vs bundle's average slope - handles degrading contexts
        my_slope = self.performance_tracker.weighted_score(self.farmer)
        bundle_slope = perf.weighted_trend_score(self.farmer)
        if bundle_slope < my_slope:
            return 0.5  # Neutral for bundles declining more than me

        # My current absolute values (avoid division by zero)
        my_yield = max(self.farmer.cropyield, 1e-6)
        my_soil = max(self.farmer.soilc, 1e-6)
        my_moisture = max(self.farmer.root_moisture, 1e-6)

        # Compare my performance to country average for this bundle
        avg_yield = perf.avg_yield
        avg_soilc = perf.avg_soilc
        avg_moisture = perf.avg_moisture

        # Absolute comparisons (ratio - 1)
        yield_cmp = avg_yield / my_yield - 1 if my_yield > 0 else 0.0
        soil_cmp = avg_soilc / my_soil - 1 if my_soil > 0 else 0.0
        moisture_cmp = avg_moisture / my_moisture - 1 if my_moisture > 0 else 0.0

        # Weighted sum of comparisons (no slope multiplication)
        raw_score = (
            self.farmer.weight_yield * yield_cmp
            + self.farmer.weight_soil * soil_cmp
            + self.farmer.weight_moisture * moisture_cmp
        )

        return sigmoid(raw_score)

    def compute_attitude_social_learning_cluster(self, new_bundle):
        """Compute attitude from AGROECOLOGICAL CLUSTER-level bundle performance.

        Cross-border social learning: farmers learn from countries with similar
        agroecological conditions. This captures diffusion of agricultural innovations
        across national boundaries within similar agro-ecological zones.
        Bundles with average slope worse than the farmer's own slope are filtered out.

        Scientific basis:
        - Social comparison theory (Festinger 1954): relative performance evaluation
        - Adaptive learning (Boyd & Richerson 1985): filter out worse strategies

        The relative slope comparison handles degrading contexts where all practices
        decline but some decline slower than others (e.g., Paraguay post-land-use-change).

        Uses aggregated cluster statistics from world.statistic (computed from
        peer countries in the same agroecological cluster).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Attitude score in [0, 1].
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return 0.5  # Neutral if no cluster

        # Access via country._world which works on both driver and workers
        cluster_stats = country._world.statistic.get("cluster_management_performance", {})
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return 0.5  # Neutral if no cluster data

        perf = store.get(new_bundle)
        if perf.count == 0:
            return 0.5  # Neutral if no data for this bundle

        # -----------------------------------------------------------------
        # Filter: skip bundles performing worse than me (relative comparison)
        # -----------------------------------------------------------------
        # My slope vs bundle's average slope - handles degrading contexts
        my_slope = self.performance_tracker.weighted_score(self.farmer)
        bundle_slope = perf.weighted_trend_score(self.farmer)
        if bundle_slope < my_slope:
            return 0.5  # Neutral for bundles declining more than me

        # My current absolute values (avoid division by zero)
        my_yield = max(self.farmer.cropyield, 1e-6)
        my_soil = max(self.farmer.soilc, 1e-6)
        my_moisture = max(self.farmer.root_moisture, 1e-6)

        # Get cluster average performance for this bundle
        avg_yield = perf.avg_yield
        avg_soilc = perf.avg_soilc
        avg_moisture = perf.avg_moisture

        # Absolute comparisons (ratio - 1)
        yield_cmp = avg_yield / my_yield - 1 if my_yield > 0 else 0.0
        soil_cmp = avg_soilc / my_soil - 1 if my_soil > 0 else 0.0
        moisture_cmp = avg_moisture / my_moisture - 1 if my_moisture > 0 else 0.0

        # Weighted sum of comparisons (no slope multiplication)
        raw_score = (
            self.farmer.weight_yield * yield_cmp
            + self.farmer.weight_soil * soil_cmp
            + self.farmer.weight_moisture * moisture_cmp
        )

        return sigmoid(raw_score)

    # =========================================================================
    # TPB COMPONENT: SOCIAL NORM
    # =========================================================================

    def compute_social_norm_local(self, new_bundle):
        """Compute social norm based on LOCAL neighbourhood practice distribution.

        Social norm reflects "what others are doing" (descriptive norm).
        Uses similarity-weighted average: higher if neighbours use similar
        bundles and grow similar crops.

        Follows the simple approach from tillage_farmer.py: fraction of
        neighbours using the practice → sigmoid transformation.

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

        # Average similarity to neighbours for this bundle
        # High similarity = neighbours use similar bundles and crops
        total_similarity = sum(
            self.total_similarity(new_bundle, n)
            for n in self.farmer.neighbourhood
        )
        avg_similarity = total_similarity / len(self.farmer.neighbourhood)

        # Shifted sigmoid: adoption threshold acts as the neutrality point
        # (Granovetter 1978 heterogeneous-threshold diffusion model).
        # Below threshold -> drag, above -> boost, smoothly transitioning.
        # Threshold is AFT-specific: pioneers feel "normed" at lower adoption,
        # traditionalists require broader local uptake to feel normative.
        threshold = self.get_aft_param("threshold_social_norm_local")
        return sigmoid(avg_similarity - threshold)

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
        store = _country_performance_store(self.farmer.cell.country)
        if store is None or store.total_farmers < 2:
            return 0.5  # Neutral if no country data

        bundle_count = store.bundle_counts.get(new_bundle, 0)
        total_farmers = store.total_farmers

        # Fraction of farmers using this bundle
        bundle_fraction = bundle_count / total_farmers

        # Shifted sigmoid: threshold = country-adoption level at which this
        # bundle feels normative (Granovetter 1978). Thresholds differ per
        # AFT (pioneers adopt the country signal earlier) and are typically
        # higher than the local threshold, because country adoption is more
        # abstract/statistical than direct observation of neighbours.
        threshold = self.get_aft_param("threshold_social_norm_country")
        return sigmoid(bundle_fraction - threshold)

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
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            Social norm score in [0, 1].
        """
        country = self.farmer.cell.country
        cluster_id = getattr(country, "agroecological_cluster", -1)

        if cluster_id < 0:
            return 0.5  # Neutral if no cluster

        # Access via country._world which works on both driver and workers
        cluster_stats = country._world.statistic.get("cluster_management_performance", {})
        store = cluster_stats.get(cluster_id)
        if not isinstance(store, RegionManagementPerformanceStore):
            return 0.5  # Neutral if no cluster data

        total_in_cluster = store.total_farmers
        bundle_adopters = store.bundle_counts.get(new_bundle, 0)

        if total_in_cluster == 0:
            return 0.5  # Neutral if no peer farmers

        # Adoption rate in cluster
        adoption_rate = bundle_adopters / total_in_cluster

        # Shifted sigmoid with AFT-specific threshold
        threshold = getattr(self.farmer, "threshold_social_norm_cluster", 0.1)
        return sigmoid(adoption_rate - threshold)

    # =========================================================================
    # COST CALCULATIONS
    # =========================================================================

    def get_bundle_direct_cost(self, bundle):
        """Compute annual direct cost of a practice bundle.

        Parameters
        ----------
        bundle : Bundle
            Practice bundle.

        Returns
        -------
        float
            Annual direct cost (scaled by farm size).
        """
        return bundle.direct_cost_per_ha(self.farmer.practice_costs) * self.farmer.net_farm_size

    def total_transition_cost(self, old_bundle, new_bundle):
        """Compute one-time transition cost for changing practices.

        Parameters
        ----------
        old_bundle : Bundle
            Current practice bundle.
        new_bundle : Bundle
            Target practice bundle.

        Returns
        -------
        float
            Total transition cost (scaled by farm size).
        """
        return (
            old_bundle.transition_cost_per_ha(new_bundle, self.farmer.practice_costs)
            * self.farmer.gross_farm_size
        )

    # =========================================================================
    # AFFORDABILITY ADJUSTMENTMENT BASED ON CAPITAL
    # =========================================================================

    def affordable_bundle(self, target_bundle):
        """Find affordable subset of target bundle.

        If farmer can't afford full target bundle, add changes cheapest-first
        to maximize what can be adopted within capital constraints.

        Parameters
        ----------
        target_bundle : Bundle
            Desired practice bundle.

        Returns
        -------
        Bundle
            Affordable bundle (may equal current if nothing affordable).
        """
        current = self.practice_bundle
        total_cost = self.total_transition_cost(current, target_bundle)

        # -----------------------------------------------------------------
        # Check if full target is affordable
        # -----------------------------------------------------------------
        if self.farmer.capital >= total_cost:
            return target_bundle

        # -----------------------------------------------------------------
        # Build affordable subset: add changes cheapest-first
        # -----------------------------------------------------------------
        changes = [
            (
                field,
                getattr(target_bundle, field),
                getattr(self.farmer.practice_costs, field).transition
                * self.farmer.gross_farm_size,
            )
            for field in PRACTICE_FIELDS
            if getattr(current, field) != getattr(target_bundle, field)
        ]

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

    def pbc_for_bundle(self, new_bundle):
        """Compute Perceived Behavioral Control for a bundle.

        PBC reflects "can I actually do this?" - lower when costs are
        high relative to available capital.

        Formula: PBC = pbc_base × cost_factor

        Where:
        - pbc_base: AFT-specific baseline (pioneers higher, traditionalists lower)
        - cost_factor: decreases as transition cost approaches disposable capital

        AFT differences in risk aversion are captured via pbc_base, not as a
        separate multiplicative factor (avoids double-counting).

        Parameters
        ----------
        new_bundle : tuple
            Bundle being evaluated.

        Returns
        -------
        float
            PBC score in [0, pbc_base].
        """
        # -----------------------------------------------------------------
        # Calculate cost impact of transitioning
        # -----------------------------------------------------------------

        # One-time transition cost
        transition_cost = self.total_transition_cost(self.practice_bundle, new_bundle)

        # Change in annual direct costs
        current_direct = self.get_bundle_direct_cost(self.practice_bundle)
        new_direct = self.get_bundle_direct_cost(new_bundle)
        direct_cost_increase = max(0, new_direct - current_direct)

        # Total cost impact
        cost_impact = transition_cost + direct_cost_increase

        # -----------------------------------------------------------------
        # Calculate disposable capital (above minimum threshold)
        # -----------------------------------------------------------------
        disposable = max(self.farmer.capital - self.farmer.min_capital, 1e-6)

        # -----------------------------------------------------------------
        # PBC decreases as cost approaches disposable capital
        # -----------------------------------------------------------------
        cost_factor = 1.0 / (1.0 + cost_impact / disposable)

        return self.farmer.pbc_base * cost_factor

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
        disabled via config (spreading_levels), and their weights are redistributed
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
        self._attitude_own_land = self.compute_attitude_own_land()

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
